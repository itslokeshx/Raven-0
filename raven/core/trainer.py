import math
import os
import random
import sys

import torch

# Add the raven root directory to sys.path so Colab can run:
#   cd raven && python3 core/trainer.py
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer


def artifact_dir():
    path = config.checkpoint_dir or root_dir
    os.makedirs(path, exist_ok=True)
    return path


def artifact_path(filename):
    return os.path.join(artifact_dir(), filename)


def get_batch(data):
    ix = torch.randint(len(data) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    return x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)


def get_lr(iter_num):
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / config.warmup_iters

    if iter_num >= config.max_iters:
        return config.learning_rate * config.min_lr_ratio

    decay_ratio = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + coeff * (config.learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data)
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_training_state(path, model, optimizer, scaler, step, best_val_loss):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
        },
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def load_training_state(path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    return checkpoint


def load_or_train_tokenizer(corpus_path):
    tokenizer_file = artifact_path(config.tokenizer_file)
    if os.path.exists(tokenizer_file):
        tokenizer = RavenTokenizer.load(tokenizer_file)
        print(f"Loaded BPE tokenizer: {tokenizer_file}")
    else:
        print(f"Training BPE tokenizer with vocab_size={config.vocab_size}...")
        tokenizer = RavenTokenizer.train_from_file(corpus_path, config.vocab_size, tokenizer_file)
        print(f"Saved BPE tokenizer: {tokenizer_file}")

    config.vocab_size = tokenizer.vocab_size
    return tokenizer


def load_or_encode_tokens(corpus_path, tokenizer):
    token_cache = artifact_path(config.token_cache_file)
    corpus_stat = os.stat(corpus_path)
    expected_meta = {
        "corpus_size": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "vocab_size": tokenizer.vocab_size,
    }

    if os.path.exists(token_cache):
        print(f"Loading token cache: {token_cache}")
        cached = torch.load(token_cache, map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and cached.get("meta") == expected_meta:
            data = cached["tokens"]
        elif isinstance(cached, torch.Tensor):
            print("Old token cache format found; re-encoding.")
            data = None
        else:
            print("Token cache is stale; re-encoding.")
            data = None
    else:
        data = None

    if data is None:
        print("Encoding corpus with BPE tokenizer...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        dtype = torch.int32 if tokenizer.vocab_size <= 2**31 - 1 else torch.int64
        data = torch.tensor(ids, dtype=dtype)
        torch.save({"meta": expected_meta, "tokens": data}, token_cache)
        print(f"Saved token cache: {token_cache}")

    if len(data) <= config.block_size + 1:
        raise ValueError("Corpus is too small for the configured block_size.")

    print(f"Token count: {len(data):,}")
    return data.long()


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )


def train():
    corpus_path = os.path.join(root_dir, "data", "corpus.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Missing corpus: {corpus_path}\n"
            "Create raven/data/corpus.txt before training."
        )

    save_dir = artifact_dir()
    print(f"Artifacts will save to: {save_dir}")
    print(f"Device: {config.device} | AMP: {config.use_amp}")

    tokenizer = load_or_train_tokenizer(corpus_path)
    data = load_or_encode_tokens(corpus_path, tokenizer)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    model = RavenModel().to(config.device)
    print(f"Model parameters: {model.get_num_params() / 1_000_000:.2f}M")

    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    weights_path = artifact_path(config.weights_file)
    latest_checkpoint_path = artifact_path(config.latest_checkpoint_file)
    best_checkpoint_path = artifact_path(config.best_checkpoint_file)

    best_val_loss = float("inf")
    start_step = 0

    if os.path.exists(latest_checkpoint_path):
        checkpoint = load_training_state(latest_checkpoint_path, model, optimizer, scaler)
        start_step = checkpoint.get("step", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from {latest_checkpoint_path} at step {start_step}")
    elif os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=config.device, weights_only=True))
        print(f"Loaded existing weights without optimizer state: {weights_path}")

    last_step = start_step - 1
    print(
        "Training started. "
        f"micro_batch={config.batch_size}, "
        f"accumulation={config.gradient_accumulation_steps}, "
        f"effective_batch={config.batch_size * config.gradient_accumulation_steps}"
    )

    try:
        for step in range(start_step, config.max_iters):
            last_step = step
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for _ in range(config.gradient_accumulation_steps):
                xb, yb = get_batch(train_data)
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    _, loss = model(xb, yb)
                    loss = loss / config.gradient_accumulation_steps
                running_loss += loss.item()
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            should_eval = step % config.eval_interval == 0 or step == config.max_iters - 1
            should_checkpoint = step % config.checkpoint_interval == 0 or step == config.max_iters - 1

            if should_eval:
                losses = estimate_loss(model, train_data, val_data)
                print(
                    f"Step {step:5d}: "
                    f"Train Loss {losses['train']:.4f} | "
                    f"Val Loss {losses['val']:.4f} | "
                    f"LR {lr:.2e}"
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(model.state_dict(), weights_path)
                    save_training_state(best_checkpoint_path, model, optimizer, scaler, step, best_val_loss)
                    print(f"Saved new best model: {weights_path}")
            elif step % 50 == 0:
                print(f"Step {step:5d}: Loss {running_loss:.4f} | LR {lr:.2e}")

            if should_checkpoint:
                save_training_state(latest_checkpoint_path, model, optimizer, scaler, step, best_val_loss)
                torch.save(model.state_dict(), weights_path)
                print(f"Saved latest checkpoint: {latest_checkpoint_path}")

    except KeyboardInterrupt:
        save_training_state(latest_checkpoint_path, model, optimizer, scaler, last_step, best_val_loss)
        torch.save(model.state_dict(), weights_path)
        print(f"\nInterrupted. Saved latest checkpoint at step {last_step}: {latest_checkpoint_path}")
        return
    except Exception:
        save_training_state(latest_checkpoint_path, model, optimizer, scaler, last_step, best_val_loss)
        torch.save(model.state_dict(), weights_path)
        print(f"\nTraining failed. Saved latest checkpoint at step {last_step}: {latest_checkpoint_path}")
        raise

    print("Training complete.")
    print(f"Latest weights: {weights_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    train()
