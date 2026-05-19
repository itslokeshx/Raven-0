import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config
from core.model import RavenModel
from core.tokenizer import RavenTokenizer, iter_text_chunks_with_sizes


def artifact_dir():
    return Path(config.get_artifact_dir())


def unique_dirs(paths):
    seen = set()
    result = []
    for path in paths:
        path = Path(path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_artifact(obj, filename, label):
    errors = []
    dirs = unique_dirs([artifact_dir(), *config.artifact_search_dirs()])
    for directory in dirs:
        path = directory / filename
        try:
            atomic_torch_save(obj, path)
            config.checkpoint_dir = str(directory)
            return path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    message = "\n".join(errors) if errors else "no writable artifact directories found"
    raise RuntimeError(f"Could not save {label}:\n{message}")


def atomic_copy_file(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(f"{destination_path.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, destination_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_file_artifact(source_path, filename, label):
    errors = []
    dirs = unique_dirs([artifact_dir(), *config.artifact_search_dirs()])
    for directory in dirs:
        path = directory / filename
        try:
            atomic_copy_file(source_path, path)
            config.checkpoint_dir = str(directory)
            return path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    message = "\n".join(errors) if errors else "no writable artifact directories found"
    raise RuntimeError(f"Could not save {label}:\n{message}")


def torch_load(path, map_location=None, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_reproducible_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_batch(data):
    upper = len(data) - config.block_size - 1
    if upper <= 0:
        raise ValueError("Token data is too small for the configured block_size.")

    ix = torch.randint(0, upper, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    non_blocking = config.device == "cuda"
    return (
        x.to(config.device, dtype=torch.long, non_blocking=non_blocking),
        y.to(config.device, dtype=torch.long, non_blocking=non_blocking),
    )


def get_lr(iter_num):
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / max(1, config.warmup_iters)

    if iter_num >= config.max_iters:
        return config.learning_rate * config.min_lr_ratio

    decay_iters = max(1, config.max_iters - config.warmup_iters)
    decay_ratio = min(1.0, (iter_num - config.warmup_iters) / decay_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + coeff * (config.learning_rate - min_lr)


def autocast_context():
    return torch.cuda.amp.autocast(enabled=config.use_amp)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.empty(config.eval_iters)
        for index in range(config.eval_iters):
            x, y = get_batch(data)
            with autocast_context():
                _, loss = model(x, y)
            losses[index] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def training_config_snapshot():
    return {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_iters": config.warmup_iters,
        "min_lr_ratio": config.min_lr_ratio,
    }


def build_training_state(model, optimizer, scaler, step, best_val_loss):
    state = {
        "format_version": 2,
        "step": int(step),
        "next_step": int(step) + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "config": training_config_snapshot(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "saved_at_unix": time.time(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def save_training_state(filename, model, optimizer, scaler, step, best_val_loss):
    state = build_training_state(model, optimizer, scaler, step, best_val_loss)
    return save_artifact(state, filename, "training checkpoint")


def save_model_weights(filename, model):
    return save_artifact(model.state_dict(), filename, "model weights")


def load_training_state(path, model, optimizer, scaler):
    checkpoint = torch_load(path, map_location=config.device, weights_only=False)
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
    tokenizer_path = config.find_artifact(config.tokenizer_file)
    if tokenizer_path:
        tokenizer = RavenTokenizer.load(tokenizer_path)
        print(f"Loaded BPE tokenizer: {tokenizer_path}")
    else:
        temp_path = Path("/tmp") / f"{config.tokenizer_file}.train.{os.getpid()}"
        print(f"Training byte-level BPE tokenizer with vocab_size={config.vocab_size}...")
        try:
            tokenizer = RavenTokenizer.train_from_file(
                corpus_path,
                config.vocab_size,
                temp_path,
                chunk_size=config.token_chunk_size,
            )
            save_path = save_file_artifact(temp_path, config.tokenizer_file, "tokenizer")
        finally:
            temp_path.unlink(missing_ok=True)
        print(f"Saved BPE tokenizer: {save_path}")

    config.vocab_size = tokenizer.vocab_size
    return tokenizer


def corpus_meta(corpus_path, tokenizer):
    stat = os.stat(corpus_path)
    return {
        "corpus_size": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "vocab_size": tokenizer.vocab_size,
    }


def compact_token_dtype(vocab_size):
    if vocab_size <= torch.iinfo(torch.int16).max:
        return torch.int16
    if vocab_size <= torch.iinfo(torch.int32).max:
        return torch.int32
    return torch.int64


def load_valid_token_cache(token_cache, expected_meta):
    if not token_cache or not os.path.exists(token_cache):
        return None

    print(f"Loading token cache: {token_cache}")
    try:
        cached = torch_load(token_cache, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Token cache unreadable; re-encoding. Details: {exc}")
        return None

    if isinstance(cached, dict) and cached.get("meta") == expected_meta and "tokens" in cached:
        data = cached["tokens"].cpu()
        print(f"Token count: {len(data):,}")
        return data

    print("Token cache stale; re-encoding.")
    return None


def encode_corpus_in_chunks(corpus_path, tokenizer):
    print("Encoding corpus in 5MB chunks to save RAM...")
    total_size = os.path.getsize(corpus_path)
    processed = 0
    chunks = []
    dtype = compact_token_dtype(tokenizer.vocab_size)

    for text, bytes_read in iter_text_chunks_with_sizes(corpus_path, config.token_chunk_size):
        processed += bytes_read
        ids = tokenizer.encode(text)
        if ids:
            chunks.append(torch.tensor(ids, dtype=dtype))
        print(f"  Encoded {processed / 1e6:.1f}MB / {total_size / 1e6:.1f}MB", end="\r")

    print()
    if not chunks:
        return torch.empty(0, dtype=dtype)

    data = torch.cat(chunks)
    print(f"Total tokens: {len(data):,}")
    return data


def load_or_encode_tokens(corpus_path, tokenizer):
    expected_meta = corpus_meta(corpus_path, tokenizer)
    token_cache_path = config.find_artifact(config.token_cache_file)
    data = load_valid_token_cache(token_cache_path, expected_meta)
    if data is not None:
        if len(data) <= config.block_size + 1:
            raise ValueError("Corpus too small for block_size.")
        return data

    data = encode_corpus_in_chunks(corpus_path, tokenizer)
    if len(data) <= config.block_size + 1:
        raise ValueError("Corpus too small for block_size.")

    cache_path = save_artifact(
        {"meta": expected_meta, "tokens": data},
        config.token_cache_file,
        "token cache",
    )
    print(f"Saved token cache: {cache_path}")
    return data


def build_optimizer(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )


def split_data(data):
    min_split = config.block_size + 2
    if len(data) <= 2 * min_split:
        raise ValueError("Corpus is too small to create train and validation splits.")

    split_index = int(config.train_split * len(data))
    split_index = max(min_split, split_index)
    split_index = min(split_index, len(data) - min_split)
    return data[:split_index], data[split_index:]


def find_latest_checkpoint():
    return config.find_artifact(config.latest_checkpoint_file)


def find_weights():
    return config.find_artifact(config.weights_file)


def install_signal_handlers():
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, interrupt)
        except (OSError, ValueError):
            pass


def train():
    configure_torch_runtime()
    set_reproducible_seed(config.seed)
    install_signal_handlers()

    corpus_path = Path(os.environ.get("RAVEN_CORPUS_PATH", str(config.CORPUS_PATH)))
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing corpus: {corpus_path}\n"
            "Create raven/data/corpus.txt or set RAVEN_CORPUS_PATH."
        )

    save_dir = artifact_dir()
    print(f"Artifacts will save to: {save_dir}")
    print(f"Device: {config.device} | AMP: {config.use_amp}")

    tokenizer = load_or_train_tokenizer(corpus_path)
    data = load_or_encode_tokens(corpus_path, tokenizer)
    train_data, val_data = split_data(data)

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    model = RavenModel().to(config.device)
    print(f"Model parameters: {model.get_num_params() / 1_000_000:.2f}M")

    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    best_val_loss = float("inf")
    start_step = 0
    last_completed_step = -1

    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = load_training_state(latest_checkpoint, model, optimizer, scaler)
        start_step = int(checkpoint.get("next_step", checkpoint.get("step", -1) + 1))
        last_completed_step = start_step - 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed from {latest_checkpoint} at step {start_step}")
    else:
        weights_path = find_weights()
        if weights_path:
            state_dict = torch_load(weights_path, map_location=config.device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded existing weights without optimizer state: {weights_path}")

    print(
        "Training started. "
        f"micro_batch={config.batch_size}, "
        f"accumulation={config.gradient_accumulation_steps}, "
        f"effective_batch={config.batch_size * config.gradient_accumulation_steps}, "
        f"save_every_step={config.save_latest_every_step}"
    )

    try:
        for step in range(start_step, config.max_iters):
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            for _ in range(config.gradient_accumulation_steps):
                x, y = get_batch(train_data)
                with autocast_context():
                    _, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps
                running_loss += loss.item()
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            last_completed_step = step

            if config.save_latest_every_step:
                save_training_state(
                    config.latest_checkpoint_file,
                    model,
                    optimizer,
                    scaler,
                    last_completed_step,
                    best_val_loss,
                )

            should_eval = step % config.eval_interval == 0 or step == config.max_iters - 1
            should_snapshot = step % config.checkpoint_interval == 0 or step == config.max_iters - 1

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
                    weights_path = save_model_weights(config.weights_file, model)
                    best_path = save_training_state(
                        config.best_checkpoint_file,
                        model,
                        optimizer,
                        scaler,
                        last_completed_step,
                        best_val_loss,
                    )
                    print(f"Saved new best weights: {weights_path}")
                    print(f"Saved best checkpoint: {best_path}")
            elif step % config.log_interval == 0:
                print(f"Step {step:5d}: Loss {running_loss:.4f} | LR {lr:.2e}")

            if should_snapshot:
                latest_path = save_training_state(
                    config.latest_checkpoint_file,
                    model,
                    optimizer,
                    scaler,
                    last_completed_step,
                    best_val_loss,
                )
                weights_path = save_model_weights(config.weights_file, model)
                print(f"Saved latest checkpoint: {latest_path}")
                print(f"Saved latest weights: {weights_path}")

    except KeyboardInterrupt as exc:
        latest_path = save_training_state(
            config.latest_checkpoint_file,
            model,
            optimizer,
            scaler,
            last_completed_step,
            best_val_loss,
        )
        weights_path = save_model_weights(config.weights_file, model)
        print(f"\nInterrupted ({exc}).")
        print(f"Saved latest checkpoint at step {last_completed_step}: {latest_path}")
        print(f"Saved latest weights: {weights_path}")
        return
    except Exception:
        latest_path = save_training_state(
            config.latest_checkpoint_file,
            model,
            optimizer,
            scaler,
            last_completed_step,
            best_val_loss,
        )
        weights_path = save_model_weights(config.weights_file, model)
        print(f"\nTraining failed. Saved checkpoint at step {last_completed_step}: {latest_path}")
        print(f"Saved latest weights: {weights_path}")
        raise

    latest_path = save_training_state(
        config.latest_checkpoint_file,
        model,
        optimizer,
        scaler,
        last_completed_step,
        best_val_loss,
    )
    weights_path = save_model_weights(config.weights_file, model)
    print("Training complete.")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Latest weights: {weights_path}")


if __name__ == "__main__":
    train()
