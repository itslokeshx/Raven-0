import os
import sys
import math
import torch
import random

# Add the 'raven' root directory to sys.path so we can import config and core modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import config
from core.tokenizer import RavenTokenizer
from core.model import RavenModel

latest_checkpoint_path = os.path.join(root_dir, 'raven_checkpoint_latest.pt')
best_checkpoint_path = os.path.join(root_dir, 'raven_checkpoint_best.pt')
weights_path = os.path.join(root_dir, 'raven_weights.pt')
tokenizer_path = os.path.join(root_dir, 'raven_vocab.json')

def save_training_state(path, model, optimizer, tokenizer, iteration, best_val_loss):
    """
    Saves everything needed to continue training after Colab disconnects,
    keyboard interrupts, or runtime restarts.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'best_val_loss': best_val_loss,
        'vocab_size': tokenizer.vocab_size,
        'config': {
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'dropout': config.dropout,
        },
        'torch_rng_state': torch.get_rng_state(),
        'python_rng_state': random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)

def load_training_state(path, model, optimizer):
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])

    return checkpoint

def get_batch(data):
    """
    Generates a small batch of data of inputs x and targets y.
    """
    # Generate random starting indices for the blocks
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # Stack the context sequences (x) and the target sequences (y)
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])

    # Move the tensors to the configured device (CPU/GPU)
    x, y = x.to(config.device), y.to(config.device)
    return x, y

def get_lr(iter_num):
    """
    Linear warmup followed by cosine decay. Small transformers are sensitive to
    a high fixed LR, especially on small personality datasets.
    """
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
    """
    Evaluates the model on both train and validation sets without tracking gradients.
    Averages the loss over multiple batches to reduce noise.
    """
    out = {}
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    for split, data in [('train', train_data), ('val', val_data)]:
        eval_iters = config.eval_iters
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set model back to training mode
    return out

def train():
    # 1. Load the corpus
    corpus_path = os.path.join(root_dir, 'data', 'corpus.txt')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded corpus. Total characters: {len(text)}")

    # 2. Initialize the tokenizer on the full text
    tokenizer = RavenTokenizer(text)

    # Dynamically update the vocab_size in config before initializing the model
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size learned: {config.vocab_size}")

    # Save vocabulary immediately. If training is interrupted later, the saved
    # weights/checkpoints still have the matching tokenizer available.
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer vocabulary saved to: {tokenizer_path}")

    # 3. Encode the entire corpus into a PyTorch tensor
    encoded_text = tokenizer.encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)

    # 4. Split into 90% train, 10% validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train data size: {len(train_data)} tokens")
    print(f"Validation data size: {len(val_data)} tokens")

    # Initialize the Raven Model and move it to the correct device
    model = RavenModel().to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    print(f"Starting training on device: {config.device}...")
    best_val_loss = float("inf")
    start_iter = 0

    if os.path.exists(latest_checkpoint_path):
        try:
            checkpoint = load_training_state(latest_checkpoint_path, model, optimizer)
            start_iter = checkpoint.get('iteration', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            print(f"Resumed from checkpoint: {latest_checkpoint_path} at step {start_iter}")
        except RuntimeError as e:
            print("Found a checkpoint, but it does not match the current model config.")
            print("Delete raven_checkpoint_latest.pt or restore the matching config to resume it.")
            raise e
    elif os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=config.device, weights_only=True))
            print(f"Loaded existing model weights from: {weights_path}")
            print("No optimizer checkpoint found, so training will continue with a fresh optimizer.")
        except RuntimeError as e:
            print("Found existing weights, but they do not match the current model config.")
            print("Delete raven_weights.pt or restore the matching config to reuse them.")
            raise e

    # 6. Run training loop for max_iters steps
    last_iter = start_iter - 1
    try:
        for iter in range(start_iter, config.max_iters):
            last_iter = iter

            # 7. Evaluate and print train/val loss every eval_interval steps
            if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
                losses = estimate_loss(model, train_data, val_data)
                print(f"Step {iter:5d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    torch.save(model.state_dict(), weights_path)
                    save_training_state(best_checkpoint_path, model, optimizer, tokenizer, iter, best_val_loss)
                    print(f"Saved best checkpoint with val loss {best_val_loss:.4f}")

            # Sample a batch of data
            xb, yb = get_batch(train_data)

            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass: Evaluate the loss
            logits, loss = model(xb, yb)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Update weights
            optimizer.step()

            if iter % config.checkpoint_interval == 0 or iter == config.max_iters - 1:
                save_training_state(latest_checkpoint_path, model, optimizer, tokenizer, iter, best_val_loss)
                torch.save(model.state_dict(), weights_path)
                print(f"Saved latest checkpoint at step {iter}")
    except KeyboardInterrupt:
        save_training_state(latest_checkpoint_path, model, optimizer, tokenizer, last_iter, best_val_loss)
        torch.save(model.state_dict(), weights_path)
        print(f"\nTraining interrupted. Saved recoverable checkpoint at step {last_iter}: {latest_checkpoint_path}")
        print(f"Latest model weights saved to: {weights_path}")
        return
    except Exception:
        save_training_state(latest_checkpoint_path, model, optimizer, tokenizer, last_iter, best_val_loss)
        torch.save(model.state_dict(), weights_path)
        print(f"\nTraining failed. Saved recoverable checkpoint at step {last_iter}: {latest_checkpoint_path}")
        print(f"Latest model weights saved to: {weights_path}")
        raise

    print("Training complete.")

    print(f"Latest model weights saved to: {weights_path}")
    print(f"Best full checkpoint saved to: {best_checkpoint_path}")

if __name__ == '__main__':
    train()
