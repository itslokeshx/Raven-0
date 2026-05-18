import os
import sys
import torch

# Add the 'raven' root directory to sys.path so we can import config and core modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import config
from core.tokenizer import RavenTokenizer
from core.model import RavenModel

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

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    Evaluates the model on both train and validation sets without tracking gradients.
    Averages the loss over multiple batches to reduce noise.
    """
    out = {}
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    for split, data in [('train', train_data), ('val', val_data)]:
        eval_iters = 200
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
    
    # Setup the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"Starting training on device: {config.device}...")
    
    # 6. Run training loop for max_iters steps
    for iter in range(config.max_iters):
        
        # 7. Evaluate and print train/val loss every eval_interval steps
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {iter:4d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(train_data)

        # Forward pass: Evaluate the loss
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Update weights
        optimizer.step()

    print("Training complete.")

    # 8. Save the trained model weights to raven_weights.pt
    model_save_path = os.path.join(root_dir, 'raven_weights.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to: {model_save_path}")

    # 9. Save the tokenizer vocabulary to raven_vocab.json
    tokenizer_save_path = os.path.join(root_dir, 'raven_vocab.json')
    tokenizer.save(tokenizer_save_path)
    print(f"Tokenizer vocabulary saved to: {tokenizer_save_path}")

if __name__ == '__main__':
    train()
