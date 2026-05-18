import torch

# RAVEN v1 — 25M Parameter Config

# Context window
block_size = 256

# Sequences processed in parallel
batch_size = 64

# Total training steps
max_iters = 10000

# Learning rate
learning_rate = 3e-4

# Print loss every N steps
eval_interval = 500

# Embedding dimensions
n_embd = 512

# Attention heads
n_head = 8

# Transformer layers
n_layer = 8

# Dropout
dropout = 0.2

# Auto detect GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set by tokenizer at runtime
vocab_size = 65
