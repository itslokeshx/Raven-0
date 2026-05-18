import torch

# RAVEN Hyperparameters

# block_size: How many characters the model can "see" at once to predict the next one (context window)
block_size = 128 

# batch_size: How many independent sequences will be processed in parallel during training
batch_size = 32

# max_iters: Total number of training steps to run
max_iters = 5000

# learning_rate: The step size for the optimizer to update model weights
learning_rate = 1e-3

# eval_interval: How often to evaluate the model on train/val sets and print the loss
eval_interval = 250

# n_embd: The size of the embedding vectors (how many dimensions represent one character)
n_embd = 64

# n_head: Number of attention heads (must divide n_embd evenly)
n_head = 4

# n_layer: Number of transformer blocks (layers of processing)
n_layer = 4

# dropout: Percentage of neurons to randomly turn off to prevent overfitting (0.0 to 1.0)
dropout = 0.0

# device: Auto-detect if a GPU is available, otherwise fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# vocab_size: Placeholder for vocabulary size (number of unique characters). 
# This will be updated by the tokenizer once it reads the text.
vocab_size = 65
