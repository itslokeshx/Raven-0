import torch

block_size = 128
batch_size = 64
max_iters = 20000
learning_rate = 3e-4
eval_interval = 500
eval_iters = 100
checkpoint_interval = 500
warmup_iters = 1000
min_lr_ratio = 0.1
weight_decay = 0.1
grad_clip = 1.0

n_embd = 64
n_head = 4
n_layer = 4

dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 65
