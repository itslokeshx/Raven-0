import os
import torch

# RAVEN v2 target: about 30M parameters on a free-tier Colab T4.
# Weight tying in the model keeps this near 30M instead of drifting toward 35M.
vocab_size = 8192
block_size = 256

n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

# T4-friendly: use micro-batches of 16 and accumulate to an effective batch of 64.
batch_size = 16
gradient_accumulation_steps = 4

max_iters = 15000
learning_rate = 5e-4
min_lr_ratio = 0.1
warmup_iters = 500
weight_decay = 0.1
grad_clip = 1.0

eval_interval = 500
eval_iters = 50
checkpoint_interval = 500

tokenizer_file = "raven_tokenizer.json"
weights_file = "raven_weights.pt"
latest_checkpoint_file = "raven_checkpoint_latest.pt"
best_checkpoint_file = "raven_checkpoint_best.pt"
token_cache_file = "raven_tokens.pt"

default_drive_dir = "/content/drive/MyDrive/raven_checkpoints"
if os.path.exists("/content/drive/MyDrive"):
    checkpoint_dir = os.environ.get("RAVEN_CHECKPOINT_DIR", default_drive_dir)
else:
    checkpoint_dir = os.environ.get("RAVEN_CHECKPOINT_DIR", "")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
