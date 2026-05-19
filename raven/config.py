import os
from pathlib import Path

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PACKAGE_DIR / "data" / "corpus.txt"

DRIVE_CHECKPOINT_DIR = Path("/content/drive/MyDrive/raven_checkpoints")
LOCAL_CHECKPOINT_DIR = Path("/content/raven")

vocab_size = int(os.environ.get("RAVEN_VOCAB_SIZE", "8192"))
block_size = 256

n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

batch_size = 16
gradient_accumulation_steps = 4

max_iters = int(os.environ.get("RAVEN_MAX_ITERS", "15000"))
learning_rate = float(os.environ.get("RAVEN_LEARNING_RATE", "5e-4"))
min_lr_ratio = 0.1
warmup_iters = 500
weight_decay = 0.1
grad_clip = 1.0

train_split = 0.9
eval_interval = int(os.environ.get("RAVEN_EVAL_INTERVAL", "500"))
eval_iters = int(os.environ.get("RAVEN_EVAL_ITERS", "50"))
checkpoint_interval = int(os.environ.get("RAVEN_CHECKPOINT_INTERVAL", "500"))
log_interval = int(os.environ.get("RAVEN_LOG_INTERVAL", "50"))
save_latest_every_step = os.environ.get("RAVEN_SAVE_EVERY_STEP", "1") != "0"

token_chunk_size = 5 * 1024 * 1024
seed = int(os.environ.get("RAVEN_SEED", "1337"))

tokenizer_file = "raven_tokenizer.json"
weights_file = "raven_weights.pt"
latest_checkpoint_file = "raven_checkpoint_latest.pt"
best_checkpoint_file = "raven_checkpoint_best.pt"
token_cache_file = "raven_tokens.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"


def _dedupe_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def artifact_search_dirs():
    dirs = []
    env_dir = os.environ.get("RAVEN_CHECKPOINT_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    if Path("/content/drive/MyDrive").exists():
        dirs.append(DRIVE_CHECKPOINT_DIR)
    dirs.append(LOCAL_CHECKPOINT_DIR)
    dirs.append(PACKAGE_DIR)
    return _dedupe_paths(dirs)


def _is_writable_dir(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".raven_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def get_artifact_dir():
    for path in artifact_search_dirs():
        if _is_writable_dir(path):
            return str(path)
    return str(PACKAGE_DIR)


def artifact_path(filename):
    return str(Path(get_artifact_dir()) / filename)


def find_artifact(filename):
    for directory in artifact_search_dirs():
        path = directory / filename
        if path.exists():
            return str(path)
    return None


checkpoint_dir = get_artifact_dir()
