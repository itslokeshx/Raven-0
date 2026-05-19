# RAVEN

RAVEN is a tiny character-driven language model built from scratch with PyTorch.
The target voice is dark, sarcastic, psychologically sharp, and concise.

## Target

| Setting | Value |
| --- | --- |
| Parameters | about 30M |
| Tokenizer | HuggingFace byte-level BPE |
| Vocabulary | 8,192 |
| Context | 256 tokens |
| Embedding | 512 |
| Heads | 8 |
| Layers | 8 |
| Dropout | 0.1 |
| Training target | Google Colab free tier T4 |

The model uses tied token embedding and language-model head weights.

## Files

```text
raven/
  config.py
  main.py
  core/
    model.py
    tokenizer.py
    trainer.py
  data/
    build_corpus.py
    corpus.txt
    parts/
  personality/
    system_prompt.py
```

Generated artifacts are not committed:

- `raven_tokenizer.json`
- `raven_tokens.pt`
- `raven_weights.pt`
- `raven_checkpoint_latest.pt`
- `raven_checkpoint_best.pt`

## Install

```bash
pip install -r requirements.txt
```

On Colab, choose `Runtime -> Change runtime type -> T4 GPU`.

## Checkpoint Location

The trainer saves tokenizer, token cache, weights, and checkpoints to:

```text
/content/drive/MyDrive/raven_checkpoints/
```

If Google Drive is not mounted or is not writable, it falls back to:

```text
/content/raven/
```

For local development, set `RAVEN_CHECKPOINT_DIR` if you want another artifact
directory.

## Dataset

The trainer reads:

```text
raven/data/corpus.txt
```

You can assemble it from curated parts:

```bash
python3 raven/data/build_corpus.py
```

For Colab, copy or upload your final corpus to:

```text
Raven/raven/data/corpus.txt
```

## Train

```bash
cd raven
python3 core/trainer.py
```

Training behavior:

- trains or loads `raven_tokenizer.json`
- encodes the corpus in 5MB byte chunks
- saves `raven_tokens.pt` and reuses it on later runs
- invalidates the token cache when corpus size or modification time changes
- uses AMP on CUDA and full precision on CPU
- uses batch size 16 with gradient accumulation 4
- uses warmup plus cosine learning-rate decay
- clips gradients at 1.0
- writes atomic latest checkpoints for resume safety

By default, `raven_checkpoint_latest.pt` is saved after every optimizer step.
This is the safest resume mode. To reduce Drive writes, set:

```bash
export RAVEN_SAVE_EVERY_STEP=0
```

Then the trainer still saves at `checkpoint_interval`, at best validation loss,
on clean interrupt, and on exceptions.

## Resume

Run the same command again:

```bash
cd raven
python3 core/trainer.py
```

If `raven_checkpoint_latest.pt` exists, optimizer state, AMP scaler state, model
state, Python RNG state, and PyTorch RNG state are restored.

## Chat

After training:

```bash
cd raven
python3 main.py
```

`main.py` loads `raven_weights.pt` when available. If weights are missing, it can
load the model state from `raven_checkpoint_latest.pt`.

Commands:

```text
/help
/status
/mode default
/mode savage
/mode philosopher
/mode chaos
/mode nox
/mode durden
/clear
/exit
```

## Quick Checks

```bash
python3 -m py_compile \
  raven/config.py \
  raven/core/model.py \
  raven/core/tokenizer.py \
  raven/core/trainer.py \
  raven/main.py \
  raven/data/build_corpus.py \
  raven/personality/system_prompt.py
```

```bash
cd raven
python3 - <<'PY'
import config
from core.model import RavenModel

model = RavenModel()
print(model.get_num_params())
PY
```
