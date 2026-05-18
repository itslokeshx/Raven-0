# RAVEN

> Not an assistant. A presence.

RAVEN is a small language model project built from scratch with PyTorch. It is trained to speak as a sharp, sarcastic, psychologically observant, cinematic system: calm, dark, direct, and never corporate.

This repo is currently aimed at a **30M parameter local model** trained on a RAVEN-heavy dataset using a small byte-level BPE tokenizer.

---

## Current Target

| Area | Current Setting |
| --- | --- |
| Model size | ~30M parameters |
| Tokenizer | Byte-level BPE |
| Vocabulary | 8,192 tokens |
| Context window | 256 tokens |
| Embedding size | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Training target | Google Colab Tesla T4 |
| Checkpoint safety | Latest + best checkpoints |

RAVEN is not using pretrained weights. The model, tokenizer, trainer, and chat runtime are part of this codebase.

---

## Project Structure

```text
raven/
├── config.py
├── main.py
├── core/
│   ├── model.py
│   ├── tokenizer.py
│   └── trainer.py
├── data/
│   ├── books/              # local raw dataset parts, ignored by git
│   ├── parts/              # optional curated parts, .txt ignored by git
│   ├── build_corpus.py
│   └── corpus.txt          # final training corpus, ignored by git
└── personality/
    └── system_prompt.py
```

Generated training artifacts are also ignored by git:

- `raven/raven_weights.pt`
- `raven/raven_tokenizer.json`
- `raven/raven_checkpoint_latest.pt`
- `raven/raven_checkpoint_best.pt`
- `raven/raven_tokens.pt`

Do not commit datasets, weights, token caches, or checkpoints. GitHub rejects files over 100 MB, and model/data files belong in Drive or local storage.

---

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Check the code imports:

```bash
python3 -m py_compile raven/config.py raven/core/model.py raven/core/tokenizer.py raven/core/trainer.py raven/main.py
```

---

## Dataset

The trainer reads one file:

```text
raven/data/corpus.txt
```

If you already have many `.txt` files inside `raven/data/books/`, combine them into the final corpus:

```bash
output="raven/data/raven-1-dataset.txt"
: > "$output"
while IFS= read -r -d '' file; do
  printf '\n\n' >> "$output"
  cat "$file" >> "$output"
  printf '\n' >> "$output"
done < <(find raven/data/books -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)

ln -f raven/data/raven-1-dataset.txt raven/data/corpus.txt
```

That makes `corpus.txt` point to the same data without wasting another copy of disk space.

For the best RAVEN behavior, keep the dataset weighted toward:

| Data Type | Target Share |
| --- | ---: |
| RAVEN dialogue | 60% |
| Psychology / behavior observations | 10% |
| Dark humor / sarcasm | 10% |
| Basic math, science, and facts | 15% |
| Philosophy | 5% |

Clean data matters more than raw size. A smaller sharp corpus will beat a huge noisy one.

---

## Colab Training

Use a GPU runtime:

```text
Runtime -> Change runtime type -> T4 GPU
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone the repo:

```bash
git clone https://github.com/itslokeshx/Raven.git
cd Raven
pip install -r requirements.txt
```

Upload or copy your corpus to:

```text
Raven/raven/data/corpus.txt
```

Start training:

```bash
cd raven
python3 core/trainer.py
```

When Drive is mounted, checkpoints save automatically to:

```text
/content/drive/MyDrive/raven_checkpoints/
```

The important files are:

| File | Purpose |
| --- | --- |
| `raven_tokenizer.json` | BPE tokenizer used by training and chat |
| `raven_weights.pt` | latest model weights for `main.py` |
| `raven_checkpoint_latest.pt` | resumable training checkpoint |
| `raven_checkpoint_best.pt` | best validation checkpoint |
| `raven_tokens.pt` | cached tokenized corpus |

If Colab disconnects, run the same trainer command again. It will resume from `raven_checkpoint_latest.pt` when that file exists.

---

## Run RAVEN Locally

After training, place these files in `raven/` or keep them in the configured checkpoint directory:

```text
raven/raven_tokenizer.json
raven/raven_weights.pt
```

Run:

```bash
cd raven
python3 main.py
```

The chat runtime includes:

- cinematic boot sequence
- mode commands
- 3-turn short-term conversation history
- temperature and top-k sampling
- response trimming to keep replies short
- graceful `/exit` and Ctrl+C handling

Useful commands:

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

---

## Training Notes

Current config is tuned for a T4:

```text
batch_size = 16
gradient_accumulation_steps = 4
effective batch = 64
max_iters = 15000
learning_rate = 5e-4
eval_interval = 500
checkpoint_interval = 500
```

If Colab runs out of memory, lower `batch_size` to `8` and raise `gradient_accumulation_steps` to `8`.

If training is too slow, reduce `max_iters` for a test run first. Do not judge model quality from a tiny test run; use it only to verify the pipeline.

---

## Git Hygiene

Before pushing, check that no dataset or model files are staged:

```bash
git status --short
git diff --cached --name-only
```

If a large data file is staged, unstage it:

```bash
git restore --staged raven/data/corpus.txt
git restore --staged raven/data/books/
git restore --staged raven/*.pt
git restore --staged raven/*.json
```

Keep code in Git. Keep data and weights outside Git.

---

## Next Steps

1. Finish cleaning the dataset mix so RAVEN dialogue dominates.
2. Train the 30M BPE model on Colab T4.
3. Download or copy `raven_tokenizer.json` and `raven_weights.pt`.
4. Run `python3 main.py` locally and test the mood modes.
5. Save good sample conversations and use them to improve the next dataset.
6. For v2.1, add a small evaluation script for prompt quality, repetition, and format drift.

---

## Scope

RAVEN is currently a local text-generation model. It has no web access, no tools, no memory database, no voice, and no external API dependency.

The goal is simple: build a small model from scratch that speaks with a distinct identity and can be understood at every layer of the stack.
