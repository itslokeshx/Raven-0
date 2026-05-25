# RAVEN-0

RAVEN-0 is a 30 million parameter language model built from scratch using raw PyTorch.  
A learning project. A character experiment. An honest attempt to understand how language models actually work by building one from zero.

---

## Honest Overview

This is not a production AI. It is not competing with GPT or Gemini.

> [!WARNING]
> **PRE-TRAINING FAILURE & RANDOM OUTPUTS**
> In its current state, the model frequently throws completely random, garbled, or gibberish outputs for given inputs. It is crucial to document that this iteration failed to achieve general coherence due to two fundamental issues:
>
> 1. **Unclean Pre-Training Corpus:** The 520MB raw dataset used during pre-training was highly contaminated, uncurated, and unclean. The mix of classic literature, noisy dialogue templates, formatting artifacts, and unstructured prose caused the model to fit to raw noise rather than clean, structured language patterns.
> 2. **Character-Level / Byte-Level Limitations:** The model is extremely small (30M parameters) and tokenizes/generates at a character/byte level rather than a high-level word/subword BPE level. Character-level models have a much higher cognitive burden to learn spelling, grammar, and syntax. When trained on an unclean corpus, this character-level bottleneck causes the model's generation to easily degenerate into chaotic, random character sequences and gibberish.

RAVEN-0 is a tiny transformer trained on a free GPU in under 3 hours. It has a strong personality on emotional and psychological topics, and it hallucinates on everything else. That is expected, documented, and part of what makes it interesting as a learning project.

If you talk to RAVEN-0 about procrastination, self-deception, discipline, or ego — it will often say something sharp and on-character. If you ask it about math, history, or factual topics — it will confidently generate something that sounds plausible and is often wrong.

That is what a 30M parameter model trained on 520MB of mixed data does. Knowing why is the whole point.


---

## What RAVEN-0 Can Do

**Works well:**
- Psychological observations about human behavior
- Short, sharp responses on emotional topics
- Dark humor and dry sarcasm
- Procrastination, ego, self-deception, discipline, burnout
- Staying in character for 2-4 line responses

**Does not work reliably:**
- Math — will sometimes generate correct answers, often won't
- Science — will generate confident-sounding explanations that may be wrong
- History, geography, specific facts — hallucinates freely
- Long responses — drifts after 4-5 lines
- Consistent multi-turn memory — context window is 256 tokens

**Known issues:**
- Old English / literary text bleeds through occasionally (corpus contained classic literature)
- Responses sometimes drift into unrelated territory
- Repetition can occur in longer generations
- Math and factual queries intercepted by hardcoded logic in the web app to prevent pure hallucination

---

## Character

RAVEN-0 is a rogue intelligence system. Calm. Sarcastic. Psychologically observant. It does not apologize for existing. It does not pad responses with enthusiasm.

```
User: I keep saying I'll start tomorrow.

RAVEN-0:
Tomorrow is where ambition goes to die.
You've been its landlord for months now.
```

```
User: I feel stuck.

RAVEN-0:
Stuck is the story you tell
so you don't have to name
what you're avoiding.
```

```
User: I want to change.

RAVEN-0:
No you don't.
You want to have changed.
Different thing entirely.
```

Inspired by Batman's tactical observation, Tyler Durden's psychological rebellion, Tony Stark's dry sarcasm, and Thanos's calm conviction. Never a copy of any of them.

---

## Architecture

```
Raw Text Input
      ↓
BPE Tokenizer              ← byte-level, vocab 8,192
      ↓
Token Embedding            ← integers → 512-dim vectors
      ↓
Positional Embedding       ← position information added
      ↓
8× Transformer Block
  ├── LayerNorm
  ├── Multi-Head Attention  ← 8 heads × 64 dim each
  ├── Residual Connection
  ├── LayerNorm
  ├── FeedForward MLP       ← 512 → 2048 → 512
  └── Residual Connection
      ↓
Final LayerNorm
      ↓
LM Head (weight tied)      ← 512 → 8,192 logits
      ↓
Sampling                   ← temperature + top-k + top-p + repetition penalty
      ↓
Output token
```

| Setting | Value |
|---|---|
| Parameters | 29.53M |
| Vocabulary | 8,192 tokens (byte-level BPE) |
| Context window | 256 tokens (~150-200 words) |
| Embedding size | 512 |
| Attention heads | 8 |
| Layers | 8 |
| Feedforward size | 2,048 |
| Dropout | 0.1 |
| Weight tying | Yes |

---

## Training

**Hardware:** Google Colab T4 GPU (free tier, 16GB VRAM)  
**Framework:** PyTorch with mixed precision (AMP)  
**Optimizer:** AdamW (betas 0.9, 0.95, weight decay 0.1)  
**Schedule:** Cosine decay with linear warmup  
**Total time:** ~2.5 hours  

| Setting | Value |
|---|---|
| Steps | 15,000 |
| Micro batch size | 16 |
| Gradient accumulation | 4 |
| Effective batch size | 64 |
| Learning rate | 5e-4 → 5e-5 |
| Warmup steps | 500 |
| Gradient clip | 1.0 |

**Loss curve:**

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 315.09 | 320.36 |
| 1,000 | 1.07 | 8.40 |
| 3,000 | 0.76 | 6.32 |
| 6,000 | 0.66 | 5.45 |
| 10,000 | 0.55 | 5.03 |
| 15,000 | 0.56 | 4.88 |

Val loss dropped consistently from 320 to 4.88. Every evaluation checkpoint produced a new best model. The gap between train and val loss reflects the mixed corpus — the model learned the training distribution better than it generalized.

---

## Training Data

520MB of synthetic and curated text across five categories:

| Category | Size | Purpose |
|---|---|---|
| RAVEN-0 Dialogue | 300MB | Voice and personality conditioning |
| Psychological Observations | 80MB | Internal monologue and tone |
| Dark Humor and Sarcasm | 60MB | Comedic timing |
| Science Q&A | 40MB | Knowledge with personality overlay |
| Math Q&A | 40MB | Logical reasoning structure |

Total encoded: 154,587,911 tokens  
Train: 139,129,119 tokens  
Validation: 15,458,792 tokens

**Why the model hallucinates and generates random output:**  
1. **Unclean Pre-Training Corpus:** The 520MB mixed pre-training dataset was highly unclean and contaminated. Because it contained old literature, noisy web dumps, unstructured prose, and inconsistent conversational formats, the model learned raw noise patterns. Instead of generalizing to clean sentence structures, the model fit to random textual noise.
2. **Character-Level / Byte-Level Limitations:** Operating at a tiny, character-level (or byte-level) resolution with only 30M parameters, the model does not utilize a high-level semantic word vocabulary. Without an exceptionally clean and vast dataset to guide character-level transition patterns, the model fails to build stable semantic associations. As a result, when faced with standard user inputs, it easily drifts and outputs chaotic, random sequences of characters.

Additionally, the science and math sections taught RAVEN-0 only the *format* and *tone* of answering those questions, not the actual facts. A 30M parameter model cannot reliably store and retrieve factual knowledge from a small, noisy subset of examples. The ~43% of the corpus consisting of miscellaneous philosophy and literature also bleeds through constantly, producing lines of archaic, 19th-century prose rather than RAVEN-0's actual voice. This pre-training failure is the primary motivator for a future SFT (Supervised Fine-Tuning) run on a clean, curated conversational corpus.


---

## Mood Modes

| Mode | Character | Temp | Top-K | Top-P |
|---|---|---|---|---|
| `DEFAULT` | Calm, sharp, observant | 0.72 | 30 | 0.85 |
| `SAVAGE` | Brutally honest | 0.90 | 40 | 0.92 |
| `PHILOSOPHER` | Slow, existential | 0.62 | 22 | 0.80 |
| `CHAOS` | Unpredictable, dark humor | 1.05 | 60 | 0.95 |
| `NOX` | Minimal, cold | 0.48 | 12 | 0.75 |
| `DURDEN` | Anti-system, aggressive | 0.95 | 50 | 0.92 |

---

## Run Locally

```bash
git clone https://github.com/itslokeshx/Raven-0.git
cd Raven-0/raven
pip install torch tokenizers numpy
python3 main.py
```

Requires `raven_weights.pt` and `raven_tokenizer.json` in the `raven/` directory. Download from the Hugging Face Space files tab.

---

## File Structure

```
raven/
├── config.py                   ← hyperparameters
├── main.py                     ← terminal chat interface
├── core/
│   ├── model.py                ← transformer architecture
│   ├── tokenizer.py            ← byte-level BPE wrapper
│   └── trainer.py              ← training loop
├── data/
│   └── corpus.txt              ← training data (not committed)
└── personality/
    └── system_prompt.py        ← RAVEN-0's few-shot examples

Artifacts (not committed):
raven_weights.pt                ← 128MB trained weights
raven_tokenizer.json            ← 537KB vocabulary
raven_checkpoint_latest.pt      ← full training state for resume
raven_checkpoint_best.pt        ← best validation checkpoint
```

---

## Tech Stack

| | |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Tokenizer | HuggingFace tokenizers |
| Web interface | Gradio |
| Training GPU | Google Colab T4 (free) |
| Checkpoint storage | Google Drive |
| OS | Linux Mint |

---

## What This Project Is

RAVEN-0 is a learning project. It was built to understand — not to impress.

Building it required learning tokenization, embeddings, self-attention, multi-head attention, residual connections, layer normalization, feedforward networks, training loops, gradient accumulation, mixed precision, cosine scheduling, gradient clipping, checkpoint systems, temperature sampling, top-k and nucleus sampling. Every piece understood before it was used.

The output is imperfect. The hallucinations are real. The personality works better than expected for a 30M model. The book text bleeding is annoying and fixable. The math issue is fundamental to what a language model is.

All of that is worth knowing. That is why it was built this way.

---

## Roadmap

```
v1  →  pretraining done, character present, hallucinations expected  ← current
v2  →  SFT on pure RAVEN-0 dialogue, personality locked, book text removed
v3  →  larger context window, better memory, BPE upgrade
v4  →  tool use for math/science, reliable factual grounding
```

---

```
RAVEN-0 v1
The machine watches.
Built from scratch. Honest about its limits.
```