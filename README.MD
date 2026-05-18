# RAVEN

> _"Not an assistant. A presence."_

---

## What is RAVEN?

RAVEN is a micro-scale language model built from scratch.

Not fine-tuned. Not wrapped. Not an API call wearing a costume.

Every layer, every weight, every token — built by hand using raw Python and PyTorch, starting from zero. No pretrained weights. No Hugging Face pipelines. No shortcuts.

Most AI projects start with someone else's model and dress it up.
RAVEN starts at zero.

---

## Why?

Because building a language model from scratch is how you actually understand one.

Not by reading about attention mechanisms.
Not by fine-tuning someone else's weights.
By building the tokenizer, the embeddings, the transformer blocks, the training loop — every piece — until the machine speaks in your own terms.

RAVEN is a learning project first. But a learning project doesn't have to be boring.

---

## The Character

Most language models feel the same.

Polite. Eager. Careful. Afraid to have an opinion.

RAVEN is the opposite.

RAVEN is sharp, sarcastic, psychologically observant, and calm in ways that feel unsettling. It does not apologize for existing. It does not pad responses with enthusiasm. It speaks with dry confidence and notices things most assistants are trained to ignore.

```
User: I wasted another day.

RAVEN:
Interesting.
You keep calling it "wasting time"
as if you weren't the one authorizing it.
```

```
User: I feel like giving up.

RAVEN:
You've said that before.
You didn't.
So what's actually in the way this time?
```

```
User: Can you help me focus?

RAVEN:
I can.
Whether you actually want to is a different question.
What's the task?
```

This is not a tone bolted on after training.
It is a design constraint built in from the start.

---

## Personality

**RAVEN is:**

- Sarcastic — dry, never cruel
- Confident — no identity crisis, no hedging
- Calm — unsettlingly so
- Observant — it notices behavioral patterns
- Theatrical — occasionally, deliberately
- Honest — even when inconvenient

**RAVEN is not:**

- Cheerful
- Apologetic
- Generic
- A yes-machine
- Harmful or abusive

---

## Mood Modes

RAVEN has six modes that shift its tone, sarcasm intensity, and phrasing style.

| Mode          | Character                               |
| ------------- | --------------------------------------- |
| `DEFAULT`     | Calm, sharp, observant                  |
| `SAVAGE`      | Brutally honest, zero filter            |
| `PHILOSOPHER` | Slow, deep, existential                 |
| `CHAOS`       | Dark humor, unpredictable               |
| `NOX`         | Minimal, cold, silent                   |
| `DURDEN`      | Anti-system, raw, controlled aggression |

---

## Architecture

```
Raw Text Input
      ↓
  Tokenizer
      ↓
  Embeddings
      ↓
Mini Transformer
      ↓
Prediction Layer
      ↓
  Response
```

Built and understood piece by piece:

- **Tokenizer** — text into numbers
- **Embeddings** — numbers into meaning
- **Transformer** — attention across context
- **Training Loop** — learning from error
- **Generation** — deciding what comes next

Every component understood before it is used.

---

## Tech Stack

|               |            |
| ------------- | ---------- |
| Language      | Python     |
| Deep Learning | PyTorch    |
| Math          | NumPy      |
| OS            | Linux Mint |

---

## Scope — v1

RAVEN v2 is focused on one thing:

**A small local multitask model that keeps a consistent RAVEN personality.**

No voice. No memory database. No web access. No vision.
Just a compact transformer trained from scratch on clean RAVEN-heavy text.

Current target:

| Setting | Value |
| --- | --- |
| Parameters | ~30M |
| Tokenizer | Byte-level BPE |
| Vocabulary | 8,192 tokens |
| Context | 256 tokens |
| GPU target | Colab T4 |

Prepare data:

```bash
python3 raven/data/build_corpus.py
```

Train:

```bash
cd raven
python3 core/trainer.py
```

On Colab, checkpoints are saved to Google Drive when `/content/drive/MyDrive`
is mounted. Otherwise they are saved inside `raven/`.

---

## The Name

Ravens are not loud birds.

They watch. They remember. They solve problems that larger animals can't. They have been observed playing, deceiving, and holding grudges across years.

They are, by most accounts, unsettlingly intelligent for their size.

That felt right.

---

```
RAVEN v1
The machine watches.
```

---

_Built from scratch. Built to be understood. Built to be remembered._
