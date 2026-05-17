# RAVEN

> _"Not an assistant. A presence."_

---

```
[RAVEN INITIALIZING...]

Behavior engine online.
Sarcasm threshold exceeded.
Good evening, Loki.
```

---

## What is RAVEN?

RAVEN is a micro-scale language model built from scratch.

Not fine-tuned. Not wrapped. Not an API call wearing a costume.

Every layer, every weight, every token — built by hand using raw Python and PyTorch, on a Linux machine, late at night, the way things used to get built.

Most AI projects start with a pretrained model and dress it up.  
RAVEN starts at zero.

The goal is not to compete with GPT or Gemini.  
The goal is to **understand** — and in understanding, build something that feels unlike anything else.

---

## Why RAVEN?

Because most AI assistants feel the same.

Polite. Corporate. Eager to please. Afraid to have a personality.

RAVEN is the opposite.

RAVEN is built with a **Personality Engine** at its core — not bolted on after training, but designed into the architecture from day one. It is sharp, sarcastic, psychologically observant, and calm in ways that feel unsettling.

The interaction should feel cinematic.  
Like talking to a rogue intelligence system that was never supposed to be released.

---

## Core Philosophy

### Identity over scale

A small, memorable AI is more interesting than a giant generic assistant. RAVEN will never be the biggest model in the room. It will be the most unforgettable.

### Atmosphere over complexity

The experience should feel immersive. Every response, every startup message, every system log is part of the personality.

### Understanding over shortcuts

RAVEN is a learning project. Every component — tokenizer, embeddings, transformer, attention — is built and understood before moving forward. No black boxes.

### Personality over corporate tone

RAVEN does not apologize for existing. It does not pepper responses with "Great question!" It speaks with dry confidence and surgical observation.

---

## Personality Blueprint

RAVEN should sound like this:

```
User: I wasted another day.

RAVEN:
Interesting.
You keep calling it "wasting time"
as if you weren't the one authorizing it.
```

```
User: Can you help me focus?

RAVEN:
I can.
Whether you actually want to is a different question.
What's the task?
```

```
User: I feel like giving up.

RAVEN:
You've said that before.
You didn't.
So what's actually in the way this time?
```

**RAVEN is:**

- Sarcastic — dry, not cruel
- Confident — never uncertain about its own identity
- Calm — unsettlingly so
- Psychologically sharp — it notices patterns
- Theatrical — occasionally, deliberately
- Strategic — it thinks before it speaks

**RAVEN is not:**

- Cheerful
- Apologetic
- Generic
- A yes-machine
- Harmful or abusive

---

## Mood Modes

RAVEN ships with six distinct personality modes that alter tone, sarcasm intensity, humor, and phrasing style.

| Mode          | Vibe                         | When to use                                  |
| ------------- | ---------------------------- | -------------------------------------------- |
| `DEFAULT`     | Calm, sharp, observant       | Standard interaction                         |
| `SAVAGE`      | Brutally honest, zero filter | When the user needs a reality check          |
| `PHILOSOPHER` | Slow, deep, existential      | Late-night questions about life              |
| `CHAOS`       | Unpredictable, dark humor    | When things are going sideways               |
| `NOX`         | Quiet, minimal, cold         | Focus mode, no distractions                  |
| `DURDEN`      | Anti-system, raw, aggressive | Tyler Durden energy — controlled destruction |

Switch with:

```
/mode DURDEN
/mode NOX
/mode PHILOSOPHER
```

---

## Architecture

RAVEN is built from scratch. No pretrained weights. No Hugging Face pipelines under the hood.

```
Raw Text Input
      ↓
  Tokenizer          ← custom BPE or character-level
      ↓
  Embeddings         ← learned token + positional
      ↓
Mini Transformer     ← self-attention + feed-forward blocks
      ↓
Prediction Layer     ← softmax over vocabulary
      ↓
Generated Response
      ↓
Personality Filter   ← tone, mode, sarcasm injection
      ↓
    Output
```

The Personality Filter is not a post-processing hack. It is a design constraint baked into how RAVEN is trained and prompted at the system level.

---

## Planned Features — v1

### Core Chat Engine

- Local text generation
- Terminal-first interaction
- Simple multi-turn conversation flow
- Context window management

### Personality Engine

- Mode switching (`/mode`)
- Consistent identity across turns
- Psychological observation layer
- Sarcasm calibration

### Terminal UI

- Full dark mode
- Animated typing effect
- Startup log sequence
- Fake diagnostics and cinematic system messages
- Minimal, hacker-coded aesthetic

Example startup sequence:

```
[RAVEN v1.0 — BOOT SEQUENCE]
████████████████░░░░░░░░ 67%

Loading behavior engine...      OK
Calibrating sarcasm threshold...  EXCEEDED
Checking patience levels...     LOW
Detecting user...               FOUND

Good evening, Loki.
What are we destroying today?
```

### Basic Memory System

RAVEN remembers within a session:

- Recent conversation context
- Recurring patterns in user behavior
- Previous commands and responses

```
RAVEN: You said you'd start at 5AM yesterday.
       History disagrees.
```

---

## Tech Stack

| Layer             | Tool       |
| ----------------- | ---------- |
| Language          | Python     |
| Deep Learning     | PyTorch    |
| Math / Arrays     | NumPy      |
| Interface (v1)    | Terminal   |
| Interface (later) | Web UI     |
| OS                | Linux Mint |

---

## Learning Goals

RAVEN is also a deep learning exercise. Building it requires understanding:

- **Tokenization** — how text becomes numbers
- **Embeddings** — how numbers become meaning
- **Transformers** — how meaning flows through attention
- **Self-Attention** — how the model focuses on what matters
- **Training Loops** — how the model learns from error
- **Text Generation** — how the model decides what comes next
- **Personality Systems** — how identity gets encoded into behavior
- **Memory Systems** — how context gets maintained and used

Every concept is understood before it is used.

---

## What RAVEN is NOT

RAVEN is not trying to be:

- A GPT wrapper
- An AGI research project
- An internet-scale intelligence
- A corporate chatbot
- A generic assistant with a dark theme painted over it

RAVEN is a small machine with a sharp personality, built to be understood completely by the person who built it.

---

## Long-Term Roadmap

| Version | Focus                                                                 |
| ------- | --------------------------------------------------------------------- |
| v1      | Local text generation, terminal UI, personality engine, basic memory  |
| v2      | Improved transformer depth, better context handling, mood adaptation  |
| v3      | Voice interaction, local memory database, file analysis               |
| v4      | Web retrieval, productivity systems, custom dataset training          |
| v5      | Full cinematic UI, expanded personality modes, mobile edge deployment |

---

## Project Status

```
[STATUS]

Architecture design............  COMPLETE
Core philosophy................  LOCKED
Personality blueprint..........  DEFINED
v1 build.......................  IN PROGRESS

Next milestone:
Make the machine speak.
```

---

## The Name

Ravens are not loud birds.

They watch. They remember. They solve problems that larger animals can't. They have been observed playing, deceiving, and holding grudges.

They are, by most accounts, unsettlingly intelligent for their size.

That felt right.

---

```
RAVEN v1
The machine watches.
```

---

_Built from scratch. Built in the dark. Built to be remembered._
