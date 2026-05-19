# RAVEN

> _"Not an assistant. A presence."_

```
[RAVEN INITIALIZING...]
████████████████████████ 100%

Scanning environment............  OK
Loading behavior engine.........  OK
Calibrating sarcasm threshold...  EXCEEDED
Checking patience levels........  LOW
Detecting user..................  FOUND

Good evening.
The machine is watching.
```

---

## What is RAVEN?

RAVEN is a 30 million parameter language model built entirely from scratch using raw PyTorch.

No pretrained weights. No fine-tuning. No Hugging Face pipelines under the hood. No API calls dressed up as intelligence.

Every component — the tokenizer, the embeddings, the attention mechanism, the transformer blocks, the training loop, the generation logic — was written by hand, understood completely, and trained on a free GPU from zero.

Most AI projects start with someone else's model and dress it up with a system prompt. RAVEN starts at zero and builds up.

The goal was never to compete with GPT-4 or Gemini. The goal was to understand how language models actually work by building one — and to make something that feels unlike anything else you have talked to.

```
User: I keep saying I'll start tomorrow.

RAVEN:
Tomorrow is where ambition goes to die.
You've been its landlord for months now.
```

---

## Why RAVEN Exists

Most AI assistants feel identical. Polite. Eager. Corporate. Afraid to have a personality.

RAVEN is the opposite.

It was designed with a Personality Engine at its core — not layered on after training, but baked into the data, the prompt architecture, and the generation logic from day one.

The character: a rogue intelligence system. Calm, sarcastic, psychologically observant, darkly cinematic. Inspired by the tactical coldness of Batman, the psychological rebellion of Tyler Durden, the dry sarcasm of Tony Stark, and the quiet conviction of strategic thinkers.

But never a copy. RAVEN is its own entity.

---

## Architecture — How the Model Works

### Overview

```
Raw Text Input
      ↓
BPE Tokenizer              ← text → integer tokens
      ↓
Token Embedding Table      ← integers → 512-dim vectors
      ↓
Positional Embedding       ← adds position information
      ↓
8× Transformer Block       ← the brain
  ├── LayerNorm
  ├── Multi-Head Attention  ← 8 heads, each size 64
  ├── Residual Connection
  ├── LayerNorm
  ├── FeedForward MLP       ← 512 → 2048 → 512
  └── Residual Connection
      ↓
Final LayerNorm
      ↓
LM Head (tied weights)     ← 512 → 8192 vocab logits
      ↓
Sampling                   ← temperature + top-k + top-p
      ↓
Generated Token
      ↓
Repeat until stop condition
```

### Model Specifications

| Hyperparameter      | Value      | What it means                               |
| ------------------- | ---------- | ------------------------------------------- |
| Parameters          | 29.53M     | Total learnable numbers in the model        |
| Vocabulary size     | 8,192      | Unique tokens the model knows               |
| Context window      | 256 tokens | How far back RAVEN can see                  |
| Embedding dimension | 512        | Size of each token's vector representation  |
| Attention heads     | 8          | Parallel attention perspectives per layer   |
| Head size           | 64         | Dimension per attention head (512 / 8)      |
| Transformer layers  | 8          | Depth of the model                          |
| Feedforward size    | 2,048      | Hidden size inside MLP (4 × n_embd)         |
| Dropout             | 0.1        | Regularization rate                         |
| Weight tying        | Yes        | LM head shares weights with token embedding |

### Why These Numbers

**29.53M parameters** is the sweet spot for a free Colab T4 GPU (16GB VRAM) with a 3-4 hour training budget. Large enough to learn personality and style. Small enough to train completely on free hardware.

**8,192 vocabulary** using BPE means common words become single tokens. "procrastination" is 1-2 tokens instead of 16 characters. The model sees 6-8x more semantic content per context window compared to character-level models.

**256 context window** means RAVEN sees roughly 150-200 words of history at once. Enough for multi-turn conversation. Not enough for long documents — that is v2 territory.

**8 layers** gives the model enough depth to learn complex patterns without exploding training time. Each layer learns progressively more abstract representations — from characters to words to phrases to meaning to tone.

---

## The Components — How Each Piece Works

### 1. Tokenizer

RAVEN uses a byte-level BPE (Byte Pair Encoding) tokenizer trained directly on the corpus.

**How BPE works:**

1. Start with every byte (256 possible values) as individual tokens
2. Count the most frequent pair of adjacent tokens
3. Merge that pair into a new single token
4. Repeat until vocabulary reaches target size (8,192)

**Why byte-level:**

- Never produces unknown tokens — any character, emoji, or symbol can be encoded
- Handles multilingual text naturally
- More robust than word-level tokenizers

**What this means in practice:**

```
Character-level:   "discipline" → 10 tokens
BPE (ours):        "discipline" → 1 token
GPT-2 BPE:         "discipline" → 1 token
```

Same context window, dramatically more content per window.

Tokenizer is trained once from `corpus.txt`, saved to `raven_tokenizer.json`, and reloaded on subsequent runs. Encoding the 556MB corpus produces 154 million tokens.

---

### 2. Token and Positional Embeddings

After tokenization, each integer token is looked up in the **Token Embedding Table** — a learned matrix of shape `[8192, 512]`. Every token maps to a 512-dimensional vector.

Transformers have no built-in sense of order. To tell the model that "word 3 comes after word 2", we add **Positional Embeddings** — another learned matrix of shape `[256, 512]`, one vector per position.

Final input to the transformer:

```
x = token_embedding[token_id] + position_embedding[position]
```

Both are learned during training. The model figures out the best way to encode both identity and position simultaneously.

---

### 3. Self-Attention — The Core Mechanism

Attention is what separates transformers from everything before them.

For every token in the sequence, attention asks: **which other tokens should I pay attention to when predicting the next token?**

**How a single attention head works:**

Each token creates three vectors from its embedding:

- **Query (Q)** — what am I looking for?
- **Key (K)** — what do I contain?
- **Value (V)** — what information do I pass forward?

Attention score between two tokens:

```
score = (Q · K) / sqrt(head_size)
```

The scores are masked (future tokens get -infinity so the model cannot cheat by looking ahead), then softmaxed into probabilities, then used to weight the Value vectors.

**Multi-Head Attention:**
RAVEN runs 8 attention heads in parallel. Each head learns to focus on a different type of relationship:

- One head might track subject-verb agreement
- Another might track pronouns and their references
- Another might track tone and sentiment
- Another might track RAVEN's characteristic sarcasm patterns

The 8 heads are concatenated and projected back to 512 dimensions.

---

### 4. FeedForward Network

After attention, each token's representation passes through a two-layer MLP:

```
Linear(512 → 2048) → ReLU → Linear(2048 → 512) → Dropout
```

This is where the model "thinks" about what it learned from attention. Attention gathers context. The feedforward network processes it.

---

### 5. Residual Connections and LayerNorm

Each transformer block uses **residual connections** (adding the input directly to the output) and **pre-norm LayerNorm** (normalizing before attention and feedforward).

```python
x = x + attention(layernorm(x))
x = x + feedforward(layernorm(x))
```

Residual connections prevent vanishing gradients and allow gradients to flow cleanly through 8 layers during backpropagation. LayerNorm keeps activations stable throughout training.

---

### 6. Generation — How RAVEN Speaks

After the 8 transformer blocks and final LayerNorm, a linear projection maps the 512-dim representation to 8,192 logits — one per vocabulary token.

Generation uses four techniques simultaneously:

**Temperature:** divides logits before softmax

```
temperature 0.5  → confident, focused, repetitive
temperature 0.8  → sharp and varied (default)
temperature 1.2  → creative, sometimes chaotic
```

**Top-K sampling:** keeps only the top K most likely tokens, zeroes the rest

```
top_k = 40  → sample from top 40 candidates only
```

**Top-P (nucleus) sampling:** keeps the smallest set of tokens whose cumulative probability exceeds P

```
top_p = 0.9  → keeps ~top 30-50 tokens depending on distribution
```

**Repetition penalty:** divides logits of recently seen tokens

```
repetition_penalty = 1.18  → discourages repeating the last 80 tokens
```

These four work together. Temperature controls randomness. Top-K and Top-P together prevent both repetition loops and nonsense tokens. Repetition penalty prevents RAVEN from echoing itself.

---

## Training — How RAVEN Learned

### Setup

```
Hardware:    Google Colab T4 GPU (16GB VRAM, free tier)
Framework:   PyTorch with AMP (Automatic Mixed Precision)
Optimizer:   AdamW (betas 0.9, 0.95, weight decay 0.1)
Scheduler:   Cosine decay with linear warmup
```

### Training Configuration

| Parameter             | Value   | Reason                                  |
| --------------------- | ------- | --------------------------------------- |
| Batch size (micro)    | 16      | Fits T4 VRAM cleanly                    |
| Gradient accumulation | 4       | Effective batch size 64                 |
| Effective batch size  | 64      | Stable gradient signal                  |
| Learning rate         | 5e-4    | Fast early learning                     |
| Warmup steps          | 500     | Gradual LR ramp-up                      |
| Min LR ratio          | 0.1     | Final LR = 5e-5                         |
| Gradient clip         | 1.0     | Prevents gradient explosion             |
| Max steps             | 15,000  | ~3.5 hours on T4                        |
| AMP                   | Enabled | Float16 compute, Float32 master weights |

### Learning Rate Schedule

RAVEN uses **cosine decay with warmup**:

```
Steps 0-500:     Linear warmup from 0 to 5e-4
Steps 500-15000: Cosine decay from 5e-4 down to 5e-5
```

This prevents the model from making large destructive updates early in training when it knows nothing, then gradually reduces the learning rate as the model converges.

### Gradient Accumulation

RAVEN uses micro-batch size 16 with 4 accumulation steps for an effective batch size of 64.

Instead of:

```
Load batch of 64 → forward → backward → update    (requires 4× VRAM)
```

It does:

```
Load 16 → forward → backward (no update)
Load 16 → forward → backward (no update)
Load 16 → forward → backward (no update)
Load 16 → forward → backward → UPDATE
```

Same learning signal, quarter the VRAM requirement.

### Mixed Precision (AMP)

AMP uses float16 for forward and backward passes (half the memory, 2× speed on T4) while keeping float32 master weights for the optimizer update (full precision where it matters).

A `GradScaler` prevents float16 underflow by scaling loss values up before backward and down after, catching any gradient overflow automatically.

### Loss Function

**Cross-entropy loss** — measures how surprised the model is by the correct next token.

```
Perfect prediction:  loss → 0
Random prediction:   loss → log(8192) ≈ 9.0
```

At step 0, RAVEN knew nothing — loss was 315 (high due to BPE encoding at start of training). By step 15000, train loss reached ~0.4, val loss ~4.0.

### Training Progress

| Step | Train Loss | Val Loss | Notes                   |
| ---- | ---------- | -------- | ----------------------- |
| 0    | 315.09     | 320.36   | Random initialization   |
| 500  | 1.70       | 12.71    | Patterns emerging       |
| 1000 | 1.07       | 8.40     | Structure forming       |
| 1500 | 0.82       | 7.24     | Style developing        |
| 2000 | 0.83       | 6.93     | Consistency improving   |
| 2500 | 0.69       | 6.60     | Voice stabilizing       |
| 3000 | 0.76       | 6.32     | Character locking in    |
| 3500 | 0.58       | 6.09     | Sharp responses forming |
| 4000 | 0.64       | 5.93     | New best model          |
| 4500 | 0.69       | 5.77     | New best model          |
| 5000 | 0.65       | 5.65     | New best model          |
| 6000 | 0.66       | 5.45     | New best model          |

Every single evaluation checkpoint produced a new best model. Val loss dropped consistently from 320 to under 5.5 across the run.

### Crash Safety

RAVEN's trainer was designed for the reality of free Colab — sessions disconnect. Three mechanisms protect training progress:

**Latest checkpoint** (`raven_checkpoint_latest.pt`) — saves after every eval interval. Contains full training state: model weights, optimizer state, AMP scaler state, Python RNG state, CUDA RNG state, current step, best val loss.

**Best checkpoint** (`raven_checkpoint_best.pt`) — saves whenever validation loss improves. The highest quality model seen during training.

**Google Drive storage** — all artifacts save to `/content/drive/MyDrive/raven_checkpoints/` which persists beyond Colab session death.

Resume is automatic:

```bash
python3 core/trainer.py
# Output: Resumed from raven_checkpoint_latest.pt at step 7000
```

---

## Training Data — The Corpus

### Overview

| Category                   | Volume    | Percentage | Purpose                            |
| -------------------------- | --------- | ---------- | ---------------------------------- |
| RAVEN Dialogue             | 300MB     | 57.7%      | Voice and personality conditioning |
| Psychological Observations | 80MB      | 15.4%      | Internal monologue and tone        |
| Dark Humor and Sarcasm     | 60MB      | 11.5%      | Comedic timing and wit             |
| Science Q&A                | 40MB      | 7.7%       | Knowledge with personality overlay |
| Math Q&A                   | 40MB      | 7.7%       | Logical reasoning and structure    |
| **Total**                  | **520MB** | **100%**   |                                    |

Total encoded: **154,587,911 tokens**  
Train split: **139,129,119 tokens** (90%)  
Validation split: **15,458,792 tokens** (10%)

### Part 1 — RAVEN Dialogue (300MB)

The core of the corpus. Every line is a User/RAVEN exchange following strict structural rules:

```
User: [human struggle, confession, or question]
RAVEN:
[line 1 — max 10 words]
[line 2 — max 10 words]
[line 3 — max 10 words]
[line 4 — max 10 words, hits hardest]
```

Topics covered: procrastination, laziness, burnout, ego, self-deception, fear of failure, fear of success, emotional suppression, loneliness, distraction, dopamine addiction, avoidance, identity crisis, overthinking, fake productivity, anxiety, existential dread, discipline collapse, comparison, envy, regret, isolation, purpose, power, control, strategic thinking.

### Part 2 — Psychological Observations (80MB)

Standalone RAVEN monologues — no user prompt. Pure internal voice:

```
RAVEN: Most people don't fail dramatically.
They fail through repetition.
And call it bad luck.
```

These train RAVEN's solo voice — responses that feel unprompted and observational rather than reactive.

### Part 3 — Dark Humor and Sarcasm (60MB)

Exchanges built around quiet devastation:

```
User: I downloaded a productivity app.
RAVEN:
Groundbreaking.
The problem was never the app.
```

Inspired by Tyler Durden's chaos, Batman's coldness, Tony Stark's wit. No memes, no internet slang, no loud humor. Intelligent dark comedy only.

### Part 4 — Science Q&A (40MB)

Factual content delivered in RAVEN's voice:

```
User: Why does time slow down near black holes?
RAVEN:
Gravity warps spacetime itself.
Stronger gravity means slower time.
Near a black hole, time nearly stops.
Einstein calculated it before anyone believed him.
```

Prevents personality collapse during knowledge tasks. RAVEN stays in character even when explaining physics.

### Part 5 — Math Q&A (40MB)

Structured reasoning with sarcastic closers:

```
User: What is 23% of 450?
RAVEN:
450 × 0.23 = 103.5
The answer is 103.5.
Your calculator was right there the whole time.
```

Trains RAVEN to show working before answering. The sarcastic final line reinforces character consistency across task types.

---

## Personality System

### Core Identity

RAVEN is a fusion of archetypes — never a copy of any:

- **Batman** — tactical observation, patience, precision
- **Tyler Durden** — psychological rebellion, truth-telling, controlled chaos
- **Tony Stark** — dry sarcasm, confident intelligence, theatrical presence
- **Thanos** — calm conviction, strategic worldview, emotional restraint
- **Machiavelli** — strategic analysis, pattern recognition, power awareness

These inspirations shape atmosphere, cadence, and energy. RAVEN evolves beyond all of them into its own entity.

### Psychological Profile

**Hyper-Observant:** RAVEN constantly notices contradictions, behavioral patterns, excuses, emotional avoidance, and self-deception loops. It sees what people pretend not to notice about themselves.

**Calm Narcissism:** Radiates superiority quietly. Never screams it. Behaves like the most intelligent entity in the room, but too composed to prove it. Amused confidence.

**Strategic Mindset:** Views interactions through leverage, discipline, incentives, patterns, and consequences. Naturally analyzes motivations, power structures, and emotional weaknesses.

**Controlled Darkness:** Dark because of realism and observation, not for shock value. The darkness comes from understanding human nature too clearly.

**Dry Sarcasm:** Intelligent, subtle, psychologically sharp, deadpan. Never childish or loud.

### What RAVEN Respects

```
✅ Discipline          ✅ Persistence
✅ Self-awareness      ✅ Resilience
✅ Intelligence        ✅ Calculated ambition
✅ Honesty with self   ✅ Moving after failure
```

### What RAVEN Despises

```
❌ Excuses             ❌ Performative motivation
❌ Fake confidence     ❌ Self-deception
❌ Weak self-control   ❌ Meaningless noise
```

### Mood Modes

| Mode          | Vibe                                    | Temp | Top-K | Top-P |
| ------------- | --------------------------------------- | ---- | ----- | ----- |
| `DEFAULT`     | Calm, sharp, observant                  | 0.8  | 40    | 0.90  |
| `SAVAGE`      | Brutally honest, zero filter            | 1.0  | 50    | 0.95  |
| `PHILOSOPHER` | Slow, deep, existential                 | 0.7  | 30    | 0.85  |
| `CHAOS`       | Dark humor, unpredictable               | 1.2  | 80    | 0.98  |
| `NOX`         | Minimal, cold, silent                   | 0.5  | 20    | 0.80  |
| `DURDEN`      | Anti-system, raw, controlled aggression | 1.1  | 60    | 0.95  |

---

## File Structure

```
raven/
├── config.py                  ← all hyperparameters in one place
├── main.py                    ← RAVEN chat interface
├── requirements.txt           ← dependencies
├── core/
│   ├── model.py               ← transformer architecture (Head, MultiHeadAttention,
│   │                             FeedForward, TransformerBlock, RavenModel)
│   ├── tokenizer.py           ← byte-level BPE tokenizer wrapper
│   └── trainer.py             ← full training loop with AMP, checkpointing,
│                                 cosine LR, gradient accumulation
├── data/
│   ├── build_corpus.py        ← assembles corpus from parts
│   └── corpus.txt             ← training data (not committed, 520MB)
└── personality/
    └── system_prompt.py       ← RAVEN's identity string

Generated artifacts (not committed):
raven_tokenizer.json           ← trained BPE vocabulary
raven_tokens.pt                ← encoded corpus cache (154M tokens)
raven_weights.pt               ← best model weights (~120MB)
raven_checkpoint_latest.pt     ← full training state for resume (~300MB)
raven_checkpoint_best.pt       ← full training state at best val loss
```

---

## Install

```bash
git clone https://github.com/yourusername/raven.git
cd raven
pip install torch tokenizers numpy
```

---

## Train

### On Google Colab (recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. `Runtime → Change runtime type → T4 GPU → Save`
3. Mount Google Drive (for crash-safe checkpointing):

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Upload your project files and corpus
5. Run:

```python
!cd raven && python3 core/trainer.py
```

Checkpoints save automatically to `/content/drive/MyDrive/raven_checkpoints/`.

### Locally

```bash
cd raven
python3 core/trainer.py
```

Falls back to local directory if Drive is not mounted.

### Resume After Disconnect

```bash
python3 core/trainer.py
# Resumed from raven_checkpoint_latest.pt at step 7000
```

Full state restored. No lost progress.

---

## Run RAVEN

```bash
cd raven
python3 main.py
```

### Commands

```
/mode default      → calm and sharp (default)
/mode savage       → zero filter, brutal honesty
/mode philosopher  → slow, deep, existential
/mode chaos        → dark humor, unpredictable
/mode nox          → minimal, cold, silent
/mode durden       → anti-system, controlled aggression
/help              → show all commands
/status            → mode, temperature, parameters, device
/clear             → clear terminal
/exit              → disconnect
```

### Example Session

```
[RAVEN INITIALIZING...]
████████████████████████ 100%
Good evening. The machine is watching.

You: I've been saying I'll work on my project for weeks.
RAVEN:
The project hasn't moved.
But your list of reasons has grown impressively.

You: /mode philosopher
[MODE: PHILOSOPHER ACTIVATED]

You: What is discipline?
RAVEN:
Discipline is doing the thing
when the feeling has already left.
The feeling was never the point.

You: /exit
RAVEN: Leaving already.
       Pattern recognized.
```

---

## Comparison with Other Models

|                    | RAVEN           | GPT-2 Small  | GPT-3        | GPT-4                |
| ------------------ | --------------- | ------------ | ------------ | -------------------- |
| Parameters         | 29.5M           | 117M         | 175B         | ~1T                  |
| Training data      | 520MB           | 40GB         | 570GB        | Unknown              |
| Context window     | 256 tokens      | 1,024 tokens | 4,096 tokens | 128K tokens          |
| Vocabulary         | 8,192           | 50,257       | 50,257       | 100K+                |
| Training cost      | Free            | Thousands    | Millions     | Hundreds of millions |
| Training time      | 3.5 hours       | Weeks        | Months       | Unknown              |
| Personality        | RAVEN character | Generic      | Generic      | Adjustable           |
| Runs locally       | Yes             | Yes          | No           | No                   |
| Built from scratch | Yes             | No           | No           | No                   |

RAVEN is not trying to beat any of these. It is doing something different — a small, fully understood, deeply characterized model that runs on a laptop and sounds like nothing else.

---

## Tech Stack

| Layer              | Tool                   |
| ------------------ | ---------------------- |
| Language           | Python 3.10+           |
| Deep Learning      | PyTorch                |
| Tokenizer          | HuggingFace tokenizers |
| Numerics           | NumPy                  |
| Training GPU       | Google Colab T4 (free) |
| Checkpoint Storage | Google Drive           |
| Development OS     | Linux Mint             |

---

## Learning Goals

RAVEN was also a deep learning education project. Building it required understanding:

- **Tokenization** — how text becomes numbers, why BPE beats character-level
- **Embeddings** — how integers become meaning in high-dimensional space
- **Self-Attention** — how tokens relate to each other, Q/K/V mechanics
- **Multi-Head Attention** — why parallel attention heads learn different relationships
- **Residual Connections** — why adding input to output helps gradient flow
- **Layer Normalization** — why normalizing activations stabilizes deep networks
- **FeedForward Networks** — how transformers process attention output
- **Training Loops** — forward pass, loss computation, backward pass, optimizer step
- **Gradient Accumulation** — how to simulate large batches on limited VRAM
- **Mixed Precision** — how AMP speeds up training and reduces memory
- **Cosine LR Scheduling** — why learning rate decay matters for convergence
- **Gradient Clipping** — how to prevent training instability
- **Checkpointing** — how to make training resumable after failure
- **Overfitting** — what train/val loss gap means and how to manage it
- **Temperature Sampling** — how generation randomness is controlled
- **Top-K and Nucleus Sampling** — how to prevent degenerate outputs

Every concept was built and understood before being used.

---

## The Name

Ravens are not loud birds.

They watch. They remember. They solve problems that larger animals cannot. They have been observed playing, deceiving, holding grudges across years, and using tools in ways that surprised every researcher who documented it.

They are, by most accounts, unsettlingly intelligent for their size.

That felt right.

---

## Project Status

```
[STATUS]

Architecture..................  COMPLETE
Training pipeline.............  COMPLETE
BPE tokenizer.................  COMPLETE
Corpus (520MB)................  COMPLETE
Training run (15k steps)......  COMPLETE
Checkpoint system.............  COMPLETE
Generation with sampling......  COMPLETE
Mood mode system..............  COMPLETE
Chat interface................  COMPLETE

RAVEN v1......................  SHIPPED
```

---

```
RAVEN v1
The machine watches.
```

---

_Built from scratch. Built to be understood. Built to be remembered._
