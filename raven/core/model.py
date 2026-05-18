import torch
import torch.nn as nn
from torch.nn import functional as F

import config

class Head(nn.Module):
    """
    A single self-attention head.
    This is where the model looks at other characters in the sequence to gather context.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # tril is a lower triangular matrix used to mask future tokens (so the model can't cheat and look ahead)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Aggregate the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads running in parallel.
    This allows the model to focus on different types of context simultaneously.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate the outputs from all heads along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity (MLP).
    This gives the model time to "think" about what it learned from the attention mechanism.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block combining attention and feed-forward layers.
    Includes residual connections (adding the input to the output) and layer normalization to stabilize training.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections: x + layer(...)
        # We apply layer norm BEFORE the attention and feed-forward (pre-norm formulation)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class RavenModel(nn.Module):
    """
    The full RAVEN language model.
    Combines character embeddings, positional embeddings, transformer blocks, and the final output layer.
    """
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Create n_layer transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Final linear layer to output the next token probabilities (logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T, C)
        
        # Add token and position embeddings together
        x = tok_emb + pos_emb # (B, T, C)
        
        # Pass through the transformer blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        
        # Get final logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=120,
        temperature=0.65,
        top_k=32,
        top_p=0.88,
        repetition_penalty=1.18,
        repetition_window=80,
        stop_tokens=None,
        stop_strings=None,
        tokenizer=None,
        min_new_tokens=12,
    ):
        """
        Controlled autoregressive generation for tiny character-level models.
        Combines temperature, top-k, nucleus sampling, repetition penalty, and
        stop conditions to reduce gibberish and repetition loops.
        """
        if idx.shape[0] != 1:
            raise ValueError("generate() currently supports batch size 1 for stop-string handling.")

        self.eval()

        stop_tokens = set(stop_tokens or [])
        stop_strings = stop_strings or []
        original_len = idx.shape[1]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty != 1.0:
                recent_tokens = idx[0, -repetition_window:].tolist()
                for token_id in set(recent_tokens):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            temperature = max(float(temperature), 1e-5)
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_remove = cumulative_probs > top_p
                sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                sorted_remove[:, 0] = False

                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(1, sorted_indices, sorted_remove)
                logits = logits.masked_fill(remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                probs = torch.ones_like(probs) / probs.size(-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            generated_len = idx.shape[1] - original_len
            if generated_len < min_new_tokens:
                continue

            if idx_next.item() in stop_tokens:
                break

            if tokenizer is not None and stop_strings:
                generated_text = tokenizer.decode(idx[0, original_len:].tolist())
                if any(stop_string in generated_text for stop_string in stop_strings):
                    break

        return idx
