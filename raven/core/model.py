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

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens one by one, appending them to the input sequence.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens, since our position embeddings 
            # only go up to block_size
            idx_cond = idx[:, -config.block_size:]
            
            # Get the predictions
            logits, loss = self(idx_cond)
            
            # Focus only on the last time step (the prediction for the next character)
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
