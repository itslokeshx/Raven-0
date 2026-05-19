import math

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import config
except ModuleNotFoundError:
    from raven import config


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        batch_size, seq_len, channels = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RavenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}.")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(positions)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, logits.size(-1)),
                targets.reshape(batch_size * seq_len),
            )

        return logits, loss

    @staticmethod
    def _apply_repetition_penalty(logits, generated, penalty, window):
        if penalty is None or penalty == 1.0:
            return logits

        for row in range(logits.size(0)):
            recent = generated[row, -window:].tolist()
            for token_id in set(recent):
                if token_id < 0 or token_id >= logits.size(-1):
                    continue
                if logits[row, token_id] > 0:
                    logits[row, token_id] /= penalty
                else:
                    logits[row, token_id] *= penalty
        return logits

    @staticmethod
    def _filter_logits(logits, top_k=None, top_p=None):
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_sorted = cumulative_probs > top_p
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False

            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(1, sorted_indices, remove_sorted)
            logits = logits.masked_fill(remove, float("-inf"))

        return logits

    @staticmethod
    def _safe_probs(logits):
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        bad_rows = row_sums.squeeze(-1) <= 0
        if bad_rows.any():
            probs[bad_rows] = 1.0 / probs.size(-1)
            row_sums = probs.sum(dim=-1, keepdim=True)
        return probs / row_sums

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=120,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15,
        repetition_window=96,
        stop_tokens=None,
        stop_strings=None,
        tokenizer=None,
        min_new_tokens=1,
    ):
        if idx.ndim != 2:
            raise ValueError("generate() expects idx with shape (batch, sequence).")
        if stop_strings and (tokenizer is None or idx.size(0) != 1):
            raise ValueError("stop_strings require a tokenizer and batch size 1.")

        was_training = self.training
        self.eval()

        stop_tokens = set(stop_tokens or [])
        stop_strings = stop_strings or []
        original_len = idx.size(1)
        temperature = max(float(temperature), 1e-5)
        repetition_window = max(1, int(repetition_window))
        min_new_tokens = max(0, int(min_new_tokens))

        try:
            for _ in range(int(max_new_tokens)):
                idx_cond = idx[:, -self.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :].float()

                logits = self._apply_repetition_penalty(
                    logits,
                    idx,
                    repetition_penalty,
                    min(repetition_window, idx.size(1)),
                )
                logits = logits / temperature
                logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
                probs = self._safe_probs(logits)

                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

                generated_len = idx.size(1) - original_len
                if generated_len < min_new_tokens:
                    continue

                if stop_tokens and all(token.item() in stop_tokens for token in idx_next[:, 0]):
                    break

                if stop_strings:
                    text = tokenizer.decode(idx[0, original_len:].tolist())
                    if any(stop_string in text for stop_string in stop_strings):
                        break
        finally:
            self.train(was_training)

        return idx
