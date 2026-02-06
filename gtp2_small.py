import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.conftest import dtype

from config import GPTSettings


class GPT2(nn.Module):
    def __init__(self, config: GPTSettings):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_dim)
        self.wpe = nn.Embedding(config.block_size, config.n_dim)
        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        #stabilize the output of transformer blocks
        self.ln_f = nn.LayerNorm(config.n_dim)

        # Layer that makes the prediction
        self.lm_head = nn.Linear(config.n_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        batch, length = idx.size()
        assert length <= self.config.block_size, "Sequence length exceeds block size"

        pos = torch.arrange(0, length, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, self.config.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -float('inf')), logits)

            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTSettings):
        super().__init__()
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)

    def forward(self, x):
        batch, length, dim = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(dim, dim=2)


        scaled_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attention_weights = F.softmax(scaled_scores, dim= -1)

        output = attention_weights @ v

        return output

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTSettings):
        super().__init__()
        assert config.n_dim % config.n_head == 0 # ensure there is not remainder
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        self.register_buffer(
            name='bias',
            tensor= torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        self.c_proj = nn.Linear(config.n_dim, config.n_dim)


    def forward(self, x):
        batch, length, dim = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(dim, dim=2)

        head_dim = dim // self.n_head

        q = q.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)

        scaled_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        scaled_scores = scaled_scores.masked_fill(self.bias[:,:,:length,:length] == 0, float('-inf'))

        attention_weights = F.softmax(scaled_scores, dim= -1)

        output = attention_weights @ v
        output = output.transpose(1, 2).contiguous().view(batch, length, dim)
        output = self.c_proj(output)

        return output

class MLP(nn.Module):
    def __init__(self, config: GPTSettings):
        super().__init__()
        self.fc = nn.Linear(config.n_dim, 4 & config.n_dim)
        self.proj = nn.Linear(4 * config.n_dim, config.n_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTSettings):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))