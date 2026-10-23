import numpy as np
import torch
import torch.nn as nn
import math

n_token, n_dim = 3, 4

size = (n_token, n_dim)
X = torch.randn(*size) #input
max_len = 1024

# Created once up to max_len (e.g., 1024 x 1024)
max_mask = torch.tril(torch.ones((max_len, max_len), dtype=torch.bool))

def attention_masked(Q, K, V):
    T, d_k = Q.size(-2), K.size(-1)
    scale = math.sqrt(d_k)
    scores = (Q @ K.transpose(-2, -1)) / scale

    # Slice the mask to match current sequence length T
    return softmax(torch.where(max_mask[:T, :T], scores, -torch.inf)) @ V


def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def single_head_attention(X, Q_w, K_w, V_w, f_attn=attention_masked):
    Q = X @ Q_w
    K = X @ K_w
    V = X @ V_w

    return f_attn(Q, K, V)

def multi_head_attention(X, heads, mixer, f_attn=attention_masked):
    X_o = [single_head_attention(X, Q_w, K_w, V_w, f_attn) for (Q_w, K_w, V_w) in heads]
    return torch.concatenate(X_o, axis=-1) @ mixer

class TBlock(nn.Module):
    def __init__(self, n_heads, n_dim, h_dim):
        assert n_dim % n_heads == 0, "embedding dimension not cleanly divisible by n_heads"
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(n_dim) for _ in range(2)])
        self.mlp_i = nn.Linear(n_dim, h_dim)
        self.mlp_o = nn.Linear(h_dim, n_dim)
        n_out = n_dim // n_heads
        self.heads = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(torch.randn(n_dim, n_out)), # Q_w
                nn.Parameter(torch.randn(n_dim, n_out)), # K_w
                nn.Parameter(torch.randn(n_dim, n_out)), # V_w
            ])
            for _ in range(n_heads)
        ])

        self.O_w = nn.Parameter(torch.randn(n_dim, n_dim))
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X + multi_head_attention(self.ln[0](X), self.heads, self.O_w, f_attn=attention_masked)
        return X + self.mlp_o(self.relu(self.mlp_i(self.ln[1](X))))

tb = TBlock(2, 4, 16)

print(tb(X))
        

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_heads, n_dim, h_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_dim)
        self.position_embedding = nn.Embedding(max_len, n_dim)

        self.blocks = nn.ModuleList([
            TBlock(n_heads, n_dim, h_dim) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_dim)
        self.lm_head = nn.Linear(n_dim, vocab_size, bias=False)

    def forward(self, tokens):
        positions = torch.arange(len(tokens), device=tokens.device)
        X = self.token_embedding(tokens) + self.position_embedding(positions)

        for block in self.blocks:
            X = block(X)

        X = self.ln_f(X)
        logits = self.lm_head(X) # shape: (n_token, vocab_size)
        return logits


t = Transformer(100, 1024, 3, 2, 4, 16)

print(t(torch.arange(5)))
