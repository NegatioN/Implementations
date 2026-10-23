import numpy as np
import torch
import torch.nn as nn
import math

n_token, n_dim = 3, 4

size = (n_token, n_dim)
X = torch.randn(*size) #input
mask = torch.tril(torch.ones((n_token, n_token), dtype=torch.bool))

def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def attention_masked(Q, K, V):
    scale = math.sqrt(K.size(-1)) # should be kept in pytorch :shrug:
    return softmax(torch.where(mask, (Q @ K.T) / scale, -torch.inf)) @ V

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
        
