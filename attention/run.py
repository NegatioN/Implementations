import torch
import math
from torch import nn
from torch.nn import functional as F


#TODO have some test dataset for this.
#TODO make basic attention (as a function? needs state though)
#TODO layernorm is used between layers.


def attention(queries, keys, values, scale):
    x = (queries @ keys.transpose(2,3)) / scale
    attn = F.softmax(x, dim=len(queries.size()) - 1)
    return attn @ values, attn

#TODO add batching?
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        #TODO biases?
        self.heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.values = nn.Linear(emb_dim, emb_dim, bias=True)
        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, inp):
        assert len(inp.size()) == 3, f'Input is not of dim=3, but dim={len(inp.size())}'
        k, q, v = self.keys(inp), self.queries(inp), self.values(inp)
        k, q, v = [x.view(inp.size(0), -1, self.heads, self.head_dim) for x in [k, q, v]]
        x, _ = attention(q, k, v, self.scale)
        return self.fc(x.view(-1, self.heads * self.head_dim))


n, emb_dim = 3, 16
vec = torch.randn(1, n, emb_dim)
a = MultiHeadAttention(emb_dim, 4)

x = a(vec)
#print(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        print(torch.sin(position * div_term))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

pe = PositionalEncoding(10, 0.1, 3)

print(pe.pe)

def positional_enc(num_entries, emb_dim, scale=10000):
    # Output shape should be: num_entries x emb_dim
    positional = scale**(2 * torch.arange(emb_dim) / emb_dim)
    per_vec_position = (torch.arange(num_entries*emb_dim) % num_entries).view(num_entries, emb_dim)
    return torch.sin(per_vec_position/positional)

my_pe = positional_enc(3, 10)

#TODO get parity between positional implementations
print(my_pe)
print((my_pe - pe.pe).sum(2))


#TODO all sequences must be same length atm rip :)
class Encoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, n_heads):
        super(Encoder, self).__init__()
        #TODO multiple layers
        self.emb = nn.Embedding(n_tokens, emb_dim)
        self.drop = nn.Dropout(0.1)
        self.net = MultiHeadAttention(emb_dim=emb_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, emb_dim, bias=True)


    def forward(self, inp):
        #TODO positional encoding
        out1 = self.norm1(self.net(inp) + inp)
        out = self.norm2(self.fc(out1) + out1)
        return out


