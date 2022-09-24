import torch
import math
from torch import nn
from torch.nn import functional as F




#TODO have some test dataset for this.
#TODO make basic attention (as a function? needs state though)
#TODO layernorm is used between layers.


def attention(queries, keys, values, scale):
    x = (queries @ keys.transpose(1,2)) / scale
    return F.softmax(x) @ values

#TODO add batching?
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.values = nn.Linear(emb_dim, emb_dim, bias=True)
        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc = nn.Linear(emb_dim, emb_dim, bias=False)
        self.scale = math.sqrt(emb_dim)

    def forward(self, inp):
        #TODO only takes input of a single sequence at this point. no batching.
        #assert len(inp.size()) == 2, f'Input is not of dim=2, but dim={len(inp.size())}'
        k, q, v = self.keys(inp), self.queries(inp), self.values(inp)
        k, q, v = [x.view(-1, self.heads, self.head_dim) for x in [k, q, v]]
        x = attention(q, k, v, self.scale).view(-1, self.heads * self.head_dim)
        return self.fc(x)


n, emb_dim = 3, 12
vec = torch.randn(n, emb_dim)
a = MultiHeadAttention(emb_dim, n)

x = a(vec)
x
