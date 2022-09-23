import torch
import math
from torch import nn
from torch.nn import functional as F




#TODO have some test dataset for this.
#TODO make basic attention (as a function? needs state though)
#TODO layernorm is used between layers.


def attention(queries, keys, values, scale):
    x = (queries @ keys) / scale
    return F.softmax(x) * values

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim):
        super(MultiHeadAttention, self).__init__()
        self.values = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.queries = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.keys = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.scale = math.sqrt(emb_dim)
    
    def forward(self, inp):
        v = inp @ self.values
        q = inp @ self.queries
        k = inp @ self.keys
        return attention(q, k, v, self.scale)


n, emb_dim = 3, 10
vec = torch.randn(emb_dim)
a = MultiHeadAttention(emb_dim)

x = a(vec)
x

nn.MultiheadAttention??
