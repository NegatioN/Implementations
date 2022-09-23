import torch
import math
from torch import nn
from torch.nn import functional as F




#TODO have some test dataset for this.
#TODO make basic attention (as a function? needs state though)
#TODO layernorm is used between layers.


def attention(queries, keys, values, scale):
    x = (queries @ keys.T) / scale
    return F.softmax(x) @ values

#TODO add batching?
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, seq_len):
        super(MultiHeadAttention, self).__init__()
        # TODO initialization scheme for parameters
        self.values = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.queries = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.keys = nn.Parameter(torch.randn((emb_dim, emb_dim)))
        self.fc = nn.Linear(seq_len * emb_dim, emb_dim, bias=False)
        self.scale = math.sqrt(emb_dim)
    
    def forward(self, inp):
        #assert len(inp.size()) == 2, f'Input is not of dim=2, but dim={len(inp.size())}'
        v = inp @ self.values
        q = inp @ self.queries
        k = inp @ self.keys
        x = attention(q, k, v, self.scale).view(-1)
        return self.fc(x)


n, emb_dim = 3, 10
vec = torch.randn(n, emb_dim)
a = MultiHeadAttention(emb_dim, n)

x = a(vec)
x
