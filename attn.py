import numpy as np
n_token, n_dim = 3, 4
size = (n_token, n_dim)
Q = np.random.randn(*size)
K = np.random.randn(*size)
V = np.random.randn(*size)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    scale = np.sqrt(K.shape[-1])
    return softmax((Q @ K.transpose()) / scale) @ V

print(attention(Q, K, V))

# learnable parameters in the model
n_out = n_dim
Q_w = np.random.randn(n_dim, n_out)
K_w = np.random.randn(n_dim, n_out)
V_w = np.random.randn(n_dim, n_out)

X = np.random.randn(*size) #input

def single_head_attention(X, Q_w, K_w, V_w, f_attn=attention):
    Q = X @ Q_w
    K = X @ K_w
    V = X @ V_w

    return f_attn(Q, K, V)

print(single_head_attention(X, Q_w, K_w, V_w))

def make_heads(n_head, n_dim, n_out):
    heads = []
    n_out /= n_head
    n_out = int(n_out)

    for _ in range(n_head):
        Q_w = np.random.randn(n_dim, n_out)
        K_w = np.random.randn(n_dim, n_out)
        V_w = np.random.randn(n_dim, n_out)
        heads.append([Q_w, K_w, V_w])

    return heads

n_out = 3
heads = make_heads(3, n_dim, n_out)
mixer = np.random.randn(n_out, n_out)

def multi_head_attention(X, heads, mixer, f_attn=attention):
    X_o = [single_head_attention(X, Q_w, K_w, V_w, f_attn) for (Q_w, K_w, V_w) in heads]
    return np.concatenate(X_o, axis=-1) @ mixer

print(multi_head_attention(X, heads, mixer))

mask = np.tril(np.ones((n_token, n_token)).astype(np.bool))
def attention_masked(Q, K, V):
    scale = np.sqrt(K.shape[-1])
    return softmax(np.where(mask, (Q @ K.transpose()) / scale, -np.inf)) @ V

print(multi_head_attention(X, heads, mixer, attention_masked))

## RULE: with residual connections, we need to ensure that the shape of the output of multi_head_attention is the same as the shape of input X.
## n_dim must be cleanly divisible by n_head so that we can always reconstruct the same shape
## is there a good formalization for this?

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)

    return gamma * x_norm + beta

def mlp(x, W_1, W_2):
    # Expand features (e.g., 4 -> 16), apply ReLU, project back (16 -> 4)
    return np.maximum(0, x @ W_1) @ W_2

def transformer_block(n_head, n_dim, h_dim):
    assert n_dim % n_head == 0, "embedding dimension not cleanly divisible by n_heads"
    gammas, betas = np.ones((2, n_dim)), np.zeros((2, n_dim))
    MLP_w1 = np.random.randn(n_dim, h_dim)
    MLP_w2 = np.random.randn(h_dim, n_dim)
    heads = make_heads(n_head, n_dim, n_dim)
    O_w = np.random.randn(n_dim, n_dim)
    def forward(X):
        X = X + multi_head_attention(layer_norm(X, gammas[0], betas[0]), heads, O_w, f_attn=attention_masked)
        norm_X2 = layer_norm(X, gammas[1], betas[1])
        return X + mlp(norm_X2, MLP_w1, MLP_w2)
    return forward



f = transformer_block(2, 4, 16)
print(f(X))

