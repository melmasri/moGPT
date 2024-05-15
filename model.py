import torch 
import torch.nn as nn
from torch.nn import functional as F
import math


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class BigramLanguageModel(nn.Module): 

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):

        logits = self.token_embedding_table(idx) # (B, T, C)
        return logits


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module implmeneted from scratch

        Follow Attention is All You Need paper: https://arxiv.org/pdf/1706.03762

        - Can add a dropout at the last stage, after the linear layer

    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        # n_embed is a multiple of number of heads
        assert config['n_embed'] % config['n_heads'] == 0
        self.n_heads = config['n_heads']
        self.n_embed = config['n_embed']
        self.bias = config['bias']
        self.dropout = config['dropout']

        # a single layer for all 3 quantitues, Q, K, V
        self.W_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=self.bias)
        self.L_atten = nn.Linear(self.n_embed, self.n_embed, bias=self.bias)
        self.att_dropout = nn.Dropout(self.dropout)
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = x.size() # batch, block size, channels (n_embed)
        Q, K, V = self.W_attn(x).split(self.n_embed, dim=-1)
        K = K.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        Q = Q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        V = V.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        d = math.sqrt(K.size(-1))

        raw_weights = (Q @ K.transpose(-2, -1) / d) # (B, n heads, T, n seq) x (B, n heads, n seq, T) --> (B, n heads, T, T)
        weights = F.softmax(raw_weights, dim=-1)

        # (B, n heads, T, T) x (B, n heads, T, n seq) --> (B, n heads, T, n seq)
        weights = self.att_dropout(weights)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)

        # Final linear layer
        out = self.L_atten(out)
        return out


class MLP(nn.Module): 
    """ Feed forward network """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_embed = config['n_embed']
        self.bias = config['bias']
        self.dropout = config['dropout']

        self.fc1 = nn.Linear(self.n_embed, 4 * self.n_embed, bias=self.bias)
        self.fc2 = nn.Linear(4 * self.n_embed, self.n_embed, bias=self.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """ Transformers are made of a number of these blocks"""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ln= LayerNorm(config['n_embed'], bias=config['bias'])
        self.mlp= MLP(config) # Feed forward network

    def forward(self, x) -> torch.tensor:
        x = x + self.attn(x)
        x = x + self.mlp(self.l(x))
        return x