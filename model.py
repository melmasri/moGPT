import torch 
import torch.nn as nn
from torch.nn import functional as F
import math

## Building GPT-2, mainly from the following papers: 
## Language Models are Unsupervised Multitask Learners https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
## Attention is all you need https://arxiv.org/abs/1706.03762
## Improved Language understanding by Generative Pretraining https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf


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
    
    def predict_next(self, x, num_tokens):
        for _ in range(num_tokens):
            logits = self(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x


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
        self.block_size = config['block_size']

        # a single layer for all 3 quantitues, Q, K, V
        # usig c_attn and c_proj names to align with GPT-2 pre-trained hugging face model
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=self.bias)
        self.att_dropout = nn.Dropout(self.dropout)

        self.causal_mask = torch.tril(torch.ones(self.block_size, 
                              self.block_size)).view(1, 1, self.block_size, self.block_size)
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = x.size() # batch, block size, channels (n_embed)
        Q, K, V = self.c_attn(x).split(self.n_embed, dim=-1)
        K = K.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        Q = Q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        V = V.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, n heads, T, n seq)
        d = math.sqrt(K.size(-1))

        raw_weights = (Q @ K.transpose(-2, -1) / d) # (B, n heads, T, n seq) x (B, n heads, n seq, T) --> (B, n heads, T, T)
        raw_weights = raw_weights.masked_fill(self.causal_mask == 0, float('-inf'))
        weights = F.softmax(raw_weights, dim=-1)

        # (B, n heads, T, T) x (B, n heads, T, n seq) --> (B, n heads, T, n seq)
        weights = self.att_dropout(weights)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)

        # Final linear layer
        out = self.c_proj(out)
        return out


class MLP(nn.Module): 
    """ Feed forward network """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_embed = config['n_embed']
        self.bias = config['bias']
        self.dropout = config['dropout']
        # using c_fc and c_proj names to align with GPT-2 in hugging face
        self.c_fc = nn.Linear(self.n_embed, 4 * self.n_embed, bias=self.bias)
        self.c_proj = nn.Linear(4 * self.n_embed, self.n_embed, bias=self.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """ Transformers are made of a number of these blocks"""

    def __init__(self, config) -> None:
        super().__init__()
        self.n_embed = config['n_embed']
        self.bias = config['bias']
        # using names to aling with GPT-2 at HuggingFace
        self.attn = MultiHeadAttention(config)
        self.ln_1= LayerNorm(self.n_embed, bias=self.bias)
        self.ln_2= LayerNorm(self.n_embed, bias=self.bias)
        self.mlp= MLP(config) # Feed forward network

    def forward(self, x) -> torch.tensor:
        x = x + self.attn(self.ln_1())
        x = x + self.mlp(self.ln_2(x))
        return x
    


class GPT(nn.Module): 
    """ GPT model"""
    def __init__(self, config) -> None:
        super().__init__()
        assert config is not None

        self.n_layers = config['n_layers']
        self.n_embed = config['n_embed']
        self.n_heads = config['n_heads']
        self.bias = config['bias']
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.block_size = config['block_size']

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.pos_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        self.dropout = nn.Dropout(self.dropout)

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layers)]), 
            ln_f = LayerNorm(self.n_embed, bias=self.bias)))
        
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)

        self.apply(self._init_weights)

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def forward(self, idx: torch.tensor) -> torch.tensor:
        B, T = idx.size() 
        device = idx.device           

        # Token and position embeddings
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=device, dtype=torch.long))

        x = token_embed + pos_embed
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_num_params(self) -> int:
        """
        Returns the number of parameters in the model without embedding

        We substract the position embeddings. 
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= sum(p.numel() for p in self.token_embedding_table.parameters())
        return n_params


    @torch.no_grad()
    def predict_next(self, x: torch.tensor, num_tokens: int) -> torch.tensor:
        """ Predict the next token given a sequence of tokens

            num_tokens: number of tokens to predict
        """
        for _ in range(num_tokens):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
            logits = self(x_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1) # extract the last token

            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return x

    def _init_weights(self, module):
        """
            Initialization of wieghts following: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)







