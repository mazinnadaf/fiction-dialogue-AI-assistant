import torch
import torch.nn as nn
import math
import copy
import warnings
warnings.filterwarnings("ignore")

def is_interactive_notebook():
    return __name__ == "__main__"

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

# Overall architecture
class DecoderOnlyLanguageModel(nn.Module):
    def __init__(self, decoder, embed, generator):
        super(DecoderOnlyLanguageModel, self).__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator # For converting the decoder output into probabilities

    def forward(self, tgt, tgt_mask):
        return self.decoder(self.embed(tgt), tgt_mask=tgt_mask)

# Converts decoder output into final probability distribution
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # Linear transformation converting decoder output to vector of size vocab
        self.proj = nn.Linear(d_model, vocab) # Size of decoder's output vectors and size of target vocabulary

    def forward(self, x):
        # return log_softmax(self.proj(x), dim=-1) # Linear transformation to vocab size
        return self.proj(x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Need layers to have mean 0 and variance 1
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Residual Connection (Adds the input back): x + sublayer_output
# Shortcut for gradients to flow backward
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    # padding tokens are added to sequences in a batch to make them all have the same length
    # tgt_mask is masking for the future tokens
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
            # x is the input to the decoder at any given step
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    # K, V of cross attention are from encoder(memory)
    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    # Dimensionality of the query vectors
    d_k = query.size(-1)
    # Query/key dot product
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # Scores times the value vector
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # d_model is just the vector size of each token as it moves through the Transformer
        self.d_k = d_model // h # Size of each head
        self.h = h # Number of heads
        # 4 Linear layers, each of first 3 transforms the input to fit. THEN head division
        self.linears = clones(nn.Linear(d_model, d_model), 4) # why not more
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # Parameters initially identical input tensors when passed in?
    # The linear layers are the things that split them (weight matrix from video, etc.)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # adds a head dimension at position 1
            # Prepares mask for broadcasting to all heads in attention computation
        nbatches = query.size(0)

        # Need to split them based on the number of heads
        # They are reshaped from [batch, seq_len, d_model] to [batch, h, seq_len, d_k]
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # Outputs from all heads are concatenated back together
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        # Pass through the final linear layer to transform back to initial dimensionality
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

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
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(
    vocab_size, N=12, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = DecoderOnlyLanguageModel(
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model