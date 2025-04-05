# embedding_positional.py
import torch
import torch.nn as nn
import math
from config import config

class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer with support for padding.
    """
    def __init__(self, vocab_size, d_model, pad_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(config.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in section 3.5 of the paper.
    Adds position-dependent signals to the token embeddings.
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)    # dim 0, 2, 4...
        pe[:, 1::2] = torch.cos(position * div_term)    # dim 1, 3, 5...

        # Register buffer to avoid model parameters, but still move with `.to(device)`
        self.register_buffer('pe', pe.unsqueeze(0))     # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]