# attention.py
import torch
import torch.nn as nn
import math
from config import config

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    Computes attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
    """
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, head, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Fill masked positions with -inf before softmax
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # shape: (batch, head, seq_len, d_v)
        return output, attn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module from section 3.2.2
    """
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections → split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        # Concatenate heads and run final linear projection
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.out_proj(concat)
        return self.dropout(output)