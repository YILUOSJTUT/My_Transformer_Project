# feed_forward.py
import torch
import torch.nn as nn
from config import config

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))

class AddNorm(nn.Module):
    """
    Residual connection + LayerNorm
    Output: LayerNorm(x + Sublayer(x))
    """
    def __init__(self, d_model):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))