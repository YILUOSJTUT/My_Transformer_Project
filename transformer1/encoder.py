# encoder.py
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward, AddNorm
from config import config

class EncoderLayer(nn.Module):
    """
    One Encoder Layer = Self-attention → AddNorm → FFN → AddNorm
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.addnorm1 = AddNorm(config.d_model)

        self.ffn = PositionwiseFeedForward(config.d_model, config.d_ff)
        self.addnorm2 = AddNorm(config.d_model)

    def forward(self, x, mask=None):
        x = self.addnorm1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.addnorm2(x, self.ffn)
        return x


class Encoder(nn.Module):
    """
    Full Encoder Stack: N repeated EncoderLayers
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x