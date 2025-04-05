# decoder.py
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward, AddNorm
from config import config

class DecoderLayer(nn.Module):
    """
    One Decoder Layer:
    - Masked self-attention → AddNorm
    - Encoder-Decoder attention → AddNorm
    - FFN → AddNorm
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.addnorm1 = AddNorm(config.d_model)

        self.cross_attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.addnorm2 = AddNorm(config.d_model)

        self.ffn = PositionwiseFeedForward(config.d_model, config.d_ff)
        self.addnorm3 = AddNorm(config.d_model)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.addnorm1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.addnorm2(x, lambda x: self.cross_attn(x, enc_output, enc_output, memory_mask))
        x = self.addnorm3(x, self.ffn)
        return x


class Decoder(nn.Module):
    """
    Full Decoder Stack: N repeated DecoderLayers
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.num_layers)])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x