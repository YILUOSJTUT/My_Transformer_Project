# transformer2/transformer.py
import torch
import torch.nn as nn
from config import config
from encoder import Encoder
from embedding_positional import TokenEmbedding, PositionalEncoding

class TransformerClassifier(nn.Module):
    """
    Transformer encoder-only model for text classification (AG News)
    """
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, config.d_model, config.pad_idx)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_len)
        self.encoder = Encoder()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Mean pooling over sequence
        self.classifier = nn.Linear(config.d_model, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        x = self.embedding(x)                             # [B, T, D]
        x = self.positional_encoding(x)                   # [B, T, D]
        x = self.encoder(x, mask=mask)                    # [B, T, D]
        x = x.mean(dim=1)                                 # [B, D]
        x = self.dropout(x)
        return self.classifier(x)                         # [B, num_classes]
