# transformer.py
import torch
import torch.nn as nn
from config import config
from encoder import Encoder
from decoder import Decoder
from embedding_positional import TokenEmbedding, PositionalEncoding

class Transformer(nn.Module):
    """
    Full Transformer model:
    - Input/Output embedding + positional encoding
    - Encoder stack
    - Decoder stack
    - Final Linear + Softmax projection to vocab
    """
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.embedding_input = TokenEmbedding(vocab_size, config.d_model, config.pad_idx)
        self.embedding_output = TokenEmbedding(vocab_size, config.d_model, config.pad_idx)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_len)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.output_layer = nn.Linear(config.d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src, tgt: (batch_size, seq_len)

        # Apply embeddings and positional encodings
        src_embed = self.positional_encoding(self.embedding_input(src))
        tgt_embed = self.positional_encoding(self.embedding_output(tgt))

        # Encoder
        memory = self.encoder(src_embed, mask=src_mask)

        # Decoder
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        # Final projection to vocabulary size
        logits = self.output_layer(output)
        probs = self.softmax(logits)
        return probs