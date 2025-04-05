# config.py
import math
import torch
import torch.nn as nn

class Config:
    def __init__(self):
        self.vocab_size = 30522   # Will match Hugging Face tokenizer
        self.num_classes = 4      # AG News has 4 classes
        self.d_model = 20         # embedding size
        self.n_heads = 2          # number of attention heads
        self.d_ff = 64            # feedforward inner layer size
        self.max_len = 128         # max sequence length
        self.dropout = 0.1
        self.num_layers = 4      # number of encoder/decoder layers
        self.pad_idx = 4          # padding index
        self.unk_idx = 5          # unknown token index

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads

config = Config()
