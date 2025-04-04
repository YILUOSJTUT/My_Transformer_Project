# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from transformer import Transformer

# --- 1. Toy data: integer token sequences ---
# Input sentence (source): [1, 2, 3, 4]
# Target sentence (decoder input): [1, 2, 3, 4]
# Target label (for loss): shifted left, [2, 3, 4, 5]

src_batch = torch.tensor([[1, 2, 3, 4]])         # (batch_size, seq_len)
tgt_input = torch.tensor([[1, 2, 3, 4]])
tgt_label = torch.tensor([[2, 3, 4, 5]])

# Pad to match max_len
pad_len = config.max_len - src_batch.size(1)
src_batch = torch.nn.functional.pad(src_batch, (0, pad_len), value=config.pad_idx)
tgt_input = torch.nn.functional.pad(tgt_input, (0, pad_len), value=config.pad_idx)
tgt_label = torch.nn.functional.pad(tgt_label, (0, pad_len), value=config.pad_idx)

# --- 2. Mask generation ---
def create_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # shape: (batch, 1, 1, seq_len)

def create_subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).bool()  # shape: (seq_len, seq_len)

src_mask = create_pad_mask(src_batch, config.pad_idx)
tgt_pad_mask = create_pad_mask(tgt_input, config.pad_idx)
tgt_sub_mask = create_subsequent_mask(tgt_input.size(1))
tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(1)

# --- 3. Model, Loss, Optimizer ---
model = Transformer(vocab_size=config.vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- 4. Forward Pass ---
output = model(src_batch, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

# output: (batch, seq_len, vocab_size) → flatten to (batch * seq_len, vocab_size)
# target: (batch, seq_len) → flatten to (batch * seq_len)
output = output.view(-1, config.vocab_size)
tgt_label = tgt_label.view(-1)

loss = criterion(output, tgt_label)
print(f"Loss: {loss.item():.4f}")

# --- 5. Backward + Optimizer Step ---
loss.backward()
optimizer.step()