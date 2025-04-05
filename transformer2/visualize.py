# visualize.py

from transformer import TransformerClassifier
from config import config
import torch
from torchinfo import summary

model = TransformerClassifier(vocab_size=30522, num_classes=4)

# Create example input of the correct type (LongTensor)
dummy_input = torch.ones((32, config.max_len), dtype=torch.long)

# Show model summary using the dummy input
summary(model, input_data=dummy_input, col_names=["input_size", "output_size", "num_params"])
