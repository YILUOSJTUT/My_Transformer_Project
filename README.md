# my_transformer_project

# Transformer1: From-Scratch Implementation in PyTorch  
**Author**: Yi Luo  
**Date**: 2025-04-04

---

## 📌 Project Overview

This project is a modular, from-scratch implementation of the Transformer model architecture as introduced in the paper  
> **"Attention Is All You Need"**  
> *Ashish Vaswani et al., 2017*

The goal of this project is twofold:

1. 🔍 **Deepen Understanding**  
   Build every component of the Transformer step-by-step in PyTorch — from multi-head attention and positional encoding to encoder-decoder stacks — to truly understand how modern attention-based models work.

2. 💼 **Portfolio-Ready Codebase**  
   Showcase clean engineering practices and NLP knowledge for job applications in machine learning, data science, and NLP engineering roles.

---

## 🧱 Project Structure

This is version **`transformer1/`**, the first iteration. Future versions (e.g., `transformer2/`) may include improvements like learnable position encoding, pretrained tokenizers, and larger datasets.


transformer1/
├── config.py                 # model hyperparameters
├── embedding_positional.py  # token + positional encodings
├── attention.py             # scaled dot-product & multi-head attention
├── feed_forward.py          # position-wise FFN
├── encoder.py               # encoder stack
├── decoder.py               # decoder stack
├── transformer.py           # full encoder-decoder Transformer
├── train.py                 # training loop on toy data



---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install torch numpy tqdm

cd transformer1
python train.py




## 📌 Project Overview

### `transformer2/` — Real Dataset Training & Visualization

A classifier-only Transformer encoder trained on [AG News](https://huggingface.co/datasets/ag_news). This version demonstrates:

- ⚙️ Hugging Face tokenizer integration
- 📊 Training on 20k news headlines for topic classification
- 🧠 Encoder-only Transformer architecture
- 📉 Loss & accuracy plots (`training_plot.png`)
- 🛑 Early stopping based on validation
- 💾 Model saving to `best_model.pt`
- 🔍 Model structure inspection using `torchinfo` in `visualize.py`


---
## 🧱 Project Structure

transformer2/
├── config.py
├── embedding_positional.py
├── attention.py
├── feed_forward.py
├── encoder.py
├── transformer.py           # encoder-only classifier
├── train.py                 # training with early stopping & plotting
├── visualize.py             # model summary with torchinfo
├── training_plot.png        # saved learning curve
├── best_model.pt            # saved weights

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install torch numpy tqdm transformers datasets scikit-learn matplotlib torchinfo

cd transformer2
python train.py       # trains model and plots training/validation curves
python visualize.py   # shows architecture summary
