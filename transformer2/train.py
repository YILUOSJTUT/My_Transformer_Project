# transformer2/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from transformer import TransformerClassifier

# --- 1. Load and split AG News dataset ---
raw_dataset = load_dataset("ag_news")
split_data = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_data = split_data["train"].select(range(20000))  # Use 20,000 samples
val_data = split_data["test"]
test_data = raw_dataset["test"].select(range(1000))     # 1k for final test

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_len
    )

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

train_data.set_format(type="torch", columns=["input_ids", "label"])
val_data.set_format(type="torch", columns=["input_ids", "label"])
test_data.set_format(type="torch", columns=["input_ids", "label"])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# --- 2. Initialize Model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = TransformerClassifier(vocab_size=tokenizer.vocab_size, num_classes=config.num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# --- 3. Training Loop ---
print("\U0001F501 Starting training...")
train_losses = []
val_accuracies = []
best_val_acc = 0
patience_counter = 0
patience_limit = 5

for epoch in range(16):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"✅ Epoch {epoch+1} complete | Avg Train Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_acc)
    print(f"\U0001F4C9 Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("\U0001F4BE Model saved!")
    else:
        patience_counter += 1
        print(f"⏳ No improvement. Patience {patience_counter}/{patience_limit}")
        if patience_counter >= patience_limit:
            print("⛔ Early stopping triggered.")
            break

# --- 4. Final Test Evaluation ---
print("\n\U0001F4CA Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"✅ Final Test Accuracy: {test_acc:.4f}")

# --- 5. Plot Training Curve ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
