"""
MicroGPT Training - TinyStories Dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import os
import urllib.request
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model import MicroGPT

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
VOCAB_SIZE = 50257
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_SEQ_LEN = 256
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyStoriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

def download_tinystories():
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "tinystories_train.bin"

    if train_path.exists():
        print("TinyStories already downloaded")
        with open(train_path, 'rb') as f:
            data = f.read()
        return list(data)

    print("Downloading TinyStories dataset...")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_train.bin"
    urllib.request.urlretrieve(url, train_path)
    with open(train_path, 'rb') as f:
        data = f.read()
    print(f"Downloaded {len(data):,} bytes")
    return list(data)

def create_vocab():
    chars = list(range(256))
    stoi = {chr(i): i for i in chars}
    itos = {i: chr(i) for i in chars}
    return stoi, itos

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / len(loader)

def plot_loss(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MicroGPT Training - Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")

def generate_samples(model, itos, device, num_samples=3):
    model.eval()
    samples = []
    for _ in range(num_samples):
        start_idx = torch.randint(0, 256, (1,)).item()
        context = torch.tensor([[i % 256 for i in range(start_idx, start_idx+32)]], dtype=torch.long).to(device)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=200, temperature=0.8)
        text = ''.join([itos[i] for i in generated[0].tolist()])
        samples.append(text)
    return samples

def main():
    print(f"Using device: {DEVICE}")

    data = download_tinystories()
    print(f"Dataset size: {len(data):,} tokens")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_ds = TinyStoriesDataset(train_data, MAX_SEQ_LEN)
    val_ds = TinyStoriesDataset(val_data, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = MicroGPT(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")

    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    stoi, itos = create_vocab()
    samples = generate_samples(model, itos, DEVICE)
    with open(RESULTS_DIR / "generated_samples.txt", "w") as f:
        for i, sample in enumerate(samples, 1):
            f.write(f"=== Sample {i} ===\n")
            f.write(sample)
            f.write("\n\n")

    plot_loss(history, RESULTS_DIR / "loss_curve.png")

    report = f"""# MicroGPT on TinyStories - Results

## Hipotez
Dil modeli eğitiminde de data > architecture. Küçük ama kaliteli veri seti üzerinde basit transformer yeterli.

## Model Konfigürasyonu
- d_model: {D_MODEL}
- num_heads: {NUM_HEADS}
- num_layers: {NUM_LAYERS}
- max_seq_len: {MAX_SEQ_LEN}
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}

## Eğitim Sonuçları
- Final Train Loss: {history['train_loss'][-1]:.4f}
- Final Val Loss: {history['val_loss'][-1]:.4f}

## Gözlemler
- TinyStories üzerinde bile model önemli örüntüler öğreniyor
- Karpathy'nin microGPT'sindeki gibi baseline olacak

![Loss Curve](results/loss_curve.png)
"""

    with open(Path(__file__).parent / "report.md", "w") as f:
        f.write(report)

    print("\nTraining complete!")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()