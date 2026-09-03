"""
Legacy: DeBERTa-v3-base sentiment fine-tune (positive/negative/neutral).
Output: models/deberta-sentiment/
"""

import os
import csv
import time
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

MODEL_PATH = Path("models/deberta-v3-base")
SAVE_PATH  = Path("models/deberta-sentiment")
TRAIN_CSV  = Path("data/complete/train.csv")
VAL_CSV    = Path("data/complete/val.csv")
TEST_CSV   = Path("data/complete/test.csv")

EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 128

LABELS    = ["positive", "negative", "neutral"]
LOG_EVERY = 200  # interim summary every N batches

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class NewsDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer):
        self.tokenizer = tokenizer
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((
                    row["text"][:512] if row["text"] else "",
                    LABELS.index(row["sentiment"]),
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(label, dtype=torch.long),
        }


class DeBERTaSentiment(nn.Module):
    def __init__(self, deberta):
        super().__init__()
        self.deberta = deberta
        self.dropout = nn.Dropout(0.1)
        self.head    = nn.Linear(768, len(LABELS))

    def forward(self, input_ids, attention_mask):
        out    = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :].float())  # cast to float32
        return self.head(pooled)


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    from collections import Counter
    all_preds = []

    pbar = tqdm(loader, desc="  train", leave=False)
    for step, batch in enumerate(pbar, 1):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(1)
        total_loss += loss.item()
        correct    += (preds == lbls).sum().item()
        total      += len(lbls)
        all_preds.extend(preds.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")

        if step % LOG_EVERY == 0:
            lr  = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / step
            acc = correct / total
            dist = Counter(LABELS[p] for p in all_preds[-LOG_EVERY*BATCH_SIZE:])
            print(
                f"\n  [Epoch {epoch} | batch {step}/{len(loader)}] "
                f"loss={avg_loss:.3f}  acc={acc:.3f}  lr={lr:.2e}"
            )
            print(
                f"  Prediction distribution (last {LOG_EVERY} batch): "
                + "  ".join(f"{k}={v}" for k, v in dist.items())
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)
        logits = model(ids, mask)
        total_loss += criterion(logits, lbls).item()
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def test_report(model, loader):
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        logits = model(ids, mask)
        all_true.extend(batch["label"].tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
    print("\n── Sentiment Test Report (DeBERTa) ──")
    print(classification_report(all_true, all_pred, target_names=LABELS))


def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    deberta   = AutoModel.from_pretrained(MODEL_PATH)
    model     = DeBERTaSentiment(deberta).float().to(DEVICE)

    print("Loading dataset...")
    train_loader = DataLoader(NewsDataset(TRAIN_CSV, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(NewsDataset(VAL_CSV,   tokenizer), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(NewsDataset(TEST_CSV,  tokenizer), batch_size=BATCH_SIZE)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion     = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    print(f"\nTraining starts — {EPOCHS} epoch, batch={BATCH_SIZE}, lr={LR}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        v_loss,  v_acc  = evaluate(model, val_loader, criterion)
        elapsed = (time.time() - t0) / 60

        print(
            f"Epoch {epoch}/{EPOCHS}  {elapsed:.1f}min  "
            f"train_loss={tr_loss:.3f}  val_loss={v_loss:.3f}  val_acc={v_acc:.3f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model.deberta.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            torch.save(model.head.state_dict(), SAVE_PATH / "head.pt")
            print(f"  ✓ Best model saved → {SAVE_PATH}")

    print("\nTest set evaluation:")
    test_report(model, test_loader)
    print(f"\n✓ Done. Model: {SAVE_PATH}")


if __name__ == "__main__":
    main()
