"""
Legacy: FinBERT multi-task fine-tune with positive/negative only; neutral was
meant to be caught at inference by a 0.40-0.60 probability band.
Output: models/finbert-multitask-po-ne/
"""

import csv
import time
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel

# ── Settings ─────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/finbert")
SAVE_PATH    = Path("models/finbert-multitask-po-ne")
TRAIN_CSV    = Path("data/complete_po_ne/train.csv")
VAL_CSV      = Path("data/complete_po_ne/val.csv")
TEST_CSV     = Path("data/complete_po_ne/test.csv")

EPOCHS       = 3
BATCH_SIZE   = 16
LR           = 2e-5
MAX_LEN      = 128

SENTIMENT_LABELS = ["positive", "negative"]
REASON_LABELS    = ["crypto", "macro", "geopolitics", "commodities", "equities", "regulatory", "other"]

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ── Dataset ──────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer):
        self.tokenizer = tokenizer
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((
                    row["text"][:512] if row["text"] else "",
                    SENTIMENT_LABELS.index(row["sentiment"]),
                    REASON_LABELS.index(row["reason"]),
                ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, s_id, r_id = self.rows[idx]
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
            "sentiment":      torch.tensor(s_id, dtype=torch.long),
            "reason":         torch.tensor(r_id, dtype=torch.long),
        }


# ── Model: FinBERT + two heads ────────────────────────────────────────────
class FinBERTMultiTask(nn.Module):
    def __init__(self, bert, num_sentiment=2, num_reason=7):
        super().__init__()
        self.bert           = bert
        self.dropout        = nn.Dropout(0.1)
        self.sentiment_head = nn.Linear(768, num_sentiment)
        self.reason_head    = nn.Linear(768, num_reason)

    def forward(self, input_ids, attention_mask):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.dropout(out.pooler_output)
        return self.sentiment_head(pooled), self.reason_head(pooled)


# ── Training ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct_s, correct_r, total = 0, 0, 0, 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        s_lbl = batch["sentiment"].to(DEVICE)
        r_lbl = batch["reason"].to(DEVICE)

        s_logits, r_logits = model(ids, mask)
        loss = criterion(s_logits, s_lbl) + criterion(r_logits, r_lbl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct_s  += (s_logits.argmax(1) == s_lbl).sum().item()
        correct_r  += (r_logits.argmax(1) == r_lbl).sum().item()
        total      += len(s_lbl)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader), correct_s / total, correct_r / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct_s, correct_r, total = 0, 0, 0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        s_lbl = batch["sentiment"].to(DEVICE)
        r_lbl = batch["reason"].to(DEVICE)

        s_logits, r_logits = model(ids, mask)
        loss = criterion(s_logits, s_lbl) + criterion(r_logits, r_lbl)

        total_loss += loss.item()
        correct_s  += (s_logits.argmax(1) == s_lbl).sum().item()
        correct_r  += (r_logits.argmax(1) == r_lbl).sum().item()
        total      += len(s_lbl)

    return total_loss / len(loader), correct_s / total, correct_r / total


@torch.no_grad()
def test_report(model, loader):
    model.eval()
    all_s_true, all_s_pred = [], []
    all_r_true, all_r_pred = [], []
    pos_probs = []  # P(positive) — for threshold/neutral analysis

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        s_logits, r_logits = model(ids, mask)

        s_prob = F.softmax(s_logits, dim=1)[:, 0]  # 0 = positive
        pos_probs.extend(s_prob.cpu().tolist())

        all_s_true.extend(batch["sentiment"].tolist())
        all_s_pred.extend(s_logits.argmax(1).cpu().tolist())
        all_r_true.extend(batch["reason"].tolist())
        all_r_pred.extend(r_logits.argmax(1).cpu().tolist())

    print("\n── Sentiment Test Report (2 classes) ──")
    print(classification_report(all_s_true, all_s_pred, target_names=SENTIMENT_LABELS))
    print("── Reason Test Report ──")
    print(classification_report(all_r_true, all_r_pred, target_names=REASON_LABELS))

    # ── Neutral-threshold analysis: P(positive) distribution ──
    print("── P(positive) distribution (for neutral threshold) ──")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = Counter()
    for p in pos_probs:
        for i in range(len(bins) - 1):
            if bins[i] <= p < bins[i + 1] or (i == len(bins) - 2 and p == 1.0):
                hist[i] += 1
                break
    total = len(pos_probs)
    for i in range(len(bins) - 1):
        n = hist[i]
        bar = "█" * int(40 * n / total)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {n:5d} {bar}")
    band = sum(1 for p in pos_probs if 0.40 <= p <= 0.60)
    print(f"\n  0.40–0.60 band (undecided→neutral candidate): {band} / {total}  ({100*band/total:.1f}%)")


# ── Main flow ────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bert      = AutoModel.from_pretrained(MODEL_PATH)
    model     = FinBERTMultiTask(bert).to(DEVICE)

    print("Loading dataset...")
    train_loader = DataLoader(NewsDataset(TRAIN_CSV, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(NewsDataset(VAL_CSV,   tokenizer), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(NewsDataset(TEST_CSV,  tokenizer), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    print(f"\nTraining starts — {EPOCHS} epoch, batch={BATCH_SIZE}, lr={LR}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_s, tr_r = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_s, v_r    = evaluate(model, val_loader, criterion)
        elapsed = (time.time() - t0) / 60

        print(
            f"Epoch {epoch}/{EPOCHS}  {elapsed:.1f}min  "
            f"train_loss={tr_loss:.3f}  val_loss={v_loss:.3f}  "
            f"val_sentiment={v_s:.3f}  val_reason={v_r:.3f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model.bert.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            torch.save({
                "sentiment_head": model.sentiment_head.state_dict(),
                "reason_head":    model.reason_head.state_dict(),
            }, SAVE_PATH / "heads.pt")
            print(f"  ✓ Best model saved → {SAVE_PATH}")

    print("\nTest set evaluation:")
    test_report(model, test_loader)
    print(f"\n✓ Training complete. Model: {SAVE_PATH}")


if __name__ == "__main__":
    main()
