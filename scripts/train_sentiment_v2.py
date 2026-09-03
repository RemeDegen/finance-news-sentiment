"""
A2 (FINAL MODEL) - FinBERT sentiment fine-tune on complete_v2 (35k dual-judge
set) with class weights. 3 epochs, batch 16, lr 2e-5, dynamic padding.

WEIGHT_MODE: "balanced" = inverse frequency (A1), "sqrt" = its square root
(A2, final), None = plain cross-entropy.
Result (sqrt): test acc 0.842 / macro F1 0.808. Details: ai_training_notes.md section 5.

Run:    uv run scripts/train_sentiment_v2.py
Output: models/finbert-sentiment-v2-{balanced|sqrt|plain}/ (+ train_log.txt)
"""

import csv
import math
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score

EPOCHS      = 3
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 128
WEIGHT_MODE = "sqrt"       # "balanced" | "sqrt" | None

MODEL_PATH = Path("models/finbert")
SAVE_PATH  = Path(f"models/finbert-sentiment-v2-{WEIGHT_MODE or 'plain'}")
LOG_PATH   = SAVE_PATH / "train_log.txt"
TRAIN_CSV  = Path("data/complete_v2/train.csv")
VAL_CSV    = Path("data/complete_v2/val.csv")
TEST_CSV   = Path("data/complete_v2/test.csv")


def log(msg: str):
    print(msg)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

LABELS = ["positive", "negative", "neutral"]

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class NewsDataset(Dataset):
    def __init__(self, csv_path: Path):
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
        return self.rows[idx]


def make_collate(tokenizer):
    """Dynamic padding: each batch is padded to its own longest sentence.
    Median is 21 tokens; padding to a fixed 128 meant ~5x wasted compute.
    Result is mathematically identical (attention mask), just faster."""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            max_length=MAX_LEN,
            padding="longest",
            pad_to_multiple_of=8,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          torch.tensor(labels, dtype=torch.long),
        }
    return collate


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Like HF group_by_length: shuffle the indices → sort by length inside
    chunks of 50 batches → shuffle the batch order. Similar-length sentences
    land in the same batch, so padding (and wasted compute) drops a lot."""

    def __init__(self, lengths, batch_size, mega=50):
        self.lengths, self.bs, self.mega = lengths, batch_size, mega

    def __iter__(self):
        idx, chunk, batches = torch.randperm(len(self.lengths)).tolist(), self.bs * self.mega, []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i:i + chunk], key=lambda j: self.lengths[j])
            batches += [part[k:k + self.bs] for k in range(0, len(part), self.bs)]
        for b in torch.randperm(len(batches)).tolist():
            yield batches[b]

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)


def sorted_batches(ds, batch_size):
    """For val/test: sort by length, fixed batch list (metrics are order-independent)."""
    idx = sorted(range(len(ds)), key=lambda j: len(ds.rows[j][0]))
    return [idx[k:k + batch_size] for k in range(0, len(idx), batch_size)]


def class_weights(dataset: NewsDataset) -> torch.Tensor | None:
    if WEIGHT_MODE is None:
        return None
    counts = Counter(lbl for _, lbl in dataset.rows)
    total = len(dataset.rows)
    w = [total / (len(LABELS) * counts[i]) for i in range(len(LABELS))]
    if WEIGHT_MODE == "sqrt":
        w = [math.sqrt(x) for x in w]
    print("Class weight (" + WEIGHT_MODE + "):")
    for name, weight, i in zip(LABELS, w, range(len(LABELS))):
        print(f"  {name:<8} n={counts[i]:>6}  w={weight:.3f}")
    return torch.tensor(w, dtype=torch.float32)


class FinBERTSentiment(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, len(LABELS))

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        return self.head(pooled)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_true, all_pred = [], []
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)
        logits = model(ids, mask)
        total_loss += criterion(logits, lbls).item()
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
        all_true.extend(batch["label"].tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader), correct / total, all_true, all_pred


def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bert      = AutoModel.from_pretrained(MODEL_PATH)
    model     = FinBERTSentiment(bert).to(DEVICE)

    print("Loading dataset...")
    collate  = make_collate(tokenizer)
    train_ds = NewsDataset(TRAIN_CSV)
    val_ds   = NewsDataset(VAL_CSV)
    test_ds  = NewsDataset(TEST_CSV)
    train_loader = DataLoader(
        train_ds, collate_fn=collate,
        batch_sampler=LengthGroupedSampler([len(t) for t, _ in train_ds.rows], BATCH_SIZE))
    val_loader  = DataLoader(val_ds,  collate_fn=collate, batch_sampler=sorted_batches(val_ds,  BATCH_SIZE))
    test_loader = DataLoader(test_ds, collate_fn=collate, batch_sampler=sorted_batches(test_ds, BATCH_SIZE))

    weights   = class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(DEVICE) if weights is not None else None)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_macro_f1 = 0.0

    log(f"\n═══ {time.strftime('%Y-%m-%d %H:%M')}  device={DEVICE}  {EPOCHS} epoch  batch={BATCH_SIZE}  "
        f"lr={LR}  max_len={MAX_LEN}  weight={WEIGHT_MODE}  padding=dynamic ═══\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc, v_true, v_pred = evaluate(model, val_loader, criterion)
        elapsed = (time.time() - t0) / 60

        log(
            f"Epoch {epoch}/{EPOCHS}  {elapsed:.1f}min  "
            f"train_loss={tr_loss:.3f}  val_loss={v_loss:.3f}  val_acc={v_acc:.3f}"
        )
        log(classification_report(v_true, v_pred, target_names=LABELS, digits=3))

        macro_f1 = f1_score(v_true, v_pred, average="macro")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model.bert.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            torch.save(model.head.state_dict(), SAVE_PATH / "head.pt")
            log(f"  ✓ Best model saved (macro_f1={macro_f1:.3f}) → {SAVE_PATH}")

    # Test runs with the BEST checkpoint on disk (not the last epoch) —
    # so the report and the saved model use the exact same weights.
    print(f"\nLoading best checkpoint ← {SAVE_PATH}")
    model = FinBERTSentiment(AutoModel.from_pretrained(SAVE_PATH)).to(DEVICE)
    model.head.load_state_dict(torch.load(SAVE_PATH / "head.pt", map_location=DEVICE))

    print("\nTest set evaluation:")
    _, _, t_true, t_pred = evaluate(model, test_loader, criterion)
    log(f"\n── Sentiment Test Report (v2, weight={WEIGHT_MODE}, best val macro_f1={best_macro_f1:.3f}) ──")
    log(classification_report(t_true, t_pred, target_names=LABELS, digits=3))
    print(f"\n✓ Done. Model: {SAVE_PATH}  Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
