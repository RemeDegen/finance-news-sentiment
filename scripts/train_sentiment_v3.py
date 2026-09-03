"""
A3 - FinBERT on complete_v3 (35k + wave 1, ~40k; dataset later deleted).
A2 recipe plus warmup 6% + linear decay, 5 epochs, best-val checkpoint.
Result 0.853 / 0.820, but mixed spec v2/v3 doctrine -> not used as final.
Imports model, collate and sampler from train_sentiment_v2.py.

Run:    uv run scripts/train_sentiment_v3.py
Output: models/finbert-sentiment-v3-sqrt/ (+ train_log.txt)
"""

import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

import train_sentiment_v2 as v2
from train_sentiment_v2 import (
    LABELS, DEVICE, NewsDataset, FinBERTSentiment,
    make_collate, LengthGroupedSampler, sorted_batches, evaluate,
)

EPOCHS       = 5
BATCH_SIZE   = 16
LR           = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_MODE  = "sqrt"

MODEL_PATH = Path("models/finbert")
SAVE_PATH  = Path("models/finbert-sentiment-v3-sqrt")
LOG_PATH   = SAVE_PATH / "train_log.txt"
DATA_DIR   = Path("data/complete_v3")
TRAIN_CSV, VAL_CSV, TEST_CSV = (DATA_DIR / f"{s}.csv" for s in ("train", "val", "test"))


def log(msg: str):
    print(msg)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader), correct / total


def main():
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise SystemExit(f"No data: {p} — build complete_v3 with build_dataset first")
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = FinBERTSentiment(AutoModel.from_pretrained(MODEL_PATH)).to(DEVICE)

    print("Loading dataset...")
    collate  = make_collate(tokenizer)
    train_ds, val_ds, test_ds = NewsDataset(TRAIN_CSV), NewsDataset(VAL_CSV), NewsDataset(TEST_CSV)
    train_loader = DataLoader(
        train_ds, collate_fn=collate,
        batch_sampler=LengthGroupedSampler([len(t) for t, _ in train_ds.rows], BATCH_SIZE))
    val_loader  = DataLoader(val_ds,  collate_fn=collate, batch_sampler=sorted_batches(val_ds,  BATCH_SIZE))
    test_loader = DataLoader(test_ds, collate_fn=collate, batch_sampler=sorted_batches(test_ds, BATCH_SIZE))

    v2.WEIGHT_MODE = WEIGHT_MODE          # class_weights reads the module constant in v2
    weights   = v2.class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_macro_f1 = 0.0

    log(f"\n═══ {time.strftime('%Y-%m-%d %H:%M')}  A3  device={DEVICE}  data={DATA_DIR}  "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  {EPOCHS} epoch  "
        f"batch={BATCH_SIZE}  lr={LR}  warmup={warmup_steps}/{total_steps} steps  "
        f"weight={WEIGHT_MODE}  padding=dynamic ═══\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        v_loss, v_acc, v_true, v_pred = evaluate(model, val_loader, criterion)
        elapsed = (time.time() - t0) / 60
        macro_f1 = f1_score(v_true, v_pred, average="macro")

        log(f"Epoch {epoch}/{EPOCHS}  {elapsed:.1f}min  train_loss={tr_loss:.3f}  "
            f"val_loss={v_loss:.3f}  val_acc={v_acc:.3f}  val_macro_f1={macro_f1:.3f}")
        log(classification_report(v_true, v_pred, target_names=LABELS, digits=3))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model.bert.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            torch.save(model.head.state_dict(), SAVE_PATH / "head.pt")
            log(f"  ✓ Best model saved (macro_f1={macro_f1:.3f}) → {SAVE_PATH}")

    # Test uses the BEST checkpoint on disk (report == saved model).
    print(f"\nLoading best checkpoint ← {SAVE_PATH}")
    model = FinBERTSentiment(AutoModel.from_pretrained(SAVE_PATH)).to(DEVICE)
    model.head.load_state_dict(torch.load(SAVE_PATH / "head.pt", map_location=DEVICE))

    print("\nTest set evaluation:")
    _, _, t_true, t_pred = evaluate(model, test_loader, criterion)
    t_macro_f1 = f1_score(t_true, t_pred, average="macro")
    log(f"\n── Sentiment Test Report (A3 v3 sqrt, best val macro_f1={best_macro_f1:.3f}, "
        f"test macro_f1={t_macro_f1:.3f}) ──")
    log(classification_report(t_true, t_pred, target_names=LABELS, digits=3))
    print(f"\n✓ Done. Model: {SAVE_PATH}  Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
