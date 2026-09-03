"""
A4 (CANCELLED) - DeBERTa-v3-base with the A3 recipe on complete_v3.
Ran out of memory on an M4 16 GB (Metal graph cache); see ai_training_notes.md
section 5. Checkpoint config is float16 -> must load as float32 or loss is NaN.
Uses micro-batch 8 x 2 accumulation + gradient checkpointing.

Run:    caffeinate -i uv run scripts/train_sentiment_v4_deberta.py
Output: models/deberta-sentiment-v3-sqrt/ (+ train_log.txt)
"""

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.90")   # BEFORE importing torch
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.75")

import math
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import classification_report, f1_score

import train_sentiment_v2 as v2
from train_sentiment_v2 import (
    LABELS, DEVICE, NewsDataset, make_collate, LengthGroupedSampler,
    sorted_batches, evaluate,
)

EPOCHS       = 3
MICRO_BATCH  = 8           # memory: samples sent to the GPU at once
ACCUM_STEPS  = 2           # 8 × 2 = effective batch 16 (same step count/lr schedule as A3)
LR           = 2e-5
WARMUP_RATIO = 0.06
WEIGHT_MODE  = "sqrt"
GRAD_CKPT    = True        # for memory (see ai_training_notes.md section 5, A4); does not change results

MODEL_PATH = Path("models/deberta-v3-base")
SAVE_PATH  = Path("models/deberta-sentiment-v3-sqrt")
LOG_PATH   = SAVE_PATH / "train_log.txt"
DATA_DIR   = Path("data/complete_v3")
TRAIN_CSV, VAL_CSV, TEST_CSV = (DATA_DIR / f"{s}.csv" for s in ("train", "val", "test"))


def log(msg: str):
    print(msg)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


class DebertaSentiment(nn.Module):
    """Wraps the HF classifier: forward(ids, mask) → logits.
    This way v2's evaluate() and collate work as they are."""

    def __init__(self, path, grad_ckpt=False):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=len(LABELS), dtype=torch.float32)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def train_epoch(model, loader, optimizer, scheduler, criterion):
    """Gradient accumulation: the loss of ACCUM_STEPS micro-batches is summed, then one
    optimizer step. Effective batch = MICRO_BATCH × ACCUM_STEPS."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  train", leave=False)
    n = len(loader)
    for i, batch in enumerate(pbar, 1):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        if not torch.isfinite(loss):
            raise RuntimeError("loss NaN/inf — is dtype fp32? (see docstring)")
        (loss / ACCUM_STEPS).backward()

        if i % ACCUM_STEPS == 0 or i == n:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if DEVICE == "mps" and i % 100 == 0:
            torch.mps.empty_cache()          # cut cache growth (for the memory ceiling)

        total_loss += loss.item()
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += len(lbls)
        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / n, correct / total


def main():
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise SystemExit(f"No data: {p}")
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = DebertaSentiment(MODEL_PATH, grad_ckpt=GRAD_CKPT).to(DEVICE)

    print("Loading dataset...")
    collate  = make_collate(tokenizer)
    train_ds, val_ds, test_ds = NewsDataset(TRAIN_CSV), NewsDataset(VAL_CSV), NewsDataset(TEST_CSV)
    train_loader = DataLoader(
        train_ds, collate_fn=collate,
        batch_sampler=LengthGroupedSampler([len(t) for t, _ in train_ds.rows], MICRO_BATCH))
    val_loader  = DataLoader(val_ds,  collate_fn=collate, batch_sampler=sorted_batches(val_ds,  MICRO_BATCH))
    test_loader = DataLoader(test_ds, collate_fn=collate, batch_sampler=sorted_batches(test_ds, MICRO_BATCH))

    v2.WEIGHT_MODE = WEIGHT_MODE
    weights   = v2.class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)   # optimizer steps
    total_steps  = EPOCHS * steps_per_epoch
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_macro_f1 = 0.0

    log(f"\n═══ {time.strftime('%Y-%m-%d %H:%M')}  A4 DeBERTa-v3-base  device={DEVICE}  data={DATA_DIR}  "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  {EPOCHS} epoch  "
        f"batch={MICRO_BATCH}x{ACCUM_STEPS}  lr={LR}  warmup={warmup_steps}/{total_steps} steps  "
        f"weight={WEIGHT_MODE}  dtype=fp32  grad_ckpt={GRAD_CKPT}  padding=dynamic ═══\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        if DEVICE == "mps":
            torch.mps.empty_cache()
        v_loss, v_acc, v_true, v_pred = evaluate(model, val_loader, criterion)
        elapsed = (time.time() - t0) / 60
        macro_f1 = f1_score(v_true, v_pred, average="macro")

        log(f"Epoch {epoch}/{EPOCHS}  {elapsed:.1f}min  train_loss={tr_loss:.3f}  "
            f"val_loss={v_loss:.3f}  val_acc={v_acc:.3f}  val_macro_f1={macro_f1:.3f}")
        log(classification_report(v_true, v_pred, target_names=LABELS, digits=3))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            model.model.save_pretrained(SAVE_PATH)      # body + classification head together
            tokenizer.save_pretrained(SAVE_PATH)
            log(f"  ✓ Best model saved (macro_f1={macro_f1:.3f}) → {SAVE_PATH}")

    print(f"\nLoading best checkpoint ← {SAVE_PATH}")
    model = DebertaSentiment(SAVE_PATH).to(DEVICE)

    print("\nTest set evaluation:")
    _, _, t_true, t_pred = evaluate(model, test_loader, criterion)
    t_macro_f1 = f1_score(t_true, t_pred, average="macro")
    log(f"\n── Sentiment Test Report (A4 DeBERTa-v3-base, v3 sqrt, best val macro_f1={best_macro_f1:.3f}, "
        f"test macro_f1={t_macro_f1:.3f}) ──")
    log(classification_report(t_true, t_pred, target_names=LABELS, digits=3))
    print(f"\n✓ Done. Model: {SAVE_PATH}  Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
