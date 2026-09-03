"""
Score an unlabeled pool with a trained checkpoint and rank rows by uncertainty
(active learning). Adds p_positive/p_negative/p_neutral, pred, confidence (top
probability) and margin (top1 - top2); rows sorted by ascending margin.

Run:  uv run scripts/score_pool.py --model models/finbert-sentiment-v2-sqrt \
          --input <csv> --output <ranked csv>
Next: uv run scripts/ai_analysis/build_packets.py --run runN --source <ranked csv> --keep-order
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

from train_sentiment_v2 import FinBERTSentiment, LABELS, MAX_LEN, DEVICE

NEW_COLS = ["p_positive", "p_negative", "p_neutral", "pred", "confidence", "margin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint folder (bert + head.pt)")
    ap.add_argument("--input", default="data/havuz_107k.csv")
    ap.add_argument("--output", default="data/havuz_107k_ranked.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (trial)")
    args = ap.parse_args()
    if args.batch_size <= 0 or args.limit < 0:
        ap.error("--batch-size > 0 and --limit >= 0 required")

    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = FinBERTSentiment(AutoModel.from_pretrained(model_dir)).to(DEVICE)
    model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location=DEVICE))
    model.eval()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]
    texts = [(r.get("text") or "")[:512] for r in rows]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sorted batches → speed
    probs = [None] * len(texts)

    print(f"Device: {DEVICE}  model: {model_dir}  rows: {len(rows)}")
    t0, n_batches = time.time(), (len(order) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for b, k in enumerate(range(0, len(order), args.batch_size), 1):
            idx = order[k:k + args.batch_size]
            enc = tokenizer([texts[i] for i in idx], max_length=MAX_LEN, padding="longest",
                            pad_to_multiple_of=8, truncation=True, return_tensors="pt").to(DEVICE)
            p = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1).cpu()
            for j, i in enumerate(idx):
                probs[i] = p[j].tolist()
            if b % 100 == 0 or b == n_batches:
                el = time.time() - t0
                print(f"  {b}/{n_batches} batch  {el/60:.1f} min  (left ~{el/b*(n_batches-b)/60:.1f} min)")

    for r, p in zip(rows, probs):
        top = sorted(p, reverse=True)
        r["p_positive"], r["p_negative"], r["p_neutral"] = (f"{x:.4f}" for x in p)
        r["pred"] = LABELS[p.index(top[0])]
        r["confidence"] = f"{top[0]:.4f}"
        r["margin"] = f"{top[0] - top[1]:.4f}"
    rows.sort(key=lambda r: float(r["margin"]))

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns + NEW_COLS)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    dist = Counter(r["pred"] for r in rows)
    unsure = sum(float(r["confidence"]) < 0.6 for r in rows)
    tight = sum(float(r["margin"]) < 0.2 for r in rows)
    print(f"\n✓ {out}  ({n} rows, {(time.time()-t0)/60:.1f} min)")
    print("  prediction distribution: " + ", ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in dist.most_common()))
    print(f"  confidence < 0.6: {unsure} ({100*unsure/n:.1f}%)   margin < 0.2: {tight} ({100*tight/n:.1f}%)")
    print(f"  first 5k rows = most uncertain 5k → 125 packets (--limit 5000)")


if __name__ == "__main__":
    main()
