"""
A8 - A2 recipe verbatim (sqrt class weights, constant lr 2e-5, 3 epochs,
batch 16, dynamic padding, best-val-macro-F1 checkpoint) on complete_v5 =
complete_v2 train + 4,997 rows from wave 3 (run4: A2's most uncertain pool
rows relabeled under spec v2). Only the data and output paths differ from
train_sentiment_v2.py, so the test delta against A2 (0.842 / 0.808) isolates
the data effect. Val/test are byte-identical to complete_v2.

Run:    uv run scripts/train_sentiment_v8.py
Output: models/finbert-sentiment-v5-sqrt/ (+ train_log.txt)
"""

from pathlib import Path

import train_sentiment_v2 as v2

DATA_DIR  = Path("data/complete_v5")
SAVE_PATH = Path("models/finbert-sentiment-v5-sqrt")

v2.WEIGHT_MODE = "sqrt"
v2.SAVE_PATH   = SAVE_PATH
v2.LOG_PATH    = SAVE_PATH / "train_log.txt"
v2.TRAIN_CSV   = DATA_DIR / "train.csv"
v2.VAL_CSV     = DATA_DIR / "val.csv"
v2.TEST_CSV    = DATA_DIR / "test.csv"


if __name__ == "__main__":
    for p in (v2.MODEL_PATH, v2.TRAIN_CSV, v2.VAL_CSV, v2.TEST_CSV):
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    v2.log(f"═══ A8: train_sentiment_v8.py — A2 recipe on {DATA_DIR} → {SAVE_PATH} ═══")
    v2.main()
