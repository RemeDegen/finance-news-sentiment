"""
Download ProsusAI FinBERT into models/finbert.
Run: uv run scripts/download_finbert.py
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SAVE_PATH = Path("models/finbert")

print("Downloading FinBERT (ProsusAI/finbert)...")
print("First download ~440MB, 1-5 min depending on connection.\n")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"\n✓ Model saved → {SAVE_PATH}")
print(f"  Classes: {model.config.id2label}")
