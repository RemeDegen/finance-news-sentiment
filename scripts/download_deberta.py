"""
Download DeBERTa-v3-base into models/deberta-v3-base.
Run: uv run scripts/download_deberta.py
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

SAVE_PATH = Path("models/deberta-v3-base")

print("Downloading DeBERTa-v3-base (microsoft/deberta-v3-base)...")
print("First download ~700MB, may take 2-5 min.\n")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"\n✓ Model saved → {SAVE_PATH}")
