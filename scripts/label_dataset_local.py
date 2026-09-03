"""
Legacy: label the dataset with a local Ollama model (free, offline).
Setup: install Ollama, `ollama pull qwen2.5:3b`, then run this script.
Output: data/dataset_labeled_local.csv (resumable via data/label_progress_local.json)
"""

import json
import time
from pathlib import Path

import httpx

DATASET_PATH = Path("data/raw_142k_deduped.csv")
OUTPUT_PATH = Path("data/dataset_labeled_local.csv")
PROGRESS_PATH = Path("data/label_progress_local.json")

MODEL = "qwen2.5:7b"         # recommended for M4 Air 16GB
OLLAMA_URL = "http://localhost:11434/api/generate"
SAVE_EVERY = 200             # write to disk every 200 rows
REQUEST_TIMEOUT = 30.0       # seconds

PROMPT_TEMPLATE = """Classify this financial news item. Reply with ONLY valid JSON, no extra text.

JSON format:
{{"sentiment": "<positive|negative|neutral>", "reason": "<crypto|macro|geopolitics|commodities|equities|regulatory|other>"}}

Definitions:
- positive: good news (price up, deal done, ceasefire, beat expectations)
- negative: bad news (crash, sanctions, conflict, miss expectations, recession)
- neutral: factual/mixed/unclear

- crypto: BTC, ETH, altcoins, DeFi, NFT, blockchain
- macro: Fed, rates, inflation, GDP, employment, CPI
- geopolitics: war, sanctions, conflict, elections, international relations
- commodities: oil, gold, silver, gas, energy
- equities: stocks, S&P, earnings, IPO, M&A, company results
- regulatory: SEC, laws, central bank rules, compliance
- other: doesn't fit above

News: {text}"""


def load_rows() -> list[dict]:
    import csv
    rows = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_progress() -> dict[int, dict]:
    if PROGRESS_PATH.exists():
        data = json.loads(PROGRESS_PATH.read_text())
        return {int(k): v for k, v in data.items()}
    return {}


def save_progress(results: dict[int, dict]) -> None:
    PROGRESS_PATH.write_text(json.dumps(results))


def call_ollama(client: httpx.Client, text: str) -> dict:
    truncated = text[:800] if text else "(empty)"
    payload = {
        "model": MODEL,
        "prompt": PROMPT_TEMPLATE.format(text=truncated),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        resp = client.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        label = json.loads(raw)
        if label.get("sentiment") not in ("positive", "negative", "neutral"):
            raise ValueError("bad sentiment")
        if label.get("reason") not in (
            "crypto", "macro", "geopolitics", "commodities",
            "equities", "regulatory", "other"
        ):
            raise ValueError("bad reason")
        return label
    except Exception:
        return {"sentiment": "neutral", "reason": "other"}


def write_output(rows: list[dict], results: dict[int, dict]) -> None:
    import csv
    fieldnames = list(rows[0].keys()) + ["sentiment", "reason"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            label = results.get(i, {"sentiment": "neutral", "reason": "other"})
            writer.writerow({**row, **label})


def check_ollama() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    if not check_ollama():
        print("ERROR: Ollama is not running.")
        print()
        print("Fix:")
        print("  1. Install Ollama from https://ollama.com")
        print("  2. In the terminal: ollama serve")
        print(f"  3. Pull the model: ollama pull {MODEL}")
        return

    print(f"Loading dataset: {DATASET_PATH}")
    rows = load_rows()
    print(f"✓ {len(rows):,} rows loaded")

    results = load_progress()
    done = len(results)
    remaining = len(rows) - done

    if done > 0:
        print(f"✓ Previous progress found: {done:,} rows completed")

    if remaining == 0:
        print("All rows are already labeled. Writing output...")
        write_output(rows, results)
        print(f"✓ Done → {OUTPUT_PATH}")
        return

    print(f"To label: {remaining:,} rows  |  Model: {MODEL}")
    print()

    times: list[float] = []
    with httpx.Client() as client:
        for i, row in enumerate(rows):
            if i in results:
                continue

            t0 = time.time()
            results[i] = call_ollama(client, row.get("text", ""))
            elapsed = time.time() - t0
            times.append(elapsed)

            completed = len(results) - done
            if completed % 10 == 0 or completed == 1:
                avg = sum(times[-50:]) / len(times[-50:])
                eta_sec = avg * (remaining - completed)
                if eta_sec < 3600:
                    eta_str = f"{eta_sec/60:.0f} min"
                else:
                    eta_str = f"{eta_sec/3600:.1f} hours"
                pct = (len(results) / len(rows)) * 100
                print(
                    f"\r{len(results):,}/{len(rows):,} ({pct:.1f}%)  "
                    f"avg {avg:.1f}s/row  ETA: {eta_str}   ",
                    end="", flush=True
                )

            if completed % SAVE_EVERY == 0:
                save_progress(results)

    print()
    save_progress(results)
    write_output(rows, results)
    PROGRESS_PATH.unlink(missing_ok=True)
    print(f"\n✓ {len(results):,} rows labeled → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
