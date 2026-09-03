"""
Legacy: label the dataset (sentiment + reason) with the Haiku Batch API.
Superseded by the dual-judge pipeline in scripts/ai_analysis/.
Run once to submit, run again to poll. Output: data/dataset_labeled.csv
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DATASET_PATH = Path("data/dataset_35k.csv")
OUTPUT_PATH = Path("data/dataset_35k_labeled.csv")
PROGRESS_PATH = Path("data/label_progress_35k.json")

MODEL = "claude-haiku-4-5"
BATCH_SIZE = 95_000  # API limit is 100k, we leave a safe margin

SYSTEM_PROMPT = """You are a financial news analyst. Classify the news item below.

Return ONLY a valid JSON object with exactly these two fields:
- "sentiment": tone of the news ("positive", "negative", or "neutral")
- "reason": main topic category

sentiment definitions:
- positive: good news (price up, deal closed, positive economic data, ceasefire, etc.)
- negative: bad news (crash, sanctions, conflict escalation, recession fears, etc.)
- neutral: factual/mixed/unclear

reason definitions:
- crypto: BTC, ETH, altcoin, DeFi, NFT, blockchain, stablecoin
- macro: Fed, interest rates, inflation, GDP, employment, CPI, economic data
- geopolitics: war, sanctions, conflict, international relations, elections
- commodities: oil, gold, silver, gas, energy prices
- equities: stocks, indices, S&P, earnings, IPO, M&A, company results
- regulatory: SEC, central bank policy, government regulation, law
- other: doesn't fit any category above"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
        },
        "reason": {
            "type": "string",
            "enum": [
                "crypto",
                "macro",
                "geopolitics",
                "commodities",
                "equities",
                "regulatory",
                "other",
            ],
        },
    },
    "required": ["sentiment", "reason"],
    "additionalProperties": False,
}


def load_rows() -> list[dict]:
    rows = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def make_request(row_idx: int, text: str) -> Request:
    # Cap text at 1500 chars (for token limit safety)
    truncated = text[:1500] if text else "(empty)"
    return Request(
        custom_id=str(row_idx),
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=50,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": truncated}],
            output_config={
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
            },
        ),
    )


def submit_batches(client: anthropic.Anthropic, rows: list[dict]) -> list[str]:
    batch_ids = []
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        requests = [make_request(i + j, row["text"]) for j, row in enumerate(chunk)]
        batch_num = len(batch_ids) + 1
        end_idx = min(i + BATCH_SIZE - 1, len(rows) - 1)
        print(
            f"Sending batch {batch_num}/{total_batches}: "
            f"rows {i:,}–{end_idx:,} ({len(chunk):,} requests)..."
        )
        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"  → Batch ID: {batch.id}")
    return batch_ids


def check_and_collect(
    client: anthropic.Anthropic, batch_ids: list[str]
) -> dict[str, dict] | None:
    """Return the results if all batches ended, otherwise return None."""
    results = {}
    all_done = True

    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        status = batch.processing_status
        c = batch.request_counts
        print(
            f"  Batch {bid[:20]}...: {status} "
            f"(✓{c.succeeded:,} ✗{c.errored:,} ⏳{c.processing:,})"
        )
        if status != "ended":
            all_done = False
            continue

        for result in client.messages.batches.results(bid):
            if result.result.type == "succeeded":
                msg = result.result.message
                raw = next(
                    (b.text for b in msg.content if b.type == "text"), "{}"
                )
                try:
                    label = json.loads(raw)
                    results[result.custom_id] = label
                except json.JSONDecodeError:
                    results[result.custom_id] = {
                        "sentiment": "neutral",
                        "reason": "other",
                    }
            else:
                results[result.custom_id] = {
                    "sentiment": "neutral",
                    "reason": "other",
                }

    return results if all_done else None


def save_output(rows: list[dict], results: dict):
    fieldnames = list(rows[0].keys()) + ["sentiment", "reason"]
    labeled = 0
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            label = results.get(str(i), {"sentiment": "neutral", "reason": "other"})
            writer.writerow({**row, **label})
            labeled += 1
    print(f"\n✓ {labeled:,} rows labeled → {OUTPUT_PATH}")


def estimate_cost(num_rows: int) -> None:
    # Haiku Batch API: input $0.50/M, output $2.50/M
    input_tokens = num_rows * 130  # system prompt + text
    output_tokens = num_rows * 30  # short JSON output
    input_cost = (input_tokens / 1_000_000) * 0.50
    output_cost = (output_tokens / 1_000_000) * 2.50
    print(f"\nEstimated cost (Batch API, 50% discount):")
    print(f"  Input: {input_tokens/1e6:.1f}M token → ~${input_cost:.2f}")
    print(f"  Output: {output_tokens/1e6:.1f}M token → ~${output_cost:.2f}")
    print(f"  Total: ~${input_cost + output_cost:.2f}")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print()
        print("To get an API key:")
        print("  1. Go to https://console.anthropic.com")
        print("  2. Settings → API Keys → Create Key")
        print("  3. In the terminal: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading dataset: {DATASET_PATH}")
    rows = load_rows()
    print(f"✓ {len(rows):,} rows loaded")

    if PROGRESS_PATH.exists():
        progress = json.loads(PROGRESS_PATH.read_text())
        batch_ids = progress["batch_ids"]
        print(f"\n{len(batch_ids)} batches were already sent, checking status...")
        print()

        results = check_and_collect(client, batch_ids)
        if results is None:
            print("\nNot finished yet. Run again in a few minutes.")
            print("(Batches usually finish within 1 hour.)")
            return

        print(f"\n✓ All batches finished! {len(results):,} labels received.")
        save_output(rows, results)
        PROGRESS_PATH.unlink()
        print("Processing complete.")

    else:
        estimate_cost(len(rows))
        print()
        confirm = input(f"Continue? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

        print()
        batch_ids = submit_batches(client, rows)
        PROGRESS_PATH.write_text(json.dumps({"batch_ids": batch_ids}))
        print(f"\n✓ {len(batch_ids)} batches sent.")
        print("Batches usually finish within 1 hour.")
        print("Run this script again to check the status.")


if __name__ == "__main__":
    main()
