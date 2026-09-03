#!/usr/bin/env python3
"""Write unblinded dispute packets for the arbiter."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--chunk-size", type=int, default=40)
    args = ap.parse_args()
    if args.chunk_size <= 0:
        ap.error("--chunk-size must be > 0")

    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    path = run_dir / "disagreements.jsonl"
    disputes = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    queue_dir = run_dir / "arbiter_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    for old in queue_dir.glob("q*.json"):
        old.unlink()

    chunks = 0
    for offset in range(0, len(disputes), args.chunk_size):
        chunks += 1
        qid = f"q{chunks:03d}"
        items = []
        for dispute in disputes[offset:offset + args.chunk_size]:
            items.append({key: dispute.get(key) for key in (
                "item", "mapping", "opus_raw", "opus_derived",
                "codex_raw", "codex_derived", "why")})
        (queue_dir / f"{qid}.json").write_text(
            json.dumps({"queue_id": qid, "items": items},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"arbiter queue: {len(disputes)} disputed items -> {chunks} chunks")


if __name__ == "__main__":
    main()
