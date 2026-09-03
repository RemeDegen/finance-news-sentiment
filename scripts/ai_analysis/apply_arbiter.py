#!/usr/bin/env python3
"""Normalize arbiter inputs and merge them into arbiter.jsonl."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_verdicts import derive  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    args = ap.parse_args()
    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    disputes = {row["item"]: row for row in
                load_jsonl(run_dir / "disagreements.jsonl")}
    inputs = load_jsonl(run_dir / "arbiter_input.jsonl")

    arbiter_path = run_dir / "arbiter.jsonl"
    existing = {}
    if arbiter_path.exists():
        existing = {row["item"]: row for row in load_jsonl(arbiter_path)}

    for verdict in inputs:
        item = verdict.get("item")
        if item not in disputes:
            raise SystemExit(f"arbiter input for unknown/undisputed item: {item}")
        final = derive(verdict)
        if final is None:
            print(f"WARNING {item}: invalid arbiter verdict")
        existing[item] = {
            "item": item, "final": final, "note": verdict.get("note", ""),
        }

    arbiter_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in existing.values()), encoding="utf-8")
    print(f"arbiter.jsonl: {len(existing)} decisions")


if __name__ == "__main__":
    main()
