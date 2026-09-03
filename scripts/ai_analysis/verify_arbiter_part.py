#!/usr/bin/env python3
"""Check one arbiter part file against its queue chunk.

uv run scripts/ai_analysis/verify_arbiter_part.py --run run2 --queue q001 --part w1c01
Exit 0 + "OK" when: every line parses, every verdict is spec-valid (derive),
item ids equal the chunk's ids exactly (no missing/extra/duplicate).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_verdicts import derive  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--part", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    queue = json.loads((run_dir / "arbiter_queue" / f"{args.queue}.json").read_text(
        encoding="utf-8"))["items"]
    expected = [i["item"] for i in queue]
    part = run_dir / "arbiter_parts" / f"{args.part}.jsonl"
    if not part.exists():
        raise SystemExit(f"MISSING {part}")

    problems, seen = [], []
    for n, line in enumerate(part.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: bad json ({e.msg})")
            continue
        if derive(row) is None:
            problems.append(f"line {n}: invalid verdict {row.get('item')} "
                            f"{row.get('profile')}/{row.get('sentiment')}/{row.get('reason')}")
        if not isinstance(row.get("note"), str):
            problems.append(f"line {n}: note must be a string ({row.get('item')})")
        seen.append(row.get("item"))

    dup = {i for i in seen if seen.count(i) > 1}
    missing = [i for i in expected if i not in seen]
    extra = [i for i in seen if i not in expected]
    if dup:
        problems.append(f"duplicate ids: {sorted(dup)[:5]}")
    if missing:
        problems.append(f"missing {len(missing)}: {missing[:5]}")
    if extra:
        problems.append(f"extra {len(extra)}: {extra[:5]}")

    if problems:
        print(f"PROBLEMS {args.part} ({len(seen)}/{len(expected)} lines):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print(f"OK {args.part}: {len(seen)} verdicts match {args.queue}")


if __name__ == "__main__":
    main()
