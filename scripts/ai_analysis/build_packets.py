#!/usr/bin/env python3
"""Build blinded sentiment-labeling packets from a source CSV.

Default source: legacy labeled 35k (run1). For pool waves (run2+) pass
--source data/havuz_107k_ranked.csv --keep-order so the uncertainty
ranking from score_pool.py decides which rows get labeled first."""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/dataset_35k_labeled.csv"
OUT_BASE = ROOT / "data/ai_analysis/runs"
TEXT_CAP = 700


def capped(text):
    truncated = len(text) > TEXT_CAP
    return (text[:TEXT_CAP] + " […]" if truncated else text), truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--packet-size", type=int, default=40)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--keep-order", action="store_true",
                    help="do not shuffle; keep source row order (ranked pools)")
    args = ap.parse_args()
    if args.limit < 0 or args.packet_size <= 0:
        ap.error("--limit must be >= 0 and --packet-size must be > 0")

    with Path(args.source).open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    rows, dropped = [], 0
    for source_row, row in enumerate(source_rows):
        text = (row.get("text") or "").strip()
        if len(text) < 10:
            dropped += 1
            continue
        rows.append((source_row, row, text))

    if not args.keep_order:
        random.Random(f"{args.run}:shuffle").shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]

    run_dir = OUT_BASE / args.run
    for sub in ("packets", "mapping", "verdicts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    packets = []
    for offset in range(0, len(rows), args.packet_size):
        packet_no = offset // args.packet_size + 1
        pid = f"p{packet_no:03d}"
        items, mappings = [], []
        for idx, (source_row, row, text) in enumerate(rows[offset:offset + args.packet_size]):
            item = f"{pid}_i{idx:02d}_r{source_row}"
            shown, truncated = capped(text)
            items.append({"item": item, "text": shown})
            mappings.append({
                "item": item, "row": source_row,
                "message_id": row["message_id"],
                "channel": row["channel_username"], "date": row["date"],
                "views": row["views"], "forwards": row["forwards"],
                "text": shown, "text_truncated": truncated,
                "haiku": {"sentiment": row.get("sentiment", ""), "reason": row.get("reason", "")},
            })
        (run_dir / "packets" / f"{pid}.json").write_text(
            json.dumps({"packet_id": pid, "items": items}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        (run_dir / "mapping" / f"{pid}.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in mappings),
            encoding="utf-8")
        packets.append({"packet_id": pid, "n_items": len(items)})

    manifest = {
        "run": args.run, "packet_size": args.packet_size,
        "source": args.source, "keep_order": args.keep_order,
        "rows_total": len(source_rows), "rows_dropped_short": dropped,
        "rows_packed": len(rows), "packets": packets,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.run}: {len(source_rows)} rows, {dropped} short dropped, "
          f"{len(rows)} packed -> {len(packets)} packets")


if __name__ == "__main__":
    main()
