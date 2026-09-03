#!/usr/bin/env python3
"""Item 5 (model as third judge): move agreed rows that the model contradicts
with high confidence into the dispute list (why=model_flag) and write them as
separate arbiter queue chunks (m*.json). The model prediction is NOT copied
into the queue — the arbiter must not see it (anchor risk)."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENTIMENTS = ("positive", "negative", "neutral")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-conf", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=40)
    args = ap.parse_args()
    if not 0 < args.min_conf <= 1 or args.chunk_size <= 0:
        ap.error("--min-conf must be in (0,1], --chunk-size > 0")

    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with (ROOT / manifest["source"]).open(newline="", encoding="utf-8") as f:
        source = list(csv.DictReader(f))
    if not source or "pred" not in source[0] or "confidence" not in source[0]:
        raise SystemExit("source csv lacks pred/confidence columns (run score_pool first)")

    agreed = load_jsonl(run_dir / "agreed.jsonl")
    disputes = load_jsonl(run_dir / "disagreements.jsonl")
    already = {d["item"] for d in disputes}

    keep, flagged = [], []
    pairs = Counter()
    for row in agreed:
        m, final = row["mapping"], row["final"]
        src = source[m["row"]]
        if src["message_id"] != m["message_id"] or src["channel_username"] != m["channel"]:
            raise SystemExit(f"mapping/source mismatch at {m['item']}")
        pred, conf = src["pred"], float(src["confidence"])
        if (final["profile"] == "news" and pred in SENTIMENTS
                and pred != final["sentiment"] and conf >= args.min_conf
                and row["item"] not in already):
            pairs[f"judges={final['sentiment']}"] += 1
            flagged.append({
                "item": row["item"], "why": "model_flag", "mapping": m,
                "opus_raw": {"item": row["item"], **final,
                             "confidence": row["confidence"]["opus"],
                             "note": row["notes"]["opus"]},
                "opus_derived": final,
                "codex_raw": {"item": row["item"], **final,
                              "confidence": row["confidence"]["codex"],
                              "note": row["notes"]["codex"]},
                "codex_derived": final,
            })
        else:
            keep.append(row)

    dump_jsonl(run_dir / "agreed.jsonl", keep)
    dump_jsonl(run_dir / "disagreements.jsonl", disputes + flagged)

    queue_dir = run_dir / "arbiter_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    for old in queue_dir.glob("m*.json"):
        old.unlink()
    for n, off in enumerate(range(0, len(flagged), args.chunk_size), 1):
        qid = f"m{n:03d}"
        (queue_dir / f"{qid}.json").write_text(json.dumps(
            {"queue_id": qid, "items": flagged[off:off + args.chunk_size]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"agreed: {len(agreed)} -> {len(keep)}; model_flag: {len(flagged)} "
          f"(conf>={args.min_conf}); disputes now {len(disputes) + len(flagged)}; "
          f"chunks: {(len(flagged) + args.chunk_size - 1) // args.chunk_size}")
    print("flagged by agreed label:", dict(pairs))


if __name__ == "__main__":
    main()
