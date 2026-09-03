#!/usr/bin/env python3
"""Merge final verdicts and build legacy-compatible training splits."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/dataset_35k_labeled.csv"
FINAL_COLUMNS = [
    "message_id", "channel_username", "date", "text", "views", "forwards",
    "sentiment", "reason", "profile", "source", "confidence_opus",
    "confidence_codex", "haiku_sentiment", "haiku_reason",
]
LEGACY_COLUMNS = [
    "message_id", "channel_username", "date", "text", "views", "forwards",
    "sentiment", "reason",
]
SENTIMENTS = ("positive", "negative", "neutral")


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution(rows):
    return {
        "sentiment": dict(Counter(row["sentiment"] for row in rows)),
        "reason": dict(Counter(row["reason"] for row in rows)),
    }


def percentage(numerator, denominator):
    return round(numerator * 100 / denominator, 2) if denominator else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out-dir", default="data/complete_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    agreed = load_jsonl(run_dir / "agreed.jsonl")
    dispute_rows = load_jsonl(run_dir / "disagreements.jsonl")
    disputes = {row["item"]: row for row in dispute_rows}
    arbiter = {row["item"]: row for row in load_jsonl(run_dir / "arbiter.jsonl")}
    unknown = [item for item in arbiter if item not in disputes]
    if unknown:
        raise SystemExit(f"arbiter lines for unknown items: {unknown[:5]}")
    unresolved = [item for item in disputes if item not in arbiter]
    if unresolved and not args.allow_partial:
        raise SystemExit(f"{len(unresolved)} disputed items lack an arbiter line "
                         f"(first: {unresolved[:5]}) — arbitrate or pass --allow-partial")

    with SOURCE.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    merged = [(row["mapping"], row["final"], "agreed",
               row["confidence"]) for row in agreed]
    for dispute in dispute_rows:
        decision = arbiter.get(dispute["item"])
        if decision is None:
            continue
        if decision.get("final") is None:
            raise SystemExit(f"invalid arbiter final for {dispute['item']}")
        confidence = {
            judge: (dispute.get(f"{judge}_raw") or {}).get("confidence")
            for judge in ("opus", "codex")
        }
        merged.append((dispute["mapping"], decision["final"],
                       "arbiter", confidence))
    merged.sort(key=lambda entry: entry[0]["row"])

    final_rows = []
    for mapping, final, source, confidence in merged:
        row_no = mapping["row"]
        if not isinstance(row_no, int) or not 0 <= row_no < len(source_rows):
            raise SystemExit(f"source row out of range for {mapping['item']}: {row_no}")
        original = source_rows[row_no]
        final_rows.append({
            "message_id": original["message_id"],
            "channel_username": original["channel_username"],
            "date": original["date"], "text": original["text"],
            "views": original["views"], "forwards": original["forwards"],
            "sentiment": final["sentiment"], "reason": final["reason"],
            "profile": final["profile"], "source": source,
            "confidence_opus": confidence.get("opus"),
            "confidence_codex": confidence.get("codex"),
            "haiku_sentiment": original["sentiment"],
            "haiku_reason": original["reason"],
        })
    write_csv(run_dir / "final_labeled.csv", final_rows, FINAL_COLUMNS)

    unique, seen = [], set()
    news_count = 0
    for row in final_rows:
        if row["profile"] != "news":
            continue
        news_count += 1
        normalized = " ".join(row["text"].lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(row)
    dedup_removed = news_count - len(unique)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for sentiment in SENTIMENTS:
        bucket = [row for row in unique if row["sentiment"] == sentiment]
        rng.shuffle(bucket)
        train_end = int(len(bucket) * 0.8)
        val_end = train_end + int(len(bucket) * 0.1)
        splits["train"].extend(bucket[:train_end])
        splits["val"].extend(bucket[train_end:val_end])
        splits["test"].extend(bucket[val_end:])
    for name, rows in splits.items():
        rng.shuffle(rows)
        write_csv(out_dir / f"{name}.csv", rows, LEGACY_COLUMNS)

    confusion = {old: {new: 0 for new in SENTIMENTS} for old in SENTIMENTS}
    comparable = [row for row in final_rows if row["profile"] == "news"
                  and row["haiku_sentiment"] in SENTIMENTS]
    sentiment_agreed = sum(row["haiku_sentiment"] == row["sentiment"]
                           for row in comparable)
    reason_comparable = [row for row in final_rows if row["profile"] == "news"
                         and row["haiku_reason"]]
    reason_agreed = sum(row["haiku_reason"] == row["reason"]
                        for row in reason_comparable)
    for row in comparable:
        confusion[row["haiku_sentiment"]][row["sentiment"]] += 1

    stats = {
        "rows_final": len(final_rows),
        "label_distributions": {
            "overall": distribution([r for r in final_rows if r["profile"] == "news"]),
            "splits": {name: distribution(rows) for name, rows in splits.items()},
        },
        "source_counts": dict(Counter(row["source"] for row in final_rows)),
        "not_news_count": sum(row["profile"] == "not_news" for row in final_rows),
        "dedup_removed_count": dedup_removed,
        "final_vs_haiku": {
            "sentiment": {"items": len(comparable), "agreed": sentiment_agreed,
                          "pct": percentage(sentiment_agreed, len(comparable))},
            "reason": {"items": len(reason_comparable), "agreed": reason_agreed,
                       "pct": percentage(reason_agreed, len(reason_comparable))},
            "sentiment_confusion_haiku_vs_final": confusion,
        },
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
    }
    if unresolved:
        stats["skipped_unresolved"] = unresolved
    (run_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"final: {len(final_rows)} rows; news dedup: {len(unique)} "
          f"({dedup_removed} removed); splits: "
          + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))


if __name__ == "__main__":
    main()
