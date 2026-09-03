#!/usr/bin/env python3
"""
Incremental dataset builder: append a run's final labels (active-learning
wave) to the base train split; val/test are copied unchanged.
Unlike build_dataset.py (run1, 80/10/10 split): no re-split, source CSV comes
from the run manifest, new rows are deduped against all three base splits.
Model predictions go only to the run's final_labeled.csv/stats, never to train.

Run: uv run scripts/ai_analysis/build_dataset_v3.py --run run2 --base data/complete_v2 --out data/complete_v3
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (  # noqa: E402
    FINAL_COLUMNS, LEGACY_COLUMNS, SENTIMENTS, distribution, load_jsonl,
    percentage, write_csv,
)

ROOT = Path(__file__).resolve().parents[2]


def norm(text):
    return " ".join(text.lower().split())


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="data/complete_v2")
    ap.add_argument("--out", default="data/complete_v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / "data/ai_analysis/runs" / args.run
    base_dir, out_dir = ROOT / args.base, ROOT / args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty — will not overwrite")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source = read_csv(ROOT / manifest["source"])
    agreed = load_jsonl(run_dir / "agreed.jsonl")
    dispute_rows = load_jsonl(run_dir / "disagreements.jsonl")
    disputes = {r["item"]: r for r in dispute_rows}
    arbiter = {r["item"]: r for r in load_jsonl(run_dir / "arbiter.jsonl")}
    unknown = [i for i in arbiter if i not in disputes]
    if unknown:
        raise SystemExit(f"arbiter lines for unknown items: {unknown[:5]}")
    unresolved = [i for i in disputes if i not in arbiter]
    if unresolved and not args.allow_partial:
        raise SystemExit(f"{len(unresolved)} disputes unarbitrated (first: {unresolved[:5]})")

    merged = [(r["mapping"], r["final"], "agreed", r["confidence"], None) for r in agreed]
    for d in dispute_rows:
        dec = arbiter.get(d["item"])
        if dec is None:
            continue
        if dec.get("final") is None:
            raise SystemExit(f"invalid arbiter final for {d['item']}")
        conf = {j: (d.get(f"{j}_raw") or {}).get("confidence") for j in ("opus", "codex")}
        merged.append((d["mapping"], dec["final"], "arbiter", conf, d.get("why")))
    merged.sort(key=lambda e: e[0]["row"])

    final_rows = []
    for mapping, final, src, conf, why in merged:
        o = source[mapping["row"]]
        if o["message_id"] != mapping["message_id"] or o["channel_username"] != mapping["channel"]:
            raise SystemExit(f"mapping/source mismatch at {mapping['item']}")
        final_rows.append({
            "message_id": o["message_id"], "channel_username": o["channel_username"],
            "date": o["date"], "text": o["text"], "views": o["views"], "forwards": o["forwards"],
            "sentiment": final["sentiment"], "reason": final["reason"], "profile": final["profile"],
            "source": src, "confidence_opus": conf.get("opus"), "confidence_codex": conf.get("codex"),
            "haiku_sentiment": o.get("sentiment", ""), "haiku_reason": o.get("reason", ""),
            "why": why or "", "secim": o.get("secim", ""),
            "model_pred": o.get("pred", ""), "model_confidence": o.get("confidence", ""),
        })
    write_csv(run_dir / "final_labeled.csv", final_rows,
              FINAL_COLUMNS + ["why", "secim", "model_pred", "model_confidence"])

    base = {s: read_csv(base_dir / f"{s}.csv") for s in ("train", "val", "test")}
    base_keys = {s: {norm(r["text"]) for r in rows} for s, rows in base.items()}
    seen = set().union(*base_keys.values())
    base_hits = Counter()
    new_rows, news_count, dup_internal = [], 0, 0
    for r in final_rows:
        if r["profile"] != "news":
            continue
        news_count += 1
        key = norm(r["text"])
        if key in seen:
            hit = [s for s in ("val", "test", "train") if key in base_keys[s]]
            if hit:
                base_hits[hit[0]] += 1
            else:
                dup_internal += 1
            continue
        seen.add(key)
        new_rows.append(r)

    rng = random.Random(args.seed)
    train = base["train"] + [{k: r[k] for k in LEGACY_COLUMNS} for r in new_rows]
    rng.shuffle(train)
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "train.csv", train, LEGACY_COLUMNS)
    for s in ("val", "test"):
        shutil.copyfile(base_dir / f"{s}.csv", out_dir / f"{s}.csv")

    news = [r for r in final_rows if r["profile"] == "news"]
    model_ok = sum(r["model_pred"] == r["sentiment"] for r in news)
    flag = [r for r in news if r["why"] == "model_flag"]
    flag_model_won = sum(r["model_pred"] == r["sentiment"] for r in flag)
    by_secim = {}
    for sec in sorted({r["secim"] for r in news}):
        grp = [r for r in news if r["secim"] == sec]
        by_secim[sec] = {"items": len(grp), "model_match_pct": percentage(
            sum(r["model_pred"] == r["sentiment"] for r in grp), len(grp))}
    stats = {
        "rows_final": len(final_rows), "news": news_count,
        "not_news": len(final_rows) - news_count,
        "source_counts": dict(Counter(r["source"] for r in final_rows)),
        "why_counts": dict(Counter(r["why"] for r in final_rows if r["why"])),
        "dedup": {"internal": dup_internal, "vs_base": dict(base_hits)},
        "added_to_train": len(new_rows),
        "label_distributions": {"new_rows": distribution(new_rows),
                                "train_total": distribution(train)},
        "split_sizes": {"train": len(train), "val": len(base["val"]), "test": len(base["test"])},
        "model_vs_final": {"news_items": len(news), "model_match_pct": percentage(model_ok, len(news)),
                           "by_secim": by_secim,
                           "model_flag": {"items": len(flag), "arbiter_sided_with_model": flag_model_won}},
    }
    if unresolved:
        stats["skipped_unresolved"] = unresolved
    (run_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"final: {len(final_rows)} rows ({news_count} news); added to train: {len(new_rows)} "
          f"(dedup internal={dup_internal}, vs base={dict(base_hits)}); "
          f"train={len(train)} val={len(base['val'])} test={len(base['test'])} → {out_dir}")
    print("new-row labels:", stats["label_distributions"]["new_rows"]["sentiment"])
    print("model vs final:", stats["model_vs_final"])


if __name__ == "__main__":
    main()
