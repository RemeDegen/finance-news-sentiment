#!/usr/bin/env python3
"""Collect two judge lanes and split matching labels from disputes."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_PROFILES = {"news", "not_news"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_REASONS = {
    "crypto", "macro", "geopolitics", "commodities",
    "equities", "regulatory", "other",
}


def extract_json_array(text):
    """Return the first parseable JSON array in possibly noisy output."""
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def load_judge(path):
    if not path.exists() or path.stat().st_size == 0:
        return {}, "missing"
    array = extract_json_array(path.read_text(encoding="utf-8"))
    if array is None:
        return {}, "unparseable"
    return {v["item"]: v for v in array
            if isinstance(v, dict) and isinstance(v.get("item"), str)}, None


def derive(verdict):
    """Normalize a judge verdict; return None when it violates the spec."""
    if not isinstance(verdict, dict) or verdict.get("profile") not in VALID_PROFILES:
        return None
    profile = verdict["profile"]
    if profile == "not_news":
        return {"profile": profile, "sentiment": None, "reason": None}
    if (verdict.get("sentiment") not in VALID_SENTIMENTS or
            verdict.get("reason") not in VALID_REASONS):
        return None
    return {"profile": profile, "sentiment": verdict["sentiment"],
            "reason": verdict["reason"]}


def agreement_key(derived):
    return tuple(derived[k] for k in ("profile", "sentiment", "reason"))


def pct(numerator, denominator):
    return round(numerator * 100 / denominator, 2) if denominator else None


def side_token(raw, derived, full=False):
    if raw is None:
        return "missing"
    if derived is None:
        return "invalid"
    if not full:
        return str(derived["sentiment"])
    return ":".join(str(derived[k]) for k in ("profile", "sentiment", "reason"))


def codex_path(verdict_dir, pid):
    json_path = verdict_dir / f"codex_{pid}.json"
    return json_path if json_path.exists() else verdict_dir / f"codex_{pid}.raw"


def haiku_summary(counter):
    n = counter["items"]
    return {
        "items": n,
        "sentiment_agreed": counter["sentiment_agreed"],
        "sentiment_pct": pct(counter["sentiment_agreed"], n),
        "reason_agreed": counter["reason_agreed"],
        "reason_pct": pct(counter["reason_agreed"], n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot1")
    args = ap.parse_args()
    run_dir = ROOT / "data/ai_analysis/runs" / args.run

    agreed, disputes = [], []
    totals = Counter()
    sentiment_dist, reason_dist, profile_dist = Counter(), Counter(), Counter()
    sentiment_pairs, full_pairs = Counter(), Counter()
    confidence = {"opus": Counter(), "codex": Counter()}
    versus_haiku = {"opus": Counter(), "codex": Counter()}
    report = {"packets": {}, "lane_errors": []}

    for map_path in sorted((run_dir / "mapping").glob("*.jsonl")):
        pid = map_path.stem
        mapping = [json.loads(line) for line in map_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        opus, opus_err = load_judge(run_dir / "verdicts" / f"opus_{pid}.json")
        cpath = codex_path(run_dir / "verdicts", pid)
        codex, codex_err = load_judge(cpath)
        if opus_err or codex_err:
            report["lane_errors"].append(
                {"packet": pid, "opus": opus_err, "codex": codex_err})

        pstat = Counter({"items": len(mapping), "agreed": 0,
                         "disputed": 0, "uncovered": 0})
        for m in mapping:
            raw = {"opus": opus.get(m["item"]), "codex": codex.get(m["item"])}
            derived = {judge: derive(value) if value is not None else None
                       for judge, value in raw.items()}
            for judge in ("opus", "codex"):
                value = raw[judge]
                confidence[judge][str(value.get("confidence"))
                                  if value is not None else "missing"] += 1
                d = derived[judge]
                if d is not None and d["profile"] == "news":
                    vh = versus_haiku[judge]
                    vh["items"] += 1
                    vh["sentiment_agreed"] += d["sentiment"] == m["haiku"]["sentiment"]
                    vh["reason_agreed"] += d["reason"] == m["haiku"]["reason"]

            if raw["opus"] is None or raw["codex"] is None:
                why = "uncovered"
                pstat["uncovered"] += 1
            elif derived["opus"] is None or derived["codex"] is None:
                why = "invalid"
                pstat["disputed"] += 1
            elif agreement_key(derived["opus"]) == agreement_key(derived["codex"]):
                pstat["agreed"] += 1
                final = derived["opus"]
                agreed.append({
                    "item": m["item"], "mapping": m, "final": final,
                    "source": "agreed",
                    "confidence": {j: raw[j].get("confidence") for j in raw},
                    "notes": {j: raw[j].get("note", "") for j in raw},
                })
                sentiment_dist[str(final["sentiment"])] += 1
                reason_dist[str(final["reason"])] += 1
                profile_dist[final["profile"]] += 1
                continue
            else:
                why = "disagreement"
                pstat["disputed"] += 1

            sentiment_pairs[
                f"{side_token(raw['opus'], derived['opus'])}|"
                f"{side_token(raw['codex'], derived['codex'])}"] += 1
            full_pairs[
                f"{side_token(raw['opus'], derived['opus'], True)}|"
                f"{side_token(raw['codex'], derived['codex'], True)}"] += 1
            disputes.append({
                "item": m["item"], "why": why, "mapping": m,
                "opus_raw": raw["opus"], "opus_derived": derived["opus"],
                "codex_raw": raw["codex"], "codex_derived": derived["codex"],
            })

        report["packets"][pid] = dict(pstat)
        totals.update(pstat)

    (run_dir / "agreed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in agreed),
        encoding="utf-8")
    (run_dir / "disagreements.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in disputes),
        encoding="utf-8")
    report.update({
        "totals": dict(totals),
        "agreed_distributions": {
            "sentiment": dict(sentiment_dist), "reason": dict(reason_dist),
            "profile": dict(profile_dist),
        },
        "dispute_pairs": {
            "sentiment": dict(sentiment_pairs), "full": dict(full_pairs),
        },
        "confidence": {judge: dict(counts) for judge, counts in confidence.items()},
        "judge_vs_haiku": {judge: haiku_summary(counts)
                            for judge, counts in versus_haiku.items()},
    })
    (run_dir / "collect_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    total = totals["items"] or 1
    print(f"items: {totals['items']}  agreed: {totals['agreed']} "
          f"({totals['agreed'] * 100 // total}%)  disputed: {totals['disputed']} "
          f" uncovered: {totals['uncovered']}")
    print("sentiment pairs:", dict(sentiment_pairs))
    print("full pairs     :", dict(full_pairs))
    if report["lane_errors"]:
        print("MISSING/UNPARSEABLE lanes:", report["lane_errors"])


if __name__ == "__main__":
    main()
