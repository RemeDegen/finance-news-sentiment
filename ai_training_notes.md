# AI Training Notes — Financial News Sentiment Analysis

> Full record of the AI side of the project: labeling, training experiments,
> data-growth attempts and the final decision. §1 is the outcome, §2–§9 the
> chronological record, §10 rules, §11 the resume recipe. Last update:
> **3 September 2026, 05:00 — PROJECT FINISHED.**
>
> **Logging rule (user, 2 Sep 2026):** every AI-training-related task
> (experiment, script, data, decision) and its outcome (prediction held or
> not, measured numbers) is written here in medium detail; a new session
> resumes from this file. This file is exempt from the 500-line rule.

## 1. FINAL STATE (3 September 2026, 04:50)

**Final model:** `models/finbert-sentiment-v2-sqrt` (A2).
**Final dataset:** `data/complete_v2/{train,val,test}.csv` =
`data/final_34968_labeled.csv` (the three splits merged, with a `split` column).

### Result (test set, 3,500 rows)

| model                  | acc       | macro F1  | pos / neg / neu F1    | mean conf | ECE   | conf ≥0.95 share / acc |
|------------------------|-----------|-----------|-----------------------|-----------|-------|------------------------|
| **A2 v2-sqrt (FINAL)** | **0.842** | **0.808** | 0.747 / 0.788 / 0.889 | 0.938     | 0.096 | 77% / 0.914            |
| A7 v2-sqrt-warmup      | 0.845     | 0.809     | 0.751 / 0.782 / 0.892 | 0.974     | 0.130 | 90% / 0.885            |

A2 was chosen: accuracy equal to A7 (+0.3 / +0.1 = ~10 headlines out of
3,500, noise), better calibration (less confident when wrong), simpler recipe.
For a product that uses a confidence threshold, A2's ≥0.95 slice (77% of
headlines) is 91% correct. Reference: the FinBERT paper reports 0.86 on the
full Financial PhraseBank and 0.97 on the all-annotators-agree subset, with
human labels on short clean sentences. On Telegram headlines with LLM labels,
0.84 / 0.81 is comparable.

### How the dataset was built (details in §3–§4)

1. **Source:** 142k English headlines / short news items from Telegram finance
   channels (`data/raw_142k_deduped.csv`, deduplicated). Channels: firstsquaw
   82.8%, WalterBloomberg 12.5%, MonitoringSituation 2.2%, 8 other channels
   2.5%. Dates Feb 2023 – Jun 2026, ~98% of rows in the last 12 months.
2. **Sample:** 35,000 random rows (seed 42); 2 too-short rows dropped, 30
   not_news rows (ads / service notices) removed → 34,968.
3. **Labeling (run1):** two blind judges from different model families
   (Claude Opus 5 and Codex gpt-5.6-sol), packets of 40, seeing only `item` +
   `text`. Rulebook `scripts/ai_analysis/judge_spec.md` v2 (3-class sentiment
   + 7 reasons; framing "is this news good or bad for an investor?").
   Agreement 85.4%; the 5,158 disputes were settled one by one by a Claude Fable 5
   arbiter using the same spec (~70% sided with Opus 5 / 27% Codex / 3% third
   label).
4. **Split:** stratified by sentiment 80/10/10 → train 27,973 / val 3,495 /
   test 3,500. Distribution: neutral 21,297 (60.9%) / negative 8,055 (23.0%) /
   positive 5,616 (16.1%). Reasons: geopolitics 14,373, macro 8,138, equities
   6,916, commodities 2,368, other 1,510, crypto 970, regulatory 693.
5. **Quality control:** two independent blind audits (300 rows each, seeds 7 /
   11) agreed with the labels 94.3% and 91.3%; 82–85% of disagreements sit on
   the neutral↔directional boundary. Expected model ceiling acc 0.83–0.88 /
   macro F1 0.78–0.85 (§9).

### How the model was trained (A2, `scripts/train_sentiment_v2.py`, 2 Sep, 63 min)

- Base ProsusAI FinBERT (`models/finbert`), new 3-class head (`head.pt`).
- Class weights sqrt(N/n_c) normalized ≈ [pos 1.441, neg 1.203, neu 0.740];
  constant lr 2e-5; 3 epochs; batch 16; max_len 128; dynamic padding +
  length-grouped sampler; MPS (MacBook Air M4 16 GB).
- Best val macro F1 checkpoint (E3) saved to disk; test run with that
  checkpoint. Val: E1 0.812/0.780 → E2 0.826/0.795 → E3 0.842/0.812.

### Tried and abandoned

- **A1** balanced weights: 0.833 / 0.805 → sqrt was better. Model deleted.
- **A3** complete_v3 (35k + 5k labeled with spec v3): 0.853 / 0.820, the
  highest score; but the v2/v3 doctrine split (war headlines: run1 68%
  negative vs waves 17% negative) produced mixed data. Dataset and model
  deleted. The +1.2 points came from the 5k v3 rows, not from warmup (A7 proof).
- **A4** DeBERTa-v3-base: OOM on 16 GB, cancelled.
- **A5** prepared for complete_v4 (45k), never run; data deleted, script kept.
- **A6** fresh 80/10/10 on 45k: E2 0.808 / 0.769 → stopped, deleted.
- **A7** A2 recipe + warmup: equal result, worse calibration; model kept.
- **Active-learning waves (run2, run3):** 10k extra labels; arbiter outputs
  remain under `data/ai_analysis/runs/run2, run3`, not used in the final set.

### Known limitations

- One channel (firstsquaw) is 83%, ~98% of rows from the last 12 months →
  performance may drop on other sources/periods (temporal holdout estimated
  3–5 points lower).
- The neutral class is broad and the "small move" threshold is inconsistent;
  pos/neg F1 0.75–0.79, neutral 0.89. 85% of errors are neutral↔directional.
- Labels are LLM-generated; no human gold set (`data/gold_set_300_blind.csv`
  is the user's blind set, not used in evaluation).
- Redistribution of Telegram content: check channel/copyright status before
  sharing.

### File inventory (final)

- `data/`: `complete_v2/`, `final_34968_labeled.csv`, `raw_142k_deduped.csv`
  (raw 142k), `gold_set_300_blind.csv`, `financial_phrasebank/`,
  `ai_analysis/runs/{pilot1,run1,run2,run3}` (judge + arbiter records),
  `ai_analysis/wave1_5k.csv`, `wave2_5k.csv` (wave sources with model scores).
- `models/`: `finbert-sentiment-v2-sqrt` (FINAL), `finbert-sentiment-v2-sqrt-warmup`
  (A7), `finbert`, `deberta-v3-base` (bases), `old_reports/`.
- `scripts/`: `train_sentiment_v2.py` (final recipe), `v3.py`, `v5.py`,
  `v7.py` (historical), `train_sentiment_v4_deberta.py` (cancelled),
  `score_pool.py`, `ai_analysis/*` (labeling pipeline); `label_dataset*.py`,
  `train_finbert*.py`, `train_deberta.py`, `train_sentiment_only.py` are
  legacy Haiku-era scripts.
- Inference: `uv run scripts/score_pool.py --model models/finbert-sentiment-v2-sqrt
  --input <csv> --output <csv>` (adds p_*/pred/confidence/margin, sorts by
  ascending margin).

## 2. Why this work was done

- The old set (`dataset_35k_labeled.csv`) was labeled with the Haiku Batch
  API; FinBERT plateaued at 77% (neutral F1 0.70, macro ~0.73). The pilot
  showed ~24% of Haiku sentiment labels were wrong (overlap 76.2%; reason
  86.7%); the full run1 measurement gave 62.75% overlap → ~37% wrong. The
  model was hitting the label-noise ceiling.
- Fix: the dual-judge blind protocol from the dimensionnews project
  (`~/Desktop/projeler/telegram aggretor poly newsterminal`,
  `scripts/ai_analiz/`) was adapted; the same 35k was relabeled from scratch.
- The target was "macro F1 / acc 90+". Honest ceiling: judge agreement 85.4%;
  the uncertainty sits on the neutral boundary. Road to 90: clean test set,
  confidence threshold (product), backbone/ensemble, label cleaning. No magic.

## 3. Labeling protocol (6 steps)

1. **Packetize** — `build_packets.py`: rows shuffled with a deterministic seed
   (`run:shuffle`), split into packets of 40. A packet holds ONLY `item` id +
   `text` (channel, date, views, old label hidden). Row↔packet mapping lives
   in `mapping/`, read only by the collector.
2. **Two judges, two model families** — A: Claude Opus 5 sub-agents
   (`verdicts/opus_pXXX.json`); B: `codex exec` lanes (gpt-5.6-sol, reasoning
   medium, read-only sandbox, `verdicts/codex_pXXX.raw`). Two copies of the
   same family make the same mistakes and hide them inside "agreement".
3. **One written spec** — `judge_spec.md` (v2); both judges read the same
   file. The spec never changes mid-run.
4. **Mechanical collection** — `collect_verdicts.py`: same `derive()` path,
   agreement key `(profile, sentiment, reason)` → `agreed.jsonl` /
   `disagreements.jsonl`.
5. **Dispute → arbiter** — Fable in the main session, blindness lifted;
   verdicts in the judges' vocabulary in `arbiter_input.jsonl`
   (`{"item","profile","sentiment","reason","note"}`), `apply_arbiter.py`
   passes them through the same `derive()`.
6. **Output** — `build_dataset.py`: agreed + arbiter, every row carries its
   source and both judges' confidence; normalized-text dedup + stratified
   80/10/10.

```
scripts/ai_analysis/
  judge_spec.md           v2 rulebook (final)            judge_spec_v3.md  v3 (262 lines, waves)
  build_packets.py        --run --limit --packet-size --source --keep-order
  collect_verdicts.py     collection + agreement report  dump_disputes.py  arbiter queue (arbiter_queue/)
  flag_model_disputes.py  item 5: rows the model contradicts at conf ≥0.9 go to the dispute list (m*.json)
  verify_arbiter_part.py  validates one 40-item arbiter part
  apply_arbiter.py        applies arbiter input            build_dataset.py  run1 → complete_v2
  build_dataset_v3.py     incremental set (--run --base --out); build_dataset.py untouched
  codex_wave_run1.zsh     run1 driver                    codex_wave.zsh <run> <spec> <n> [effort] [lanes]
data/ai_analysis/runs/pilot1  1,500 rows, spec v1 — diagnosis only, not in the final set
data/ai_analysis/runs/run1    35k, 875 packets, spec v2 — THE MAIN RUN
data/ai_analysis/runs/run2    wave 1 5k, spec v3 (unused)   run3  wave 2 5k, spec v3 (unused)
```

## 4. run1 log (31 Aug – 2 Sep 2026) — 35k, spec v2

Pilot (1,500 rows, spec v1): agreement 80.5%, 293 arbiter verdicts, spec
v1→v2. run1: 875 packets / 34,998 rows (2 too-short rows dropped).

| Wave | Packets  | Rows  | Agreement | Arbiter   | Note                                 |
|------|----------|-------|-----------|-----------|--------------------------------------|
| 1    | p001–072 | 2,880 | 84.9%     | 435 (w1c) |                                      |
| 2    | p073–182 | 4,400 | 86.8%     | 579 (w2c) | Codex 1.11M tokens                   |
| 3    | p183–292 | 4,400 | 85.1%     | 654 (w3c) | 1 Sep morning, autonomous via cron   |
| 4    | p293–402 | 4,400 | 85.6%     | 634 (w4c) |                                      |
| 4.5  | p403–432 | 1,200 | 85.5%     | 174 (w5c) | leftover-quota mini wave             |
| 5    | p433–557 | 5,000 | 84.7%     | 767 (w6c) | internet outage, 11 packets next day |
| 6    | p558–682 | 5,000 | 84.9%     | 753 (w7c) | most efficient: 8.8k tokens/packet   |
| 6.5  | p683–732 | 2,000 | 83.7%     | 326 (w8c) | mini wave                            |
| 7    | p733–875 | 5,718 | 85.4%     | 836 (w9c) | 2 Sep 07:27 cron, final              |

- Arbiter verdicts total **5,158** (`arbiter_parts/w1c–w9c`), validation 0
  errors; ~70% Opus 5 / ~27% Codex / ~3% third label. Zero Codex failures; 14
  item-id typos/uncovered items sent to the arbiter; ~15 items relabeled
  not_news.
- Codex total 7,780,933 tokens (~9–10k/packet); Opus 5 ~1.2M per wave.
- Final pipeline (2 Sep): collect (85% agreement, 5,144 disputes + 14
  uncovered) → arbiter_input (5,158) → apply_arbiter → build_dataset.
  `final_labeled.csv` 34,998 rows (29,840 agreed + 5,158 arbiter; 30
  not_news) → `complete_v2`.

## 5. Training experiments

**Rules:** the user runs training (exceptions: A3 and A7, run by Claude in
the background with explicit permission); one script per experiment
(`train_sentiment_vN.py`), earlier scripts are never edited. Label order in
the scripts is `["positive","negative","neutral"]`; balanced weights ≈
[2.076, 1.447, 0.547], sqrt ≈ [1.441, 1.203, 0.740]. Decision metric: macro
F1 + positive P/R balance. Old Haiku-set baseline: acc ~77%, neutral F1
0.70, macro ~0.73.

### Experiment table (test set)

| #      | Data         | Weights  | lr            | Ep    | Best val ep (F1) | Test acc  | Test macro F1 | Pos F1 (P/R)      | Neg F1 (P/R)      | Neu F1 (P/R)      |
|--------|--------------|----------|---------------|-------|------------------|-----------|---------------|-------------------|-------------------|-------------------|
| A1     | v2           | balanced | 2e-5          | 3     | 2 (0.801)        | 0.833     | 0.805         | 0.757 (0.70/0.82) | 0.776 (0.73/0.84) | 0.881 (0.93/0.84) |
| **A2** | **v2**       | **sqrt** | **2e-5**      | **3** | **3 (0.812)**    | **0.842** | **0.808**     | 0.747 (0.71/0.79) | 0.788 (0.80/0.78) | 0.889 (0.90/0.88) |
| A3     | v3 (32,971)  | sqrt     | 2e-5 wu6%+lin | 5     | 3 (0.817)        | 0.853     | 0.820         | 0.774 (0.76/0.79) | 0.788 (0.81/0.77) | 0.898 (0.89/0.90) |
| A6     | v4 80/10/10* | sqrt     | 2e-5 wu6%+lin | 5→2   | —                | —         | —             | —                 | —                 | —                 |
| A7     | v2           | sqrt     | 2e-5 wu6%+lin | 5     | 5 (0.809)        | 0.845     | 0.809         | 0.751 (0.75/0.75) | 0.782 (0.79/0.78) | 0.892 (0.89/0.90) |

- A1: test with the epoch-3 model; disk had epoch 1 (old script). Model deleted.
- **A2: FINAL.** 63 min. Checkpoint E3 = test model.
- A3: 123 min; E4–5 wasted (val_loss 0.57→0.90). Dataset and model deleted.
- A6: *train 35,968, test 4,496 (different test set, not comparable). E1 val
  0.781/0.742, E2 0.808/0.769 → stopped, deleted.
- A7: 102 min; equal to A2; checkpoint E5, val_loss 0.953 → poor calibration.

### Experiment readings

- **A1 (2 Sep 16:37):** val macro E1 0.758 → E2 0.801 → E3 0.796; train_loss
  0.607→0.385→0.256, val_loss 0.476→0.485→0.542 → memorization after E2, no
  decay with constant lr. Pos/neg recall > precision, neutral the reverse
  (P 0.93 / R 0.84): balanced weights over-predict pos/neg, ~16% of neutrals
  leak.
- **A2 (2 Sep 17:51):** val macro 0.780 → 0.795 → 0.812; at E3 val_loss jumped
  0.460→0.591 while F1 rose (calibration worsened, accuracy improved).
  Neg/neutral P/R became symmetric; positive still R > P (0.79/0.71), the
  weakest class.
- **A3 (2 Sep 20:44–22:47):** train 32,971 (v2 + wave-1 4,998), sqrt
  w=[1.449, 1.234, 0.732], warmup 618/10,305 steps. Val macro 0.786 → 0.810 →
  **0.817** → 0.812 → 0.817; val_loss 0.498 → 0.470 → 0.574 → 0.796 → 0.898.
  Gains from positive precision (0.705→0.759) and neutral recall
  (0.879→0.903). Flaw: data and lr changed at once, not separated → A7
  separated them.
- **A7 (3 Sep 02:52–04:34):** val acc/F1 E1 0.820/0.788 → E2 0.836/0.806 →
  E3 0.839/0.808 → E4 0.837/0.805 → E5 0.841/0.809; val_loss 0.461 → 0.462 →
  0.596 → 0.852 → 0.953. Test 0.845 / 0.809 vs A2 0.842 / 0.808: **the recipe
  effect is unmeasurable**; A3's +1.2 came from the 5k v3 rows. Calibration
  (score_pool on test): A2 mean confidence 0.938, ECE 0.096, ≥0.95 share 77% /
  acc 0.914; A7 0.974, ECE 0.130, 90% / 0.885; A7's 0.8–0.95 confidence
  slice (165 headlines) is only 40–46% correct. The two models agree on 90.9%
  of predictions; only A2 correct 137, only A7 correct 147.
- **A4 DeBERTa (CANCELLED, 2 Sep 23:47):** `train_sentiment_v4_deberta.py`
  (fp32 required; config float16 → NaN). Three attempts on M4 16 GB: bs16 →
  15 GB swap, freeze; bs16 + grad ckpt + 0.9 cap → OOM in minute 1; micro 8×2
  + grad ckpt + empty_cache → OOM at 50% of epoch 1. Cause: Metal "other
  allocations" (per-shape graph cache ~5 GB), empty_cache ineffective. On a
  Colab/Kaggle GPU it would be a one-hour job.

### Script changes (from A2 on, `train_sentiment_v2.py`)

1. Every report goes to `train_log.txt`; output `models/finbert-sentiment-v2-{mode}/`.
2. Checkpoint criterion macro F1 (not val_loss); test report uses the best
   checkpoint on disk → report = saved model.
3. Dynamic padding + length grouping (`make_collate`, `LengthGroupedSampler`,
   `sorted_batches`): mean batch length 128 → 42; epoch 35.7 → 20.7 min (1.7x;
   not 3x because of fixed per-step cost on MPS).
4. v3/v5/v7 scripts import from v2; addition: `get_linear_schedule_with_warmup`.

### Prediction ledger

| Prediction                                                | Outcome                                                                  |
|-----------------------------------------------------------|--------------------------------------------------------------------------|
| A1: 3rd epoch is too many, memorization starts            | TRUE — val_loss 0.485→0.542, val macro 0.801→0.796                       |
| sqrt cuts the positive-precision leak                     | TRUE — E1 pos P 0.57→0.72, neutral R 0.74→0.82                           |
| Dynamic padding shortens an epoch ~3x                     | FALSE (partly) — 1.7x                                                    |
| sqrt beats A1 on macro F1                                 | TRUE (narrow) — 0.805→0.808, acc 0.833→0.842; P/R balance clearly better |
| A2: E3 is meaningful                                      | TRUE — val macro 0.795→0.812; val_loss jumped (calibration)              |
| A3 test macro F1 ≥ 0.82                                   | TRUE (borderline) — 0.820; two variables at once, not separated          |
| v3 labels are neutral-heavy → A3 neutral R ↑, pos/neg R ↓ | PARTLY — neutral R +2.4; pos/neg R −0.5/−1.0; pos P +5.4 net gain        |
| 5 epochs + linear decay keep gaining after E3             | FALSE — A3 best at E3; A7's E5 only +0.1                                 |
| DeBERTa-v3-base +2–4 points                               | NOT MEASURED — OOM, cancelled                                            |
| A5 (v4, A3 recipe) 0.855–0.862 / 0.822–0.830              | NOT RUN                                                                  |
| A6 (45k fresh split) 0.85–0.87 / 0.82–0.84                | FALSE — E2 val 0.808 / 0.769, stopped                                    |
| A7 (v2 + warmup) 0.845–0.852 / 0.812–0.818                | LOWER BOUND — 0.845 / 0.809; recipe effect ~0                            |

### Ceiling analysis (2 Sep 22:55, A3 test 3,500, by run1 label source)

Judges agreed (2,955) → acc 0.882; went to arbiter (545) → 0.697; agreed +
both high confidence (1,400, 40%) → 0.961. Model confidence threshold ≥0.90
→ coverage 83%, acc 0.910; ≥0.95 → 77%, 0.926. 85% of errors are
neutral↔directional (neg→neu 142, neu→neg 112, neu→pos 95, pos→neu 87),
polarity errors 78/514. Overall accuracy sits at the label ceiling; where the
judges are sure the model is 96% right.

## 6. Data growth / active learning (2–3 Sep) — LATER ABANDONED

**Plan (2 Sep, approved):** label 5k at a time from the 107k pool, add to
train. Same pipeline (no cheap single-model labeling); val/test frozen;
versioning complete_v3, v4…; each wave = most uncertain rows (margin =
top1−top2) + random rows (pure uncertain rows turn train into a "contested"
set). Agreement drops on uncertain rows (measured 84.7% vs random 90.4%).

**Pool:** `havuz_107k.csv` = 142k − labeled 35k (id+channel) = 107,664 rows
(later deleted). **score_pool A2 (2 Sep 18:16, 17.5 min):** predictions
neutral 59% / neg 22% / pos 18%; confidence<0.6 4.9%, margin<0.2 3.8% →
"really uncertain" ~5k. Top-4k dilemmas: neu/pos 1,984, neg/neu 1,317,
neg/pos 699.

### Wave 1 (run2, 2 Sep 18:30–20:47, spec v3)

- `wave1_5k.csv` = 4,000 most uncertain (margin ≤0.193) + 1,000 random (seed
  2). Codex `codex_wave.zsh run2 judge_spec_v3.md 120` (medium, 6 lanes,
  1.71M tokens, 13.7k/packet) + Opus 5 20 agents × 6 packets; 125/125, 0 broken.
- Full agreement 82.2%, sentiment 85.8%; uncertain slice 81.3% / 84.7%, random
  85.7% / 90.4% (v3 gave +5 points on normal news vs run1). Disputes 889
  (neutral↔direction 696, reason-only 179, polarity 13). Codex confidence 81%
  high; Opus 5 32% high / 64% medium.
- Model flags (A2 conf ≥0.9, item 5): 112, all from the random slice; arbiter
  KEEP 112/112 → those rows are v3 doctrine the model does not know.
- **Arbitration:** 1,001 verdicts, 26 parts (`arbiter_parts/w1c01–26`), 6
  Fable forks in parallel (≈2.0M tokens → hence the no-fork rule). Arbiter
  sided Opus 5 64% / Codex 34% / third 2.6%; output neutral 832 / pos 89 / neg
  79 / not_news 1.
- **complete_v3:** train 32,971 = 27,973 + 4,998 (dedup 0), val/test
  byte-identical to v2. New labels neutral 3,487 (70%) / neg 772 / pos 739.
  A2's agreement with the final labels 49.7% (uncertain 42.8%, random 77.4%,
  below test acc: v3 is more neutral-heavy, first warning).

### Wave 2 (run3, 3 Sep 00:11–01:31, spec v3)

- `havuz_102k.csv` = 107k − wave 1 = 102,664 (later deleted). **score_pool A3
  (user):** neutral 61% / neg 22.7% / pos 16.3%; margin<0.2 2,725 (2.7%), <0.3
  4,165, conf<0.6 3,389. Top-3k margin cap 0.217; dilemmas neg/neu 1,489,
  neu/pos 1,162, neg/pos 349.
- `wave2_5k.csv` = 3,000 uncertain (neutral 1,385 / neg 864 / pos 751) + 2,000
  random (seed 3), shuffled, 125 packets. Opus 5 21 agents × 6 packets (125/125
  at 00:33, distribution neutral 62.5% / neg 19.6% / pos 17.9%).
- **Codex effort episode:** started medium 6 lanes; 00:30 user: "stop, rerun
  with HIGH" → 11 medium packets moved to `verdicts_medium_cancelled/`;
  `[effort]` and `[lanes]` parameters added to `codex_wave.zsh`, 20 lanes.
  00:50 quota exhausted (88/125); 00:56 reset credit → 01:05 125/125, 0
  errors. **Medium vs high (same 440 headlines):** full agreement with Opus 5
  82.3% vs 83.2%, sentiment 86.4% vs 87.3%, medium↔high 93.9% → 4 headlines
  apart, noise; high brought no quality for 5x quota. User: "high was a
  mistake".
- **collect:** full agreement 81.6%, sentiment 86.2%; disputes 919 (23 parts)
  + model flags (A3 conf ≥0.9) 208 (6 parts) = 1,127.
- **Arbitration (01:08–01:30, Fable main session, no forks):** 1,127
  verdicts, 29 parts `w2c01–23 + w2m01–06`, verify OK. Neutral 912 / pos 102 /
  neg 109 / not_news 4 (81% neutral in disputes: Codex high says "no prior →
  neutral" on data prints, Opus 5 leans directional). Model flags 208: arbiter
  sided with the model on 7; 201 were v3 doctrine. 8 verdicts revised for
  consistency (Duffy warnings, Hormuz "traffic control" cluster → neutral,
  "expectations anchored" → neutral, war-driven output loss → geopolitics).
- **complete_v4:** train 37,965 = 32,971 + 4,994 (6 not_news, dedup 0),
  val/test unchanged. New labels neutral 3,384 (67.8%) / neg 824 / pos 786.
  A3's agreement 60.9% (uncertain 47.9%, random 80.5%).
- **A5** (`train_sentiment_v5.py`, A3 recipe, v4) prepared, not run.
- **Wave 3 idea (not implemented):** run Codex like Opus 5, one context per
  agent with 6 packets (`codex exec --sandbox workspace-write`, 21 parallel);
  spec read once (~25% fewer tokens). Risk: a capacity error drops 6 packets;
  pilot with one agent first.

### Quota notes (Codex $20 Plus)

- ~9.7k tokens per packet (spec v2), ~13.7k (v3). Widget measurement:
  **medium ≈ 0.18% of a session per packet (5h ≈ 550 packets), high ≈ 1% per
  packet (5x).** The log's "tokens used" excludes reasoning; use only the
  widget for quota. Weekly ≈ 6.3 sessions; a 125-packet medium wave ≈ 22% of a
  session ≈ 3.5% of the week.
- Opus 5 20x plan, not a constraint. **Gemini is BANNED.** If Codex runs out:
  split across weeks or go all-Claude (breaks the two-family rule, avoid).

## 7. Protocol improvement decisions and spec v3 (2 Sep evening)

The user brought 7 suggestions from a separate Fable session:

| #   | Suggestion                                                      | Decision                                                       |
|-----|-----------------------------------------------------------------|----------------------------------------------------------------|
| 6   | Bake the arbiter case law into spec v3                          | DONE (below)                                                   |
| 2   | Gold test set hand-labeled by the user (300, blind)             | File exists (`gold_set_300_blind.csv`), not used in evaluation |
| 5   | Model as third judge (high-confidence contradictions → arbiter) | DONE: `flag_model_disputes.py`, 112 + 208 flags in waves 1–2   |
| 4   | Arbiter anchoring on the Haiku label                            | Measured (below); from run2 on the pool has no Haiku label     |
| 1   | Audit agreed rows with Fable                                    | Skipped; done as a blind audit on 3 Sep (§9)                   |
| 3   | Rule-bound, semi-automatic arbiter                              | NOT DONE (robustness > speed)                                  |
| 7   | Active learning                                                 | §6                                                             |

**Item 4 measurement (5,118 arbitrated rows):** the arbiter sided with Opus 5
83.7% when Haiku agreed with Opus 5, 59.3% when Haiku agreed with Codex, 72.8%
when Haiku agreed with neither (overall 70.3%); stratified odds ratio 6.2
(label pair) / 3.8 (confidence pair). Reading: strong correlation but not
proof of anchoring; since Haiku is 63% right, a correct arbiter also leans
toward Haiku's side (rough Bayes expectation 84% / 51% = observed). Separating
them would need re-arbitrating 300 rows with Haiku hidden; not done.

**Dispute analysis (run1's 5,144 disputes, Opus 5 agent):** 74.7% involve
sentiment; of those 93.8% are neutral↔directional, polarity clashes 5.5%.
Codex leans directional, Opus 5 neutral (Opus 5 64% neutral, Codex 30%); the one
exception is layoffs (Codex right, 75% negative). Largest group: neu/neg →
arbiter neutral, geopolitics, 608 rows (ongoing events). News types: military
strike 599, political statement 463, plan/intent 279, oil/energy 272, single
asset move 226, macro print 176, central banker remark 171, tariff/sanction
153, M&A 152, court 138, ceasefire 128. Reason confusions: geopolitics/other
364, commodities/geopolitics 148, macro/other 147, equities/other 112.

**Spec v3** (`judge_spec_v3.md` = v2 + 18 calibration + 3 reason rules).
Four high-volume additions: (1) known/ongoing event → neutral; (2)
intercepted / no damage → neutral; (3) central bankers exempt from "actions
not words", tone = signal (politicians are not); (4) numeric prints default
neutral, directional only if expectation and prior agree and the threshold is
crossed. Smaller gaps: earnings beat + cautious guidance → neutral; PMI small
miss but >50 → neutral; weekly claims → neutral; large layoffs → negative;
concrete plan directional, vague neutral; probe of a politician neutral, of a
company negative; routine small crypto buys neutral.
**Pilot (run2 p001–005, 200 items, 2 Sep 19:20):** Codex 5/5 zero errors
(12.5k tokens/packet); sentiment agreement 86.0% (uncertain 84.0%, random
94.6%); A2's agreement with the agreed labels 48.3% → active-learning
selection validated.

**Learned later:** v3's "ongoing event / interception → neutral" doctrine
reversed run1's v2 doctrine (every strike negative); a mixed set resulted
(§9). Lesson: if the spec changes, either re-arbitrate everything or keep a
separate set.

## 8. Arbiter case law

**v2 (run1):** actions not words; inconclusive meeting → neutral; ≤0.1–0.2% /
a few bp → neutral; commodity supply cut → positive, increase → negative
(normalization story → positive); aid delivered → positive, withheld →
negative; event contained → positive, ongoing damage → negative, distant
event with no effect → neutral; risk-off-driven asset move → negative; poll →
neutral; sanctions imposed → negative, eased → positive (geopolitics);
politician's remark → neutral, a central banker's policy signal is directional.

**Wave-2 case law (spec v4 candidates, never applied):** (1) forecast
present / prior absent → deviation ≥ threshold suffices; (2) no forecast →
sign change / crossing 50 / big jump gives direction, slowdown from a strong
level neutral; (3) PMI without forecast, ≥1 point move directional; (4) weekly
claims / MBA / ADP neutral; (5) monthly PSNCR neutral; (6) routine FX quote
neutral; (7) PBOC fixing / reverse repo neutral; (8) commodity drop clearly
due to de-escalation is risk-on → positive; (9) war-driven price shock /
supply loss → negative; output loss reason geopolitics, price commentary
commodities; (10) a decision-maker rejecting an expected de-escalation →
negative, repeated posture neutral; (11) central-bank explanatory remark
neutral, explicit policy lean directional; (12) domestic politics /
law-enforcement → regulatory, human-interest → other; OpenAI/Anthropic →
other; MSTR → crypto; (13) analyst rating/target → direction, macro forecast →
neutral; (14) "near deal / prepares bid / in-principle" → neutral. Spots where
arbiters disagreed: data prints without expectations, FX pair subject, PBOC
fixing, SPR release, "slightly", domestic-politics reason, war-driven supply loss.

## 9. Blind audit, doctrine split and the final decision (3 Sep 01:45–04:50)

- **01:45 user:** "drop the val/test freeze; split the 45k 80/10/10,
  fine-tune FinBERT, the project ends." `build_dataset_v4_split.py` →
  `complete_v4_80_10_10` (44,960; 35,968 / 4,496 / 4,496, stratified, leakage
  0, every split neutral 62.7% / neg 21.5% / pos 15.9%). **A6** started.
- **02:00 blind audit** (at the user's request, 2 context-free Fable agents,
  only the 3 CSVs, 300 rows each, seeds 7 / 11): agreement with labels A 94.3%
  (strict ~85%) / B 91.3%; disagreements 82–85% neutral↔direction, polarity
  15–18%. Shared findings: neutral is broad ("the labeler is unsure"),
  arbitrary thresholds on small % moves (A50 +0.42 positive, Nikkei −0.2
  neutral), two competing commodity frames, "killed/dead" 65% neg / 33%
  neutral, channel 83% firstsquaw / 95% two channels, time 98% last 12 months
  (Iran/Hormuz/Trump heavy), positive 16% (5.8% within geopolitics), 34
  normalized duplicate pairs (4 conflicting), 250 first-50-char near-duplicate
  groups / 44 conflicting, split leakage ≤0.2%. Expectation acc 0.83–0.88 /
  F1 0.78–0.85. Suggestions (not done): relabel ~3–4k rows with written
  thresholds and one commodity frame; temporal and per-channel test reports.
- **02:30 doctrine split measured:** geopolitics + war/strike pattern 3,140
  rows (7%): run1 (v2) 68% neg / 29% neutral, waves 1–2 (v3) 17% neg / 80%
  neutral; "intercepted/shot down" 67% vs 9%. **User decision: the shared set
  is complete_v2 (pure v2); no mixed doctrine for +1 point.**
- **02:45 A6 result:** E1 0.781 / 0.742, E2 0.808 / 0.769 → stopped; data,
  model and scripts deleted.
- **02:52 A7** (`train_sentiment_v7.py`, v2 + A3 recipe) started by Claude at
  the user's request (user asleep); finished 04:34, result in §5.
- **03:00–03:25 cleanup (user "delete"):** `models/finbert-sentiment-v2` (A1),
  `finbert-sentiment-v3-sqrt` (A3), `data/complete_v3`, `complete_v4`,
  `havuz_107k*.csv`, `havuz_102k*.csv`, `dataset_35k*.csv`, the Haiku splits
  (`dataset_notr_cikarilmis_24470`, `dataset_sadece_notr_10530`),
  `scripts/eval_neutrals.py`. `final_34968_labeled.csv` written.
- **04:50 final decision:** A2 is the final model (§1). Project finished.
- **05:00 (14:30 local) housekeeping:** notes restructured and translated to
  English; Turkish file/dir names renamed (`ai_analiz` → `ai_analysis`,
  `run2_dalga1_5k` → `wave1_5k`, `run3_dalga2_5k` → `wave2_5k`,
  `*_medium_iptal` → `*_medium_cancelled`, `altin_set_300_kor` →
  `gold_set_300_blind`, `dataset_final_deduped copy` → `raw_142k_deduped`,
  `eski_raporlar` → `old_reports`, `önemli_yerler.md` → `key_notes.md`,
  package `finans-duygu` → `finance-sentiment`); script docstrings shortened
  in English; all in-code Turkish comments and log/print strings translated
  (23 scripts, logic verified unchanged by AST comparison). New
  `train_log.txt` headers now read `device= data= steps padding=dynamic`;
  existing logs keep the old Turkish wording.

## 10. Rules and pitfalls

- **Git:** the project has no backup (0 commits, no remote). User decision
  (1 Sep): NO commits, do not propose again.
- **Gemini is BANNED.** Arbitration happens in the Fable main session; more
  than one Fable agent/fork per session is FORBIDDEN (exception: when the
  user explicitly asks, as in the blind audit).
- **Training is the user's job;** Claude runs it only with explicit
  permission. One script per experiment.
- The "val/test FROZEN" rule in CLAUDE.md was voided by the 3 Sep decision;
  to be updated with the user's approval (not edited yet).
- `collect_verdicts.py` scans the whole mapping, a missing lane = uncovered →
  run only after all packets are done. Codex `.raw` output may be noisy; the
  parser takes the first JSON array. Opus 5 agents occasionally typo an item id;
  the verification step in the brief catches it, do not trim the brief.
- The arbiter NEVER reads files that contain model predictions
  (`wave1_5k.csv`, `wave2_5k.csv`, ranked pools).
- 1 Sep: 4 old Haiku-labeled fine-tunes (~2 GB) deleted, reports in
  `models/old_reports/`; `dataset_70k.csv` deleted.

## 11. Resume recipe for a new session (if labeling is reopened)

**Firing a wave:** source CSV → `build_packets.py --run runN --source <csv>
--keep-order` → Codex `caffeinate -i /bin/zsh scripts/ai_analysis/codex_wave.zsh
runN <spec> <n> medium 6` in the background (skips finished packets, stops on
quota) + Opus 5 `general-purpose` agents (model opus), 6 packets each, in
parallel. Agent brief template (use verbatim):

> You are judge A — one of two independent blind judges in a labeling pipeline.
> Base dir: <absolute path of the repo>
> Spec (read FIRST, follow exactly, including the "Calibration rules" section): <base>/scripts/ai_analysis/<spec>.md
> Packet files: <base>/data/ai_analysis/runs/<run>/packets/<pid>.json
> Output files: <base>/data/ai_analysis/runs/<run>/verdicts/opus_<pid>.json
> Your packets: pXXX, pXXX, ...
> For each packet in order: Read the packet file, judge every item per the spec, then Write the output file containing ONLY the strict JSON array (one object per item, same order, valid JSON, no prose). Before writing, verify the array length equals the packet's item count and every item id is echoed verbatim.
> Hard rules: read ONLY the spec and your packet files; write ONLY your output files. Do not open mapping/, CSVs, or any other project file. Use only the Read and Write tools — no Bash, no web. Final report: one line per packet — packet id, item count, written ok / failed.

**Chain:** `collect_verdicts.py --run runN` → `dump_disputes.py --run runN`
→ `flag_model_disputes.py --run runN` (needs a source CSV with model
predictions) → arbitration in 40-item parts `arbiter_parts/*.jsonl`
(`{"item","profile","sentiment","reason","note"}`), each part through
`verify_arbiter_part.py` → merge into `arbiter_input.jsonl` →
`apply_arbiter.py --run runN` → `build_dataset_v3.py --run runN --base <previous
set> --out <new set>`. Write arbiter verdicts into the run directory, not a
scratchpad. If the spec changes, do not mix with the old set.
