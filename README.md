# Finance News Sentiment — FinBERT fine-tuned on 35k LLM-labeled financial news headlines

A 3-class sentiment classifier (positive / negative / neutral) for short English
financial news, built from a new dataset: **34,968 financial news headlines (collected
from public Telegram news-wire channels) labeled by two independent LLM judges from different model families, with
disputes settled by an arbiter.** ProsusAI FinBERT fine-tuned on it reaches
**test accuracy 0.842 / macro F1 0.808** on a held-out 3,500-row split.

[![Dataset](https://img.shields.io/badge/🤗%20dataset-finance--news--sentiment--35k-blue)](https://huggingface.co/datasets/remehostingservices/finance-news-sentiment-35k)
[![Model](https://img.shields.io/badge/🤗%20model-finbert--finance--news--sentiment-green)](https://huggingface.co/remehostingservices/finbert-finance-news-sentiment)
![Python](https://img.shields.io/badge/python-3.12-informational)
![License](https://img.shields.io/badge/license-MIT%20code%20%2F%20CC%20BY--NC%204.0%20data-lightgrey)

## Results

Test split (3,500 headlines, never used for training or model selection):

| Model                                | Data                                             | Accuracy | Macro F1 | Pos / Neg / Neu F1   |
|--------------------------------------|--------------------------------------------------|----------|----------|----------------------|
| **This repo** (FinBERT + sqrt class weights) | 35k news headlines, LLM-labeled (this dataset) | **0.842** | **0.808** | 0.747 / 0.788 / 0.889 |
| FinBERT, Araci (2019)                | Financial PhraseBank, human-labeled, all sentences | 0.86     | 0.84     | –                    |
| FinBERT, Araci (2019)                | Financial PhraseBank, 100%-agreement subset        | 0.97     | 0.95     | –                    |

The two rows are not a head-to-head comparison: the datasets differ. Financial
PhraseBank is short, clean, human-annotated sentences; this dataset is raw
news-wire headlines labeled by LLMs, where the judges themselves agree on only
85% of rows. Our number sits at that label-noise ceiling: on rows where both
judges agreed with high confidence, the model is 96% correct; on rows that
needed arbitration, 70%.

Calibration matters if you use the model behind a threshold. With confidence
≥ 0.95 the model covers 77% of headlines at 91.4% accuracy (mean confidence
0.938, ECE 0.096).

Per-class detail (test):

| Class    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| positive | 0.71      | 0.79   | 0.747 | 563     |
| negative | 0.80      | 0.78   | 0.788 | 806     |
| neutral  | 0.90      | 0.88   | 0.889 | 2,131   |

85% of errors are on the neutral ↔ directional boundary; polarity flips
(positive ↔ negative) are rare.

## Dataset

**Source.** 8 public Telegram finance news channels, English, February 2023 –
June 2026. A 142k deduplicated pool was sampled to 35,000 rows; 32 rows were
dropped as too short or not news. Two channels make up 95% of the data and
about 98% of rows fall in the last 12 months:

| Channel (`channel_username`) | Rows   | Share | Date range              |
|------------------------------|--------|-------|-------------------------|
| firstsquaw                   | 28,958 | 82.8% | 2025-06-06 → 2026-06-05 |
| WalterBloomberg              | 4,360  | 12.5% | 2025-06-06 → 2026-06-05 |
| MonitoringSituation          | 754    | 2.2%  | 2025-06-14 → 2026-06-03 |
| ZoomerfiedNews               | 313    | 0.9%  | 2023-03-22 → 2026-06-02 |
| TreeNewsFeed                 | 222    | 0.6%  | 2023-02-09 → 2026-05-12 |
| apewirenews                  | 195    | 0.6%  | 2026-04-11 → 2026-06-03 |
| AGGRNEWSWIRE                 | 153    | 0.4%  | 2024-07-27 → 2026-06-01 |
| cukkanews                    | 13     | 0.0%  | 2025-11-13 → 2026-04-19 |

**Size and splits** (stratified by sentiment, 80/10/10):

| Split | Rows   | positive | negative | neutral |
|-------|--------|----------|----------|---------|
| train | 27,973 | 16.1%    | 23.0%    | 60.9%   |
| val   | 3,495  | 16.1%    | 23.0%    | 60.9%   |
| test  | 3,500  | 16.1%    | 23.0%    | 60.9%   |

**Columns:** `message_id`, `channel_username`, `date`, `text`, `views`,
`forwards`, `sentiment` (positive / negative / neutral), `reason` (geopolitics
14,373 · macro 8,138 · equities 6,916 · commodities 2,368 · other 1,510 ·
crypto 970 · regulatory 693). The `reason` column is a free secondary label;
the model in this repo uses only `sentiment`.

**How it was labeled.** Every headline was judged blind (text only, no
channel, date or engagement) by two LLMs from different model families —
Claude Opus 5 and OpenAI gpt-5.6-sol via Codex — working from the same written
rulebook, [`scripts/ai_analysis/judge_spec.md`](scripts/ai_analysis/judge_spec.md).
The framing is "is this news good or bad for an investor?", with explicit
calibration rules (actions over words, tiny moves are neutral, supply cuts are
positive for the commodity, and so on). The judges agreed on 85.4% of rows;
the remaining 5,158 disputes were settled one by one by a stronger model,
Claude Fable 5, acting as arbiter with both verdicts visible. Two later blind
audits (300 random rows each, independent Claude Fable 5 sessions) agreed with the final labels
94.3% and 91.3% of the time. The full labeling pipeline is in
`scripts/ai_analysis/`.

**Get it.** Download from Hugging Face: **https://huggingface.co/datasets/remehostingservices/finance-news-sentiment-35k**. Place the three
files under `data/complete_v2/`:

```
data/complete_v2/train.csv
data/complete_v2/val.csv
data/complete_v2/test.csv
```

or load them directly:

```python
from datasets import load_dataset
ds = load_dataset("remehostingservices/finance-news-sentiment-35k")   # splits: train / validation / test
```

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/). Training was done
on a MacBook Air M4 16 GB (MPS) in 63 minutes; a CUDA GPU works unchanged.

```bash
uv sync                                   # install dependencies
uv run scripts/download_finbert.py        # base model → models/finbert
# put the dataset under data/complete_v2/ (see above)
uv run scripts/train_sentiment_v2.py      # → models/finbert-sentiment-v2-sqrt/ (+ train_log.txt)
```

Or skip training and fetch the fine-tuned checkpoint from
[Hugging Face](https://huggingface.co/remehostingservices/finbert-finance-news-sentiment):

```bash
uv run hf download remehostingservices/finbert-finance-news-sentiment \
    --local-dir models/finbert-sentiment-v2-sqrt
```

The published checkpoint is a standard `BertForSequenceClassification`, so it
also works straight from the Hub:

```python
from transformers import pipeline
clf = pipeline("text-classification", model="remehostingservices/finbert-finance-news-sentiment")
clf("Fed holds rates, signals two cuts this year")   # [{'label': 'positive', 'score': 0.77}]
```

**Score your own headlines** (CSV with a `text` column; adds
`p_positive`, `p_negative`, `p_neutral`, `pred`, `confidence`, `margin`):

```bash
uv run scripts/score_pool.py --model models/finbert-sentiment-v2-sqrt \
    --input my_headlines.csv --output my_headlines_scored.csv
```

**Minimal inference in Python:**

```python
import torch
from transformers import AutoModel, AutoTokenizer

CKPT = "models/finbert-sentiment-v2-sqrt"
LABELS = ["positive", "negative", "neutral"]

tok = AutoTokenizer.from_pretrained(CKPT)
bert = AutoModel.from_pretrained(CKPT).eval()
head = torch.nn.Linear(768, 3)
head.load_state_dict(torch.load(f"{CKPT}/head.pt", map_location="cpu"))

def predict(text):
    enc = tok(text, max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(head(bert(**enc).pooler_output), dim=1)[0]
    return LABELS[int(probs.argmax())], probs.tolist()

print(predict("Fed holds rates, signals two cuts this year"))
```

## Training recipe

| Setting          | Value                                                             |
|------------------|-------------------------------------------------------------------|
| Base model       | ProsusAI/finbert (BERT-base), new 3-class linear head on `pooler_output` |
| Class weights    | sqrt of inverse frequency, normalized ≈ [pos 1.44, neg 1.20, neu 0.74] |
| Optimizer / lr   | AdamW, constant 2e-5                                              |
| Epochs / batch   | 3 / 16, max length 128 tokens                                     |
| Batching         | dynamic padding + length-grouped sampler (1.7× faster on MPS)     |
| Model selection  | best validation macro F1 checkpoint (epoch 3); test on that checkpoint |

Why these choices: fully balanced class weights over-predicted the minority
classes (positive precision 0.70); the square root keeps the neutral class
honest while still lifting positive/negative recall. Warmup + linear decay
with 5 epochs (`train_sentiment_v7.py`) gave the same accuracy (0.845 / 0.809)
but a worse-calibrated model (ECE 0.130), so the simpler 3-epoch recipe is
the release.

## What we tried that did not help

- **Balanced class weights** (`WEIGHT_MODE="balanced"`): 0.833 / 0.805, worse
  precision on positive.
- **Warmup 6% + linear decay, 5 epochs**: equal accuracy, over-confident on
  errors. Memorization starts after epoch 2 regardless of schedule.
- **DeBERTa-v3-base**: out of memory on a 16 GB Apple Silicon machine (Metal
  graph cache), not pursued; on a CUDA GPU it is a one-hour experiment.
- **Active learning (+10k labels in two waves)**: the model's most uncertain
  headlines were labeled with a revised rulebook (`judge_spec_v3.md`). The
  revision changed the doctrine for ongoing geopolitical events (68% negative
  under v2 vs 17% under v3), so the enlarged set mixed two labeling regimes.
  It scored higher (0.853 / 0.820) but was withdrawn; the released dataset is
  the pure v2 set. Lesson: change the rulebook and you must relabel
  everything, not append.
- **Re-splitting 45k fresh 80/10/10**: stopped at epoch 2 (0.808 / 0.769 on a
  harder, mixed-doctrine validation set).

Full experiment log with every number: [`ai_training_notes.md`](ai_training_notes.md).

## Limitations

- **Source concentration.** One channel contributes 83% of rows and ~98% of
  the data is from the last 12 months (heavy on Iran / Hormuz / tariffs /
  Fed). Expect a drop on other sources or periods; a temporal holdout would
  likely score 3–5 points lower than the random split.
- **Neutral is broad.** Small price moves, routine data prints and political
  statements are neutral by design; the threshold for "small" is not perfectly
  consistent across labelers. This is where almost all model errors live.
- **LLM labels, no human gold set.** Labels reflect the written rulebook as
  interpreted by two LLMs and an LLM arbiter. Inter-judge agreement (85%)
  and the blind audits (91–94%) bound the label quality.
- **Telegram content.** Headlines are reproduced from public channels for
  research; check the channels' terms before commercial use.

## Repository layout

```
scripts/
  train_sentiment_v2.py        final training recipe (this model)
  train_sentiment_v3/v5/v7.py  other experiments (kept for the record)
  train_sentiment_v4_deberta.py DeBERTa attempt (OOM on 16 GB)
  score_pool.py                batch inference / uncertainty ranking
  download_finbert.py          fetch the base checkpoint
  ai_analysis/                 labeling pipeline: packets → two judges → collect → arbiter → dataset
    judge_spec.md              the labeling rulebook (v2, used for the released data)
models/                        base + fine-tuned checkpoints (not in git)
data/                          dataset splits (not in git)
ai_training_notes.md           full experiment log
```

## Citation

```bibtex
@misc{financial-news-sentiment-2026,
  title  = {Finance News Sentiment: FinBERT fine-tuned on 35k LLM-labeled financial news headlines},
  author = {remehostingservices},
  year   = {2026},
  url    = {https://github.com/RemeDegen/finance-news-sentiment}
}
```

Base model and reference results: Araci, D. (2019). *FinBERT: Financial
Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
Checkpoint: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).

## License

Code: MIT ([LICENSE](LICENSE)). Dataset: CC BY-NC 4.0 (headlines originate from public Telegram
channels; non-commercial research use).
