# Evaluation

`sample_gold.csv` is **synthetic** (invented headlines) and exists only so the pipeline and CI can run.
Its numbers mean nothing. The real gold set is `gold.csv`, labelled by hand from real feed items.

## Building the gold set (30-50 articles)

```bash
newswatch ingest                                  # a few runs over a couple of days gives variety
newswatch export-labels --limit 60 --out eval/to_label.csv
```

Open `eval/to_label.csv`, fill two columns, delete rows you can't decide on, save as `eval/gold.csv`:

| column | value |
|---|---|
| `relevant` | `1` if someone following the brief would want this in their feed, else `0` |
| `topics` | 1-3 keys from the taxonomy, `;`-separated, e.g. `klima_energi;politik` |
| `notes` | optional: why it was hard to call |

Brief: *klima, energi og grøn omstilling i Danmark* (full text in `src/newswatch/classify/base.py`).

Labelling rules I follow, so the labels are consistent:

1. Label from headline + teaser only. That is all the classifiers see, so it is a fair comparison.
2. Label **before** looking at any classifier output, so the gold set is not biased toward either method.
3. Aim for roughly a third relevant. If the feeds give too few, deliberately include more climate/energy
   items and keyword traps (e.g. "klimaanlæg", "energidrik"), and say so in the README.
4. Ambiguous cases: write the reason in `notes`. They are the most interesting part of the error analysis.

## Running

```bash
# Free, offline: checks the plumbing, not the quality.
NEWSWATCH_LLM=fake newswatch eval --gold eval/gold.csv

# Real model. Replies are cached in eval/.cache/, so a re-run with the same prompt costs nothing.
NEWSWATCH_LLM=anthropic ANTHROPIC_API_KEY=... \
NEWSWATCH_PRICE_IN_PER_MTOK=... NEWSWATCH_PRICE_OUT_PER_MTOK=... \
newswatch eval --gold eval/gold.csv
```

Each run writes `eval/results/eval-<timestamp>.json` with metrics, token counts, cost and every
misclassified article with the classifier's rationale.

## Metrics

* **Precision / recall / F1** on "relevant" at threshold 0.5. Recall matters more for monitoring
  (a missed story is worse than an extra one to skim), so I report both rather than only F1.
* **P@k**: of the top k articles by score, how many are relevant. This is what a user scrolling the
  feed actually experiences.
* **Topic F1** (micro, multi-label) for the topic tags.
* **failed**: LLM replies that could not be parsed. They count as "not relevant", never guessed.
* **LLM calls, tokens, cost**: the hybrid's whole point is fewer calls; this shows what that costs in recall.

With 30-50 items, one article is 2-3 percentage points. Differences smaller than that are noise.
