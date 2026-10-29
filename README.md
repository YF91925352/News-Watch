# newswatch

A small news-monitoring service: it polls public Danish RSS feeds, ranks articles by relevance to a
monitoring brief (climate, energy and the green transition in Denmark), and - the main point -
measures how well that ranking works, comparing a keyword baseline against an LLM.

Python, PostgreSQL, FastAPI, Docker Compose, GitHub Actions.

| Problem | Where it lives |
|---|---|
| Sort a stream of articles by topic, relevance and news value | `classify/`, `service.rank_score`, `GET /articles` |
| Use a language model without giving up quality | LLM behind an interface, output validated against a fixed taxonomy, failures recorded as failures |
| Make quality measurable | `evaluate.py`, hand-labelled gold set, keyword vs LLM vs hybrid in one table |
| Respect sources, citation and the law | headline + teaser + link only, conditional GET, honest User-Agent |

## Quick start

```bash
cp .env.example .env            # required; set a contact address in NEWSWATCH_USER_AGENT
docker compose up --build       # api on :8000, worker polls feeds every 15 min
open http://localhost:8000/docs # or http://localhost:8000/ for a plain list
```

Without Docker:

```bash
pip install -e ".[dev,llm]"
export DATABASE_URL=postgresql+psycopg://newswatch:newswatch@localhost:5432/newswatch
newswatch ingest
newswatch classify --method keyword
NEWSWATCH_LLM=fake newswatch classify --method hybrid   # offline stand-in for the LLM
uvicorn --factory newswatch.api:app_factory
pytest                                                   # 40 tests, SQLite, no network
```

## How it works

```
feeds.yaml ─► ingest ─► sources / articles (PostgreSQL)
                               │
                               ▼
              ┌──────── classifiers (same interface) ────────┐
              │ keyword   │ llm (LLMClient)   │ hybrid:       │
              │ baseline  │ fake | anthropic  │ keyword gate  │
              │           │                   │ → llm         │
              └───────────────────┬───────────────────────────┘
                                  ▼
          classifications (+ model, prompt version + hash, rationale, tokens, review status)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            FastAPI /articles           evaluate.py vs gold.csv
            /review (human)             precision / recall / P@k / cost
```

**Ingestion** (`ingest.py`). Same idea as my earlier scraper project: watch a fixed set of sources
continuously, and be polite about it. Each source keeps its own ETag / Last-Modified, so an unchanged
feed costs the publisher a `304`. There is a minimum interval per source, a pause between requests,
and on `429` or `5xx` it simply waits for the next cycle instead of retrying in a loop. Articles are
unique on the normalised URL because DR publishes the same story in several section feeds.

This continues my earlier project, [rednote-scraper](https://github.com/YF91925352/rednote-scraper).
There, requesting pages too quickly triggered the platform's rate limits, so pacing and stopping on
any warning became design requirements, notes were de-duplicated by ID, and re-runs resumed instead
of starting over. Its "next steps" section proposed moving from flat files to a database with one row
per item and a last-seen timestamp, turning it into a change monitor. newswatch is that step, on
sources that publish RSS for reuse, so no browser automation or login is needed.

**Classification** (`classify/`). Every classifier returns the same `Classification`. The LLM sits
behind a one-method `LLMClient` interface; tests and CI use a deterministic fake, so they are free
and offline. The real client uses the cheapest model tier (`claude-haiku-5-5` by default, configurable).
The model must answer in JSON with labels from a fixed taxonomy; unknown labels are dropped,
relevance is clamped to 0-1, and an unparseable reply is stored as `failed` with the raw output.
Article text is wrapped in delimiters and the prompt says to treat it as data, since headlines are
untrusted input.

**Ranking.** `rank_score = relevance × 0.5^(age_hours / 24)`: a deliberately simple, explainable
proxy for news value. `GET /articles?sort=rank|relevance|recent`.

## Evaluation

See [`eval/README.md`](eval/README.md) for the labelling rules and metrics.

```bash
newswatch export-labels --out eval/to_label.csv   # then label by hand → eval/gold.csv
NEWSWATCH_LLM=anthropic newswatch eval --gold eval/gold.csv
```

### Results

> **TODO: fill in after labelling the real gold set and running once with the real model.**
> The table below is the format; the numbers must come from `eval/results/*.json`.

| method | model | n | precision | recall | F1 | P@10 | topic F1 | LLM calls | cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| keyword | keyword-v1 | – | – | – | – | – | – | 0 | 0 |
| llm | claude-haiku-5-5 | – | – | – | – | – | – | – | – |
| hybrid | claude-haiku-5-5 | – | – | – | – | – | – | – | – |

What I found: *TODO, 3-5 sentences. Where does the LLM win (articles that are relevant without
using the obvious words)? Where does the keyword baseline win? How much recall does the hybrid's
prefilter cost, and how many calls does it save?*

### Cost

LLM replies are cached by (model, prompt hash, input), so re-running the evaluation after changing
metrics is free, and changing the prompt automatically invalidates the cache. Prices are not
hard-coded; set them from the provider's pricing page in `.env`.

> **TODO:** one full evaluation run on N articles = X input / Y output tokens ≈ $Z.
> Ongoing classification with the hybrid on four feeds ≈ $Z per day.

## Responsible use of the model

* Every result stores **method, model, prompt version and the SHA-256 of the prompt template**,
  so an edited prompt can never pass as the old version.
* Every result stores the model's **rationale** and raw output.
* Every result has a **review status** (`unreviewed` → `confirmed` / `rejected`) with reviewer, note
  and time, via `POST /classifications/{id}/review`. The API and list page label scores as
  suggestions; nothing downstream treats model output as a final verdict.
* A changed model or prompt version re-classifies articles as a new row; old results are kept for comparison.

## Sources, citation and copyright

* Only the headline, a teaser (HTML stripped, truncated to 400 chars), the link and the publish time
  are stored. Full text (`content:encoded`) is ignored, and a test checks it never reaches the database.
* Every item links back to the publisher; the service is a pointer, not a copy.
* Feeds are public RSS that publishers provide for exactly this kind of reuse. Before running it
  beyond a demo, the publishers' terms should still be checked.

## What went wrong

Things that broke or nearly broke while building this:

1. **"politi" matched "politik".** The keyword baseline matches word prefixes to cope with Danish
   compounds, and the crime term `politi` tagged every politics story as crime. Fixed with exact-token
   terms (`politi$`) and a regression test. Prefix matching has the same problem elsewhere
   (`klima` also matches `klimaanlæg`), which is now one of the traps in the gold set.
2. **Possible double billing.** The "already classified?" check keys on the model name. Storing the
   model name the API echoes back could differ from the requested id (e.g. a dated alias), which
   would re-classify, and re-pay for, every article on every run. It now stores the requested id.
3. **SQLite and PostgreSQL disagree on timestamps.** SQLite returns naive datetimes, PostgreSQL
   returns aware ones, and comparing them raises. Everything goes through `as_utc()`, and CI runs
   the whole suite on both databases.
4. **The hybrid has a recall ceiling.** An article the keyword gate drops never reaches the LLM.
   In the synthetic demo set, "Køer skal betale: aftale om afgift på metan fra husdyr" is relevant
   but scores 0.18 on keywords and is skipped. The evaluation reports skips separately so this cost is visible.

5. **The SDK rejected `temperature`.** The first real run crashed with
   `Messages.create() got an unexpected keyword argument 'temperature'`: the current Anthropic SDK
   no longer takes it. The fake client could not catch this, because it never touches the SDK; a
   test now binds the real call against the SDK's signature. Reproducibility comes from the reply
   cache instead of temperature 0.
6. **Keywords missed most real climate/energy news.** On the real gold set the baseline had
   precision 1.0 but recall 0.43: stories about diesel prices for island ferries and protests
   against data centres are about energy, but use none of the listed words.

> **TODO:** add what happened against the live feeds (encoding, missing dates, feed changes, etc.).

## What I deliberately did not do

* **No full-text scraping.** Headlines and teasers are enough to rank and avoid copyright problems.
* **No embeddings.** The proposal allowed keywords or embeddings for the first pass; I chose keywords
  because they are transparent, free, and make a clean baseline. Embeddings are the obvious next
  experiment, and the classifier interface is where they would go.
* **No Alembic.** `create_all` is enough for one schema version. Real migrations come with the second.
* **No authentication** on the review endpoint. It records a reviewer name but trusts it; fine for a
  demo, not for production.
* **Ranking is sorted in Python** after a filtered query. Fine for thousands of rows; at scale the
  score belongs in SQL.
* **No async, no task queue.** One worker in a loop is enough for four feeds every 15 minutes.
* **No frontend framework.** `GET /` is a server-rendered list; the API is the product.

## Project layout

```
config/feeds.yaml          sources to monitor
src/newswatch/
  ingest.py                polite RSS fetching, dedup
  classify/                keyword, LLM, hybrid; LLM clients (fake, anthropic)
  prompts/relevance_v1.md  versioned prompt template
  service.py               classify new articles, review, rank score
  evaluate.py              metrics, LLM reply cache
  api.py                   FastAPI + list page
  cli.py                   newswatch ingest | classify | run | export-labels | eval
eval/                      gold set, labelling guide, results
tests/                     40 tests, run on SQLite and PostgreSQL in CI
```
