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
docker compose up --build       # api on :8000 (NEWSWATCH_PORT), worker polls feeds every 15 min
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
pytest                                                   # 44 tests, SQLite, no network
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

61 real DR headlines + teasers, labelled by hand (14 relevant). Threshold 0.5, fixed before the run.
Run of 29 Oct 2026, prompt `relevance-v1`, thinking off, structured output
(`eval/results/eval-20261029T213019Z.json`).

| method | model | precision | recall | F1 | P@10 | topic F1 | failed | LLM calls | cost (USD) |
|---|---|---|---|---|---|---|---|---|---|
| keyword | keyword-v1 | 1.00 | 0.43 | 0.60 | 0.6 | 0.46 | 0 | 0 | 0 |
| llm | claude-haiku-5-5 | 0.85 | **0.79** | **0.82** | **0.9** | **0.71** | 0 | 61 | 0.0092 |
| hybrid | claude-haiku-5-5 | 1.00 | 0.36 | 0.53 | 0.6 | 0.46 | 0 | 6 | ≈0.001 |

What I found:

* **The LLM finds what the keywords cannot.** Recall goes from 0.43 to 0.79. The keyword baseline
  missed every story about diesel prices for island ferries and protests against data centres:
  they are about energy but use none of the listed words. The LLM caught most of them.
* **It pays for that with two false positives, and one of them is a misreading.** "Minister vil
  forbyde turbokyllinger" (a ban on fast-growing broiler chickens) got 0.6 with the rationale
  *"a minister wanting to ban turbo heaters relates to energy"*. The stored rationale is what made
  this visible in seconds, and it is the reason results start as `unreviewed`.
* **The hybrid is worse than either method alone.** The keyword gate drops exactly the articles the
  LLM is good at, so the LLM never sees them; and on "Trump har sået klimatvivl" the keywords said
  relevant but the LLM, now the final judge, said 0.4. It saves 90% of the calls, but at a cost of
  a cent per 61 articles there is nothing worth saving. Lesson: a cheap prefilter only helps if its
  recall is high.
* **The remaining misses are a scope question, not only a model error.** I labelled a UN
  biodiversity summit and data-centre stories as relevant; the brief says climate and energy, and
  the model scored them 0.35-0.4. I kept the labels and the brief as they were rather than change
  either after seeing the results.

Limits of this evaluation: one labeller (me), 61 articles, 14 relevant, so one article moves recall
by 7 points. Nine of the relevant articles were added after a first pass came out with only five,
picked by me from a larger export, which may bias the set. The first LLM run (thinking on) gave
recall 0.43 with 5 parse failures; see "What went wrong".

### Cost

Claude Haiku 5.5 at $0.10 / $0.50 per million input / output tokens (prompts under 100k tokens).
The evaluation run above: 66,329 input + 5,155 output tokens = **$0.0092 for 61 articles**, about
1,090 input and 85 output tokens per article. The system prompt is most of the input.

Estimate, not measured: at roughly 100 new articles a day across four feeds, classifying all of
them with the LLM would cost about $0.015 a day.

LLM replies are cached by (model, request settings, prompt, input), so re-running the evaluation
after changing metrics is free, and changing the prompt or settings invalidates the cache
automatically. Prices are not hard-coded; they are set in `.env` from the provider's pricing page.

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
6. **Thinking ate the output budget.** The second run had 5 of 61 replies with "no JSON object".
   Claude Haiku 5.5 thinks by default, and thinking counts towards `max_tokens` (300), so on some
   articles it ran out before writing any JSON. Fixed by turning thinking off (a short
   classification does not need it), requiring a JSON schema through structured outputs, raising
   the budget, and recording `stop_reason` on failure. Recall also went from 0.43 to 0.79 between
   the two runs, which is a reminder that request settings are part of the model and must be
   versioned; they are now part of the cache key.
7. **Keywords missed most real climate/energy news.** On the real gold set the baseline had
   precision 1.0 but recall 0.43. I did not add `diesel` or `datacenter` to the list afterwards:
   tuning the baseline on the test set would make the comparison meaningless.
8. **Infrastructure.** An older Docker Compose did not understand `env_file: {path, required}`;
   the API container started before PostgreSQL finished initialising (the health check passed on
   the Unix socket during init), so the check now uses TCP and the API restarts on failure; and
   port 8000 was taken on my machine, so the host port is now `NEWSWATCH_PORT` in `.env`.
   WPS and Excel write CSV with a byte-order mark, so the gold set is read with `utf-8-sig`.

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
* **No prompt caching or Batch API.** The system prompt is about 1,000 tokens and identical on
  every call, so prompt caching would cut input cost, and the Batch API halves it. At one cent per
  evaluation run it was not worth the complexity yet.
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
tests/                     44 tests, run on SQLite and PostgreSQL in CI
```

---

Built with AI-assisted development (Claude). I defined the problem and the brief, labelled the gold
set, ran it against the live feeds and the real model, and decided what to change and what not to.
