"""Command line: init-db, ingest, classify, run (scheduled loop), export-labels, eval."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time

from sqlalchemy import select

from .classify import HybridClassifier, KeywordClassifier, LLMClassifier, make_client
from .config import Settings, get_settings
from .db import init_db, make_engine, make_sessionmaker
from .ingest import ingest_all
from .models import Article
from .service import classify_pending, estimate_cost

log = logging.getLogger("newswatch")


def _session(settings: Settings):
    engine = make_engine(settings.database_url)
    init_db(engine)
    return make_sessionmaker(engine)()


def _classifier(method: str, settings: Settings, client=None):
    if method == "keyword":
        return KeywordClassifier()
    client = client or make_client(settings.llm_backend, settings.llm_model)
    if client is None:
        raise SystemExit(f"method {method!r} needs an LLM: set NEWSWATCH_LLM=fake or anthropic")
    llm = LLMClassifier(client)
    return llm if method == "llm" else HybridClassifier(llm)


def cmd_ingest(args, settings):
    with _session(settings) as s:
        for r in ingest_all(s, settings, force=args.force):
            print(f"{r.source:<20} {r.status:<18} new={r.new_articles} dup={r.duplicates}"
                  + (f" errors={r.errors}" if r.errors else ""))


def cmd_classify(args, settings):
    clf = _classifier(args.method, settings)
    with _session(settings) as s:
        st = classify_pending(s, clf, limit=args.limit)
    cost = estimate_cost(st.input_tokens, st.output_tokens, settings.price_in_per_mtok, settings.price_out_per_mtok)
    print(f"classified={st.classified} failed={st.failed} skipped_prefilter={st.skipped_prefilter} "
          f"tokens={st.input_tokens}/{st.output_tokens} cost_usd={'n/a' if cost is None else f'{cost:.4f}'}")


def cmd_run(args, settings):
    """Simple scheduler: ingest, then classify, then sleep. Good enough for one small worker."""
    methods = ["keyword"] + (["hybrid"] if settings.llm_backend != "off" else [])
    while True:
        try:
            cmd_ingest(argparse.Namespace(force=False), settings)
            for m in methods:
                cmd_classify(argparse.Namespace(method=m, limit=args.limit), settings)
        except Exception:  # keep the worker alive; the next cycle retries
            log.exception("cycle failed")
        if args.once:
            return
        time.sleep(args.interval)


def cmd_export_labels(args, settings):
    """Export articles to a CSV you label by hand: fill `relevant` (1/0) and `topics` (a;b)."""
    with _session(settings) as s:
        arts = s.scalars(select(Article).order_by(Article.published_at.desc().nulls_last()).limit(args.limit)).all()
        with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "url", "title", "summary", "relevant", "topics", "notes"])
            for a in arts:
                w.writerow([a.id, a.url, a.title, a.summary, "", "", ""])
    print(f"wrote {len(arts)} rows to {args.out}")


def cmd_eval(args, settings):
    from .evaluate import CachingClient, evaluate, load_gold, to_markdown, write_results

    gold = load_gold(args.gold)
    if not gold:
        raise SystemExit(f"no labelled rows in {args.gold}")
    client = None
    if any(m != "keyword" for m in args.methods):
        inner = make_client(settings.llm_backend, settings.llm_model)
        if inner is None:
            raise SystemExit("LLM methods need NEWSWATCH_LLM=fake or anthropic")
        client = CachingClient(inner, args.cache)
    results = [evaluate(_classifier(m, settings, client), gold, k=args.k,
                        price_in=settings.price_in_per_mtok, price_out=settings.price_out_per_mtok)
               for m in args.methods]
    print(f"gold set: {args.gold} ({len(gold)} labelled, {sum(g.relevant for g in gold)} relevant)\n")
    print(to_markdown(results))
    if client is not None:
        print(f"\nLLM cache: {client.hits} hits, {client.misses} paid calls")
    for r in results:
        if r.errors:
            print(f"\n{r.method}: {len(r.errors)} misclassified")
            for e in r.errors[: args.show_errors]:
                print(f"  [{'REL' if e['gold'] else 'irr'}] {e['relevance']} {e['title'][:70]} -- {e['rationale'][:90]}")
    path = write_results(results, args.out_dir, args.gold)
    print(f"\nfull results: {path}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="newswatch")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")
    sp = sub.add_parser("ingest")
    sp.add_argument("--force", action="store_true", help="ignore the per-source minimum interval")
    sp = sub.add_parser("classify")
    sp.add_argument("--method", choices=["keyword", "llm", "hybrid"], default="keyword")
    sp.add_argument("--limit", type=int, default=200)
    sp = sub.add_parser("run")
    sp.add_argument("--interval", type=int, default=900)
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--once", action="store_true")
    sp = sub.add_parser("export-labels")
    sp.add_argument("--out", default="eval/to_label.csv")
    sp.add_argument("--limit", type=int, default=60)
    sp = sub.add_parser("eval")
    sp.add_argument("--gold", default="eval/gold.csv")
    sp.add_argument("--methods", nargs="+", default=["keyword", "llm", "hybrid"],
                    choices=["keyword", "llm", "hybrid"])
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--cache", default="eval/.cache/llm_cache.jsonl")
    sp.add_argument("--out-dir", default="eval/results")
    sp.add_argument("--show-errors", type=int, default=5)

    args = p.parse_args(argv)
    settings = get_settings()
    if args.cmd == "init-db":
        _session(settings).close()
        print("database ready")
        return
    {"ingest": cmd_ingest, "classify": cmd_classify, "run": cmd_run,
     "export-labels": cmd_export_labels, "eval": cmd_eval}[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
