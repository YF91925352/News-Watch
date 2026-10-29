"""Evaluation harness: score classifiers against a hand-labelled gold set.

The gold set is a CSV with columns: id, url, title, summary, relevant (0/1), topics (";"-separated).
It carries its own title/summary, so evaluation does not depend on what is in the database and
results are reproducible.

LLM replies are cached on disk keyed by (model, prompt hash, input), so re-running the evaluation
after changing *metrics* costs nothing; changing the *prompt* invalidates the cache automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .classify.base import RELEVANCE_THRESHOLD, TOPICS, ArticleInput, Classifier
from .classify.clients import LLMClient, LLMResponse
from .service import estimate_cost


@dataclass
class GoldItem:
    id: str
    title: str
    summary: str
    relevant: bool
    topics: set[str]
    url: str = ""


def load_gold(path: str | Path) -> list[GoldItem]:
    items = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("relevant") or "").strip() == "":
                continue  # not labelled yet
            topics = {t.strip() for t in (row.get("topics") or "").split(";") if t.strip()}
            unknown = topics - set(TOPICS)
            if unknown:
                raise ValueError(f"row {row.get('id')}: unknown topics {sorted(unknown)}")
            items.append(GoldItem(
                id=row["id"], url=row.get("url", ""), title=row["title"], summary=row.get("summary", ""),
                relevant=row["relevant"].strip() in {"1", "true", "yes", "ja"}, topics=topics,
            ))
    return items


class CachingClient:
    def __init__(self, inner: LLMClient, cache_path: str | Path):
        self.inner = inner
        self.model = inner.model
        self.config = getattr(inner, "config", "")
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec
        self.hits = 0
        self.misses = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        config = getattr(self.inner, "config", "")
        key = hashlib.sha256(f"{self.model}\0{config}\0{system}\0{user}".encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            rec = self.cache[key]
            # Cached replies cost nothing this run.
            return LLMResponse(text=rec["text"], model=rec["model"], input_tokens=0, output_tokens=0)
        self.misses += 1
        resp = self.inner.complete(system, user)
        rec = {"key": key, "text": resp.text, "model": resp.model,
               "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}
        self.cache[key] = rec
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return resp


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


@dataclass
class EvalResult:
    method: str
    model: str
    n: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    precision_at_k: float
    k: int
    topic_precision: float
    topic_recall: float
    topic_f1: float
    failed: int
    skipped_prefilter: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    seconds: float
    errors: list[dict] = field(default_factory=list)


def evaluate(classifier: Classifier, gold: list[GoldItem], *, k: int = 10,
             price_in: float | None = None, price_out: float | None = None) -> EvalResult:
    t0 = time.perf_counter()
    preds = [(g, classifier.classify(ArticleInput(title=g.title, summary=g.summary, url=g.url))) for g in gold]
    seconds = round(time.perf_counter() - t0, 2)

    tp = fp = fn = tn = 0
    ttp = tfp = tfn = 0
    errors = []
    for g, c in preds:
        pred = c.relevance is not None and c.relevance >= RELEVANCE_THRESHOLD
        if pred and g.relevant:
            tp += 1
        elif pred and not g.relevant:
            fp += 1
        elif not pred and g.relevant:
            fn += 1
        else:
            tn += 1
        if pred != g.relevant or c.status == "failed":
            errors.append({"id": g.id, "title": g.title, "gold": g.relevant, "relevance": c.relevance,
                           "status": c.status, "rationale": c.rationale})
        pt = set(c.topics)
        ttp += len(pt & g.topics)
        tfp += len(pt - g.topics)
        tfn += len(g.topics - pt)

    ranked = sorted(preds, key=lambda gc: gc[1].relevance or 0.0, reverse=True)[:k]
    p_at_k = round(sum(g.relevant for g, _ in ranked) / max(1, len(ranked)), 3)

    p, r, f = _prf(tp, fp, fn)
    tp_, tr_, tf_ = _prf(ttp, tfp, tfn)
    in_tok = sum(c.input_tokens for _, c in preds)
    out_tok = sum(c.output_tokens for _, c in preds)
    llm_calls = sum(1 for _, c in preds if c.method != "keyword" and c.status != "skipped_prefilter")
    return EvalResult(
        method=classifier.method,
        model=preds[0][1].model if preds else "",
        n=len(gold), precision=p, recall=r, f1=f,
        accuracy=round((tp + tn) / max(1, len(gold)), 3),
        precision_at_k=p_at_k, k=k,
        topic_precision=tp_, topic_recall=tr_, topic_f1=tf_,
        failed=sum(c.status == "failed" for _, c in preds),
        skipped_prefilter=sum(c.status == "skipped_prefilter" for _, c in preds),
        llm_calls=llm_calls, input_tokens=in_tok, output_tokens=out_tok,
        cost_usd=estimate_cost(in_tok, out_tok, price_in, price_out),
        seconds=seconds, errors=errors,
    )


def to_markdown(results: list[EvalResult]) -> str:
    head = ("| method | model | n | precision | recall | F1 | P@k | topic F1 | failed | LLM calls "
            "| tokens in/out | cost (USD) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for r in results:
        cost = "n/a" if r.cost_usd is None else f"{r.cost_usd:.4f}"
        rows.append(f"| {r.method} | {r.model} | {r.n} | {r.precision} | {r.recall} | {r.f1} "
                    f"| {r.precision_at_k} (k={r.k}) | {r.topic_f1} | {r.failed} | {r.llm_calls} "
                    f"| {r.input_tokens}/{r.output_tokens} | {cost} |")
    return "\n".join([head, *rows])


def write_results(results: list[EvalResult], out_dir: str | Path, gold_path: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"eval-{stamp}.json"
    path.write_text(json.dumps({"gold": str(gold_path), "threshold": RELEVANCE_THRESHOLD,
                                "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
