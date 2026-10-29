from __future__ import annotations

import json
from pathlib import Path

import pytest

from newswatch.classify import FakeLLMClient, HybridClassifier, KeywordClassifier, LLMClassifier
from newswatch.cli import main
from newswatch.evaluate import CachingClient, GoldItem, evaluate, load_gold

ROOT = Path(__file__).resolve().parents[1]


def gold(*rows):
    return [GoldItem(id=str(i), title=t, summary="", relevant=r, topics=set(tp)) for i, (t, r, tp) in enumerate(rows)]


def test_metrics_are_computed_correctly():
    g = gold(
        ("Ny klimaplan", True, ["klima_energi"]),          # TP
        ("Havniveauet stiger langs kysten", True, ["klima_energi"]),  # FN (no keyword)
        ("Klimaanlæg i bussen virker ikke", False, ["andet"]),  # FP (keyword trap)
        ("Landsholdet vinder", False, ["kultur_sport"]),   # TN
    )
    r = evaluate(KeywordClassifier(), g, k=2)
    assert (r.precision, r.recall, r.accuracy) == (0.5, 0.5, 0.5)
    assert r.precision_at_k == 0.5
    assert len(r.errors) == 2
    assert r.llm_calls == 0 and r.cost_usd is None


def test_failed_llm_output_counts_as_not_relevant_and_is_reported():
    g = gold(("Ny klimaplan", True, ["klima_energi"]))
    r = evaluate(LLMClassifier(FakeLLMClient(lambda s, u: "nope")), g)
    assert r.failed == 1 and r.recall == 0.0 and r.errors[0]["status"] == "failed"


def test_cost_is_estimated_when_prices_are_set():
    g = gold(("Ny klimaplan", True, ["klima_energi"]))
    r = evaluate(LLMClassifier(FakeLLMClient()), g, price_in=1.0, price_out=5.0)
    assert r.cost_usd is not None and r.cost_usd > 0


def test_cache_makes_second_run_free(tmp_path):
    inner = FakeLLMClient()
    g = gold(("Ny klimaplan", True, ["klima_energi"]), ("Elprisen stiger", True, ["klima_energi"]))
    c1 = CachingClient(inner, tmp_path / "c.jsonl")
    evaluate(LLMClassifier(c1), g)
    c2 = CachingClient(inner, tmp_path / "c.jsonl")
    r = evaluate(LLMClassifier(c2), g)
    assert (c2.hits, c2.misses) == (2, 0) and r.input_tokens == 0
    assert len(inner.calls) == 2


def test_hybrid_reports_skips():
    g = gold(("Landsholdet vinder", False, ["kultur_sport"]), ("Ny klimaplan", True, ["klima_energi"]))
    r = evaluate(HybridClassifier(LLMClassifier(FakeLLMClient())), g)
    assert r.skipped_prefilter == 1 and r.llm_calls == 1


def test_load_gold_validates_topics(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("id,title,summary,relevant,topics\n1,a,,1,klima_energi\n2,b,,,\n3,c,,0,weather\n", encoding="utf-8")
    with pytest.raises(ValueError, match="weather"):
        load_gold(p)
    p.write_text("id,title,summary,relevant,topics\n1,a,,1,klima_energi\n2,b,,,\n", encoding="utf-8")
    assert len(load_gold(p)) == 1  # unlabelled rows are ignored


def test_sample_gold_set_loads():
    items = load_gold(ROOT / "eval" / "sample_gold.csv")
    assert len(items) == 20 and sum(i.relevant for i in items) == 10


def test_cli_eval_end_to_end_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEWSWATCH_LLM", "fake")
    main(["eval", "--gold", str(ROOT / "eval" / "sample_gold.csv"), "--cache", str(tmp_path / "c.jsonl"),
          "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "| keyword |" in out and "| llm |" in out and "| hybrid |" in out
    saved = json.loads(next((tmp_path / "out").glob("eval-*.json")).read_text())
    assert {r["method"] for r in saved["results"]} == {"keyword", "llm", "hybrid"}


def test_cache_is_keyed_on_request_config(tmp_path):
    inner = FakeLLMClient()
    g = gold(("Ny klimaplan", True, ["klima_energi"]))
    evaluate(LLMClassifier(CachingClient(inner, tmp_path / "c.jsonl")), g)
    inner.config = "thinking=off"
    c2 = CachingClient(inner, tmp_path / "c.jsonl")
    evaluate(LLMClassifier(c2), g)
    assert (c2.hits, c2.misses) == (0, 1)
