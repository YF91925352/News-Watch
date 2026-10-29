from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from newswatch.api import create_app
from newswatch.classify import FakeLLMClient, HybridClassifier, KeywordClassifier, LLMClassifier
from newswatch.ingest import ingest_all
from newswatch.models import Classification
from newswatch.service import classify_pending

from .test_ingest import FakeServer, make_settings


@pytest.fixture
def populated(session, tmp_path):
    ingest_all(session, make_settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(FakeServer())), pause_s=0)
    classify_pending(session, KeywordClassifier())
    return session


def test_classify_pending_is_idempotent(populated):
    stats = classify_pending(populated, KeywordClassifier())
    assert stats.classified == 0


def test_new_prompt_or_model_version_classifies_again(populated):
    client = FakeLLMClient()
    assert classify_pending(populated, LLMClassifier(client)).classified == 4
    assert classify_pending(populated, LLMClassifier(client)).classified == 0
    client.model = "fake-llm-v2"
    assert classify_pending(populated, LLMClassifier(client)).classified == 4


def test_hybrid_run_counts_skips_and_tokens(populated):
    client = FakeLLMClient()
    stats = classify_pending(populated, HybridClassifier(LLMClassifier(client)))
    assert stats.classified == 4
    assert stats.skipped_prefilter == 1  # the football story
    assert len(client.calls) == 3
    assert stats.input_tokens > 0


@pytest.fixture
def api(populated, session_factory):
    return TestClient(create_app(session_factory))


def test_articles_ranked_and_filtered(api):
    items = api.get("/articles", params={"min_relevance": 0.5}).json()
    titles = [a["title"] for a in items]
    assert "Landsholdet vinder efter sent mål" not in titles
    assert len(items) == 3
    assert all(a["classification"]["review_status"] == "unreviewed" for a in items)
    assert items[0]["rank_score"] >= items[-1]["rank_score"]

    by_topic = api.get("/articles", params={"topic": "kultur_sport"}).json()
    assert [a["title"] for a in by_topic] == ["Landsholdet vinder efter sent mål"]


def test_review_flow(api):
    art = api.get("/articles", params={"sort": "relevance"}).json()[0]
    cid = art["classification"]["id"]
    r = api.post(f"/classifications/{cid}/review", json={"status": "confirmed", "reviewer": "anna", "note": "ok"})
    assert r.status_code == 200 and r.json()["review_status"] == "confirmed"
    confirmed = api.get("/articles", params={"review_status": "confirmed"}).json()
    assert [a["id"] for a in confirmed] == [art["id"]]

    assert api.post(f"/classifications/{cid}/review", json={"status": "maybe", "reviewer": "x"}).status_code == 422
    assert api.post("/classifications/9999/review", json={"status": "rejected", "reviewer": "x"}).status_code == 404


def test_article_detail_shows_all_classifications(api, populated):
    classify_pending(populated, LLMClassifier(FakeLLMClient()))
    aid = api.get("/articles").json()[0]["id"]
    detail = api.get(f"/articles/{aid}").json()
    assert {c["method"] for c in detail["classifications"]} == {"keyword", "llm"}


def test_sources_and_index_page(api):
    sources = api.get("/sources").json()
    assert sum(s["article_count"] for s in sources) == 4
    page = api.get("/").text
    assert "Elprisen stiger" in page and "<script" not in page


def test_topics_stored_relationally(populated):
    row = populated.scalars(select(Classification)).first()
    assert sorted(t.topic for t in row.topic_rows) == sorted(row.topics)
    json.dumps(row.topics)
