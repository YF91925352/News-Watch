from __future__ import annotations

from dataclasses import replace

import httpx
from sqlalchemy import select

from newswatch.config import Settings
from newswatch.ingest import fetch_source, ingest_all, sync_sources
from newswatch.models import Article, Source
from newswatch.text import normalise_url, strip_html, truncate

from .conftest import FIXTURES

FEEDS = {
    "https://feeds.test/indland": (FIXTURES / "feed_indland.xml").read_bytes(),
    "https://feeds.test/politik": (FIXTURES / "feed_politik.xml").read_bytes(),
}


def make_settings(tmp_path, **kw) -> Settings:
    cfg = tmp_path / "feeds.yaml"
    cfg.write_text(
        "feeds:\n"
        "  - {name: Indland, url: 'https://feeds.test/indland'}\n"
        "  - {name: Politik, url: 'https://feeds.test/politik'}\n",
        encoding="utf-8",
    )
    return replace(Settings(), feeds_file=str(cfg), min_fetch_interval_s=600, **kw)


class FakeServer:
    """Serves fixture feeds, records request headers, honours If-None-Match."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.status_override: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status_override:
            return httpx.Response(self.status_override)
        etag = f'"v1-{request.url.path}"'
        if request.headers.get("If-None-Match") == etag:
            return httpx.Response(304)
        return httpx.Response(200, content=FEEDS[str(request.url)], headers={"ETag": etag})


def run(session, settings, server, force=False):
    client = httpx.Client(transport=httpx.MockTransport(server))
    return ingest_all(session, settings, client=client, force=force, pause_s=0)


def test_ingest_dedups_across_feeds_and_skips_bad_entries(session, tmp_path):
    reports = run(session, make_settings(tmp_path), FakeServer())
    assert [r.status for r in reports] == ["ok", "ok"]
    # indland: 3 valid + 1 untitled skipped; politik: 1 duplicate of indland (trailing slash) + 1 new
    assert reports[0].new_articles == 3 and reports[0].skipped_entries == 1
    assert reports[1].new_articles == 1 and reports[1].duplicates == 1
    assert session.scalar(select(Article).where(Article.title.like("Kommune vil%"))).source.name == "Indland"
    assert len(session.scalars(select(Article)).all()) == 4


def test_only_title_link_summary_time_are_stored(session, tmp_path):
    run(session, make_settings(tmp_path), FakeServer())
    for art in session.scalars(select(Article)):
        assert "FULL ARTICLE TEXT" not in art.summary
        assert "<" not in art.summary  # HTML stripped
    dige = session.scalar(select(Article).where(Article.title.like("Kommune vil%")))
    assert dige.summary.startswith("Byrådet afsætter penge til kystsikring")
    assert dige.published_at is not None


def test_polite_fetching_user_agent_conditional_get_and_min_interval(session, tmp_path):
    settings = make_settings(tmp_path, user_agent="newswatch-test (me@example.com)")
    server = FakeServer()
    run(session, settings, server)
    assert all(r.headers["User-Agent"] == "newswatch-test (me@example.com)" for r in server.requests)

    # Second run right away: skipped by the minimum interval, no requests at all.
    reports = run(session, settings, server)
    assert {r.status for r in reports} == {"skipped_too_soon"}
    assert len(server.requests) == 2

    # Forced run: sends If-None-Match and gets 304, nothing re-inserted.
    reports = run(session, settings, server, force=True)
    assert {r.status for r in reports} == {"not_modified"}
    assert server.requests[-1].headers["If-None-Match"].startswith('"v1-')
    assert len(session.scalars(select(Article)).all()) == 4


def test_backs_off_on_429(session, tmp_path):
    settings = make_settings(tmp_path)
    server = FakeServer()
    server.status_override = 429
    reports = run(session, settings, server)
    assert {r.status for r in reports} == {"backoff_429"}
    assert session.scalars(select(Article)).all() == []


def test_sync_sources_deactivates_removed_feeds(session, tmp_path):
    sync_sources(session, [{"name": "A", "url": "https://a"}, {"name": "B", "url": "https://b"}])
    active = sync_sources(session, [{"name": "A", "url": "https://a"}])
    assert [s.url for s in active] == ["https://a"]
    assert session.scalar(select(Source).where(Source.url == "https://b")).active is False


def test_network_error_is_recorded_not_raised(session, tmp_path):
    settings = make_settings(tmp_path)
    src = sync_sources(session, [{"name": "Down", "url": "https://down.test/feed"}])[0]

    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    report = fetch_source(session, src, httpx.Client(transport=httpx.MockTransport(boom)), settings)
    assert report.status == "network_error"
    assert src.last_status == "error:ConnectError"


def test_text_helpers():
    assert normalise_url("HTTPS://Example.dk/a/?utm_source=x&id=2#top") == "https://example.dk/a?id=2"
    assert strip_html("<p>Hej &amp; <b>farvel</b></p>") == "Hej & farvel"
    assert truncate("en to tre fire", 9) == "en to…"
