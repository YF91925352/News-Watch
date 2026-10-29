"""Polite RSS ingestion.

"Polite" means: an honest User-Agent, conditional GET (ETag / Last-Modified) so unchanged feeds
cost the publisher a 304, a minimum interval per source, a pause between requests, and backing
off on 429/5xx instead of retrying in a loop.
"""

from __future__ import annotations

import calendar
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import feedparser
import httpx
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import as_utc
from .models import Article, Source, utcnow
from .text import strip_html, truncate, url_hash

log = logging.getLogger(__name__)


@dataclass
class FetchReport:
    source: str
    status: str
    new_articles: int = 0
    duplicates: int = 0
    skipped_entries: int = 0
    errors: list[str] = field(default_factory=list)


def load_feed_config(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return list(data.get("feeds", []))


def sync_sources(session: Session, feeds: list[dict]) -> list[Source]:
    """Make the sources table match the config file (add new, re-activate, deactivate removed)."""
    by_url = {s.url: s for s in session.scalars(select(Source))}
    wanted = set()
    for feed in feeds:
        wanted.add(feed["url"])
        src = by_url.get(feed["url"])
        if src is None:
            src = Source(name=feed["name"], url=feed["url"], active=True)
            session.add(src)
        else:
            src.name = feed["name"]
            src.active = True
    for url, src in by_url.items():
        if url not in wanted:
            src.active = False
    session.commit()
    return list(session.scalars(select(Source).where(Source.active.is_(True)).order_by(Source.id)))


def _entry_published(entry) -> datetime | None:
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not struct:
        return None
    return datetime.fromtimestamp(calendar.timegm(struct), tz=UTC)


def fetch_source(
    session: Session,
    source: Source,
    client: httpx.Client,
    settings: Settings,
    *,
    force: bool = False,
) -> FetchReport:
    now = utcnow()
    last = as_utc(source.last_fetched_at)
    if not force and last and now - last < timedelta(seconds=settings.min_fetch_interval_s):
        return FetchReport(source=source.name, status="skipped_too_soon")

    headers = {"User-Agent": settings.user_agent, "Accept": "application/rss+xml, application/xml"}
    if source.etag:
        headers["If-None-Match"] = source.etag
    if source.last_modified:
        headers["If-Modified-Since"] = source.last_modified

    try:
        resp = client.get(source.url, headers=headers, timeout=settings.request_timeout_s)
    except httpx.HTTPError as exc:
        source.last_fetched_at = now
        source.last_status = f"error:{type(exc).__name__}"
        session.commit()
        return FetchReport(source=source.name, status="network_error", errors=[str(exc)])

    source.last_fetched_at = now
    source.last_status = str(resp.status_code)

    if resp.status_code == 304:
        session.commit()
        return FetchReport(source=source.name, status="not_modified")
    if resp.status_code == 429 or resp.status_code >= 500:
        # Back off: we simply wait for the next scheduled run (min interval still applies).
        session.commit()
        return FetchReport(source=source.name, status=f"backoff_{resp.status_code}")
    if resp.status_code != 200:
        session.commit()
        return FetchReport(source=source.name, status=f"http_{resp.status_code}")

    source.etag = resp.headers.get("ETag")
    source.last_modified = resp.headers.get("Last-Modified")

    parsed = feedparser.parse(resp.content)
    report = FetchReport(source=source.name, status="ok")
    if parsed.bozo and not parsed.entries:
        report.status = "parse_error"
        report.errors.append(str(parsed.get("bozo_exception")))
        session.commit()
        return report

    candidates: dict[str, dict] = {}
    for entry in parsed.entries:
        link = entry.get("link")
        title = strip_html(entry.get("title", ""))
        if not link or not title:
            report.skipped_entries += 1
            continue
        h = url_hash(link)
        if h in candidates:
            report.duplicates += 1
            continue
        candidates[h] = {
            "url": link,
            "guid": entry.get("id"),
            "title": truncate(title, 1000),
            "summary": truncate(strip_html(entry.get("summary", "")), settings.summary_max_chars),
            "published_at": _entry_published(entry),
        }

    existing = set()
    if candidates:
        existing = set(
            session.scalars(select(Article.url_hash).where(Article.url_hash.in_(list(candidates))))
        )
    for h, data in candidates.items():
        if h in existing:
            report.duplicates += 1
            continue
        session.add(Article(source_id=source.id, url_hash=h, fetched_at=now, **data))
        report.new_articles += 1

    session.commit()
    return report


def ingest_all(
    session: Session,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
    pause_s: float = 2.0,
) -> list[FetchReport]:
    sources = sync_sources(session, load_feed_config(settings.feeds_file))
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    reports = []
    try:
        for i, src in enumerate(sources):
            if i and pause_s:
                time.sleep(pause_s)
            report = fetch_source(session, src, client, settings, force=force)
            log.info("%s: %s (+%d new, %d dup)", report.source, report.status, report.new_articles, report.duplicates)
            reports.append(report)
    finally:
        if own_client:
            client.close()
    return reports
