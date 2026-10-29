"""FastAPI query interface plus a minimal HTML list page."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .db import as_utc, init_db, make_engine, make_sessionmaker
from .models import Article, ClassificationTopic, Source, utcnow
from .models import Classification as ClassificationRow
from .service import rank_score, review


class ClassificationOut(BaseModel):
    id: int
    method: str
    model: str
    prompt_version: str
    status: str
    relevance: float | None
    topics: list[str]
    rationale: str
    review_status: str
    reviewer: str | None
    reviewed_at: datetime | None
    created_at: datetime


class ArticleOut(BaseModel):
    id: int
    source: str
    title: str
    url: str
    summary: str
    published_at: datetime | None
    rank_score: float | None = None
    classification: ClassificationOut | None = None


class ArticleDetail(ArticleOut):
    classifications: list[ClassificationOut]


class SourceOut(BaseModel):
    id: int
    name: str
    url: str
    active: bool
    last_fetched_at: datetime | None
    last_status: str | None
    article_count: int


class ReviewIn(BaseModel):
    status: Literal["confirmed", "rejected"]
    reviewer: str
    note: str | None = None


def _cls_out(c: ClassificationRow) -> ClassificationOut:
    return ClassificationOut(
        id=c.id, method=c.method, model=c.model, prompt_version=c.prompt_version, status=c.status,
        relevance=c.relevance, topics=list(c.topics or []), rationale=c.rationale,
        review_status=c.review_status, reviewer=c.reviewer, reviewed_at=as_utc(c.reviewed_at),
        created_at=as_utc(c.created_at),
    )


def create_app(session_factory: sessionmaker[Session] | None = None) -> FastAPI:
    if session_factory is None:
        engine = make_engine(get_settings().database_url)
        init_db(engine)
        session_factory = make_sessionmaker(engine)

    app = FastAPI(
        title="newswatch",
        description="Danish news feeds ranked by relevance to a monitoring brief. "
        "Model output is a suggestion until a human confirms it.",
        version="0.1.0",
    )

    def get_session():
        with session_factory() as s:
            yield s

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/sources", response_model=list[SourceOut])
    def sources(session: Session = Depends(get_session)):
        counts = dict(session.execute(select(Article.source_id, func.count()).group_by(Article.source_id)).all())
        return [
            SourceOut(id=s.id, name=s.name, url=s.url, active=s.active, last_fetched_at=as_utc(s.last_fetched_at),
                      last_status=s.last_status, article_count=counts.get(s.id, 0))
            for s in session.scalars(select(Source).order_by(Source.id))
        ]

    def _query_articles(session: Session, method: str, topic: str | None, min_relevance: float,
                        review_status: str | None, q: str | None, sort: str, limit: int, offset: int):
        # Latest classification per article for the chosen method.
        latest = (
            select(func.max(ClassificationRow.id).label("cid"))
            .where(ClassificationRow.method == method)
            .group_by(ClassificationRow.article_id)
            .subquery()
        )
        stmt = (
            select(Article, ClassificationRow)
            .join(ClassificationRow, ClassificationRow.article_id == Article.id)
            .join(latest, latest.c.cid == ClassificationRow.id)
            .where(func.coalesce(ClassificationRow.relevance, 0.0) >= min_relevance)
        )
        if topic:
            stmt = stmt.join(ClassificationTopic, ClassificationTopic.classification_id == ClassificationRow.id).where(
                ClassificationTopic.topic == topic
            )
        if review_status:
            stmt = stmt.where(ClassificationRow.review_status == review_status)
        if q:
            stmt = stmt.where(Article.title.ilike(f"%{q}%"))

        now = utcnow()
        rows = session.execute(stmt).all()
        if sort == "recent":
            rows.sort(key=lambda r: as_utc(r[0].published_at) or datetime.min.replace(tzinfo=now.tzinfo), reverse=True)
        elif sort == "relevance":
            rows.sort(key=lambda r: r[1].relevance or 0.0, reverse=True)
        else:
            rows.sort(key=lambda r: rank_score(r[1].relevance, r[0].published_at, now), reverse=True)
        out = []
        for art, cls in rows[offset: offset + limit]:
            out.append(ArticleOut(
                id=art.id, source=art.source.name, title=art.title, url=art.url, summary=art.summary,
                published_at=as_utc(art.published_at), rank_score=round(rank_score(cls.relevance, art.published_at, now), 4),
                classification=_cls_out(cls),
            ))
        return out

    @app.get("/articles", response_model=list[ArticleOut])
    def articles(
        method: Literal["keyword", "llm", "hybrid"] = "keyword",
        topic: str | None = None,
        min_relevance: float = Query(0.0, ge=0.0, le=1.0),
        review_status: Literal["unreviewed", "confirmed", "rejected"] | None = None,
        q: str | None = Query(None, max_length=200),
        sort: Literal["rank", "relevance", "recent"] = "rank",
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ):
        return _query_articles(session, method, topic, min_relevance, review_status, q, sort, limit, offset)

    @app.get("/articles/{article_id}", response_model=ArticleDetail)
    def article(article_id: int, session: Session = Depends(get_session)):
        art = session.get(Article, article_id)
        if art is None:
            raise HTTPException(404, "article not found")
        classes = sorted(art.classifications, key=lambda c: c.id, reverse=True)
        return ArticleDetail(
            id=art.id, source=art.source.name, title=art.title, url=art.url, summary=art.summary,
            published_at=as_utc(art.published_at), classifications=[_cls_out(c) for c in classes],
        )

    @app.post("/classifications/{classification_id}/review", response_model=ClassificationOut)
    def post_review(classification_id: int, body: ReviewIn, session: Session = Depends(get_session)):
        try:
            row = review(session, classification_id, body.status, body.reviewer, body.note)
        except LookupError:
            raise HTTPException(404, "classification not found") from None
        return _cls_out(row)

    @app.get("/", response_class=HTMLResponse)
    def index(method: Literal["keyword", "llm", "hybrid"] = "keyword", min_relevance: float = 0.5,
              session: Session = Depends(get_session)):
        items = _query_articles(session, method, None, min_relevance, None, None, "rank", 100, 0)
        rows = "".join(
            f"<li><a href='{html.escape(a.url)}' rel='noopener' target='_blank'>{html.escape(a.title)}</a>"
            f"<div class='meta'>{html.escape(a.source)} · relevance {a.classification.relevance} · "
            f"{html.escape(', '.join(a.classification.topics))} · <em>{a.classification.review_status}</em></div>"
            f"<div class='why'>{html.escape(a.classification.rationale)}</div></li>"
            for a in items
        ) or "<li>No articles yet. Run <code>newswatch ingest</code> and <code>newswatch classify</code>.</li>"
        return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>newswatch</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}}
li{{margin:0 0 1rem}}.meta{{color:#555;font-size:.85rem}}.why{{color:#333;font-size:.85rem}}</style>
<h1>newswatch</h1><p>Method: <b>{html.escape(method)}</b>, relevance ≥ {min_relevance}.
Scores are model suggestions until reviewed. Links go to the publisher; only headlines and teasers are stored.</p>
<ol>{rows}</ol></html>"""

    return app


def app_factory() -> FastAPI:  # used by uvicorn --factory
    return create_app()
