"""Database schema.

Three ideas drive the design:
* sources and articles are separate, so politeness state (ETag, last fetch) lives per source;
* an article is unique on its normalised URL, because the same story appears in several feeds;
* a classification is never "the answer". It records which method, model and prompt version
  produced it, why, and whether a human has confirmed it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str] = mapped_column(String(1000), unique=True)
    active: Mapped[bool] = mapped_column(default=True)
    etag: Mapped[str | None] = mapped_column(String(500))
    last_modified: Mapped[str | None] = mapped_column(String(200))
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(100))

    articles: Mapped[list[Article]] = relationship(back_populates="source")


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    url_hash: Mapped[str] = mapped_column(String(64), unique=True)
    url: Mapped[str] = mapped_column(String(2000))
    guid: Mapped[str | None] = mapped_column(String(2000))
    title: Mapped[str] = mapped_column(String(1000))
    # Short teaser from the feed, HTML stripped and truncated. Never full text.
    summary: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source: Mapped[Source] = relationship(back_populates="articles")
    classifications: Mapped[list[Classification]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )


REVIEW_STATUSES = ("unreviewed", "confirmed", "rejected")


class Classification(Base):
    __tablename__ = "classifications"
    __table_args__ = (
        UniqueConstraint("article_id", "method", "model", "prompt_version", name="uq_classification_run"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)

    # Provenance: what produced this result.
    method: Mapped[str] = mapped_column(String(50))  # keyword | llm | hybrid
    model: Mapped[str] = mapped_column(String(100))  # e.g. keyword-v1, claude-haiku-5-5
    prompt_version: Mapped[str] = mapped_column(String(50))
    prompt_sha256: Mapped[str | None] = mapped_column(String(64))

    # Output.
    status: Mapped[str] = mapped_column(String(30))  # ok | failed | skipped_prefilter
    relevance: Mapped[float | None] = mapped_column(Float)
    topics: Mapped[list] = mapped_column(JSON, default=list)
    rationale: Mapped[str] = mapped_column(Text, default="")
    raw_output: Mapped[str | None] = mapped_column(Text)

    # Cost.
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Human in the loop: model output is a suggestion until someone confirms it.
    review_status: Mapped[str] = mapped_column(String(20), default="unreviewed", index=True)
    reviewer: Mapped[str | None] = mapped_column(String(200))
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    article: Mapped[Article] = relationship(back_populates="classifications")
    topic_rows: Mapped[list[ClassificationTopic]] = relationship(cascade="all, delete-orphan")


class ClassificationTopic(Base):
    """One row per (classification, topic) so topic filters are a plain indexed join in any database."""

    __tablename__ = "classification_topics"

    classification_id: Mapped[int] = mapped_column(ForeignKey("classifications.id"), primary_key=True)
    topic: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
