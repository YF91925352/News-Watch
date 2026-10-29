"""Glue between classifiers and the database: classify what is new, record reviews."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from .classify.base import ArticleInput, Classification, Classifier
from .db import as_utc
from .models import REVIEW_STATUSES, Article, ClassificationTopic, utcnow
from .models import Classification as ClassificationRow


@dataclass
class RunStats:
    classified: int = 0
    failed: int = 0
    skipped_prefilter: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def estimate_cost(input_tokens: int, output_tokens: int, price_in: float | None, price_out: float | None) -> float | None:
    if price_in is None or price_out is None:
        return None
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def _model_key(classifier: Classifier) -> tuple[str, str]:
    """(model, prompt_version) the classifier will write, used to skip already-classified articles."""
    if hasattr(classifier, "llm"):
        return classifier.llm.client.model, classifier.llm.prompt_version
    if hasattr(classifier, "client"):
        return classifier.client.model, classifier.prompt_version
    from .classify.keyword import MODEL_NAME

    return MODEL_NAME, "n/a"


def save_classification(session: Session, article_id: int, result: Classification) -> ClassificationRow:
    row = ClassificationRow(
        article_id=article_id,
        method=result.method,
        model=result.model,
        prompt_version=result.prompt_version,
        prompt_sha256=result.prompt_sha256,
        status=result.status,
        relevance=result.relevance,
        topics=list(result.topics),
        rationale=result.rationale,
        raw_output=result.raw_output,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    row.topic_rows = [ClassificationTopic(topic=t) for t in dict.fromkeys(result.topics)]
    session.add(row)
    return row


def classify_pending(session: Session, classifier: Classifier, *, limit: int = 200) -> RunStats:
    model, prompt_version = _model_key(classifier)
    already = exists().where(
        and_(
            ClassificationRow.article_id == Article.id,
            ClassificationRow.method == classifier.method,
            ClassificationRow.model == model,
            ClassificationRow.prompt_version == prompt_version,
        )
    )
    articles = session.scalars(
        select(Article).where(~already).order_by(Article.published_at.desc().nulls_last()).limit(limit)
    ).all()

    stats = RunStats()
    for art in articles:
        result = classifier.classify(ArticleInput(title=art.title, summary=art.summary, url=art.url))
        save_classification(session, art.id, result)
        stats.classified += 1
        stats.failed += result.status == "failed"
        stats.skipped_prefilter += result.status == "skipped_prefilter"
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        session.commit()  # commit per article: a crash halfway does not lose paid-for results
    return stats


def review(session: Session, classification_id: int, status: str, reviewer: str, note: str | None = None) -> ClassificationRow:
    if status not in REVIEW_STATUSES or status == "unreviewed":
        raise ValueError(f"status must be 'confirmed' or 'rejected', got {status!r}")
    row = session.get(ClassificationRow, classification_id)
    if row is None:
        raise LookupError(classification_id)
    row.review_status = status
    row.reviewer = reviewer
    row.review_note = note
    row.reviewed_at = utcnow()
    session.commit()
    return row


def rank_score(relevance: float | None, published_at: datetime | None, now: datetime, half_life_h: float = 24.0) -> float:
    """Simple, explainable news-value proxy: relevance decayed by age (half-life in hours)."""
    if relevance is None:
        return 0.0
    published = as_utc(published_at)
    if published is None:
        return relevance * 0.5
    age_h = max(0.0, (now - published).total_seconds() / 3600)
    return relevance * math.pow(0.5, age_h / half_life_h)
