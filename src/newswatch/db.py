from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def make_engine(url: str) -> Engine:
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        # Used by the test-suite only. In-memory SQLite needs a single shared connection.
        from sqlalchemy.pool import StaticPool

        kwargs.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return create_engine(url, **kwargs)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    # Deliberately create_all instead of Alembic for a project this size; see README.
    Base.metadata.create_all(engine)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on the way back; PostgreSQL keeps it. Normalise both to aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
