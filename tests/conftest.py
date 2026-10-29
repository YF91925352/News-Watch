"""Tests run on in-memory SQLite by default and on real PostgreSQL in CI
(set TEST_DATABASE_URL=postgresql+psycopg://...)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from newswatch.db import make_engine, make_sessionmaker
from newswatch.models import Base

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def engine():
    eng = make_engine(os.getenv("TEST_DATABASE_URL", "sqlite://"))
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return make_sessionmaker(engine)


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s
