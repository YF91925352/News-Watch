"""Settings, read from environment variables so the same code runs locally, in Docker and in CI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg://newswatch:newswatch@localhost:5432/newswatch"
        )
    )
    feeds_file: str = field(default_factory=lambda: os.getenv("NEWSWATCH_FEEDS", "config/feeds.yaml"))

    # Politeness when fetching feeds.
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "NEWSWATCH_USER_AGENT",
            "newswatch/0.1 (portfolio project; contact: set NEWSWATCH_USER_AGENT)",
        )
    )
    min_fetch_interval_s: int = field(
        default_factory=lambda: int(os.getenv("NEWSWATCH_MIN_FETCH_INTERVAL_S", "600"))
    )
    request_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("NEWSWATCH_REQUEST_TIMEOUT_S", "15"))
    )
    summary_max_chars: int = field(
        default_factory=lambda: int(os.getenv("NEWSWATCH_SUMMARY_MAX_CHARS", "400"))
    )

    # LLM: "off", "fake" (deterministic, offline) or "anthropic".
    llm_backend: str = field(default_factory=lambda: os.getenv("NEWSWATCH_LLM", "off"))
    llm_model: str = field(default_factory=lambda: os.getenv("NEWSWATCH_LLM_MODEL", "claude-haiku-5-5"))
    # Prices are deliberately not hard-coded: check the provider's pricing page and set them.
    price_in_per_mtok: float | None = field(
        default_factory=lambda: _float_or_none(os.getenv("NEWSWATCH_PRICE_IN_PER_MTOK"))
    )
    price_out_per_mtok: float | None = field(
        default_factory=lambda: _float_or_none(os.getenv("NEWSWATCH_PRICE_OUT_PER_MTOK"))
    )


def get_settings() -> Settings:
    return Settings()
