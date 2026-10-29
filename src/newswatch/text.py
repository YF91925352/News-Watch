"""Small text helpers: HTML stripping, truncation, URL normalisation."""

from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRACKING_PARAMS = {"fbclid", "gclid", "ref", "cmpid"}


def strip_html(value: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value or ""))).strip()


def truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    cut = value[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "…"


def normalise_url(url: str) -> str:
    """Lower-case scheme/host, drop fragment, tracking params and trailing slash."""
    parts = urlsplit(url.strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalise_url(url).encode("utf-8")).hexdigest()


def fold(value: str) -> str:
    """Lower-case for matching. Danish letters are kept as-is."""
    return value.casefold()
