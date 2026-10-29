"""LLM classifier and the keyword-prefilter hybrid."""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources

from .base import BRIEF, TOPICS, ArticleInput, Classification
from .clients import LLMClient
from .keyword import KeywordClassifier

PROMPT_VERSION = "relevance-v1"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_prompt(version: str = PROMPT_VERSION) -> tuple[str, str]:
    """Return (rendered system prompt, sha256 of the template). The hash is stored with every
    result, so an edited prompt can never silently masquerade as the old version."""
    name = version.replace("relevance-", "relevance_") + ".md"
    template = resources.files("newswatch.prompts").joinpath(name).read_text(encoding="utf-8")
    sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    topics = "\n".join(f"- {k}: {v}" for k, v in TOPICS.items())
    return template.format(brief=BRIEF, topics=topics), sha


def render_user_message(article: ArticleInput) -> str:
    # Strip anything that looks like our delimiter so a headline cannot close the data block.
    clean = lambda s: s.replace("<article>", "").replace("</article>", "")
    return f"<article>\nHeadline: {clean(article.title)}\nTeaser: {clean(article.summary)}\n</article>"


class ParseError(ValueError):
    pass


def parse_output(text: str) -> tuple[list[str], float, str]:
    match = _JSON_RE.search(text or "")
    if not match:
        raise ParseError("no JSON object in output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    try:
        relevance = float(data["relevance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParseError("missing or non-numeric relevance") from exc
    relevance = min(1.0, max(0.0, relevance))
    raw_topics = data.get("topics") or []
    if not isinstance(raw_topics, list):
        raise ParseError("topics is not a list")
    # Drop labels outside the taxonomy instead of trusting them.
    topics = [t for t in raw_topics if isinstance(t, str) and t in TOPICS][:3] or ["andet"]
    rationale = str(data.get("rationale", ""))[:500]
    return topics, round(relevance, 3), rationale


class LLMClassifier:
    method = "llm"

    def __init__(self, client: LLMClient, prompt_version: str = PROMPT_VERSION):
        self.client = client
        self.prompt_version = prompt_version
        self.system_prompt, self.prompt_sha = load_prompt(prompt_version)

    def classify(self, article: ArticleInput) -> Classification:
        resp = self.client.complete(self.system_prompt, render_user_message(article))
        base = dict(
            method=self.method,
            # The *requested* model id, not resp.model: the provider may echo a dated alias,
            # which would break the "already classified?" check and re-bill every article.
            model=self.client.model,
            prompt_version=self.prompt_version,
            prompt_sha256=self.prompt_sha,
            raw_output=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
        try:
            topics, relevance, rationale = parse_output(resp.text)
        except ParseError as exc:
            # A failed parse is recorded as a failure, never guessed into a score.
            why = f"parse error: {exc}"
            if resp.stop_reason and resp.stop_reason != "end_turn":
                why += f" (stop_reason={resp.stop_reason})"
            return Classification(status="failed", relevance=None, topics=[], rationale=why, **base)
        return Classification(status="ok", relevance=relevance, topics=topics, rationale=rationale, **base)


class HybridClassifier:
    """Keyword prefilter, then LLM only for articles that pass it. Saves cost; may lose recall on
    articles that are relevant without using any of the keywords - the eval measures exactly that."""

    method = "hybrid"

    def __init__(self, llm: LLMClassifier, prefilter_min: float = 0.2):
        self.keyword = KeywordClassifier()
        self.llm = llm
        self.prefilter_min = prefilter_min

    def classify(self, article: ArticleInput) -> Classification:
        kw = self.keyword.classify(article)
        if (kw.relevance or 0.0) < self.prefilter_min:
            return Classification(
                method=self.method,
                model=self.llm.client.model,
                prompt_version=self.llm.prompt_version,
                prompt_sha256=self.llm.prompt_sha,
                status="skipped_prefilter",
                relevance=kw.relevance,
                topics=kw.topics,
                rationale="Skipped LLM: keyword prefilter score below threshold. " + kw.rationale,
            )
        result = self.llm.classify(article)
        result.method = self.method
        return result
