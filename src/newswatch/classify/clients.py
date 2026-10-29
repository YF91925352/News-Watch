"""LLM clients behind one small interface.

`LLMClassifier` only depends on `LLMClient.complete`. Tests and CI use `FakeLLMClient`, which is
deterministic, free and offline; the real provider is imported lazily so the package installs and
tests run without the SDK or an API key.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None


class LLMClient(Protocol):
    model: str
    # Request settings that change the output (thinking, output format...). Part of the eval
    # cache key, so changing them can never silently reuse replies produced under old settings.
    config: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


class FakeLLMClient:
    """Deterministic stand-in. By default it answers with the keyword baseline's verdict as JSON,
    so the whole pipeline runs end to end offline. Pass `responder` to script exact replies."""

    model = "fake-llm"
    config = "fake"

    def __init__(self, responder: Callable[[str, str], str] | None = None):
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self._responder is not None:
            text = self._responder(system, user)
        else:
            from .base import ArticleInput
            from .keyword import KeywordClassifier

            body = user.split("<article>", 1)[-1].split("</article>", 1)[0]
            r = KeywordClassifier().classify(ArticleInput(title=body))
            text = json.dumps({"topics": r.topics, "relevance": r.relevance, "rationale": "fake: " + r.rationale})
        # Rough token estimate (~4 chars/token) so cost reporting has something to show.
        return LLMResponse(text=text, model=self.model, input_tokens=(len(system) + len(user)) // 4,
                           output_tokens=len(text) // 4)


def response_schema() -> dict:
    from .base import TOPICS

    return {
        "type": "object",
        "properties": {
            "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
            "relevance": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["topics", "relevance", "rationale"],
        "additionalProperties": False,
    }


class AnthropicClient:
    """Thinking is turned off: this is a short classification, and with thinking on (the model's
    default) the reasoning used up the token budget before any JSON was written. Structured
    outputs make the reply match the schema, so parse failures should not happen."""

    config = "thinking=off;format=json_schema;max_tokens=1000"

    def __init__(self, model: str, max_tokens: int = 1000):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'newswatch[llm]' to use the Anthropic backend") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(max_retries=3)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - needs network
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": response_schema()}},
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return LLMResponse(text=text, model=msg.model, input_tokens=msg.usage.input_tokens,
                           output_tokens=msg.usage.output_tokens,
                           stop_reason=getattr(msg, "stop_reason", None))


def make_client(backend: str, model: str) -> LLMClient | None:
    if backend == "off":
        return None
    if backend == "fake":
        return FakeLLMClient()
    if backend == "anthropic":
        return AnthropicClient(model)
    raise ValueError(f"Unknown LLM backend: {backend!r} (expected off, fake or anthropic)")
