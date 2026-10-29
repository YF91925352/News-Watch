"""Shared types for all classifiers.

Every classifier - keyword, LLM, or hybrid - returns the same `Classification`, so the API,
the database and the evaluation script never need to know which one produced a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# The monitoring brief: what "relevant" means for this mini product.
BRIEF = (
    "Klima, energi og grøn omstilling i Danmark: klimapolitik, CO2-udledning, vedvarende energi, "
    "elpriser og energiforsyning, klimatilpasning (oversvømmelse, stormflod, skybrud) og "
    "landbrugets klimaaftryk."
)

TOPICS: dict[str, str] = {
    "klima_energi": "Klima, energi, grøn omstilling, klimatilpasning",
    "politik": "Dansk politik, Folketinget, regering, kommuner",
    "oekonomi": "Økonomi, erhverv, arbejdsmarked, privatøkonomi",
    "sundhed": "Sundhed, hospitaler, psykiatri, ældrepleje",
    "teknologi": "Teknologi, digitalisering, AI, cybersikkerhed",
    "krimi_retsvaesen": "Kriminalitet, politi, retssager",
    "udland": "Udenrigs- og internationale forhold",
    "kultur_sport": "Kultur, medier, sport, underholdning",
    "andet": "Alt andet",
}

RELEVANCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ArticleInput:
    title: str
    summary: str = ""
    url: str = ""


@dataclass
class Classification:
    method: str
    model: str
    prompt_version: str
    status: str  # ok | failed | skipped_prefilter
    relevance: float | None
    topics: list[str] = field(default_factory=list)
    rationale: str = ""
    prompt_sha256: str | None = None
    raw_output: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def is_relevant(self) -> bool:
        return self.relevance is not None and self.relevance >= RELEVANCE_THRESHOLD


class Classifier(Protocol):
    method: str

    def classify(self, article: ArticleInput) -> Classification: ...
