"""Keyword baseline. Cheap, transparent, and the bar the LLM has to beat.

Terms are matched as word *prefixes* ("klima" matches "klimaforandringerne"), which suits
Danish compounds. Multi-word terms are matched as phrases in the folded text.
"""

from __future__ import annotations

import math
import re

from ..text import fold
from .base import ArticleInput, Classification

MODEL_NAME = "keyword-v1"
_TOKEN_RE = re.compile(r"[\wæøå₂]+(?:-[\wæøå₂]+)*", re.UNICODE)

# Relevance to the brief: (term, weight). Strong terms alone are enough to call it relevant.
BRIEF_TERMS: list[tuple[str, float]] = [
    ("klima", 1.0), ("co2", 1.0), ("co₂", 1.0), ("drivhusgas", 1.0), ("udledning", 1.0),
    ("grøn omstilling", 1.0), ("vedvarende energi", 1.0), ("vindmølle", 1.0), ("havvind", 1.0),
    ("solcelle", 1.0), ("energiø", 1.0), ("power-to-x", 1.0), ("fjernvarme", 1.0),
    ("varmepumpe", 1.0), ("biogas", 1.0), ("brint", 0.8), ("elpris", 1.0), ("strømpris", 1.0),
    ("oversvømmelse", 1.0), ("stormflod", 1.0), ("skybrud", 1.0), ("kystsikring", 1.0),
    ("hedebølge", 0.8), ("tørke", 0.8), ("kvælstof", 0.8), ("fossil", 1.0), ("cop3", 1.0),
    ("energi", 0.5), ("elbil", 0.5), ("naturgas", 0.6), ("olie", 0.4), ("afgift", 0.2),
    ("landbrug", 0.3), ("trepart", 0.6),
]

TOPIC_TERMS: dict[str, list[str]] = {
    "klima_energi": [t for t, w in BRIEF_TERMS if w >= 0.5],
    "politik": ["folketing", "minister", "regering", "partiet", "venstre", "socialdemokrat",
                "moderaterne", "valg", "kommunalvalg", "borgmester", "lovforslag", "aftale"],
    "oekonomi": ["økonomi", "inflation", "rente", "aktie", "virksomhed", "ledighed", "løn",
                 "skat", "bolig", "nationalbank", "pension", "konkurs", "eksport"],
    "sundhed": ["sygehus", "hospital", "patient", "læge", "sundhed", "psykiatri", "ældre",
                "vaccine", "sygdom", "medicin"],
    "teknologi": ["ai$", "kunstig intelligens", "digital", "hacker", "cyber", "app$", "data",
                  "algoritme", "software", "chip"],
    # Not the prefix "politi": it also matches "politik" and tagged every politics story as crime.
    "krimi_retsvaesen": ["politi$", "politiet", "politibetjent", "anholdt", "sigtet", "dømt", "retten", "drab", "fængsel",
                         "tyveri", "bande", "anklage"],
    "udland": ["usa", "ukraine", "rusland", "kina", "eu$", "nato", "israel", "gaza", "tyskland",
               "sverige", "trump"],
    "kultur_sport": ["fodbold", "landshold", "koncert", "film", "musik", "museum", "håndbold",
                     "teater", "superliga", "festival"],
}


def _matches(term: str, tokens: list[str], folded: str) -> bool:
    if " " in term:
        return term in folded
    if term.endswith("$"):  # exact token, for short words like "ai" or "eu"
        return term[:-1] in tokens
    return any(tok.startswith(term) for tok in tokens)


class KeywordClassifier:
    method = "keyword"

    def classify(self, article: ArticleInput) -> Classification:
        folded = fold(f"{article.title} {article.summary}")
        tokens = _TOKEN_RE.findall(folded)

        hits = [(t, w) for t, w in BRIEF_TERMS if _matches(t, tokens, folded)]
        score_sum = sum(w for _, w in hits)
        relevance = round(1 - math.exp(-score_sum), 3)

        topics = [
            topic for topic, terms in TOPIC_TERMS.items() if any(_matches(t, tokens, folded) for t in terms)
        ] or ["andet"]

        rationale = (
            "Matched brief terms: " + ", ".join(t for t, _ in hits) if hits else "No brief terms matched."
        )
        return Classification(
            method=self.method,
            model=MODEL_NAME,
            prompt_version="n/a",
            status="ok",
            relevance=relevance,
            topics=topics,
            rationale=rationale,
        )
