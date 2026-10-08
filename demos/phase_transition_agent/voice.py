"""Measure the *voice* of an answer: how scientific vs. chatty/generic it reads.

Crude, transparent heuristics - no model involved - so the same numbers can be recomputed by anyone. They describe style,
not correctness (correctness is the job of the evidence checks in tools.py). Each metric is documented where it is
computed; `register_score` combines them into one 0-100 number for the summary table, and the weights are in one dict so
they can be argued with.
"""

import re
from dataclasses import asdict, dataclass

# Phrases typical of a chatty assistant rather than of a paper's discussion section.
CHATTY = [
    "here is a breakdown", "here's a breakdown", "let's break", "in summary", "in short", "in conclusion",
    "it's important to", "it is important to", "important to note", "you should", "great question",
    "as an ai", "i hope this", "feel free", "dangerous", "catastrophic", "explosion", "undesirable",
    "significant stress", "major hazard", "the primary concern",
]
# Hedges that a careful scientist uses precisely (counted as a mild positive, capped).
SCIENTIFIC_CUES = [
    "reported", "observed", "attributed to", "consistent with", "whereas", "thereby", "owing to", "induced by",
    "irreversible", "reversible", "lattice", "in situ", "operando", "state of charge", "delithiat", "intercalat",
]
UNIT = re.compile(r"\b\d+(?:\.\d+)?\s?(?:v|mv|%|ma|mah/g|c|°c|k|nm|µm|um|å|gpa|wh/kg|h)\b", re.I)
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
CHEM = re.compile(r"\b(?:[A-Z][a-z]?\d*(?:\.\d+)?){2,}\b|\bH[1-4]\b|\bNMC\d*|\bNCM\d*|\bSEI\b|\bCEI\b")
CITE = re.compile(r"\[\d+\]|【\d+[^】]*】|doi:\S+|\(\w+ et al\.")

WEIGHTS = {  # contribution to register_score (0-100); negative = penalty
    "technical_density": 30, "quantitative_density": 20, "cue_density": 15, "citation_density": 15,
    "prose_ratio": 10, "no_chatty": 10,
}


@dataclass
class Voice:
    words: int
    avg_sentence_words: float
    technical_terms_per_100w: float
    numbers_with_units_per_100w: float
    scientific_cues_per_100w: float
    citations_per_100w: float
    bullet_or_header_lines_pct: float
    chatty_phrases: int
    register_score: float

    def row(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def analyse(text: str) -> Voice:
    clean = re.sub(r"[*_`#]", "", text)
    words = re.findall(r"\S+", clean)
    n = max(len(words), 1)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.split()) >= 3]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    structured = [ln for ln in lines if re.match(r"\s*(?:[-*•]|\d+[.)]|#{1,6}\s|\|)", ln)]
    low = clean.lower()

    per100 = lambda count: 100 * count / n  # noqa: E731
    technical = per100(len(CHEM.findall(clean)))
    quantitative = per100(len(UNIT.findall(clean)))
    cues = per100(sum(low.count(c) for c in SCIENTIFIC_CUES))
    cites = per100(len(CITE.findall(text)))
    structured_pct = 100 * len(structured) / max(len(lines), 1)
    chatty = sum(low.count(c) for c in CHATTY)

    score = (
        WEIGHTS["technical_density"] * min(technical / 6, 1)
        + WEIGHTS["quantitative_density"] * min(quantitative / 2.5, 1)
        + WEIGHTS["cue_density"] * min(cues / 4, 1)
        + WEIGHTS["citation_density"] * min(cites / 4, 1)
        + WEIGHTS["prose_ratio"] * (1 - min(structured_pct / 60, 1))
        + WEIGHTS["no_chatty"] * (1 - min(chatty / 3, 1))
    )
    return Voice(
        words=len(words), avg_sentence_words=len(words) / max(len(sentences), 1), technical_terms_per_100w=technical,
        numbers_with_units_per_100w=quantitative, scientific_cues_per_100w=cues, citations_per_100w=cites,
        bullet_or_header_lines_pct=structured_pct, chatty_phrases=chatty, register_score=score,
    )
