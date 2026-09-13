"""A cheap, no-LLM check that a claimed piece of evidence actually comes from its source chunk.

Extraction models sometimes state something true-sounding but not actually present in the given text (from
background knowledge rather than the chunk). Exact substring matching is too strict — models paraphrase
slightly even when asked to quote — so this uses word-overlap: most of the significant words in the evidence
sentence should appear somewhere in the source text.
"""

import re

_WORD = re.compile(r"[a-z0-9]+")
_MIN_WORD_LEN = 3


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= _MIN_WORD_LEN}


def overlap_ratio(evidence: str, source_text: str) -> float:
    evidence_words = _significant_words(evidence)
    if not evidence_words:
        return 0.0
    source_words = _significant_words(source_text)
    return len(evidence_words & source_words) / len(evidence_words)


def is_grounded(evidence: str, source_text: str, threshold: float = 0.6) -> bool:
    return overlap_ratio(evidence, source_text) >= threshold
