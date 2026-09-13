"""License gate plus keyword relevance scoring for discovered documents.

Decisions (stored on documents.status with a reason in status_reason):
  rejected    license not allowed/unknown, or no scope term in title/abstract
  borderline  in scope but weakly (few topic signals, or a comparison-only chemistry in the title);
              resolved later by an LLM yes/no on the `screen` route
  accepted    in scope with enough topic signal
Duplicates (duplicate_of set) are never screened; their canonical document is.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import Document
from batterygemma.screen.license import FLAGGED, REJECTED, UNKNOWN, evaluate_license

ACCEPT_SCORE = 3.0
TITLE_WEIGHT = 2.0
ABSTRACT_WEIGHT = 1.0
MAX_SCOPE_POINTS = 3.0

# Publisher metadata often uses typographic dashes and spaces ("Lithium‐Ion", "Li–S", "4.2 V"), which would
# otherwise never match ASCII scope terms like "lithium-ion" (seen on ~70% of rejected OpenAlex titles/abstracts).
_UNICODE_DASHES = re.compile("[‐‑‒–—−]")
_UNICODE_SPACES = re.compile("[    ]")


def _normalize(text: str | None) -> str:
    return _UNICODE_SPACES.sub(" ", _UNICODE_DASHES.sub("-", (text or "").lower()))


@dataclass
class Relevance:
    score: float
    in_scope: bool
    tags: list[str] = field(default_factory=list)
    comparison_only_title: bool = False


def _compile(term: str) -> re.Pattern[str]:
    # Whole tokens with an optional plural: "SEI" must not hit "seismic", but "cathode" should hit "cathodes"
    # and "lithium battery" should hit "lithium batteries".
    stem = re.escape(term.lower())
    if term.lower().endswith("y"):
        stem = rf"{stem[:-1]}(?:y|ies)"
    return re.compile(rf"(?<![a-z0-9]){stem}(?:s|es)?(?![a-z0-9])")


def score_relevance(title: str, abstract: str | None, scope: dict[str, Any]) -> Relevance:
    title_text = _normalize(title)
    abstract_text = _normalize(abstract)

    scope_points = 0.0
    for group in scope["must"]:
        patterns = [_compile(t) for t in group]
        group_points = sum(TITLE_WEIGHT for p in patterns if p.search(title_text))
        group_points += sum(ABSTRACT_WEIGHT for p in patterns if p.search(abstract_text))
        if group_points == 0:
            return Relevance(score=0.0, in_scope=False)
        scope_points += group_points

    tags = [
        category
        for category, terms in scope.get("boost", {}).items()
        if any(_compile(t).search(title_text) or _compile(t).search(abstract_text) for t in terms)
    ]
    comparison_only_title = any(_compile(t).search(title_text) for t in scope.get("comparison_only", []))
    score = min(MAX_SCOPE_POINTS, scope_points) + len(tags)
    return Relevance(score=score, in_scope=True, tags=tags, comparison_only_title=comparison_only_title)


def screen_documents(session: Session, cfg: dict[str, Any], *, rescreen: bool = False) -> Counter[str]:
    statuses = ["discovered", "borderline"] + (["accepted", "rejected"] if rescreen else [])
    query = select(Document).where(Document.duplicate_of.is_(None), Document.status.in_(statuses))
    counts: Counter[str] = Counter()
    for doc in session.scalars(query):
        decision = evaluate_license(doc.license, cfg["license_allow"], cfg["license_flag"])
        relevance = score_relevance(doc.title, doc.abstract, cfg["scope"])
        doc.relevance = relevance.score
        doc.topic_tags = relevance.tags
        doc.license_flagged = decision == FLAGGED

        if decision in (REJECTED, UNKNOWN):
            doc.status, doc.status_reason = "rejected", f"license:{decision}:{doc.license}"
        elif not relevance.in_scope:
            doc.status, doc.status_reason = "rejected", "relevance:out_of_scope"
        elif relevance.comparison_only_title or relevance.score < ACCEPT_SCORE:
            reason = "comparison_only_title" if relevance.comparison_only_title else f"score={relevance.score:g}"
            doc.status, doc.status_reason = "borderline", f"relevance:{reason}"
        else:
            doc.status, doc.status_reason = "accepted", f"relevance:score={relevance.score:g}"
        counts[doc.status] += 1
    return counts
