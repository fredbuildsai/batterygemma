"""Stage 9 (Verify): near-duplicate detection.

v1: pairwise word-overlap within small buckets (same question_type + component), reusing the same Jaccard-
style overlap metric stage 7 uses for grounding. This is O(bucket_size^2) but buckets stay small at pilot
scale (a few hundred rows). `datasketch` (MinHash + LSH) is already declared as the optional `dedupe` extra
in pyproject.toml for when the corpus grows past what pairwise comparison can handle (tens of thousands of
rows per bucket) - swap this module's implementation for it then, keeping the same `mark_duplicates` API.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from batterygemma.annotate.grounding import overlap_ratio

DEFAULT_THRESHOLD = 0.85


def find_duplicate_ids(items: Iterable[tuple[str, str, str]], threshold: float = DEFAULT_THRESHOLD) -> set[str]:
    """`items` is (id, bucket_key, text). Within each bucket, every item after the first near-duplicate match
    is flagged. Returns the set of ids to reject; the first-seen item in each duplicate group is kept.
    """
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item_id, bucket_key, text in items:
        buckets[bucket_key].append((item_id, text))

    duplicates: set[str] = set()
    for group in buckets.values():
        kept: list[tuple[str, str]] = []
        for item_id, text in group:
            if any(overlap_ratio(text, kept_text) >= threshold for _, kept_text in kept):
                duplicates.add(item_id)
            else:
                kept.append((item_id, text))
    return duplicates


def mark_duplicates(session, model_cls, *, bucket_columns: tuple[str, ...], text_fn: Callable[[Any], str],
                    threshold: float = DEFAULT_THRESHOLD, status_filter: str = "accepted") -> int:
    """Marks near-duplicate rows of `model_cls` (matching `status_filter`) as rejected. Returns how many.

    `text_fn(row)` extracts the text to compare - a plain column, or something computed (e.g. a QA row's
    question, which lives inside its `turns` JSON list rather than its own column).
    """
    from sqlalchemy import select

    rows = session.scalars(select(model_cls).where(model_cls.status == status_filter)).all()
    items = [
        (row.id, "|".join(str(getattr(row, col)) for col in bucket_columns), text_fn(row))
        for row in rows
    ]
    duplicate_ids = find_duplicate_ids(items, threshold=threshold)
    for row in rows:
        if row.id in duplicate_ids:
            row.status = "rejected"
            row.reject_reason = "near_duplicate"
    return len(duplicate_ids)
