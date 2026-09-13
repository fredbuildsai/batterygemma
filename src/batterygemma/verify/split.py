"""Stage 9 (Verify): assign a deterministic train/eval split, by paper.

Splitting must happen at the *document* level, not per-example: several QA/negative/ideation rows can come
from the same paper, and letting some of a paper's rows land in eval while others train on the same source
text would leak information across the split. Each document's `doc_id` hashes to a stable train/eval
assignment, then every derived row (QA, Ideation, Negative, DPOPair, ClaimPair) inherits its source
document's split at export time (see `export.unsloth_jsonl`).
"""

import hashlib

DEFAULT_EVAL_FRACTION = 0.10


def split_for_doc(doc_id: str, eval_fraction: float = DEFAULT_EVAL_FRACTION) -> str:
    """Deterministic: the same doc_id always gets the same split, with no persisted state needed to repeat it."""
    bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 1000
    return "eval" if bucket < eval_fraction * 1000 else "train"


def assign_document_splits(session, eval_fraction: float = DEFAULT_EVAL_FRACTION) -> dict[str, int]:
    """Sets `Document.split` for every chunked document that doesn't already have one. Returns split counts."""
    from collections import Counter

    from sqlalchemy import select

    from batterygemma.db.models import Document

    counts: Counter[str] = Counter()
    docs = session.scalars(select(Document).where(Document.status == "chunked", Document.split.is_(None)))
    for doc in docs:
        doc.split = split_for_doc(doc.doc_id, eval_fraction)
        counts[doc.split] += 1
    return dict(counts)
