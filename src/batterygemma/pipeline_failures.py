"""Roll up every kind of pipeline failure back to the document level, in one place.

Two distinct failure sources exist and neither alone answers "which documents need attention":
- `bg fetch-failures` already covers documents `bg fetch` never got full text for (see `fetch.py`).
- Every annotate/generate/judge stage records its own per-chunk/per-row failures as `GenTask` rows with
  `status="failed"` (see `annotate.tasks`), keyed by a stage-specific id - a chunk_id for extract/generate
  tasks, but a QA-id or Ideation-id (`"<chunk_id>#qa<N>"` / `"<chunk_id>#idea"`) for judge/dpo tasks, since
  those operate on already-generated rows rather than raw chunks.

`resolve_chunk_id` undoes that: it tries the full remainder as a chunk_id first (the common case), and falls
back to stripping the last `#`-segment for the qa/idea-keyed tasks. This is a document-level view, so multiple
failed chunks/rows for the same document collapse into one entry with all of that document's failures listed.
"""

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import Chunk, Document, GenTask

# Every GenTask key is "<task_type>:<rest>". Tasks in this set are keyed directly by chunk_id; every other
# task type is keyed by a QA-id or Ideation-id ("<chunk_id>#qa<N>" / "<chunk_id>#idea"), so a lookup miss on
# the full remainder falls back to stripping the last '#...' segment.
CHUNK_KEYED_TASKS = {"extract_facts", "extract_claims", "generate_qa", "generate_false_premise", "generate_ideation"}


def resolve_chunk_id(session: Session, task_type: str, rest: str) -> str | None:
    if task_type in CHUNK_KEYED_TASKS:
        return rest if session.get(Chunk, rest) else None
    if "#" in rest:
        candidate = rest.rsplit("#", 1)[0]
        if session.get(Chunk, candidate):
            return candidate
    return None


def failed_documents(session: Session) -> list[dict[str, Any]]:
    """One entry per document that has at least one failed GenTask, each carrying every failure found for
    that document across every stage (task_type, key, error). Documents with no resolvable chunk (e.g. a
    stale/deleted chunk) are skipped - dangling GenTask rows are not a document-level concern."""
    failed = session.scalars(select(GenTask).where(GenTask.status == "failed")).all()

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doc_ids_seen: dict[str, str] = {}  # chunk_id -> doc_id, to avoid re-querying Chunk per failure
    for task in failed:
        task_type, _, rest = task.key.partition(":")
        chunk_id = resolve_chunk_id(session, task_type, rest)
        if chunk_id is None:
            continue
        doc_id = doc_ids_seen.get(chunk_id)
        if doc_id is None:
            chunk = session.get(Chunk, chunk_id)
            if chunk is None:
                continue
            doc_id = doc_ids_seen[chunk_id] = chunk.doc_id
        by_doc[doc_id].append({"task_type": task_type, "key": task.key, "attempts": task.attempts,
                               "error": task.last_error or ""})

    results = []
    for doc_id, failures in by_doc.items():
        doc = session.get(Document, doc_id)
        results.append({
            "doc_id": doc_id, "title": doc.title if doc else "", "failure_count": len(failures),
            "stages": sorted({f["task_type"] for f in failures}), "failures": failures,
        })
    results.sort(key=lambda r: r["failure_count"], reverse=True)
    return results


DEFAULT_BLACKLIST_THRESHOLD = 20  # a document accumulating this many failed tasks is failing every time,
                                    # not just hitting an occasional rate limit - see cli.pipeline_failures_cmd


def blacklist_repeat_failures(
    session: Session, *, threshold: int = DEFAULT_BLACKLIST_THRESHOLD
) -> list[dict[str, Any]]:
    """Blacklist every not-yet-blacklisted document with `failure_count >= threshold`. Blacklisting a
    document stops `bg annotate`/`bg generate` from selecting its chunks in future runs (see cli._pending_
    chunk_ids and the `annotate`/`generate negatives` document queries) - it does not touch or delete
    anything already generated, and does not retroactively fail anything in progress. Returns the newly
    blacklisted documents (already-blacklisted ones are skipped, so re-running this is idempotent and its
    report only ever shows what changed)."""
    newly_blacklisted = []
    for result in failed_documents(session):
        if result["failure_count"] < threshold:
            continue
        doc = session.get(Document, result["doc_id"])
        if doc is None or doc.blacklisted:
            continue
        doc.blacklisted = True
        doc.blacklist_reason = (
            f"{result['failure_count']} failed tasks across stages: {', '.join(result['stages'])}"
        )
        newly_blacklisted.append(result)
    return newly_blacklisted
