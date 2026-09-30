"""Stage 11 (Eval): build a held-out gold benchmark from accepted stage-8 records on eval-split documents.

Items are drawn only from documents already assigned `split == "eval"` by `verify.split` - the same papers
excluded from every training export - so the benchmark never leaks into what the model was tuned on. Closed
and open Q&A come from judge-accepted `QA` rows; negative-detection items come from `Negative` rows (no judge
step exists for negatives - see `export.unsloth_jsonl`'s module docstring); ideation prompts come from
judge-accepted `Ideation` rows. Each item keeps its `source_chunk_ids` so `run_eval` can hand a judge the same
source excerpt used at generation time (for ideation) or simply for provenance (for Q&A/negatives).
"""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import Document, Ideation, Negative, QA

# Plan's pilot targets (see plan's "Gold eval set" section); build_gold_set caps each category at whatever
# is actually available, so a small corpus yields a small (but still correctly-shaped) benchmark.
DEFAULT_TARGET_COUNTS = {"closed_qa": 600, "open_qa": 400, "negative_detection": 300, "ideation": 200}


def _is_eval_doc(session: Session, doc_ids: list[Any]) -> bool:
    if not doc_ids:
        return False
    doc = session.get(Document, doc_ids[0])
    return bool(doc and doc.split == "eval")


def _closed_qa_items(session: Session, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = session.scalars(select(QA).where(QA.status == "accepted", QA.answer_type == "CLOSED")).all()
    for qa in rows:
        if len(items) >= limit:
            break
        if not _is_eval_doc(session, qa.doc_ids):
            continue
        items.append({
            "gold_id": f"gold-closed_qa-{qa.id}", "category": "closed_qa", "source_id": qa.id,
            "source_chunk_ids": qa.chunk_ids, "component": qa.component, "question_type": qa.question_type,
            "prompt": [{"role": "user", "content": qa.turns[0]["content"]}],
            "reference_answer": qa.turns[1]["content"], "closed_label": qa.closed_label,
        })
    return items


def _open_qa_items(session: Session, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = session.scalars(select(QA).where(QA.status == "accepted", QA.answer_type == "OPEN")).all()
    for qa in rows:
        if len(items) >= limit:
            break
        if not _is_eval_doc(session, qa.doc_ids):
            continue
        items.append({
            "gold_id": f"gold-open_qa-{qa.id}", "category": "open_qa", "source_id": qa.id,
            "source_chunk_ids": qa.chunk_ids, "component": qa.component, "question_type": qa.question_type,
            "prompt": [{"role": "user", "content": qa.turns[0]["content"]}],
            "reference_answer": qa.turns[1]["content"],
        })
    return items


def _negative_items(session: Session, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = session.scalars(select(Negative).where(Negative.status == "generated")).all()
    for neg in rows:
        if len(items) >= limit:
            break
        if not _is_eval_doc(session, neg.doc_ids):
            continue
        items.append({
            "gold_id": f"gold-negative_detection-{neg.id}", "category": "negative_detection", "source_id": neg.id,
            "source_chunk_ids": neg.chunk_ids, "kind": neg.kind, "flawed_element": neg.flawed_element,
            "prompt": [{"role": "user", "content": neg.prompt}], "reference_answer": neg.expert_response,
        })
    return items


def _ideation_items(session: Session, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = session.scalars(select(Ideation).where(Ideation.status == "accepted")).all()
    for idea in rows:
        if len(items) >= limit:
            break
        if not _is_eval_doc(session, idea.doc_ids):
            continue
        problem = idea.problem
        if idea.constraints:
            problem += "\n\nConstraints: " + "; ".join(idea.constraints)
        items.append({
            "gold_id": f"gold-ideation-{idea.id}", "category": "ideation", "source_id": idea.id,
            "source_chunk_ids": idea.chunk_ids, "prompt": [{"role": "user", "content": problem}],
            "reference_ideas": idea.ideas,
        })
    return items


def build_gold_set(
    session: Session, output_path: Path, *, target_counts: dict[str, int] | None = None
) -> dict[str, int]:
    """Select held-out benchmark items and write them to `output_path` as JSONL. Returns counts per category."""
    targets = target_counts or DEFAULT_TARGET_COUNTS
    items = [
        *_closed_qa_items(session, targets.get("closed_qa", 0)),
        *_open_qa_items(session, targets.get("open_qa", 0)),
        *_negative_items(session, targets.get("negative_detection", 0)),
        *_ideation_items(session, targets.get("ideation", 0)),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts
