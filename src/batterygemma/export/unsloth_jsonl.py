"""Stage 9 (Export): write CPT/SFT/DPO JSONL exactly as `train.data` reads them.

Splits are by paper (`Document.split`, set by `verify.split`), so every row inherits its source document's
split - a row is never assigned its own independent split, which is what would let one paper's content leak
across train/eval. QA rows are exported only when `status == "accepted"` (judge-approved: `bg generate dpo`
also only builds pairs from accepted QA, so its `chosen` answer is always judge-verified, never an
unverified "generated" one). Negatives and DPO pairs have no separate judge step yet - their own grounding
check (stage 8) is the quality gate, so `generated` is exported for them. `rejected` rows are always left in
the database for inspection, never exported.
"""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import DPOPair, Chunk, Document, Negative, QA

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist. Reason from mechanisms, structure-property "
    "relationships and experimental evidence; use specific numbers and units; state uncertainty and "
    "trade-offs; correct false premises; and propose ideas that are testable and grounded in the literature."
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _doc_split(session: Session, doc_id: str) -> str | None:
    doc = session.get(Document, doc_id)
    return doc.split if doc else None


def export_cpt(session: Session, output_dir: Path) -> dict[str, int]:
    counts: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for chunk in session.scalars(select(Chunk)).all():
        split = _doc_split(session, chunk.doc_id)
        if split and chunk.text.strip():
            counts[split].append({"text": chunk.text})
    return {split: _write_jsonl(output_dir / f"cpt_{split}.jsonl", rows) for split, rows in counts.items()}


def export_sft(session: Session, output_dir: Path) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}

    for qa in session.scalars(select(QA).where(QA.status == "accepted")).all():
        split = _doc_split(session, qa.doc_ids[0]) if qa.doc_ids else None
        if not split:
            continue
        messages = [{"role": "system", "content": qa.system or SYSTEM_PROMPT}, *qa.turns]
        rows_by_split[split].append({"messages": messages})

    for neg in session.scalars(select(Negative).where(Negative.status == "generated")).all():
        split = _doc_split(session, neg.doc_ids[0]) if neg.doc_ids else None
        if not split:
            continue
        rows_by_split[split].append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": neg.prompt},
                {"role": "assistant", "content": neg.expert_response},
            ]
        })

    return {split: _write_jsonl(output_dir / f"sft_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


def export_dpo(session: Session, output_dir: Path) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for pair in session.scalars(select(DPOPair).where(DPOPair.status == "generated")).all():
        split = _doc_split(session, pair.doc_ids[0]) if pair.doc_ids else None
        if not split:
            continue
        rows_by_split[split].append({"prompt": pair.prompt, "chosen": pair.chosen, "rejected": pair.rejected})
    return {split: _write_jsonl(output_dir / f"dpo_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


def export_all(session: Session, output_dir: Path) -> dict[str, dict[str, int]]:
    return {
        "cpt": export_cpt(session, output_dir),
        "sft": export_sft(session, output_dir),
        "dpo": export_dpo(session, output_dir),
    }
