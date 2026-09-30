"""Stage 9 (Export): write CPT/SFT/DPO JSONL exactly as `train.data` reads them.

Splits are by paper (`Document.split`, set by `verify.split`), so every row inherits its source document's
split - a row is never assigned its own independent split, which is what would let one paper's content leak
across train/eval. QA rows are exported only when `status == "accepted"` (judge-approved: `bg generate dpo`
also only builds pairs from accepted QA, so its `chosen` answer is always judge-verified, never an
unverified "generated" one). Negatives and DPO pairs have no separate judge step yet - their own grounding
check (stage 8) is the quality gate, so `generated` is exported for them. `rejected` rows are always left in
the database for inspection, never exported.
"""

import logging
from pathlib import Path
from typing import Any

from corpusforge.export.corpus import (
    AttributionManifest,
    ExtraCptRow,
    get_doc,
    row_images,
    write_jsonl,
)
from corpusforge.export.corpus import (
    export_cpt as export_corpus_cpt,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import QA, DPOPair, Negative

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist. Reason from mechanisms, structure-property "
    "relationships and experimental evidence; use specific numbers and units; state uncertainty and "
    "trade-offs; correct false premises; and propose ideas that are testable and grounded in the literature."
)


def ontology_cpt_rows() -> list[ExtraCptRow]:
    """The lithium-ion-relevant EMMO/BattINFO definition rows (`ontology.lithium_cpt`) as corpusforge CPT extras.

    They carry no source paper, so they are attributed to the ontology itself (CC-BY-4.0) and always go to the
    train split - see `corpusforge.export.corpus.ExtraCptRow`.
    """
    from batterygemma.ontology.lithium_cpt import generate_ontology_cpt_rows

    return [
        ExtraCptRow(
            text=row["text"], attribution_key=f"ontology:{row['domain']}", license="CC-BY-4.0",
            title="EMMO/BattINFO domain ontology (chemical-substance, electrochemistry, battery)",
        )
        for row in generate_ontology_cpt_rows()
    ]


def export_cpt(session: Session, output_dir: Path, *, include_images: bool = False,
               pack_tokens: int | None = None, include_ontology: bool = True) -> dict[str, int]:
    """CPT export: corpusforge's paper-derived rows, plus (`include_ontology`) the ontology definition rows.

    `pack_tokens`: pack a paper's consecutive chunks into rows of about this many tokens (see
    `corpusforge.export.corpus.pack_cpt_chunks`); unset = one row per chunk.
    """
    return export_corpus_cpt(
        session, output_dir, include_images=include_images, pack_tokens=pack_tokens,
        extra_rows=ontology_cpt_rows() if include_ontology else (),
    )


def export_sft(session: Session, output_dir: Path, *, include_images: bool = False) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    attribution = AttributionManifest()

    for qa in session.scalars(select(QA).where(QA.status == "accepted")).all():
        doc = get_doc(session, qa.doc_ids[0]) if qa.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        messages = [{"role": "system", "content": qa.system or SYSTEM_PROMPT}, *qa.turns]
        row: dict[str, Any] = {"messages": messages}
        if include_images and (images := row_images(session, qa.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)

    for neg in session.scalars(select(Negative).where(Negative.status == "generated")).all():
        doc = get_doc(session, neg.doc_ids[0]) if neg.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": neg.prompt},
                {"role": "assistant", "content": neg.expert_response},
            ]
        }
        if include_images and (images := row_images(session, neg.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)

    attribution.write(output_dir, "sft")
    return {split: write_jsonl(output_dir / f"sft_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


def export_dpo(session: Session, output_dir: Path, *, include_images: bool = False) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    attribution = AttributionManifest()
    for pair in session.scalars(select(DPOPair).where(DPOPair.status == "generated")).all():
        doc = get_doc(session, pair.doc_ids[0]) if pair.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        row: dict[str, Any] = {"prompt": pair.prompt, "chosen": pair.chosen, "rejected": pair.rejected}
        if include_images and (images := row_images(session, pair.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)
    attribution.write(output_dir, "dpo")
    return {split: write_jsonl(output_dir / f"dpo_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


def export_all(session: Session, output_dir: Path, *, include_images: bool = False,
                cpt_pack_tokens: int | None = None, include_ontology: bool = True) -> dict[str, dict[str, int]]:
    """`include_images` defaults to off: Gemma 4 E2B's text-only SFT/DPO/CPT training doesn't consume the
    `images` field, so most exports shouldn't carry image paths around. Turn it on to audit which figures
    back a given example, or to prepare for a future multimodal training run.

    `cpt_pack_tokens`: passed straight through to `export_cpt`'s `pack_tokens` - see
    `cli.py`'s `export` command for how this is derived from `train_cpt.yaml`'s `max_seq_length`.
    `include_ontology`: passed straight through to `export_cpt`'s `include_ontology`."""
    result = {
        "cpt": export_cpt(session, output_dir, include_images=include_images, pack_tokens=cpt_pack_tokens,
                          include_ontology=include_ontology),
        "sft": export_sft(session, output_dir, include_images=include_images),
        "dpo": export_dpo(session, output_dir, include_images=include_images),
    }
    logger.info("export complete -> %s: %s", output_dir, result,
               extra={"context": {"output_dir": str(output_dir), "counts": result}})
    return result
