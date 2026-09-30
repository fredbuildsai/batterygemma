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
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import DPOPair, Chunk, Document, Negative, QA
from batterygemma.images import resolve_chunk_image_paths
from batterygemma.parse.chunk import strip_overlap_prefix

logger = logging.getLogger(__name__)

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


def _get_doc(session: Session, doc_id: str) -> Document | None:
    return session.get(Document, doc_id)


def _doc_split(session: Session, doc_id: str) -> str | None:
    doc = _get_doc(session, doc_id)
    return doc.split if doc else None


class AttributionManifest:
    """Tracks, per source document, which row indices of an export file were derived from it.

    Rows are traced back to their source document's DOI (falling back to `doc_id` when a
    document has none, e.g. some theses/OSTI records) with its license, so a redistributed
    export carries per-record provenance sufficient for CC-BY-style attribution - required at
    the redistribution level, not just satisfied by a single blanket credit in a README.
    """

    def __init__(self) -> None:
        self._by_split: dict[str, dict[str, dict[str, Any]]] = {}

    def record(self, split: str, doc: Document, row_index: int) -> None:
        key = doc.doi or doc.doc_id
        entry = self._by_split.setdefault(split, {}).setdefault(
            key, {"doc_id": doc.doc_id, "license": doc.license, "title": doc.title, "chunks": []}
        )
        entry["chunks"].append(row_index)

    def record_external(self, split: str, key: str, *, license: str, title: str, row_index: int) -> None:
        """Like `record`, for a row with no source `Document` - e.g. an ontology-derived CPT row (see
        `export_cpt`'s `include_ontology`), attributed to the ontology itself rather than a paper."""
        entry = self._by_split.setdefault(split, {}).setdefault(
            key, {"doc_id": key, "license": license, "title": title, "chunks": []}
        )
        entry["chunks"].append(row_index)

    def write(self, output_dir: Path, name: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for split, manifest in self._by_split.items():
            path = output_dir / f"{name}_{split}.attribution.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            paths[split] = path
        return paths


def _row_images(session: Session, chunk_ids: list[str]) -> list[str]:
    """Downloaded figure paths for every chunk `chunk_ids` points to, deduped, in chunk order. This is how a
    generated example's `images` field links back to the actual figures its source chunk(s) referenced -
    see images.py for how a chunk's caption filenames resolve to real downloaded files."""
    paths: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        chunk = session.get(Chunk, chunk_id)
        if not chunk or not chunk.images:
            continue
        for path in resolve_chunk_image_paths(session, chunk.doc_id, chunk.images):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _cpt_text(chunk: Chunk) -> str:
    """Drop the leading overlap-with-previous-chunk sentences `parse/chunk.py` prepends for
    retrieval-style use (so a fact split across a chunk boundary isn't lost when only one chunk is
    retrieved). CPT has no such boundary - the model trains on the literal token sequence, so that
    overlap is pure redundancy: the same tokens get extra gradient updates for no benefit."""
    return strip_overlap_prefix(chunk.text, chunk.overlap_prev_tokens)


def _pack_cpt_chunks(chunks: list[Chunk], pack_tokens: int) -> list[list[Chunk]]:
    """Group chunks by `doc_id` only (no section boundary at all - see below), then greedily pack
    each document's chunks, in document order (`order` is a global, monotonically increasing sequence
    number for the whole document, not reset per section or subsection - confirmed against real
    chunk_ids, e.g. `#s00-c00, #s00-c01, #s01-c00, ...`), into batches whose combined (overlap-
    stripped) length stays under `pack_tokens`.

    An earlier version of this function stopped at the top-level section boundary (matching
    `parse.chunk`'s own rule, added there so chunks never cross a chapter and mix unrelated content
    for a *retrieval* unit). For CPT that concern doesn't apply the same way: there is no retrieval
    boundary being violated, only a plain-text sequence being trained on, and a real corpus
    measurement showed the section-respecting version left 94.2% of sections short of even a 2000-
    token budget - most documents simply don't have one section that long. Dropping the boundary
    entirely (grouping by doc_id only) was measured, on this same real corpus, to raise the average
    packed row from 897 to 1,632 tokens with only 3% of rows still under 500 tokens, so it's kept
    document-scoped (never crossing into a different paper) but no longer section-scoped within one.
    Cross-document row order in the output doesn't matter: each packed row becomes an independent,
    shuffled training example either way."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    groups: list[list[Chunk]] = []
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: c.order)
        current: list[Chunk] = []
        current_tokens = 0
        for chunk in doc_chunks:
            piece_tokens = chunk.tokens - chunk.overlap_prev_tokens
            if current and current_tokens + piece_tokens > pack_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(chunk)
            current_tokens += piece_tokens
        if current:
            groups.append(current)
    return groups


def export_cpt(session: Session, output_dir: Path, *, include_images: bool = False,
                pack_tokens: int | None = None, include_ontology: bool = True) -> dict[str, int]:
    """`pack_tokens`: if set, concatenate consecutive same-section chunks up to this many tokens per
    row instead of exporting one row per chunk - see `_pack_cpt_chunks`. Leave unset (one row per
    chunk, current default) for exports where per-chunk granularity itself matters.

    `include_ontology`: also append the lithium-ion-relevant EMMO/BattINFO definition rows from
    `ontology.lithium_cpt` (materials, electrodes/electrolytes, battery types) to the train split - see
    that module's docstring for what's in scope and why. These carry no source `Document` (attributed to
    the ontology itself, CC-BY-4.0, via `AttributionManifest.record_external`), so they're always put in
    `train`, never held out to `eval`: there's no paper-level split to inherit, and holding out a handful
    of definition sentences would not test anything an eval split is meant to test.
    """
    counts: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    attribution = AttributionManifest()
    all_chunks = session.scalars(select(Chunk)).all()
    eligible = []
    for chunk in all_chunks:
        doc = _get_doc(session, chunk.doc_id)
        if doc and doc.split and not doc.blacklisted and chunk.text.strip():
            eligible.append(chunk)
    groups = _pack_cpt_chunks(eligible, pack_tokens) if pack_tokens else [[c] for c in eligible]

    for group in groups:
        doc = _get_doc(session, group[0].doc_id)
        text = "\n\n".join(_cpt_text(c) for c in group)
        row: dict[str, Any] = {"text": text}
        if include_images:
            images: list[str] = []
            seen: set[str] = set()
            for c in group:
                for path in resolve_chunk_image_paths(session, c.doc_id, c.images):
                    if path not in seen:
                        seen.add(path)
                        images.append(path)
            if images:
                row["images"] = images
        attribution.record(doc.split, doc, len(counts[doc.split]))
        counts[doc.split].append(row)

    if include_ontology:
        from batterygemma.ontology.lithium_cpt import generate_ontology_cpt_rows

        for ontology_row in generate_ontology_cpt_rows():
            attribution.record_external(
                "train", f"ontology:{ontology_row['domain']}", license="CC-BY-4.0",
                title="EMMO/BattINFO domain ontology (chemical-substance, electrochemistry, battery)",
                row_index=len(counts["train"]),
            )
            counts["train"].append({"text": ontology_row["text"]})

    attribution.write(output_dir, "cpt")
    return {split: _write_jsonl(output_dir / f"cpt_{split}.jsonl", rows) for split, rows in counts.items()}


def export_sft(session: Session, output_dir: Path, *, include_images: bool = False) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    attribution = AttributionManifest()

    for qa in session.scalars(select(QA).where(QA.status == "accepted")).all():
        doc = _get_doc(session, qa.doc_ids[0]) if qa.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        messages = [{"role": "system", "content": qa.system or SYSTEM_PROMPT}, *qa.turns]
        row: dict[str, Any] = {"messages": messages}
        if include_images and (images := _row_images(session, qa.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)

    for neg in session.scalars(select(Negative).where(Negative.status == "generated")).all():
        doc = _get_doc(session, neg.doc_ids[0]) if neg.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": neg.prompt},
                {"role": "assistant", "content": neg.expert_response},
            ]
        }
        if include_images and (images := _row_images(session, neg.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)

    attribution.write(output_dir, "sft")
    return {split: _write_jsonl(output_dir / f"sft_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


def export_dpo(session: Session, output_dir: Path, *, include_images: bool = False) -> dict[str, int]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    attribution = AttributionManifest()
    for pair in session.scalars(select(DPOPair).where(DPOPair.status == "generated")).all():
        doc = _get_doc(session, pair.doc_ids[0]) if pair.doc_ids else None
        if not doc or not doc.split or doc.blacklisted:
            continue
        row: dict[str, Any] = {"prompt": pair.prompt, "chosen": pair.chosen, "rejected": pair.rejected}
        if include_images and (images := _row_images(session, pair.chunk_ids)):
            row["images"] = images
        attribution.record(doc.split, doc, len(rows_by_split[doc.split]))
        rows_by_split[doc.split].append(row)
    attribution.write(output_dir, "dpo")
    return {split: _write_jsonl(output_dir / f"dpo_{split}.jsonl", rows) for split, rows in rows_by_split.items()}


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
