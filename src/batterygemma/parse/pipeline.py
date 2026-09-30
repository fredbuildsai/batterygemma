"""Turn a fetched document's full text into stored, quality-flagged chunks (idempotent per document).

JATS XML is preferred when present; otherwise the PDF is parsed with the injected PDF parser (Docling).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from batterygemma.db.models import Chunk, Document
from batterygemma.parse.chunk import TokenCounter, chunk_sections, strip_overlap_prefix
from batterygemma.parse.clean import garble_ratio
from batterygemma.parse.jats import ParsedDocument, Section, parse_jats

PdfParser = Callable[[Path], ParsedDocument]


def chunk_id(doc_id: str, section_index: int, order_in_section: int) -> str:
    return f"{doc_id}#s{section_index:02d}-c{order_in_section:02d}"


def _write_chunk_drafts(session: Session, doc: Document, drafts: list[Any], chunking: dict[str, Any],
                         method: str) -> int:
    """Shared tail of both chunking paths: replace `doc`'s existing chunks with freshly chunked
    `drafts` (from `chunk_sections`), update its status. Returns the chunk count."""
    session.execute(delete(Chunk).where(Chunk.doc_id == doc.doc_id))
    for position, draft in enumerate(drafts):
        ratio = garble_ratio(draft.text)
        session.add(
            Chunk(
                chunk_id=chunk_id(doc.doc_id, draft.section_index, draft.order),
                doc_id=doc.doc_id,
                section_path=draft.section_path,
                section_type=draft.section_type,
                order=position,
                tokens=draft.tokens,
                overlap_prev_tokens=draft.overlap_prev_tokens,
                text=draft.text,
                captions=draft.captions,
                images=draft.images,
                quality={"garble_ratio": round(ratio, 3),
                         "flags": ["garbled"] if ratio > chunking["max_garble_ratio"] else []},
            )
        )

    if drafts:
        doc.status, doc.status_reason = "chunked", f"parse:{method}:{len(drafts)}_chunks"
    else:
        doc.status_reason = f"parse:{method}:no_body_text"
    return len(drafts)


def parse_and_chunk(
    session: Session,
    doc: Document,
    count_tokens: TokenCounter,
    chunking: dict[str, Any],
    pdf_parser: PdfParser | None = None,
) -> int:
    """Parse the document's full-text file into chunks, replacing any previous chunks. Returns the chunk count."""
    xml_file = next((f for f in doc.files if f.kind == "xml"), None)
    pdf_file = next((f for f in doc.files if f.kind == "pdf"), None)
    if xml_file is not None:
        parsed, method = parse_jats(Path(xml_file.path).read_bytes()), "jats"
    elif pdf_file is not None and pdf_parser is not None:
        parsed, method = pdf_parser(Path(pdf_file.path)), "docling"
    else:
        doc.status_reason = "parse:no_parsable_file"
        return 0

    if not doc.abstract and parsed.abstract:
        doc.abstract = parsed.abstract

    drafts = chunk_sections(
        parsed.sections,
        count_tokens,
        target_tokens=chunking["target_tokens"],
        max_tokens=chunking["max_tokens"],
        overlap_tokens=chunking["overlap_tokens"],
    )
    return _write_chunk_drafts(session, doc, drafts, chunking, method)


def reconstruct_sections_from_chunks(chunks: list[Chunk]) -> list[Section]:
    """Rebuild `Section` objects (path, paragraphs, captions/images) from a document's *existing*
    chunk rows, well enough to re-run `chunk_sections` at a different `target_tokens`/`max_tokens`
    without re-parsing the original PDF/XML file at all.

    Each chunk after the first in its section stores an overlap prefix that must be stripped first
    (see `strip_overlap_prefix`) - failing to strip it would duplicate text at every old chunk
    boundary in the reconstructed section. Chunks are grouped by their full `section_path` (distinct
    subsections stay distinct Section objects, matching how they were originally parsed) and ordered
    within a group by `order`, which is a global, monotonically increasing sequence number for the
    whole document - confirmed against real chunk_ids - so the resulting section order (by each
    group's minimum `order`) reproduces the document's real original section sequence.

    Paragraph boundaries are approximated: the original chunking joined paragraph-level units with
    "\\n\\n" before splitting into chunks, so re-splitting the reconstructed text on "\\n\\n" recovers
    something functionally equivalent for re-chunking purposes, even though exact original
    paragraph-by-paragraph fidelity isn't guaranteed to survive a round trip through chunking.
    Captions/images (stored only on each section's first chunk originally) are carried over from
    each group's first chunk - `caption_images` is reconstructed as a single group holding all of
    that chunk's flattened image filenames, since the original per-caption alignment isn't preserved
    once flattened into `Chunk.images`.
    """
    groups: dict[tuple[str, ...], list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(tuple(chunk.section_path), []).append(chunk)

    ordered_groups: list[tuple[int, Section]] = []
    for path, group_chunks in groups.items():
        group_chunks.sort(key=lambda c: c.order)
        pieces = [strip_overlap_prefix(c.text, c.overlap_prev_tokens) for c in group_chunks]
        paragraphs = [p for p in "\n\n".join(pieces).split("\n\n") if p.strip()]
        first = group_chunks[0]
        ordered_groups.append((first.order, Section(
            path=list(path),
            section_type=first.section_type or "other",
            paragraphs=paragraphs,
            captions=list(first.captions or []),
            caption_images=[list(first.images)] if first.images else [],
        )))
    ordered_groups.sort(key=lambda t: t[0])
    return [section for _, section in ordered_groups]


def rechunk_from_existing_chunks(session: Session, doc: Document, count_tokens: TokenCounter,
                                  chunking: dict[str, Any]) -> int:
    """Re-chunk a document at new `target_tokens`/`max_tokens` using its *existing* chunks as the
    source text, instead of re-parsing the original PDF/XML file - avoids re-running Docling (slow)
    when only the chunk size changed, not the underlying extraction. See
    `reconstruct_sections_from_chunks` for how the source text is recovered. Only usable for a
    document that already has chunks; returns 0 and leaves the document untouched otherwise."""
    existing = list(session.scalars(select(Chunk).where(Chunk.doc_id == doc.doc_id)))
    if not existing:
        return 0
    sections = reconstruct_sections_from_chunks(existing)
    drafts = chunk_sections(
        sections,
        count_tokens,
        target_tokens=chunking["target_tokens"],
        max_tokens=chunking["max_tokens"],
        overlap_tokens=chunking["overlap_tokens"],
    )
    return _write_chunk_drafts(session, doc, drafts, chunking, "rechunk_from_existing")
