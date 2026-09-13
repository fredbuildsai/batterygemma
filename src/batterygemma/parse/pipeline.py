"""Turn a fetched document's full text into stored, quality-flagged chunks (idempotent per document).

JATS XML is preferred when present; otherwise the PDF is parsed with the injected PDF parser (Docling).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from batterygemma.db.models import Chunk, Document
from batterygemma.parse.chunk import TokenCounter, chunk_sections
from batterygemma.parse.clean import garble_ratio
from batterygemma.parse.jats import ParsedDocument, parse_jats

PdfParser = Callable[[Path], ParsedDocument]


def chunk_id(doc_id: str, section_index: int, order_in_section: int) -> str:
    return f"{doc_id}#s{section_index:02d}-c{order_in_section:02d}"


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
                quality={"garble_ratio": round(ratio, 3),
                         "flags": ["garbled"] if ratio > chunking["max_garble_ratio"] else []},
            )
        )

    if drafts:
        doc.status, doc.status_reason = "chunked", f"parse:{method}:{len(drafts)}_chunks"
    else:
        doc.status_reason = f"parse:{method}:no_body_text"
    return len(drafts)
