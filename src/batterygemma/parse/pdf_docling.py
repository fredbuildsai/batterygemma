"""Parse born-digital PDFs with Docling into the same section structure the JATS parser produces.

Docling handles layout (reading order, columns, running headers/footers, tables); this module maps its
document items onto `Section`s: section headers open a new section path, body text becomes paragraphs,
tables become markdown with their captions, figure captions are kept, and page furniture, footnotes,
reference lists and boilerplate sections (acknowledgements, competing interests, ...) are dropped.

PDF-specific noise that JATS markup avoids is filtered at paragraph level:
- bibliographies Docling does not label as references (e.g. under a page-number heading, as list items);
- front matter: author lists with affiliation numbers, "Received/Accepted" lines, journal citation lines;
- licence boilerplate ("Open Access This article is licensed under a Creative Commons ...");
- citation numbers flattened from superscripts, where unambiguous: right after a sentence end
  ("observations. 81") or at the very start of a paragraph ("67 These results ..."). Docling does not expose
  superscript formatting, so mid-sentence numbers ("study 17 showed") are left alone rather than risk
  deleting real quantities ("cycled 17 times").
"""

import re
from pathlib import Path
from typing import Any

from batterygemma.parse.clean import clean_text
from batterygemma.parse.jats import ParsedDocument, Section, classify_section, is_boilerplate_section

SKIP_LABELS = {"page_header", "page_footer", "footnote", "reference", "document_index", "caption", "marker"}
TEXT_LABELS = {"text", "paragraph", "list_item", "code", "formula"}
FIGURE_LABELS = {"picture", "chart"}
REFERENCE_HEADINGS = {"references", "reference", "bibliography", "literature cited", "notes and references"}
MAX_REFERENCE_ENTRY_CHARS = 600
MIN_CITATIONS_FOR_REFERENCE_SECTION = 3
MAX_AUTHOR_LIST_CHARS = 400
MAX_CITATION_LINE_CHARS = 250

_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-zA-Z'-]+,\s*(?:[A-Z]\.\s*-?)+")  # "Broussely, M." / "Kee, R. J."
_CITATION_TAIL = re.compile(
    r"\(\d{4}[a-z]?\)\.?\s*$"  # "... (2011)."
    r"|\b\d{1,5}\s*[-–]\s*\d{1,5}\s*\(\d{4}\)"  # "5109 - 5114 (2011)"
    r"|\bdoi:\s*10\.|https?://doi\.org/10\.",
    re.IGNORECASE,
)
_FRONT_MATTER = re.compile(
    r"^(received|accepted|revised|published(\s+online)?|keywords?|e-?mail|correspondence|corresponding author)\b"
    r"|©|\bthe author\(s\)",
    re.IGNORECASE,
)
_LICENCE_BOILERPLATE = re.compile(
    r"open access this article|creative commons|this article is (?:licensed|distributed) under"
    r"|to view a copy of this licen[cs]e|permits unrestricted use",
    re.IGNORECASE,
)
_DOI_LINE = re.compile(r"\bdoi:\s*10\.\d{4,}", re.IGNORECASE)  # "npj Computational Materials (2018) 4:15; doi:10.1038/..."
# "Tianyu Li 1,3 · Xiao-Zi Yuan 1 · ..." or "Aiping Wang 1,2, Sanket Kadam 3, ... and Yue Qi 3"
_AUTHOR_LIST = re.compile(
    r"^(?:(?:and\s+)?[A-Z][A-Za-z'.-]*(?:[ -][A-Z][A-Za-z'.-]*)+\s*\d+(?:\s*,\s*\d+)*\*?\s*(?:[·,]\s*)?)+$"
)
_SENTENCE_END_CITATION = re.compile(r"(?<=[.!?])\s+\d{1,3}(?:\s*[,–-]\s*\d{1,3})*(?=\s+[A-Z]|\s*$)")
_LEADING_CITATION = re.compile(r"^\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s+(?=[A-Z][a-z])")


def build_converter(
    *, device: str = "mps", do_ocr: bool = False, formula_enrichment: bool = False, num_threads: int = 4
) -> Any:
    """Docling PDF converter. OCR is off by default: open-access papers are born-digital PDFs."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = do_ocr
    options.do_table_structure = True
    options.do_formula_enrichment = formula_enrichment
    options.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice(device))
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def is_citation_entry(text: str) -> bool:
    return (
        len(text) <= MAX_REFERENCE_ENTRY_CHARS
        and bool(_AUTHOR_INITIALS.search(text))
        and bool(_CITATION_TAIL.search(text))
    )


def is_author_list(text: str) -> bool:
    return len(text) <= MAX_AUTHOR_LIST_CHARS and any(ch.isdigit() for ch in text) and bool(_AUTHOR_LIST.match(text))


def is_front_matter(text: str) -> bool:
    return (
        bool(_FRONT_MATTER.search(text))
        or bool(_LICENCE_BOILERPLATE.search(text))
        or (len(text) <= MAX_CITATION_LINE_CHARS and bool(_DOI_LINE.search(text)))
        or is_author_list(text)
    )


def strip_flattened_citations(text: str) -> str:
    text = _LEADING_CITATION.sub("", text)
    return _SENTENCE_END_CITATION.sub("", text).strip()


def _normalized_heading(text: str) -> str:
    return text.lower().strip(" .:0123456789")


def docling_to_parsed(document: Any) -> ParsedDocument:
    title: str | None = None
    abstract_parts: list[str] = []
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current = Section(path=["Body"], section_type="other")
    skipping = False
    in_abstract = False

    def flush() -> None:
        citations = [p for p in current.paragraphs if is_citation_entry(p)]
        if len(citations) >= MIN_CITATIONS_FOR_REFERENCE_SECTION and len(citations) * 2 >= len(current.paragraphs):
            return  # an unlabeled bibliography
        current.paragraphs = [p for p in current.paragraphs if not is_citation_entry(p)]
        if current.paragraphs or current.captions:
            sections.append(current)

    for item, _depth in document.iterate_items():
        label = getattr(getattr(item, "label", None), "value", None)
        if label is None or label in SKIP_LABELS:
            continue
        if label == "title":
            title = title or clean_text(item.text) or None
            continue
        if label == "section_header":
            heading = clean_text(item.text)
            level = getattr(item, "level", 1) or 1
            flush()
            heading_stack = [(lvl, text) for lvl, text in heading_stack if lvl < level] + [(level, heading)]
            path = [text for _, text in heading_stack]
            normalized = _normalized_heading(heading)
            skipping = normalized in REFERENCE_HEADINGS or is_boilerplate_section(heading, None)
            in_abstract = normalized == "abstract"
            current = Section(path=path, section_type=classify_section(path))
            continue
        if skipping:
            continue
        if label in TEXT_LABELS:
            text = clean_text(getattr(item, "text", "") or "")
            if not text or is_front_matter(text):
                continue
            if not is_citation_entry(text):
                text = strip_flattened_citations(text)
            if text:
                (abstract_parts if in_abstract else current.paragraphs).append(text)
        elif label == "table":
            caption = clean_text(item.caption_text(document) or "")
            current.captions.append(f"{caption}\n{item.export_to_markdown(document)}".strip())
        elif label in FIGURE_LABELS:
            if caption := clean_text(item.caption_text(document) or ""):
                current.captions.append(caption)
    flush()
    return ParsedDocument(title=title, abstract=" ".join(abstract_parts) or None, sections=sections)


def parse_pdf(path: Path, converter: Any) -> ParsedDocument:
    return docling_to_parsed(converter.convert(str(path)).document)
