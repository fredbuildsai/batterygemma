"""Parse JATS XML (PMC / Europe PMC open-access subset) into ordered, section-aware text.

Keeps section titles as a path, paragraphs, display/inline formulas (as TeX when available) and figure/table
captions (tables flattened to rows). Drops bibliographic citation markers (including the separators between
consecutive citations, e.g. the dash in a superscript range), boilerplate sections such as acknowledgements or
competing interests, the reference list and back matter.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from batterygemma.parse.clean import clean_text

CITATION_REF_TYPES = {"bibr", "fn"}
FLOAT_TAGS = {"fig", "table-wrap", "fig-group"}
FORMULA_TAGS = {"inline-formula", "disp-formula"}
PARAGRAPH_TAGS = {"p", "disp-formula", "list", "disp-quote", "statement"}
MAX_TABLE_ROWS = 30
_CITATION_SEPARATOR = re.compile(r"^[\s,;–—-]*$")

BOILERPLATE_TITLE_KEYWORDS = (
    "acknowledg", "author contribution", "competing interest", "conflict of interest", "conflicts of interest",
    "declaration of competing", "data availability", "code availability", "funding", "additional information",
    "supplementary", "supporting information", "ethics", "abbreviations", "peer review", "publisher's note",
    "reporting summary", "associated content", "author information", "credit authorship",
    "ethical standards", "open access", "rights and permissions",
)
BOILERPLATE_SEC_TYPE_KEYWORDS = ("coi", "contribution", "availability", "funding", "supplementary", "ack", "ethics")

SECTION_TYPES = [
    ("introduction", ("introduction", "background")),
    ("methods", ("method", "experimental", "materials and", "synthesis", "computational detail", "characterization")),
    ("results", ("result",)),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "summary", "outlook", "perspective")),
]


@dataclass
class Section:
    path: list[str]
    section_type: str
    paragraphs: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    caption_images: list[list[str]] = field(default_factory=list)  # caption_images[i] aligns with captions[i]:
                                                                    # the figure filename(s) that caption refers
                                                                    # to (resolved to actual files by images.py)


@dataclass
class ParsedDocument:
    title: str | None
    abstract: str | None
    sections: list[Section]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(root: ET.Element, name: str) -> ET.Element | None:
    return next((el for el in root.iter() if _local(el.tag) == name), None)


def _is_citation(node: ET.Element | None) -> bool:
    return node is not None and _local(node.tag) == "xref" and node.get("ref-type") in CITATION_REF_TYPES


def classify_section(path: list[str]) -> str:
    joined = " ".join(path).lower()
    for section_type, keywords in SECTION_TYPES:
        if any(k in joined for k in keywords):
            return section_type
    return "other"


def is_boilerplate_section(title: str, sec_type: str | None) -> bool:
    lowered_title = title.lower()
    lowered_type = (sec_type or "").lower()
    return any(k in lowered_title for k in BOILERPLATE_TITLE_KEYWORDS) or any(
        k in lowered_type for k in BOILERPLATE_SEC_TYPE_KEYWORDS
    )


def _collect(node: ET.Element, parts: list[str]) -> None:
    tag = _local(node.tag)
    if _is_citation(node) or tag in FLOAT_TAGS:
        return
    if tag in FORMULA_TAGS:
        tex = next((n.text for n in node.iter() if _local(n.tag) == "tex-math" and n.text), None)
        parts.append(f" ${tex.strip()}$ " if tex else " " + " ".join(t.strip() for t in node.itertext() if t.strip()) + " ")
        return
    if node.text:
        parts.append(node.text)
    children = list(node)
    for index, child in enumerate(children):
        _collect(child, parts)
        if not child.tail:
            continue
        following = children[index + 1] if index + 1 < len(children) else None
        if _is_citation(child) and _is_citation(following) and _CITATION_SEPARATOR.match(child.tail):
            continue  # separator between two removed citations, e.g. "1–3" or "1, 2"
        parts.append(child.tail)


def element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    parts: list[str] = []
    _collect(node, parts)
    return clean_text(" ".join("".join(parts).split()))


def _href(el: ET.Element) -> str | None:
    return next((v for k, v in el.attrib.items() if k.endswith("href")), None)


def caption_images(float_el: ET.Element) -> list[str]:
    """Bare figure filenames (e.g. "ncomms8898-f1.jpg") referenced by this float's own graphic elements -
    not its descendants' (a table-wrap's own <graphic>, not one nested inside an unrelated child)."""
    return [
        name.rsplit("/", 1)[-1]
        for g in float_el.iter()
        if _local(g.tag) in {"graphic", "inline-graphic"} and (href := _href(g))
        for name in [href]
    ]


def caption_text(float_el: ET.Element) -> str:
    label = element_text(next((c for c in float_el if _local(c.tag) == "label"), None))
    caption = element_text(next((c for c in float_el if _local(c.tag) == "caption"), None))
    text = f"{label}: {caption}" if label and caption else label or caption
    rows = [
        " | ".join(element_text(cell) for cell in row if _local(cell.tag) in {"td", "th"})
        for row in float_el.iter()
        if _local(row.tag) == "tr"
    ]
    if rows:
        text += "\n" + "\n".join(rows[:MAX_TABLE_ROWS])
    return text.strip()


def _section_title(sec: ET.Element) -> str:
    return element_text(next((c for c in sec if _local(c.tag) == "title"), None))


def _visit(sec: ET.Element, path: list[str], sections: list[Section]) -> None:
    title = _section_title(sec)
    current_path = path + [title] if title else path
    section_path = current_path or ["Body"]
    section = Section(path=section_path, section_type=classify_section(section_path))

    def flush() -> None:
        nonlocal section
        if section.paragraphs or section.captions:
            sections.append(section)
        section = Section(path=section_path, section_type=classify_section(section_path))

    for child in sec:
        tag = _local(child.tag)
        if tag == "sec":
            if is_boilerplate_section(_section_title(child), child.get("sec-type")):
                continue
            flush()
            _visit(child, current_path, sections)
        elif tag in PARAGRAPH_TAGS:
            if text := element_text(child):
                section.paragraphs.append(text)
            for f in child.iter():
                if _local(f.tag) in FLOAT_TAGS:
                    section.captions.append(caption_text(f))
                    section.caption_images.append(caption_images(f))
        elif tag in FLOAT_TAGS:
            section.captions.append(caption_text(child))
            section.caption_images.append(caption_images(child))
    flush()


def parse_jats(xml: bytes) -> ParsedDocument:
    root = ET.fromstring(xml)
    meta = _find(root, "article-meta")
    title = element_text(_find(meta, "article-title")) if meta is not None else None
    abstract_el = _find(meta, "abstract") if meta is not None else None
    abstract = None
    if abstract_el is not None:
        abstract = " ".join(t for p in abstract_el.iter() if _local(p.tag) == "p" and (t := element_text(p))) or None
    sections: list[Section] = []
    if (body := _find(root, "body")) is not None:
        _visit(body, [], sections)
    return ParsedDocument(title=title or None, abstract=abstract, sections=sections)
