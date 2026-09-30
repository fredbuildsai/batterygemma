"""Section-aware chunking with a small sentence overlap and Gemma token counts.

Chunks never cross section boundaries (the previous trial's chunks spanned chapters and repeated about half
of the previous chunk). Paragraphs are packed up to `target_tokens`; a paragraph longer than `max_tokens` is
split by sentences. Each chunk after the first in a section starts with trailing sentences of the previous
chunk totalling at most `overlap_tokens`.
"""

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from batterygemma.parse.jats import Section

TokenCounter = Callable[[str], int]
GEMMA_REPO_DIR = "models--unsloth--gemma-4-E2B-it"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


@dataclass
class ChunkDraft:
    section_index: int
    order: int
    section_path: list[str]
    section_type: str
    text: str
    tokens: int
    overlap_prev_tokens: int
    captions: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


def load_token_counter() -> TokenCounter:
    """Gemma 4 tokenizer from the local Hugging Face cache; falls back to a word-based estimate."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(next((hub / GEMMA_REPO_DIR / "snapshots").glob("*/tokenizer.json"))))
    except (ImportError, StopIteration, OSError, ValueError):
        return lambda text: max(1, round(len(text.split()) * 1.35))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def strip_overlap_prefix(text: str, overlap_prev_tokens: int) -> str:
    """The inverse of what `chunk_sections` prepends: a chunk after the first in a section stores
    `f"{overlap_sentences}\\n\\n{body_text}"`, where `overlap_sentences` is exactly
    `overlap_prev_tokens` worth of trailing sentences from the previous chunk, joined by a single
    space. Splitting on the first "\\n\\n" reliably separates them, since `overlap_sentences` itself
    never contains "\\n\\n" (verified against real chunks: it's built by `' '.join(tail)`, never
    `'\\n\\n'.join`). Returns `text` unchanged when there's no overlap to strip."""
    if overlap_prev_tokens <= 0:
        return text
    return text.split("\n\n", 1)[-1]


def _flat_images(section: Section) -> list[str]:
    seen: dict[str, None] = {}
    for names in section.caption_images:
        for name in names:
            seen.setdefault(name, None)
    return list(seen)


def chunk_sections(
    sections: Sequence[Section],
    count_tokens: TokenCounter,
    *,
    target_tokens: int = 600,
    max_tokens: int = 900,
    overlap_tokens: int = 60,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for section_index, section in enumerate(sections):
        units: list[str] = []
        for paragraph in section.paragraphs:
            units.extend([paragraph] if count_tokens(paragraph) <= max_tokens else split_sentences(paragraph))

        if not units:
            if section.captions:
                text = "\n\n".join(section.captions)
                drafts.append(ChunkDraft(section_index, 0, section.path, section.section_type, text,
                                         count_tokens(text), 0, list(section.captions), _flat_images(section)))
            continue

        body: list[str] = []
        body_tokens = 0
        overlap: list[str] = []
        overlap_count = 0
        order = 0

        def flush() -> None:
            nonlocal body, body_tokens, overlap, overlap_count, order
            body_text = "\n\n".join(body)
            text = f"{' '.join(overlap)}\n\n{body_text}" if overlap else body_text
            drafts.append(ChunkDraft(section_index, order, section.path, section.section_type, text,
                                     count_tokens(text), overlap_count,
                                     list(section.captions) if order == 0 else [],
                                     _flat_images(section) if order == 0 else []))
            tail: list[str] = []
            tail_tokens = 0
            for sentence in reversed(split_sentences(body_text)):
                sentence_tokens = count_tokens(sentence)
                if tail_tokens + sentence_tokens > overlap_tokens:
                    break
                tail.insert(0, sentence)
                tail_tokens += sentence_tokens
            overlap, overlap_count = tail, tail_tokens
            body, body_tokens = [], 0
            order += 1

        for unit in units:
            unit_tokens = count_tokens(unit)
            if body and body_tokens + unit_tokens > target_tokens:
                flush()
            body.append(unit)
            body_tokens += unit_tokens
        flush()
    return drafts
