"""Text cleaning and a garble metric for extracted scientific text."""

import re
from functools import lru_cache
from pathlib import Path

WORD_LIST = Path("/usr/share/dict/words")

_UNICODE_SPACES = re.compile("[    ]")  # no-break, figure, thin, narrow no-break spaces
_PRIVATE_USE = re.compile(r"[-]")  # PDF symbol-font glyphs such as 
_SOFT_HYPHEN = re.compile("­")  # invisible hyphenation points left in PDF text
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "stor-\ning" -> "storing"
# A PDF ligature glyph extracted as its own token: "af fi nity", "SEI fi lms", "the fi rst".
_LIGATURE_SPLIT = re.compile(
    r"(?:\b(?P<left>[A-Za-z]+) |(?<![A-Za-z]))(?P<lig>ffi|ffl|fi|fl|ff) (?P<right>[a-z][a-z-]*)"
)
_BRACKET_CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")  # [12], [3, 4], [5–7]
_EMPTY_BRACKETS = re.compile(r"\s*[\[(][\s,;–-]*[\])]")  # left behind after removing citation markup
_SPACES = re.compile(r"[ \t\r\f\v]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])(?=\s|$)")
_BLANK_LINES = re.compile(r"\n\s*\n+")

GARBLE_MIN_RUN = 4


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    """Lowercase English word list; empty if unavailable (then ligatures are only joined rightward)."""
    try:
        return frozenset(line.strip().lower() for line in WORD_LIST.read_text(encoding="utf-8").splitlines())
    except OSError:
        return frozenset()


def _is_word(word: str) -> bool:
    words = _words()
    w = word.lower()
    candidates = {w}
    for suffix, replacements in (
        ("ication", ("y",)), ("ically", ("ic",)), ("ally", ("al", "")), ("ies", ("y",)), ("ied", ("y",)),
        ("ing", ("", "e")), ("ly", ("",)), ("ed", ("", "e")), ("es", ("", "e")), ("s", ("",)),
    ):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            candidates.update(w[: -len(suffix)] + r for r in replacements)
    return any(c in words for c in candidates)


def _rejoin_ligature(match: re.Match[str]) -> str:
    left, lig, right = match.group("left"), match.group("lig"), match.group("right")
    head = right.split("-", 1)[0]  # "fi rst-principles" -> check "first"
    if left and _is_word(f"{left}{lig}{head}"):
        return f"{left}{lig}{right}"  # "af fi nity" -> "affinity"
    return f"{left} {lig}{right}" if left else f"{lig}{right}"  # "the fi rst" -> "the first"


def clean_text(text: str) -> str:
    text = _UNICODE_SPACES.sub(" ", text)
    text = _PRIVATE_USE.sub("", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _LIGATURE_SPLIT.sub(_rejoin_ligature, text)
    text = _BRACKET_CITATION.sub("", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def garble_ratio(text: str) -> float:
    """Share of tokens inside runs of >= 4 consecutive single-character tokens.

    Figure-axis soup extracted from PDFs ("0 . 9 H D ) i n / m") is long runs of single characters, whereas
    normal scientific prose only has isolated ones ("4.2 V", "Li +", "5109 - 5114"), which score ~0.
    """
    tokens = text.split()
    if not tokens:
        return 1.0
    in_runs = run = 0
    for token in [*tokens, ""]:  # trailing sentinel closes the last run
        if len(token) == 1:
            run += 1
            continue
        if run >= GARBLE_MIN_RUN:
            in_runs += run
        run = 0
    return in_runs / len(tokens)
