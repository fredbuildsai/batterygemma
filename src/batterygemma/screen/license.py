"""License normalization and the public-release license gate.

Sources report licenses as URLs (Crossref, arXiv), short codes (OpenAlex "cc-by") or free text
(Europe PMC "cc by-nc-nd"). Everything is normalized to a canonical id such as "CC-BY-4.0",
"CC-BY" (version unknown), "CC0-1.0" or "public-domain"; unrecognized values become "other:<raw>".
Gate decisions compare license *families* (the id without its version).
"""

import re

_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)(?:/(\d\.\d))?", re.IGNORECASE)
_CC_CODE = re.compile(r"cc-(by(?:-(?:nc|sa|nd))*)(?:-(\d\.\d))?")
_VERSION_SUFFIX = re.compile(r"-\d\.\d$")

ALLOWED = "allowed"
FLAGGED = "flagged"
REJECTED = "rejected"
UNKNOWN = "unknown"


def normalize_license(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip().lower()

    if match := _CC_URL.search(text):
        kind, code, version = match.groups()
        if kind == "publicdomain":
            return "CC0-1.0" if code == "zero" else "public-domain"
        return f"CC-{code.upper()}" + (f"-{version}" if version else "")
    if "arxiv.org/licenses/nonexclusive-distrib" in text:
        return "arXiv-nonexclusive-1.0"

    token = re.sub(r"[\s_]+", "-", text)  # "cc by-nc-nd 4.0" -> "cc-by-nc-nd-4.0"
    if token.startswith("cc0") or token == "cc-zero":
        return "CC0-1.0"
    if token in {"public-domain", "publicdomain", "pd", "public-domain-mark"}:
        return "public-domain"
    if match := _CC_CODE.fullmatch(token):
        code, version = match.groups()
        return f"CC-{code.upper()}" + (f"-{version}" if version else "")
    return f"other:{raw.strip()}"


def license_family(license_id: str) -> str:
    return _VERSION_SUFFIX.sub("", license_id)


def evaluate_license(license_id: str | None, allow: list[str], flag: list[str]) -> str:
    """Return allowed / flagged / rejected / unknown for a normalized license id."""
    if not license_id:
        return UNKNOWN
    family = license_family(license_id)
    if family in allow:
        return ALLOWED
    if family in flag:
        return FLAGGED
    return REJECTED
