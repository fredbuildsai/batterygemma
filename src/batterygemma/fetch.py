"""Download full text for accepted documents.

Order of preference:
1. JATS XML from Europe PMC's open-access subset, when Europe PMC reports an allowed license for that copy.
2. The document's own licensed PDF link (`pdf_url`, which `sources.store` keeps paired with the license it
   came with).
3. Unpaywall, queried by DOI, as a fallback when (2) fails. Unpaywall indexes green-OA mirrors (university
   and subject repositories) in addition to the publisher's own copy; those mirrors are often served with no
   bot protection at all even when the publisher's site blocks scraping (confirmed: an IOP/ECS article whose
   own site returns a Radware CAPTCHA has a plain, unprotected PDF on its author's institutional repository).
   Repository mirrors are tried before the publisher's own listing, since the latter duplicates what (2)
   already tried. This also covers documents with no cached `pdf_url` at all ("no_url").

Bot challenges (Cloudflare, Radware, and similar) are recorded and skipped, never bypassed — see
`sources.base.PoliteClient` for what is detected. Successful downloads are stored under data/raw/<source>/
with a sha256 and a `files` row, and the document moves to status "fetched".
"""

import hashlib
import re
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from batterygemma.db.models import Document, File
from batterygemma.screen.license import ALLOWED, evaluate_license, normalize_license
from batterygemma.sources.base import BlockedByBotProtection, PoliteClient

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
UNPAYWALL_API = "https://api.unpaywall.org/v2/{doi}"
MAX_BYTES = 50 * 1024 * 1024


def _safe_name(external_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", external_id)


def find_open_access_pmcid(client: PoliteClient, doi: str, allow: list[str], flag: list[str]) -> str | None:
    response = client.get(
        EUROPEPMC_SEARCH, params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 5}
    )
    for result in response.json().get("resultList", {}).get("result", []):
        license_id = normalize_license(result.get("license"))
        if result.get("pmcid") and result.get("isOpenAccess") == "Y" and evaluate_license(license_id, allow, flag) == ALLOWED:
            return result["pmcid"]
    return None


def unpaywall_candidate_urls(client: PoliteClient, doi: str, contact_email: str, exclude: set[str]) -> list[str]:
    """Candidate full-text URLs from Unpaywall for `doi`, excluding any already tried.

    Repository (green OA) copies are ordered before the publisher's own copy: the publisher URL is usually
    the one already attempted and that failed, while a repository mirror is frequently unprotected.
    """
    email = contact_email or "batterygemma@example.org"  # Unpaywall requires *a* contact address, even a placeholder
    try:
        response = client.get(UNPAYWALL_API.format(doi=doi), params={"email": email})
    except (BlockedByBotProtection, httpx.HTTPStatusError):
        return []  # unknown DOI (404) or Unpaywall itself unavailable; not fatal to the overall fetch
    locations = sorted(response.json().get("oa_locations") or [], key=lambda loc: loc.get("host_type") != "repository")
    urls: list[str] = []
    for loc in locations:
        url = loc.get("url_for_pdf") or loc.get("url")
        if url and url not in exclude and url not in urls:
            urls.append(url)
    return urls


def _store(session: Session, doc: Document, raw_dir: Path, kind: str, content: bytes) -> File:
    directory = raw_dir / doc.source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(doc.external_id)}.{kind}"
    path.write_bytes(content)
    file = File(doc_id=doc.doc_id, kind=kind, path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content))
    session.add(file)
    return file


def _try_pdf_url(client: PoliteClient, url: str) -> tuple[str, bytes | None, str | None]:
    """One attempt at downloading `url` as a PDF. Returns (outcome, content, detail)."""
    try:
        response = client.get(url)
    except BlockedByBotProtection:
        return "blocked", None, None
    except httpx.HTTPStatusError as exc:
        return f"http_{exc.response.status_code}", None, None
    if len(response.content) > MAX_BYTES:
        return "too_large", None, None
    is_pdf = response.content.startswith(b"%PDF") or "pdf" in response.headers.get("content-type", "")
    if not is_pdf:
        return "not_pdf", None, response.headers.get("content-type", "?")[:40]
    return "pdf", response.content, None


def _reason(outcome: str, detail: str | None) -> str:
    if outcome == "blocked":
        return "fetch:bot_protection"
    if outcome == "no_url":
        return "fetch:no_fulltext_url"
    return f"fetch:{outcome}:{detail}" if detail else f"fetch:{outcome}"


def fetch_document(
    session: Session,
    doc: Document,
    client: PoliteClient,
    raw_dir: Path,
    *,
    allow: list[str],
    flag: list[str],
    contact_email: str = "",
) -> str:
    """Fetch one document's full text.

    Returns the outcome: xml, pdf, pdf_unpaywall, not_pdf, too_large, no_url, blocked, or http_<code> (the
    last four reflect whichever attempt got furthest, when every attempt fails).
    """
    try:
        if doc.doi and (pmcid := find_open_access_pmcid(client, doc.doi, allow, flag)):
            url = EUROPEPMC_XML.format(pmcid=pmcid)
            response = client.get(url)
            if response.content.lstrip().startswith(b"<"):
                _store(session, doc, raw_dir, "xml", response.content)
                doc.xml_url, doc.status, doc.status_reason = url, "fetched", f"fetch:xml:{pmcid}"
                return "xml"
    except BlockedByBotProtection:
        doc.status_reason = "fetch:bot_protection"
        return "blocked"
    except httpx.HTTPStatusError as exc:
        doc.status_reason = f"fetch:http_{exc.response.status_code}"
        return f"http_{exc.response.status_code}"

    tried_urls: set[str] = set()
    best_outcome, best_detail = "no_url", None

    if doc.pdf_url:
        tried_urls.add(doc.pdf_url)
        outcome, content, detail = _try_pdf_url(client, doc.pdf_url)
        if outcome == "pdf":
            _store(session, doc, raw_dir, "pdf", content)
            doc.status, doc.status_reason = "fetched", "fetch:pdf"
            return "pdf"
        best_outcome, best_detail = outcome, detail

    if doc.doi:
        for url in unpaywall_candidate_urls(client, doc.doi, contact_email, tried_urls):
            tried_urls.add(url)
            outcome, content, detail = _try_pdf_url(client, url)
            if outcome == "pdf":
                _store(session, doc, raw_dir, "pdf", content)
                doc.pdf_url = url  # replace the stale/blocked link with the one that actually worked
                doc.status, doc.status_reason = "fetched", "fetch:pdf:unpaywall"
                return "pdf_unpaywall"
            if best_outcome == "no_url":
                best_outcome, best_detail = outcome, detail

    doc.status_reason = _reason(best_outcome, best_detail)
    return best_outcome
