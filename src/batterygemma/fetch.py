"""Download full text for accepted documents.

Order of preference:
1. JATS XML from Europe PMC's open-access subset, when Europe PMC reports an allowed license for that copy.
2. The document's own licensed PDF link (`pdf_url`, which `sources.store` keeps paired with the license it
   came with).
3. CORE (api.core.ac.uk), queried by DOI. CORE aggregates and re-hosts full text harvested from thousands of
   institutional and subject repositories on its own servers - when it has a copy, the download comes from
   core.ac.uk itself, not the publisher's site, so it carries none of the publisher's bot management
   (confirmed live 2026-09-15: a Wiley "aenm" article whose own site 403s has a plain CORE-hosted PDF mirror).
   Works without a key at a low anonymous rate limit; set CORE_API_KEY (free at core.ac.uk/services/api) for
   a much higher limit.
4. Unpaywall, queried by DOI, as a fallback when (2)/(3) fail. Unpaywall indexes green-OA mirrors (university
   and subject repositories) in addition to the publisher's own copy; those mirrors are often served with no
   bot protection at all even when the publisher's site blocks scraping (confirmed: an IOP/ECS article whose
   own site returns a Radware CAPTCHA has a plain, unprotected PDF on its author's institutional repository).
   Repository mirrors are tried before the publisher's own listing, since the latter duplicates what (2)
   already tried. This also covers documents with no cached `pdf_url` at all ("no_url").

A repository/CORE/Unpaywall candidate is often a landing page rather than a direct PDF link. When that
landing page's HTML carries a `citation_pdf_url` <meta> tag - the same tag Google Scholar, Zotero and
Mendeley rely on, published by essentially every DSpace/EPrints/Pure repository specifically so automated
tools can find the PDF - we follow it one extra hop (confirmed live 2026-09-15 against Bristol, Imperial and
TUM repository pages). This is reading metadata the repository publishes for exactly this purpose, not bot
evasion.

Bot challenges (Cloudflare, Radware, and similar) are recorded and skipped, never bypassed - see
`sources.base.PoliteClient` for what is detected. Successful downloads are stored under data/raw/<source>/
with a sha256 and a `files` row, and the document moves to status "fetched".
"""

import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from sqlalchemy.orm import Session

from batterygemma.db.models import Document, File
from batterygemma.screen.license import ALLOWED, evaluate_license, normalize_license
from batterygemma.sources.base import BlockedByBotProtection, PoliteClient

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
UNPAYWALL_API = "https://api.unpaywall.org/v2/{doi}"
CORE_SEARCH_API = "https://api.core.ac.uk/v3/search/works/"
_CITATION_PDF_URL_RE = re.compile(
    rb'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE
)
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
    except (BlockedByBotProtection, httpx.HTTPStatusError, httpx.TransportError):
        return []  # unknown DOI (404) or Unpaywall itself unavailable; not fatal to the overall fetch
    locations = sorted(response.json().get("oa_locations") or [], key=lambda loc: loc.get("host_type") != "repository")
    urls: list[str] = []
    for loc in locations:
        url = loc.get("url_for_pdf") or loc.get("url")
        if url and url not in exclude and url not in urls:
            urls.append(url)
    return urls


def core_candidate_url(client: PoliteClient, doi: str, api_key: str = "") -> str | None:
    """CORE's own re-hosted PDF copy for `doi`, if it has harvested one. See module docstring for why this
    frequently succeeds where the publisher's own site is bot-walled."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        response = client.get(CORE_SEARCH_API, params={"q": f'doi:"{doi}"'}, headers=headers)
    except (BlockedByBotProtection, httpx.HTTPStatusError, httpx.TransportError):
        return None
    results = response.json().get("results") or []
    if not results:
        return None
    return results[0].get("downloadUrl") or None


def _citation_pdf_url(html: bytes, page_url: str) -> str | None:
    """The `citation_pdf_url` <meta> tag from a repository landing page, resolved to an absolute URL.

    This is metadata the repository itself publishes so tools like Google Scholar can find the PDF - see
    the module docstring for why following it is not bot evasion.
    """
    if match := _CITATION_PDF_URL_RE.search(html[:65536]):
        return urljoin(page_url, match.group(1).decode("utf-8", "ignore"))
    return None


def _store(session: Session, doc: Document, raw_dir: Path, kind: str, content: bytes) -> File:
    directory = raw_dir / doc.source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(doc.external_id)}.{kind}"
    path.write_bytes(content)
    file = File(doc_id=doc.doc_id, kind=kind, path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content))
    session.add(file)
    return file


def _try_pdf_url(client: PoliteClient, url: str, *, follow_citation_meta: bool = True) -> tuple[str, bytes | None, str | None]:
    """One attempt at downloading `url` as a PDF. Returns (outcome, content, detail).

    If the response is an HTML landing page (not a PDF), follows a `citation_pdf_url` <meta> tag one hop
    when present - see module docstring for why. Only one hop is taken, so a repository whose own
    citation_pdf_url points back to another landing page still just reports `not_pdf`.
    """
    try:
        response = client.get(url)
    except BlockedByBotProtection:
        return "blocked", None, None
    except httpx.HTTPStatusError as exc:
        return f"http_{exc.response.status_code}", None, None
    except httpx.TransportError:
        return "network_error", None, None
    if len(response.content) > MAX_BYTES:
        return "too_large", None, None
    is_pdf = response.content.startswith(b"%PDF") or "pdf" in response.headers.get("content-type", "")
    if not is_pdf:
        if follow_citation_meta and (pdf_url := _citation_pdf_url(response.content, str(response.url))) and pdf_url != url:
            return _try_pdf_url(client, pdf_url, follow_citation_meta=False)
        return "not_pdf", None, response.headers.get("content-type", "?")[:40]
    return "pdf", response.content, None


def best_effort_url(doc: Document) -> str:
    """The most useful link to hand a human for manual retrieval: the cached PDF link, else the landing
    page, else a DOI resolver link, else empty."""
    return doc.pdf_url or doc.url or (f"https://doi.org/{doc.doi}" if doc.doi else "")


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
    core_api_key: str = "",
) -> str:
    """Fetch one document's full text.

    Returns the outcome: xml, pdf, pdf_core, pdf_unpaywall, not_pdf, too_large, no_url, blocked, or
    http_<code> (the last four reflect whichever attempt got furthest, when every attempt fails).
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
    except httpx.TransportError:
        doc.status_reason = "fetch:network_error"
        return "network_error"

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

    if doc.doi and (core_url := core_candidate_url(client, doc.doi, core_api_key)) and core_url not in tried_urls:
        tried_urls.add(core_url)
        outcome, content, detail = _try_pdf_url(client, core_url)
        if outcome == "pdf":
            _store(session, doc, raw_dir, "pdf", content)
            doc.status, doc.status_reason = "fetched", "fetch:pdf:core"
            return "pdf_core"
        if best_outcome == "no_url":
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
