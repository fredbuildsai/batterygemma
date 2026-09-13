"""Download full text for accepted documents.

Order of preference:
1. JATS XML from Europe PMC's open-access subset, when Europe PMC reports an allowed license for that copy.
2. The document's licensed PDF (`pdf_url`, which `sources.store` keeps paired with the license it came with).

Bot challenges (e.g. Cloudflare) are recorded and skipped, never bypassed. Successful downloads are stored
under data/raw/<source>/ with a sha256 and a `files` row, and the document moves to status "fetched".
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


def _store(session: Session, doc: Document, raw_dir: Path, kind: str, content: bytes) -> File:
    directory = raw_dir / doc.source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(doc.external_id)}.{kind}"
    path.write_bytes(content)
    file = File(doc_id=doc.doc_id, kind=kind, path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content))
    session.add(file)
    return file


def fetch_document(
    session: Session, doc: Document, client: PoliteClient, raw_dir: Path, *, allow: list[str], flag: list[str]
) -> str:
    """Fetch one document's full text. Returns the outcome: xml, pdf, not_pdf, too_large, no_url, blocked, http_<code>."""
    try:
        if doc.doi and (pmcid := find_open_access_pmcid(client, doc.doi, allow, flag)):
            url = EUROPEPMC_XML.format(pmcid=pmcid)
            response = client.get(url)
            if response.content.lstrip().startswith(b"<"):
                _store(session, doc, raw_dir, "xml", response.content)
                doc.xml_url, doc.status, doc.status_reason = url, "fetched", f"fetch:xml:{pmcid}"
                return "xml"

        if not doc.pdf_url:
            doc.status_reason = "fetch:no_fulltext_url"
            return "no_url"

        response = client.get(doc.pdf_url)
        if len(response.content) > MAX_BYTES:
            doc.status_reason = "fetch:too_large"
            return "too_large"
        is_pdf = response.content.startswith(b"%PDF") or "pdf" in response.headers.get("content-type", "")
        if not is_pdf:
            doc.status_reason = f"fetch:not_pdf:{response.headers.get('content-type', '?')[:40]}"
            return "not_pdf"
        _store(session, doc, raw_dir, "pdf", response.content)
        doc.status, doc.status_reason = "fetched", "fetch:pdf"
        return "pdf"
    except BlockedByBotProtection:
        doc.status_reason = "fetch:bot_protection"
        return "blocked"
    except httpx.HTTPStatusError as exc:
        doc.status_reason = f"fetch:http_{exc.response.status_code}"
        return f"http_{exc.response.status_code}"
