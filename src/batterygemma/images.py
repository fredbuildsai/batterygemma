"""Resolve and download the figures referenced from a document's JATS XML full text.

JATS XML from Europe PMC (`fetch.py`'s first-choice route) stores figures as `<graphic>`/`<inline-graphic>`
elements with a bare filename in `xlink:href` (e.g. "ncomms8898-f1.jpg"), relative to the article's own
package - the XML never carries a resolvable URL. Europe PMC/PMC's old per-article image endpoints
(`.../bin/<filename>`, the `oa.fcgi` OA Web Service) were retired in 2026; the replacement is the official
PMC Open Access S3 bucket (registry.opendata.aws/ncbi-pmc), which republishes every OA article's package -
XML, PDF and every figure file - under `s3://pmc-oa-opendata/<PMCID>.<version>/`, served over plain HTTPS
with no auth and no bot management. We list that per-article prefix and match it against the filenames the
XML already told us to expect.

Images are stored at data/images/<source>/<external_id>/<original_filename> - grouped by article so they're
found the same way as the article's own raw XML/PDF under data/raw/<source>/<external_id>.* - and recorded
as File(kind="image") rows against the same doc_id, so every image is linked to its article the same way the
XML/PDF already are.
"""

import hashlib
import re
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import Document, File
from batterygemma.sources.base import BlockedByBotProtection, PoliteClient

PMC_ID_RE = re.compile(r"/(PMC\d+)/fullTextXML")
GRAPHIC_HREF_RE = re.compile(
    rb'<(?:graphic|inline-graphic|media)\b[^>]*\bxlink:href=["\']([^"\']+)["\']', re.IGNORECASE
)
PMC_OA_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
S3_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
# Only the nested <CommonPrefixes><Prefix>...</Prefix></CommonPrefixes> - S3 also echoes the query's own
# `prefix` param as a top-level <Prefix> sibling of <Name>, which must not be matched here.
S3_COMMON_PREFIX_RE = re.compile(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".svg")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def pmcid_from_xml_url(xml_url: str | None) -> str | None:
    if not xml_url:
        return None
    match = PMC_ID_RE.search(xml_url)
    return match.group(1) if match else None


def graphic_hrefs(xml_bytes: bytes) -> list[str]:
    """Bare figure filenames referenced by `<graphic>`/`<inline-graphic>`/`<media>` tags, de-duplicated,
    in document order. Non-image media (e.g. a linked supplementary PDF/dataset) is filtered out here."""
    seen: dict[str, None] = {}
    for match in GRAPHIC_HREF_RE.finditer(xml_bytes):
        href = match.group(1).decode("utf-8", "ignore")
        name = href.rsplit("/", 1)[-1]
        if name.lower().endswith(IMAGE_EXTENSIONS):
            seen.setdefault(name, None)
    return list(seen)


def _latest_version_prefix(client: PoliteClient, pmcid: str) -> str | None:
    """The S3 key prefix for the newest version of `pmcid` (e.g. "PMC4532849.1/"), or None if the bucket
    has no package for it (common for articles outside PMC's biomedical scope - most of this project's
    materials-science literature isn't in PMC at all, so a miss here is expected, not an error)."""
    response = client.get(PMC_OA_BUCKET, params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"})
    prefixes = S3_COMMON_PREFIX_RE.findall(response.text)
    if not prefixes:
        return None
    return max(prefixes, key=lambda p: int(p.rstrip("/").rsplit(".", 1)[-1]))


def pmc_oa_image_keys(client: PoliteClient, pmcid: str) -> dict[str, str]:
    """Map of {filename: full S3 key} for every file in `pmcid`'s latest PMC-OA package."""
    prefix = _latest_version_prefix(client, pmcid)
    if not prefix:
        return {}
    response = client.get(PMC_OA_BUCKET, params={"list-type": "2", "prefix": prefix})
    keys = S3_KEY_RE.findall(response.text)
    return {key.rsplit("/", 1)[-1]: key for key in keys}


def resolve_chunk_image_paths(session: Session, doc_id: str, filenames: list[str]) -> list[str]:
    """Downloaded file paths for `filenames` (a `Chunk.images` list) that actually have a File(kind="image")
    row for `doc_id`. A filename with no matching row means the image was never downloaded (e.g. the article
    is outside the PMC-OA bucket's scope, or `bg images` hasn't run yet) - silently omitted, not an error."""
    if not filenames:
        return []
    wanted = set(filenames)
    rows = session.scalars(select(File).where(File.doc_id == doc_id, File.kind == "image")).all()
    return [f.path for f in rows if Path(f.path).name in wanted]


def download_images_for_document(
    session: Session, doc: Document, xml_path: Path, client: PoliteClient, images_dir: Path
) -> list[File]:
    """Resolve and download every figure referenced in `doc`'s stored XML. Returns the newly-created
    File(kind="image") rows (already added to `session`, not yet committed). Idempotent: skips filenames
    that already have an image File row for this doc."""
    pmcid = pmcid_from_xml_url(doc.xml_url)
    if not pmcid:
        return []
    hrefs = graphic_hrefs(xml_path.read_bytes())
    if not hrefs:
        return []

    existing = {Path(f.path).name for f in doc.files if f.kind == "image"}
    wanted = [name for name in hrefs if name not in existing]
    if not wanted:
        return []

    try:
        keys_by_name = pmc_oa_image_keys(client, pmcid)
    except (BlockedByBotProtection, httpx.HTTPStatusError, httpx.TransportError):
        return []

    directory = images_dir / doc.source / _safe_name(doc.external_id)
    new_files: list[File] = []
    for name in wanted:
        key = keys_by_name.get(name)
        if not key:
            continue
        try:
            response = client.get(PMC_OA_BUCKET + key)
        except (BlockedByBotProtection, httpx.HTTPStatusError, httpx.TransportError):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _safe_name(name)
        path.write_bytes(response.content)
        file = File(doc_id=doc.doc_id, kind="image", path=str(path),
                    sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content))
        session.add(file)
        new_files.append(file)
    return new_files
