"""Ingest local PDF files directly into the corpus, bypassing discover/screen/fetch.

Two tracks exist by design: the default, publicly-releasable corpus is limited to CC0/CC-BY/public-domain
content (see `screen.license`). Local files added here are commercial or otherwise not openly licensed by
default (`all-rights-reserved`), so they are recorded honestly and stay excluded from that default track —
`screen.license.evaluate_license` returns `REJECTED` for them, exactly as it does for any other non-open
license. They remain fully usable for local fine-tuning under the "all sources, including commercial works"
track; only pass an actual open `license=` if you hold the rights to release that specific file's content
publicly.
"""

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from batterygemma.db.models import Document, File

DEFAULT_LOCAL_LICENSE = "all-rights-reserved"


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def attach_local_pdf(session: Session, path: Path, raw_dir: Path, doc_id: str) -> Document:
    """Attach a manually-retrieved file to an *existing* Document (e.g. one `bg fetch` couldn't reach).

    Unlike `add_local_pdf`, this never creates a new Document - it is for closing the loop on
    `bg fetch-failures`: the paper was already discovered and screened (it has a real DOI, license, title),
    fetching it automatically just failed, and a human retrieved a copy through legitimate means (library
    access, the author, etc). Attaching preserves that original metadata instead of creating a disconnected
    `local:<hash>` document with none of it. Raises `KeyError` if `doc_id` doesn't exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = session.get(Document, doc_id)
    if doc is None:
        raise KeyError(f"no document {doc_id!r} in the corpus")

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if not any(f.sha256 == digest for f in doc.files):
        directory = raw_dir / "local"
        directory.mkdir(parents=True, exist_ok=True)
        stored_path = directory / f"{digest[:16]}.pdf"
        stored_path.write_bytes(content)
        doc.files.append(File(kind="pdf", path=str(stored_path), sha256=digest, bytes=len(content)))
    doc.status, doc.status_reason = "fetched", f"local:manually_attached:{path.name}"
    session.flush()
    return doc


def add_local_pdf(
    session: Session, path: Path, raw_dir: Path, *, license: str = DEFAULT_LOCAL_LICENSE, title: str | None = None
) -> Document:
    """Copy `path` into the corpus as a new (or existing, if already added) local document, ready to parse.

    The doc_id is derived from the file's content hash, so re-adding the same file is a no-op that returns
    the existing Document rather than duplicating it. To attach a file to a document `bg fetch` already
    discovered but couldn't download, use `attach_local_pdf` instead (preserves the original DOI/license).
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    doc_id = f"local:{digest[:16]}"

    existing = session.get(Document, doc_id)
    if existing is not None:
        return existing

    resolved_title = title or _title_from_filename(path)
    doc = Document(
        doc_id=doc_id, source="local", external_id=digest[:16], title=resolved_title,
        norm_title=resolved_title.lower(), license=license,
        license_evidence=f"local:user-provided:{path.name}", status="fetched", status_reason="local:added",
        raw_metadata={"original_path": str(path)},
    )
    directory = raw_dir / "local"
    directory.mkdir(parents=True, exist_ok=True)
    stored_path = directory / f"{digest[:16]}.pdf"
    stored_path.write_bytes(content)
    doc.files.append(File(kind="pdf", path=str(stored_path), sha256=digest, bytes=len(content)))
    session.add(doc)
    session.flush()
    return doc
