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


def add_local_pdf(
    session: Session, path: Path, raw_dir: Path, *, license: str = DEFAULT_LOCAL_LICENSE, title: str | None = None
) -> Document:
    """Copy `path` into the corpus as a new (or existing, if already added) local document, ready to parse.

    The doc_id is derived from the file's content hash, so re-adding the same file is a no-op that returns
    the existing Document rather than duplicating it.
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
