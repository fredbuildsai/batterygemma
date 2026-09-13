"""Persist discovered records with cross-source deduplication.

A record whose DOI (or long normalized title) matches an existing canonical document is stored with
status "duplicate" and `duplicate_of` set. If the duplicate carries an allowed license and the canonical
does not, the canonical adopts the duplicate's license *together with* its PDF/XML links, so the text we
later fetch is always the copy that license applies to (preprint and published versions can differ).
"""

from collections import Counter
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.db.models import Document
from batterygemma.screen.license import ALLOWED, evaluate_license, normalize_license
from batterygemma.sources.base import DiscoveredRecord, normalize_doi, normalize_title

MIN_TITLE_MATCH_CHARS = 30


def upsert_records(
    session: Session, records: Iterable[DiscoveredRecord], *, allow: list[str], flag: list[str]
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        doi = normalize_doi(record.doi)
        norm_title = normalize_title(record.title)
        license_id = normalize_license(record.license_raw)

        existing = session.get(Document, record.doc_id)
        if existing is not None:
            for attr in ("doi", "abstract", "pdf_url", "xml_url", "year", "venue"):
                if getattr(existing, attr) is None and getattr(record, attr) is not None:
                    setattr(existing, attr, doi if attr == "doi" else getattr(record, attr))
            counts["updated"] += 1
            continue

        canonical = _find_canonical(session, doi, norm_title)
        document = Document(
            doc_id=record.doc_id, source=record.source, external_id=record.external_id, doi=doi,
            title=record.title, norm_title=norm_title, authors=record.authors, year=record.year,
            venue=record.venue, abstract=record.abstract, url=record.url, pdf_url=record.pdf_url,
            xml_url=record.xml_url, license=license_id, license_evidence=record.license_evidence,
            raw_metadata=record.raw,
        )
        if canonical is None:
            counts["new"] += 1
        else:
            document.status = "duplicate"
            document.duplicate_of = canonical.doc_id
            counts["duplicate"] += 1
            if (
                evaluate_license(license_id, allow, flag) == ALLOWED
                and evaluate_license(canonical.license, allow, flag) != ALLOWED
            ):
                canonical.license = license_id
                canonical.license_evidence = record.license_evidence
                canonical.pdf_url = record.pdf_url
                canonical.xml_url = record.xml_url
                canonical.raw_metadata = {**canonical.raw_metadata, "license_copy_from": record.doc_id}
                counts["license_upgraded"] += 1
            if canonical.abstract is None and record.abstract:
                canonical.abstract = record.abstract
        session.add(document)
        session.flush()
    return counts


def _find_canonical(session: Session, doi: str | None, norm_title: str) -> Document | None:
    base = select(Document).where(Document.duplicate_of.is_(None))
    if doi and (match := session.scalars(base.where(Document.doi == doi)).first()):
        return match
    if len(norm_title) >= MIN_TITLE_MATCH_CHARS:
        return session.scalars(base.where(Document.norm_title == norm_title)).first()
    return None
