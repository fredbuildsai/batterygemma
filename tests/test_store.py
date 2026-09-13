from sqlalchemy import select

from batterygemma.db.models import Document
from batterygemma.db.session import get_session
from batterygemma.sources.base import DiscoveredRecord
from batterygemma.sources.store import upsert_records

ALLOW, FLAG = ["CC0", "CC-BY", "public-domain"], ["CC-BY-SA"]
TITLE = "Suppressing intergranular cracking in single-crystal NMC811 cathodes"


def test_duplicate_by_doi_upgrades_canonical_license_with_matching_pdf(engine):
    preprint = DiscoveredRecord(source="chemrxiv", external_id="chemrxiv-1", title=TITLE, doi="10.1/abc",
                                license_raw="cc by-nc-nd", pdf_url="https://chemrxiv.org/pdf/1")
    published = DiscoveredRecord(source="openalex", external_id="W9", title=TITLE, doi="https://doi.org/10.1/ABC",
                                 license_raw="cc-by", pdf_url="https://publisher.org/w9.pdf",
                                 license_evidence="openalex:best_oa_location.license")

    with get_session(engine) as s:
        assert upsert_records(s, [preprint], allow=ALLOW, flag=FLAG) == {"new": 1}
        counts = upsert_records(s, [published], allow=ALLOW, flag=FLAG)
    assert counts == {"duplicate": 1, "license_upgraded": 1}

    with get_session(engine) as s:
        canonical = s.get(Document, "chemrxiv:chemrxiv-1")
        duplicate = s.get(Document, "openalex:W9")
        assert canonical.license == "CC-BY" and canonical.pdf_url == "https://publisher.org/w9.pdf"
        assert canonical.raw_metadata["license_copy_from"] == "openalex:W9"
        assert duplicate.status == "duplicate" and duplicate.duplicate_of == "chemrxiv:chemrxiv-1"


def test_title_match_dedupes_when_doi_missing_and_rerun_updates(engine):
    a = DiscoveredRecord(source="arxiv", external_id="2501.1", title=TITLE, license_raw="cc-by")
    b = DiscoveredRecord(source="openalex", external_id="W2", title=TITLE.upper() + "!", abstract="text")
    with get_session(engine) as s:
        upsert_records(s, [a, b], allow=ALLOW, flag=FLAG)
        assert upsert_records(s, [a], allow=ALLOW, flag=FLAG) == {"updated": 1}
    with get_session(engine) as s:
        statuses = dict(s.execute(select(Document.doc_id, Document.status)).all())
    assert statuses == {"arxiv:2501.1": "discovered", "openalex:W2": "duplicate"}
