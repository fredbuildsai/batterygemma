import hashlib

import httpx
from sqlalchemy import select

from batterygemma.db.models import Document, File
from batterygemma.db.session import get_session
from batterygemma.fetch import fetch_document
from batterygemma.sources.base import PoliteClient

ALLOW, FLAG = ["CC0", "CC-BY", "public-domain"], ["CC-BY-SA"]
PDF = b"%PDF-1.7 fake pdf bytes"


def client_for(handler):
    return PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda _: None, min_interval=0)


def epmc(results):
    return httpx.Response(200, json={"resultList": {"result": results}})


def make_doc(s, **kw):
    doc = Document(doc_id="openalex:W1", source="openalex", external_id="W1", title="t", norm_title="t",
                   status="accepted", license="CC-BY", **kw)
    s.add(doc)
    return doc


def run(engine, tmp_path, handler, **doc_kw):
    with get_session(engine) as s:
        doc = make_doc(s, **doc_kw)
        outcome = fetch_document(s, doc, client_for(handler), tmp_path, allow=ALLOW, flag=FLAG)
    with get_session(engine) as s:
        return outcome, s.get(Document, "openalex:W1"), s.scalars(select(File)).all()


def test_prefers_europepmc_xml_when_its_license_is_allowed(engine, tmp_path):
    def handler(request):
        if request.url.path.endswith("/search"):
            return epmc([{"pmcid": "PMC123", "isOpenAccess": "Y", "license": "cc by"}])
        return httpx.Response(200, content=b"<article>jats</article>")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/x", pdf_url="https://pub.org/x.pdf")
    assert outcome == "xml" and doc.status == "fetched"
    assert files[0].kind == "xml" and files[0].sha256 == hashlib.sha256(b"<article>jats</article>").hexdigest()


def test_falls_back_to_pdf_when_europepmc_copy_is_not_allowed(engine, tmp_path):
    def handler(request):
        if request.url.path.endswith("/search"):
            return epmc([{"pmcid": "PMC9", "isOpenAccess": "Y", "license": "cc by-nc"}])
        return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/x", pdf_url="https://pub.org/x.pdf")
    assert outcome == "pdf" and doc.status == "fetched"
    assert (tmp_path / "openalex" / "W1.pdf").read_bytes() == PDF


def test_html_landing_page_is_not_stored_as_pdf(engine, tmp_path):
    handler = lambda request: httpx.Response(200, content=b"<html>login</html>", headers={"content-type": "text/html"})
    outcome, doc, files = run(engine, tmp_path, handler, pdf_url="https://pub.org/x.pdf")
    assert outcome == "not_pdf" and doc.status == "accepted" and files == []


def test_bot_challenge_is_recorded_not_bypassed(engine, tmp_path):
    handler = lambda request: httpx.Response(403, headers={"cf-mitigated": "challenge"})
    outcome, doc, files = run(engine, tmp_path, handler, pdf_url="https://chemrxiv.org/doi/pdf/x")
    assert outcome == "blocked" and doc.status_reason == "fetch:bot_protection" and files == []


def test_missing_urls(engine, tmp_path):
    outcome, doc, _ = run(engine, tmp_path, lambda request: epmc([]))
    assert outcome == "no_url" and doc.status_reason == "fetch:no_fulltext_url"
