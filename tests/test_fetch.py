import hashlib

import httpx
from sqlalchemy import select

from batterygemma.db.models import Document, File
from batterygemma.db.session import get_session
from batterygemma.fetch import best_effort_url, fetch_document
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


def unpaywall(locations):
    return httpx.Response(200, json={"oa_locations": locations})


def core_empty():
    return httpx.Response(200, json={"results": []})


def test_unpaywall_repository_mirror_recovers_a_blocked_publisher_pdf(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "www.iop.org":  # the primary, blocked publisher link
            return httpx.Response(200, content=b"<html>captcha landing page</html>", headers={"content-type": "text/html"})
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([
                {"host_type": "publisher", "url_for_pdf": "https://www.iop.org/article/x/pdf"},
                {"host_type": "repository", "url_for_pdf": "https://mediatum.example.org/doc/1/1.pdf"},
            ])
        if host == "mediatum.example.org":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1149/x", pdf_url="https://www.iop.org/article/x/pdf")

    assert outcome == "pdf_unpaywall"
    assert doc.status == "fetched" and doc.status_reason == "fetch:pdf:unpaywall"
    assert doc.pdf_url == "https://mediatum.example.org/doc/1/1.pdf"  # replaced with the link that actually worked
    assert files[0].kind == "pdf"


def test_unpaywall_skips_the_url_already_tried_and_tries_the_next_one(engine, tmp_path):
    attempted = []

    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "pub.example.org":
            attempted.append(str(request.url))
            return httpx.Response(403, headers={"cf-mitigated": "challenge"})
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([
                {"host_type": "publisher", "url_for_pdf": "https://pub.example.org/x.pdf"},  # same as pdf_url
                {"host_type": "repository", "url_for_pdf": "https://repo.example.org/x.pdf"},
            ])
        if host == "repo.example.org":
            attempted.append(str(request.url))
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, _ = run(engine, tmp_path, handler, doi="10.1/y", pdf_url="https://pub.example.org/x.pdf")

    assert outcome == "pdf_unpaywall"
    assert attempted == ["https://pub.example.org/x.pdf", "https://repo.example.org/x.pdf"]  # not tried twice


def test_unpaywall_recovers_a_document_with_no_cached_pdf_url(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([{"host_type": "repository", "url_for_pdf": "https://repo.example.org/z.pdf"}])
        if host == "repo.example.org":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/z")  # no pdf_url at all

    assert outcome == "pdf_unpaywall" and doc.status == "fetched"
    assert files[0].kind == "pdf"


def test_unpaywall_unknown_doi_does_not_crash_the_fetch(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return httpx.Response(404)  # Unpaywall's response for a DOI it doesn't know
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/unknown")

    assert outcome == "no_url" and files == []


def test_core_mirror_recovers_a_blocked_publisher_pdf(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "onlinelibrary.example.org":  # the primary, blocked publisher link
            return httpx.Response(200, content=b"<html>login wall</html>", headers={"content-type": "text/html"})
        if host == "api.core.ac.uk":
            return httpx.Response(200, json={"results": [{"downloadUrl": "https://core.ac.uk/download/1.pdf"}]})
        if host == "core.ac.uk":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1002/x", pdf_url="https://onlinelibrary.example.org/x.pdf")

    assert outcome == "pdf_core"
    assert doc.status == "fetched" and doc.status_reason == "fetch:pdf:core"
    assert files[0].kind == "pdf"


def test_core_is_tried_before_unpaywall(engine, tmp_path):
    """CORE's own re-hosted copy is checked before falling through to Unpaywall's candidate list."""
    called_unpaywall = False

    def handler(request):
        nonlocal called_unpaywall
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "api.core.ac.uk":
            return httpx.Response(200, json={"results": [{"downloadUrl": "https://core.ac.uk/download/2.pdf"}]})
        if host == "core.ac.uk":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        if host == "api.unpaywall.org":
            called_unpaywall = True
            return unpaywall([])
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/core-first")

    assert outcome == "pdf_core" and not called_unpaywall


def test_citation_pdf_url_meta_tag_is_followed_one_hop(engine, tmp_path):
    """A repository landing page with no direct PDF link but a `citation_pdf_url` <meta> tag is followed -
    this is metadata the repository itself publishes, not bot evasion (see fetch.py's module docstring)."""
    landing = (
        b'<html><head><meta name="citation_pdf_url" content="https://repo.example.org/files/real.pdf">'
        b"</head><body>landing page</body></html>"
    )

    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([{"host_type": "repository", "url": "https://repo.example.org/landing/1"}])
        if str(request.url) == "https://repo.example.org/landing/1":
            return httpx.Response(200, content=landing, headers={"content-type": "text/html"})
        if str(request.url) == "https://repo.example.org/files/real.pdf":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected url {request.url}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/citmeta")

    assert outcome == "pdf_unpaywall" and doc.status == "fetched"
    assert files[0].kind == "pdf"


def test_best_effort_url_prefers_pdf_url_then_url_then_doi():
    assert best_effort_url(Document(doc_id="d", source="t", external_id="1", title="t", norm_title="t",
                                    pdf_url="https://pub.org/x.pdf", url="https://pub.org/x", doi="10.1/x")) == "https://pub.org/x.pdf"
    assert best_effort_url(Document(doc_id="d", source="t", external_id="1", title="t", norm_title="t",
                                    url="https://pub.org/x", doi="10.1/x")) == "https://pub.org/x"
    assert best_effort_url(Document(doc_id="d", source="t", external_id="1", title="t", norm_title="t",
                                    doi="10.1/x")) == "https://doi.org/10.1/x"
    assert best_effort_url(Document(doc_id="d", source="t", external_id="1", title="t", norm_title="t")) == ""


def test_network_error_on_pdf_url_does_not_crash_and_falls_back_to_unpaywall(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "flaky.example.org":
            raise httpx.ReadTimeout("timed out", request=request)
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([{"host_type": "repository", "url_for_pdf": "https://repo.example.org/ok.pdf"}])
        if host == "repo.example.org":
            return httpx.Response(200, content=PDF, headers={"content-type": "application/pdf"})
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/flaky", pdf_url="https://flaky.example.org/x.pdf")

    assert outcome == "pdf_unpaywall" and doc.status == "fetched"
    assert files[0].kind == "pdf"


def test_network_error_with_no_fallback_is_reported_not_raised(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        raise httpx.ConnectError("connection refused", request=request)

    outcome, doc, files = run(engine, tmp_path, handler, pdf_url="https://flaky.example.org/x.pdf")

    assert outcome == "network_error" and doc.status_reason == "fetch:network_error" and files == []


def test_unpaywall_all_candidates_fail_reports_the_best_outcome_seen(engine, tmp_path):
    def handler(request):
        host = request.url.host
        if host == "www.ebi.ac.uk":
            return epmc([])
        if host == "pub.example.org":
            return httpx.Response(200, content=b"<html>login wall</html>", headers={"content-type": "text/html"})
        if host == "api.core.ac.uk":
            return core_empty()
        if host == "api.unpaywall.org":
            return unpaywall([{"host_type": "repository", "url_for_pdf": "https://repo.example.org/dead.pdf"}])
        if host == "repo.example.org":
            return httpx.Response(404)
        raise AssertionError(f"unexpected host {host}")

    outcome, doc, files = run(engine, tmp_path, handler, doi="10.1/w", pdf_url="https://pub.example.org/x.pdf")

    assert outcome == "not_pdf" and files == []  # the primary attempt's more informative outcome wins
