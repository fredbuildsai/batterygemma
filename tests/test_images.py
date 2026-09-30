import httpx
from sqlalchemy import select

from batterygemma.db.models import Document, File
from batterygemma.db.session import get_session
from batterygemma.images import (
    download_images_for_document,
    graphic_hrefs,
    pmc_oa_image_keys,
    pmcid_from_xml_url,
)
from batterygemma.sources.base import PoliteClient

JPG = b"\xff\xd8\xff fake jpeg bytes"

XML_WITH_FIGURES = b"""
<article>
  <body>
    <fig><graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="ncomms8898-f1.jpg"/></fig>
    <fig><graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="ncomms8898-f2.jpg"/></fig>
    <p>inline <inline-graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="ncomms8898-i1.jpg"/></p>
    <supplementary-material><media xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="ncomms8898-s1.pdf"/></supplementary-material>
  </body>
</article>
"""


def client_for(handler):
    return PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda _: None, min_interval=0)


def s3_listing(keys, prefix=None):
    inner = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    prefix_xml = f"<Prefix>{prefix}</Prefix>" if prefix else ""
    return httpx.Response(200, content=f'<?xml version="1.0"?><ListBucketResult>{prefix_xml}{inner}</ListBucketResult>'.encode())


def s3_delimited_listing(prefixes, query_prefix="PMC4532849."):
    """A delimiter listing, including the top-level <Prefix> S3 echoes back for the query itself (a real,
    confirmed-live gotcha: that echo has no version suffix and must not be parsed as a version prefix)."""
    inner = "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    return httpx.Response(
        200,
        content=f'<?xml version="1.0"?><ListBucketResult><Name>pmc-oa-opendata</Name>'
        f"<Prefix>{query_prefix}</Prefix>{inner}</ListBucketResult>".encode(),
    )


def test_pmcid_from_xml_url():
    assert pmcid_from_xml_url("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4532849/fullTextXML") == "PMC4532849"
    assert pmcid_from_xml_url(None) is None
    assert pmcid_from_xml_url("https://example.org/not-a-match") is None


def test_graphic_hrefs_extracts_images_and_drops_non_images_and_dupes():
    xml = XML_WITH_FIGURES + XML_WITH_FIGURES  # duplicated on purpose
    hrefs = graphic_hrefs(xml)
    assert hrefs == ["ncomms8898-f1.jpg", "ncomms8898-f2.jpg", "ncomms8898-i1.jpg"]  # s1.pdf filtered, no dupes


def test_pmc_oa_image_keys_picks_the_latest_version():
    def handler(request):
        if request.url.params.get("delimiter") == "/":
            return s3_delimited_listing(["PMC4532849.1/", "PMC4532849.2/"])
        return s3_listing(["PMC4532849.2/ncomms8898-f1.jpg", "PMC4532849.2/ncomms8898-f2.jpg"])

    keys = pmc_oa_image_keys(client_for(handler), "PMC4532849")
    assert keys == {"ncomms8898-f1.jpg": "PMC4532849.2/ncomms8898-f1.jpg",
                    "ncomms8898-f2.jpg": "PMC4532849.2/ncomms8898-f2.jpg"}


def test_pmc_oa_image_keys_returns_empty_when_article_is_outside_pmc_scope():
    handler = lambda request: s3_delimited_listing([])  # no prefix found - most materials-science papers
    assert pmc_oa_image_keys(client_for(handler), "PMC0") == {}


def test_download_images_for_document_stores_files_and_links_to_the_doc(engine, tmp_path):
    xml_path = tmp_path / "src.xml"
    xml_path.write_bytes(XML_WITH_FIGURES)

    def handler(request):
        if request.url.params.get("delimiter") == "/":
            return s3_delimited_listing(["PMC4532849.1/"])
        if "prefix" in request.url.params:
            return s3_listing([
                "PMC4532849.1/ncomms8898-f1.jpg", "PMC4532849.1/ncomms8898-f2.jpg",
                "PMC4532849.1/ncomms8898-i1.jpg",
            ])
        return httpx.Response(200, content=JPG, headers={"content-type": "image/jpeg"})

    with get_session(engine) as s:
        doc = Document(doc_id="openalex:W1", source="openalex", external_id="W1", title="t", norm_title="t",
                       status="chunked", license="CC-BY",
                       xml_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4532849/fullTextXML")
        s.add(doc)
        s.flush()
        new_files = download_images_for_document(s, doc, xml_path, client_for(handler), tmp_path / "images")

    assert len(new_files) == 3
    assert (tmp_path / "images" / "openalex" / "W1" / "ncomms8898-f1.jpg").read_bytes() == JPG

    with get_session(engine) as s:
        rows = s.scalars(select(File).where(File.doc_id == "openalex:W1", File.kind == "image")).all()
        assert len(rows) == 3
        assert {r.path.rsplit("/", 1)[-1] for r in rows} == {"ncomms8898-f1.jpg", "ncomms8898-f2.jpg", "ncomms8898-i1.jpg"}


def test_download_images_is_idempotent_and_skips_already_downloaded_names(engine, tmp_path):
    xml_path = tmp_path / "src.xml"
    xml_path.write_bytes(XML_WITH_FIGURES)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.params.get("delimiter") == "/":
            return s3_delimited_listing(["PMC4532849.1/"])
        if "prefix" in request.url.params:
            return s3_listing(["PMC4532849.1/ncomms8898-f2.jpg", "PMC4532849.1/ncomms8898-i1.jpg"])
        return httpx.Response(200, content=JPG, headers={"content-type": "image/jpeg"})

    with get_session(engine) as s:
        doc = Document(doc_id="openalex:W2", source="openalex", external_id="W2", title="t", norm_title="t",
                       status="chunked", license="CC-BY",
                       xml_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4532849/fullTextXML")
        s.add(doc)
        s.add(File(doc_id="openalex:W2", kind="image", path=str(tmp_path / "images/openalex/W2/ncomms8898-f1.jpg"),
                   sha256="x", bytes=1))
        s.flush()
        s.refresh(doc)
        new_files = download_images_for_document(s, doc, xml_path, client_for(handler), tmp_path / "images")

    assert {f.path.rsplit("/", 1)[-1] for f in new_files} == {"ncomms8898-f2.jpg", "ncomms8898-i1.jpg"}


def test_missing_pmcid_or_no_graphics_returns_empty(engine, tmp_path):
    xml_path = tmp_path / "src.xml"
    xml_path.write_bytes(b"<article><body>no figures here</body></article>")

    with get_session(engine) as s:
        doc = Document(doc_id="openalex:W3", source="openalex", external_id="W3", title="t", norm_title="t",
                       status="chunked", license="CC-BY", xml_url=None)
        s.add(doc)
        s.flush()
        assert download_images_for_document(s, doc, xml_path, client_for(lambda r: httpx.Response(200)), tmp_path) == []

    with get_session(engine) as s:
        doc = Document(doc_id="openalex:W4", source="openalex", external_id="W4", title="t", norm_title="t",
                       status="chunked", license="CC-BY",
                       xml_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML")
        s.add(doc)
        s.flush()
        xml_path2 = tmp_path / "empty.xml"
        xml_path2.write_bytes(b"<article><body>no figures here</body></article>")
        assert download_images_for_document(s, doc, xml_path2, client_for(lambda r: httpx.Response(200)), tmp_path) == []
