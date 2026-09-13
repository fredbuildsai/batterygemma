import httpx

from batterygemma.sources.arxiv import ArxivOaiSource
from batterygemma.sources.base import PoliteClient
from batterygemma.sources.crossref import CrossrefChemRxivSource, base_doi
from batterygemma.sources.openalex import OpenAlexSource, rebuild_abstract


def client_for(handler):
    return PoliteClient(transport=httpx.MockTransport(handler), sleep=lambda _: None, min_interval=0)


def test_openalex_pages_with_cursor_and_maps_fields():
    requests = []
    pages = {
        "*": {"meta": {"next_cursor": "c2"}, "results": [{
            "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1038/S41467-020-15355-0",
            "title": "A reflection on lithium-ion battery cathode chemistry", "publication_year": 2020,
            "type": "article", "authorships": [{"author": {"display_name": "A. Manthiram"}}],
            "abstract_inverted_index": {"cathodes": [1], "Layered": [0], "matter": [2]},
            "best_oa_location": {"license": "cc-by", "pdf_url": "https://example.org/w1.pdf",
                                 "landing_page_url": "https://doi.org/10.1038/s41467-020-15355-0"},
            "primary_location": {"source": {"display_name": "Nature Communications"}},
        }]},
        "c2": {"meta": {"next_cursor": None}, "results": []},
    }

    def handler(request):
        requests.append(request.url.params)
        return httpx.Response(200, json=pages[request.url.params["cursor"]])

    records = list(OpenAlexSource(client_for(handler), api_key="").discover(["lithium-ion battery"], limit=10))

    assert len(records) == 1
    record = records[0]
    assert record.doc_id == "openalex:W1"
    assert record.doi == "10.1038/s41467-020-15355-0"
    assert record.abstract == "Layered cathodes matter"
    assert record.license_raw == "cc-by" and record.pdf_url == "https://example.org/w1.pdf"
    assert record.venue == "Nature Communications"
    assert requests[0]["filter"] == "is_oa:true,best_oa_location.license:cc-by|cc0|public-domain"
    assert "mailto" not in requests[0]


def test_openalex_splits_limit_across_terms():
    def work(work_id):
        return {"id": f"https://openalex.org/{work_id}", "title": f"Paper {work_id}", "best_oa_location": {"license": "cc-by"}}

    pages = {
        "cathode": [work("C1"), work("C2"), work("C3")],
        "electrolyte": [work("E1"), work("C1"), work("E2"), work("E3")],  # C1 repeats across queries
    }
    requested = []

    def handler(request):
        term = request.url.params["search"]
        requested.append((term, request.url.params["per-page"]))
        return httpx.Response(200, json={"meta": {"next_cursor": None}, "results": pages[term]})

    records = list(OpenAlexSource(client_for(handler), api_key="").discover(["cathode", "electrolyte"], limit=4))

    assert [r.external_id for r in records] == ["C1", "C2", "E1", "E2"]  # 2 per term; duplicate C1 skipped
    assert requested == [("cathode", "2"), ("electrolyte", "2")]


def test_rebuild_abstract_handles_missing_index():
    assert rebuild_abstract(None) is None


def test_crossref_merges_versions_and_takes_license_from_any_version():
    payload = {"message": {"next-cursor": None, "items": [
        {"DOI": "10.26434/chemrxiv-2025-rvp45/v2", "title": ["Sulfite electrolytes"],
         "link": [{"URL": "https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-rvp45/v2"}],
         "posted": {"date-parts": [[2026, 2, 17]]}},
        {"DOI": "10.26434/chemrxiv-2025-rvp45", "title": ["Sulfite electrolytes"],
         "abstract": "<jats:p>Sulfite <jats:italic>solvents</jats:italic> tune solvation.</jats:p>",
         "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
         "link": [{"URL": "https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-rvp45"}],
         "author": [{"given": "Ada", "family": "Lovelace"}], "posted": {"date-parts": [[2025, 12, 1]]}},
    ]}}
    source = CrossrefChemRxivSource(client_for(lambda request: httpx.Response(200, json=payload)))

    records = list(source.discover(["lithium-ion battery"], limit=10))

    assert len(records) == 1
    record = records[0]
    assert record.doc_id == "chemrxiv:chemrxiv-2025-rvp45"
    assert record.license_raw == "https://creativecommons.org/licenses/by/4.0/"
    assert record.pdf_url.endswith("/v2")
    assert record.abstract == "Sulfite solvents tune solvation."
    assert record.raw["versions"] == ["10.26434/chemrxiv-2025-rvp45/v2", "10.26434/chemrxiv-2025-rvp45"]


def test_base_doi_strips_both_version_styles():
    assert base_doi("10.26434/chemrxiv-2025-rvp45/v2") == "10.26434/chemrxiv-2025-rvp45"
    assert base_doi("10.26434/chemrxiv.14226542.v3") == "10.26434/chemrxiv.14226542"


ARXIV_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record><header><identifier>oai:arXiv.org:2501.00001</identifier></header><metadata>
      <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
        <id>2501.00001</id><version version="v1"><date>Wed, 1 Jan 2025 10:00:00 GMT</date></version>
        <title>Suppressing   cracking in NMC811 cathodes for lithium-ion batteries</title>
        <authors>Jane Doe, John Roe and Ada Lovelace</authors>
        <categories>cond-mat.mtrl-sci physics.chem-ph</categories>
        <license>http://creativecommons.org/licenses/by/4.0/</license>
        <abstract>We study intergranular cracking.</abstract>
      </arXivRaw></metadata></record>
    <record><header><identifier>oai:arXiv.org:2501.00002</identifier></header><metadata>
      <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
        <id>2501.00002</id><title>Superconductivity in twisted bilayers</title>
        <authors>X. Y.</authors><categories>cond-mat.supr-con</categories>
        <abstract>No batteries here.</abstract>
      </arXivRaw></metadata></record>
    <record><header status="deleted"><identifier>oai:arXiv.org:2501.00003</identifier></header></record>
    <resumptionToken></resumptionToken>
  </ListRecords>
</OAI-PMH>"""


def test_arxiv_oai_filters_by_category_and_terms_and_reads_license():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, text=ARXIV_PAGE)

    source = ArxivOaiSource(client_for(handler), sets=["physics:cond-mat"], categories=["cond-mat.mtrl-sci"],
                            from_date="2025-01-01")
    records = list(source.discover(["lithium-ion"], limit=10))

    assert [r.external_id for r in records] == ["2501.00001"]
    record = records[0]
    assert record.title == "Suppressing cracking in NMC811 cathodes for lithium-ion batteries"
    assert record.authors == ["Jane Doe", "John Roe", "Ada Lovelace"]
    assert record.year == 2025
    assert record.license_raw == "http://creativecommons.org/licenses/by/4.0/"
    assert seen[0] == {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "physics:cond-mat", "from": "2025-01-01"}


def test_arxiv_no_records_match_is_not_an_error():
    empty = '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch"/></OAI-PMH>'
    source = ArxivOaiSource(client_for(lambda request: httpx.Response(200, text=empty)), sets=["s"], categories=[])
    assert list(source.discover(["lithium"], limit=5)) == []
