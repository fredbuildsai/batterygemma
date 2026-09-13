from sqlalchemy import func, select

from batterygemma.db.models import Chunk, Document, File
from batterygemma.db.session import get_session
from batterygemma.parse.jats import ParsedDocument, Section
from batterygemma.parse.pipeline import parse_and_chunk
from tests.test_parse import JATS

CHUNKING = {"target_tokens": 600, "max_tokens": 900, "overlap_tokens": 60, "max_garble_ratio": 0.05}


def words(text):
    return len(text.split())


def add_doc(s, doc_id, kind=None, path=None):
    doc = Document(doc_id=doc_id, source="openalex", external_id=doc_id.split(":")[1], title="t", norm_title="t",
                   status="fetched")
    if kind:
        doc.files.append(File(kind=kind, path=str(path), sha256="x", bytes=1))
    s.add(doc)
    return doc


def test_xml_becomes_chunks_with_global_ids_and_rerun_is_idempotent(engine, tmp_path):
    xml_path = tmp_path / "W1.xml"
    xml_path.write_bytes(JATS)
    with get_session(engine) as s:
        add_doc(s, "openalex:W1", "xml", xml_path)

    for _ in range(2):
        with get_session(engine) as s:
            assert parse_and_chunk(s, s.get(Document, "openalex:W1"), words, CHUNKING) == 2

    with get_session(engine) as s:
        chunks = s.scalars(select(Chunk).order_by(Chunk.order)).all()
        doc = s.get(Document, "openalex:W1")
        assert s.scalar(select(func.count()).select_from(Chunk)) == 2
    assert [c.chunk_id for c in chunks] == ["openalex:W1#s00-c00", "openalex:W1#s01-c00"]
    assert chunks[0].section_type == "introduction" and chunks[0].captions
    assert chunks[1].section_path == ["Experimental", "Electrochemical testing"]
    assert chunks[0].quality["flags"] == []
    assert doc.status == "chunked" and doc.status_reason == "parse:jats:2_chunks"
    assert doc.abstract == "We study intergranular cracking."


def test_pdf_uses_injected_parser_and_flags_garbled_chunks(engine, tmp_path):
    pdf_path = tmp_path / "W2.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    parsed = ParsedDocument(title="t", abstract="An abstract.", sections=[
        Section(path=["Results"], section_type="results", paragraphs=["Clean sentence about NMC811 cracking."]),
        Section(path=["Figure soup"], section_type="other", paragraphs=["0 . 9 H D ) i n / m 0 . 7 m ( n t e"]),
    ])
    seen = []

    def fake_pdf_parser(path):
        seen.append(path)
        return parsed

    with get_session(engine) as s:
        doc = add_doc(s, "openalex:W2", "pdf", pdf_path)
        assert parse_and_chunk(s, doc, words, CHUNKING, pdf_parser=fake_pdf_parser) == 2
        assert doc.status_reason == "parse:docling:2_chunks"
    with get_session(engine) as s:
        flags = [c.quality["flags"] for c in s.scalars(select(Chunk).order_by(Chunk.order))]
    assert seen == [pdf_path] and flags == [[], ["garbled"]]


def test_document_without_parsable_file_is_left_untouched(engine, tmp_path):
    with get_session(engine) as s:
        doc = add_doc(s, "openalex:W3")
        assert parse_and_chunk(s, doc, words, CHUNKING) == 0
        assert doc.status == "fetched" and doc.status_reason == "parse:no_parsable_file"
        pdf_only = add_doc(s, "openalex:W4", "pdf", tmp_path / "x.pdf")
        assert parse_and_chunk(s, pdf_only, words, CHUNKING, pdf_parser=None) == 0  # no PDF parser supplied
