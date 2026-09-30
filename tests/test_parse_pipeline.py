from sqlalchemy import func, select

from batterygemma.db.models import Chunk, Document, File
from batterygemma.db.session import get_session
from batterygemma.parse.jats import ParsedDocument, Section
from batterygemma.parse.pipeline import parse_and_chunk, reconstruct_sections_from_chunks, rechunk_from_existing_chunks
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


def test_reconstruct_sections_from_chunks_strips_overlap_and_regroups_by_section_path():
    chunks = [
        Chunk(chunk_id="d#s00-c00", doc_id="d", section_path=["Intro"], section_type="introduction",
              order=0, tokens=4, overlap_prev_tokens=0, text="First paragraph of the intro.",
              captions=["Fig 1."], images=["fig1.jpg"]),
        Chunk(chunk_id="d#s00-c01", doc_id="d", section_path=["Intro"], section_type="introduction",
              order=1, tokens=5, overlap_prev_tokens=2,
              text="of the intro.\n\nSecond paragraph, new content only."),
        Chunk(chunk_id="d#s01-c00", doc_id="d", section_path=["Methods"], section_type="methods",
              order=2, tokens=3, overlap_prev_tokens=0, text="Methods paragraph."),
    ]
    sections = reconstruct_sections_from_chunks(chunks)

    assert [s.path for s in sections] == [["Intro"], ["Methods"]]  # order preserved via min(order) per group
    intro = sections[0]
    # the overlap prefix ("of the intro.") from chunk 2 must not be duplicated
    assert intro.paragraphs == ["First paragraph of the intro.", "Second paragraph, new content only."]
    assert intro.captions == ["Fig 1."]
    assert intro.caption_images == [["fig1.jpg"]]
    assert sections[1].paragraphs == ["Methods paragraph."]
    assert sections[1].captions == []


def test_rechunk_from_existing_chunks_merges_small_chunks_without_reparsing_the_raw_file(engine):
    """The whole point: re-chunk using only what's already in the chunks table, never touching a
    PDF/XML file or a pdf_parser - the document here doesn't even have a File row."""
    with get_session(engine) as s:
        doc = Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", status="chunked")
        s.add(doc)
        s.add(Chunk(chunk_id="d1#s00-c00", doc_id="d1", section_path=["Body"], order=0, tokens=6,
                    overlap_prev_tokens=0, text="Short first chunk from the old small-chunk scheme."))
        s.add(Chunk(chunk_id="d1#s00-c01", doc_id="d1", section_path=["Body"], order=1, tokens=6,
                    overlap_prev_tokens=0, text="Short second chunk, still the same section."))

    big_chunking = {"target_tokens": 600, "max_tokens": 900, "overlap_tokens": 60, "max_garble_ratio": 0.05}
    with get_session(engine) as s:
        n = rechunk_from_existing_chunks(s, s.get(Document, "d1"), words, big_chunking)
        assert n == 1  # both old chunks now fit in one, well under the much bigger target

    with get_session(engine) as s:
        chunks = s.scalars(select(Chunk).where(Chunk.doc_id == "d1")).all()
        doc = s.get(Document, "d1")
    assert len(chunks) == 1
    assert chunks[0].text == ("Short first chunk from the old small-chunk scheme.\n\n"
                               "Short second chunk, still the same section.")
    assert doc.status_reason == "parse:rechunk_from_existing:1_chunks"


def test_rechunk_from_existing_chunks_returns_zero_when_document_has_no_chunks(engine):
    with get_session(engine) as s:
        doc = Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", status="chunked")
        s.add(doc)
    with get_session(engine) as s:
        assert rechunk_from_existing_chunks(s, s.get(Document, "d2"), words, CHUNKING) == 0
