from pathlib import Path

import pytest
from sqlalchemy import select

from batterygemma.db.models import Document, File
from batterygemma.db.session import get_session
from batterygemma.screen.license import REJECTED, evaluate_license
from batterygemma.sources.local import DEFAULT_LOCAL_LICENSE, add_local_pdf, attach_local_pdf

PDF = b"%PDF-1.7 a small fake book"
ALLOW, FLAG = ["CC0", "CC-BY", "public-domain"], ["CC-BY-SA"]


def test_add_local_pdf_defaults_to_a_non_open_license_and_is_immediately_fetched(engine, tmp_path):
    book = tmp_path / "source" / "Handbook_Of_Batteries.pdf"
    book.parent.mkdir()
    book.write_bytes(PDF)

    with get_session(engine) as s:
        doc = add_local_pdf(s, book, tmp_path / "raw")

    assert doc.source == "local" and doc.status == "fetched"
    assert doc.license == DEFAULT_LOCAL_LICENSE
    assert doc.title == "Handbook Of Batteries"
    assert evaluate_license(doc.license, ALLOW, FLAG) == REJECTED  # excluded from the open/CC-only track

    with get_session(engine) as s:
        stored = s.get(Document, doc.doc_id)
        files = s.scalars(select(File).where(File.doc_id == doc.doc_id)).all()
    assert stored.status == "fetched"
    assert files[0].kind == "pdf" and files[0].bytes == len(PDF)
    assert Path(files[0].path).read_bytes() == PDF


def test_re_adding_the_same_file_is_a_no_op(engine, tmp_path):
    book = tmp_path / "b.pdf"
    book.write_bytes(PDF)

    with get_session(engine) as s:
        first = add_local_pdf(s, book, tmp_path / "raw")
    with get_session(engine) as s:
        second = add_local_pdf(s, book, tmp_path / "raw")

    assert first.doc_id == second.doc_id
    with get_session(engine) as s:
        assert s.scalar(select(File).where(File.doc_id == first.doc_id).with_only_columns(File.id)) is not None
        count = len(s.scalars(select(File).where(File.doc_id == first.doc_id)).all())
    assert count == 1  # not duplicated


def test_explicit_open_license_is_honoured_for_files_the_user_has_rights_to_release(engine, tmp_path):
    book = tmp_path / "c.pdf"
    book.write_bytes(PDF)
    with get_session(engine) as s:
        doc = add_local_pdf(s, book, tmp_path / "raw", license="CC-BY-4.0", title="My Own Open Notes")
    assert doc.license == "CC-BY-4.0" and doc.title == "My Own Open Notes"
    assert evaluate_license(doc.license, ALLOW, FLAG) != REJECTED


def test_missing_file_raises(tmp_path, engine):
    with get_session(engine) as s, pytest.raises(FileNotFoundError):
        add_local_pdf(s, tmp_path / "does-not-exist.pdf", tmp_path / "raw")


def test_attach_local_pdf_preserves_the_original_document_metadata(engine, tmp_path):
    with get_session(engine) as s:
        s.add(Document(
            doc_id="openalex:W1", source="openalex", external_id="W1", title="A Real Paper", norm_title="a real paper",
            doi="10.1/x", license="CC-BY", status="accepted", status_reason="fetch:bot_protection",
        ))

    manual = tmp_path / "manually_downloaded.pdf"
    manual.write_bytes(PDF)
    with get_session(engine) as s:
        doc = attach_local_pdf(s, manual, tmp_path / "raw", "openalex:W1")

    assert doc.doc_id == "openalex:W1"
    assert doc.title == "A Real Paper" and doc.doi == "10.1/x" and doc.license == "CC-BY"  # unchanged
    assert doc.status == "fetched" and doc.status_reason == "local:manually_attached:manually_downloaded.pdf"

    with get_session(engine) as s:
        files = s.scalars(select(File).where(File.doc_id == "openalex:W1")).all()
    assert len(files) == 1 and files[0].kind == "pdf"
    assert Path(files[0].path).read_bytes() == PDF


def test_attach_local_pdf_unknown_doc_id_raises(engine, tmp_path):
    manual = tmp_path / "x.pdf"
    manual.write_bytes(PDF)
    with get_session(engine) as s, pytest.raises(KeyError):
        attach_local_pdf(s, manual, tmp_path / "raw", "openalex:does-not-exist")


def test_attach_local_pdf_reattaching_the_same_file_does_not_duplicate(engine, tmp_path):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="openalex", external_id="1", title="t", norm_title="t",
                       license="CC-BY", status="accepted"))
    manual = tmp_path / "x.pdf"
    manual.write_bytes(PDF)
    with get_session(engine) as s:
        attach_local_pdf(s, manual, tmp_path / "raw", "d1")
    with get_session(engine) as s:
        attach_local_pdf(s, manual, tmp_path / "raw", "d1")
    with get_session(engine) as s:
        files = s.scalars(select(File).where(File.doc_id == "d1")).all()
    assert len(files) == 1
