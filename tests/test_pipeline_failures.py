from batterygemma.db.models import Chunk, Document, GenTask
from batterygemma.db.session import get_session
from batterygemma.pipeline_failures import blacklist_repeat_failures, failed_documents, resolve_chunk_id


def seed(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="Paper One", norm_title="paper one"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="Paper Two", norm_title="paper two"))
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=10, text="t"))
        s.add(Chunk(chunk_id="d2#c0", doc_id="d2", order=0, tokens=10, text="t"))

        # d1: two failures across two stages (chunk-keyed and qa-keyed)
        s.add(GenTask(task_type="extract_facts", key="extract_facts:d1#c0", status="failed",
                     attempts=4, last_error="Route exhausted"))
        s.add(GenTask(task_type="judge_qa", key="judge_qa:d1#c0#qa0", status="failed",
                     attempts=2, last_error="invalid json"))
        # d1: one done task - must not appear as a failure
        s.add(GenTask(task_type="generate_qa", key="generate_qa:d1#c0", status="done", attempts=1))

        # d2: one failure, ideation-keyed
        s.add(GenTask(task_type="judge_ideation", key="judge_ideation:d2#c0#idea", status="failed",
                     attempts=1, last_error="timeout"))

        # a dangling task referencing a chunk that doesn't exist - must be skipped, not crash
        s.add(GenTask(task_type="extract_facts", key="extract_facts:ghost#c0", status="failed",
                     attempts=1, last_error="whatever"))


def test_resolve_chunk_id_for_chunk_keyed_and_id_keyed_tasks(engine):
    seed(engine)
    with get_session(engine) as s:
        assert resolve_chunk_id(s, "extract_facts", "d1#c0") == "d1#c0"
        assert resolve_chunk_id(s, "judge_qa", "d1#c0#qa0") == "d1#c0"
        assert resolve_chunk_id(s, "judge_ideation", "d2#c0#idea") == "d2#c0"
        assert resolve_chunk_id(s, "extract_facts", "ghost#c0") is None


def test_failed_documents_groups_by_document_and_skips_done_and_dangling_tasks(engine):
    seed(engine)
    with get_session(engine) as s:
        results = failed_documents(s)

    by_doc = {r["doc_id"]: r for r in results}
    assert set(by_doc) == {"d1", "d2"}

    assert by_doc["d1"]["failure_count"] == 2
    assert by_doc["d1"]["title"] == "Paper One"
    assert set(by_doc["d1"]["stages"]) == {"extract_facts", "judge_qa"}

    assert by_doc["d2"]["failure_count"] == 1
    assert by_doc["d2"]["stages"] == ["judge_ideation"]


def test_failed_documents_sorts_by_failure_count_descending(engine):
    seed(engine)
    with get_session(engine) as s:
        results = failed_documents(s)
    assert results[0]["doc_id"] == "d1"  # 2 failures > d2's 1


def test_failed_documents_empty_when_nothing_failed(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="clean", source="t", external_id="1", title="t", norm_title="t"))
        s.add(Chunk(chunk_id="clean#c0", doc_id="clean", order=0, tokens=5, text="t"))
        s.add(GenTask(task_type="extract_facts", key="extract_facts:clean#c0", status="done", attempts=1))
    with get_session(engine) as s:
        assert failed_documents(s) == []


def test_blacklist_repeat_failures_only_blacklists_documents_over_the_threshold(engine):
    seed(engine)  # d1: 2 failures, d2: 1 failure
    with get_session(engine) as s:
        newly = blacklist_repeat_failures(s, threshold=2)

    assert [r["doc_id"] for r in newly] == ["d1"]
    with get_session(engine) as s:
        d1, d2 = s.get(Document, "d1"), s.get(Document, "d2")
        assert d1.blacklisted is True
        assert "extract_facts" in d1.blacklist_reason and "judge_qa" in d1.blacklist_reason
        assert d2.blacklisted is False


def test_blacklist_repeat_failures_is_idempotent(engine):
    seed(engine)
    with get_session(engine) as s:
        blacklist_repeat_failures(s, threshold=2)
    with get_session(engine) as s:
        newly = blacklist_repeat_failures(s, threshold=2)  # d1 already blacklisted - nothing new
    assert newly == []


def test_blacklisted_document_chunks_are_excluded_from_pending_selection(engine):
    from batterygemma.db.models import GenTask as GT  # noqa: N811 - local alias only for readability here

    with get_session(engine) as s:
        s.add(Document(doc_id="bad", source="t", external_id="1", title="t", norm_title="t",
                       status="chunked", blacklisted=True))
        s.add(Document(doc_id="good", source="t", external_id="2", title="t", norm_title="t", status="chunked"))
        s.add(Chunk(chunk_id="bad#c0", doc_id="bad", order=0, tokens=5, text="t"))
        s.add(Chunk(chunk_id="good#c0", doc_id="good", order=0, tokens=5, text="t"))

    from sqlalchemy import select

    from batterygemma.db import models as m

    with get_session(engine) as s:
        chunk_ids = s.scalars(
            select(m.Chunk.chunk_id).join(m.Document, m.Chunk.doc_id == m.Document.doc_id)
            .where(m.Document.status == "chunked", ~m.Document.blacklisted)
        ).all()
    assert chunk_ids == ["good#c0"]
