from batterygemma.db.models import Chunk, Document, QA
from batterygemma.db.session import get_session
from batterygemma.verify.dedupe import find_duplicate_ids, mark_duplicates


def test_find_duplicate_ids_flags_near_identical_text_within_a_bucket():
    items = [
        ("a", "mechanism|cathode", "Why does capacity fade above 4.2 V in NMC811?"),
        ("b", "mechanism|cathode", "Why does capacity fade above 4.2V in NMC811 cathodes?"),  # near-duplicate of a
        ("c", "mechanism|cathode", "Why does silicon anode capacity fade during cycling?"),  # distinct
    ]
    duplicates = find_duplicate_ids(items, threshold=0.6)
    assert duplicates == {"b"}  # "a" is kept (first seen), "b" is flagged, "c" is distinct enough


def test_find_duplicate_ids_does_not_compare_across_buckets():
    items = [
        ("a", "mechanism|cathode", "Why does capacity fade above 4.2 V?"),
        ("b", "trade_off|cathode", "Why does capacity fade above 4.2 V?"),  # identical text, different bucket
    ]
    assert find_duplicate_ids(items, threshold=0.9) == set()


def test_find_duplicate_ids_handles_empty_input():
    assert find_duplicate_ids([]) == set()


def add_qa(s, qa_id, question, question_type="mechanism", component="cathode", status="accepted"):
    s.add(QA(
        id=qa_id, doc_ids=["doc:1"], chunk_ids=["doc:1#s00-c00"], status=status,
        task_format="instruction_response", polarity="positive", question_type=question_type,
        component=component, answer_type="OPEN",
        turns=[{"role": "user", "content": question}, {"role": "assistant", "content": "answer"}],
    ))


def test_mark_duplicates_rejects_near_duplicate_qa_rows(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:1", source="t", external_id="1", title="t", norm_title="t"))
        s.add(Chunk(chunk_id="doc:1#s00-c00", doc_id="doc:1", order=0, tokens=10, text="x"))
        add_qa(s, "qa1", "Why does capacity fade above 4.2 V in NMC811?")
        add_qa(s, "qa2", "Why does capacity fade above 4.2V in NMC811 cathodes?")  # near-dup of qa1
        add_qa(s, "qa3", "How does FEC additive affect silicon anode SEI?")
        add_qa(s, "qa4", "Why does capacity fade above 4.2 V in NMC811?", status="rejected")  # not "accepted"; ignored

    with get_session(engine) as s:
        n = mark_duplicates(s, QA, bucket_columns=("question_type", "component"),
                            text_fn=lambda row: row.turns[0]["content"], threshold=0.6)
    assert n == 1

    with get_session(engine) as s:
        assert s.get(QA, "qa1").status == "accepted"
        assert s.get(QA, "qa2").status == "rejected" and s.get(QA, "qa2").reject_reason == "near_duplicate"
        assert s.get(QA, "qa3").status == "accepted"
        assert s.get(QA, "qa4").status == "rejected" and s.get(QA, "qa4").reject_reason is None  # untouched
