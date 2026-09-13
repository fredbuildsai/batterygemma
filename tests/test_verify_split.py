from batterygemma.db.models import Document
from batterygemma.db.session import get_session
from batterygemma.verify.split import assign_document_splits, split_for_doc


def test_split_for_doc_is_deterministic():
    assert split_for_doc("openalex:W1") == split_for_doc("openalex:W1")


def test_split_for_doc_roughly_matches_the_requested_fraction():
    splits = [split_for_doc(f"doc:{i}", eval_fraction=0.10) for i in range(2000)]
    eval_share = splits.count("eval") / len(splits)
    assert 0.06 < eval_share < 0.14


def test_assign_document_splits_only_touches_chunked_unsplit_documents(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", status="chunked"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", status="chunked",
                       split="train"))  # already split - must not be reassigned
        s.add(Document(doc_id="d3", source="t", external_id="3", title="t", norm_title="t", status="accepted"))

    with get_session(engine) as s:
        counts = assign_document_splits(s)

    with get_session(engine) as s:
        d1, d2, d3 = s.get(Document, "d1"), s.get(Document, "d2"), s.get(Document, "d3")
    assert d1.split in ("train", "eval")
    assert d2.split == "train"  # untouched
    assert d3.split is None  # not chunked, so not split
    assert sum(counts.values()) == 1


def test_assign_document_splits_is_idempotent(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", status="chunked"))
    with get_session(engine) as s:
        assign_document_splits(s)
    with get_session(engine) as s:
        first_split = s.get(Document, "d1").split
    with get_session(engine) as s:
        counts = assign_document_splits(s)  # nothing left to assign
    assert counts == {}
    with get_session(engine) as s:
        assert s.get(Document, "d1").split == first_split
