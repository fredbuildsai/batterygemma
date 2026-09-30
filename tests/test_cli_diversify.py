from batterygemma.cli import _diversify_by_doc


def test_round_robins_across_documents_preserving_each_docs_order():
    chunk_ids = [
        "big:doc#c0", "big:doc#c1", "big:doc#c2", "big:doc#c3",
        "small:doc#c0",
        "medium:doc#c0", "medium:doc#c1",
    ]
    result = _diversify_by_doc(chunk_ids)
    # first pass takes one from each doc (in first-seen order), then continues round-robin
    assert result == [
        "big:doc#c0", "small:doc#c0", "medium:doc#c0",
        "big:doc#c1", "medium:doc#c1",
        "big:doc#c2",
        "big:doc#c3",
    ]


def test_a_single_huge_document_cannot_starve_a_bounded_limit():
    huge = [f"huge:doc#c{i}" for i in range(700)]
    others = [f"paper{i}:doc#c0" for i in range(50)]
    result = _diversify_by_doc(huge + others)[:100]
    doc_ids = {c.split("#", 1)[0] for c in result}
    assert len(doc_ids) == 51  # every other document got at least one slot within the first 100


def test_empty_and_single_doc_inputs():
    assert _diversify_by_doc([]) == []
    assert _diversify_by_doc(["a:d#c0", "a:d#c1"]) == ["a:d#c0", "a:d#c1"]
