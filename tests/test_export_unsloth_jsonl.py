import json

import pytest

from corpusforge.models import Chunk, Document, File

from batterygemma.db.models import QA, DPOPair, Negative
from batterygemma.db.session import get_session
from batterygemma.export.unsloth_jsonl import export_all, export_cpt, export_dpo, export_sft


def read_attribution(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", split="eval"))
        s.add(Document(doc_id="d3", source="t", external_id="3", title="t", norm_title="t", split=None))  # not split yet
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=10, text="Train chunk text.",
                    images=["fig1.jpg", "missing.jpg"]))  # missing.jpg was never downloaded
        s.add(Chunk(chunk_id="d2#c0", doc_id="d2", order=0, tokens=10, text="Eval chunk text."))
        s.add(File(doc_id="d1", kind="image", path="/data/images/t/1/fig1.jpg", sha256="x", bytes=1))
        s.add(Chunk(chunk_id="d3#c0", doc_id="d3", order=0, tokens=10, text="Unsplit chunk - must be excluded."))
        s.add(Chunk(chunk_id="d1#c1", doc_id="d1", order=1, tokens=0, text="   "))  # blank - must be excluded

        s.add(QA(
            id="qa1", doc_ids=["d1"], chunk_ids=["d1#c0"], status="accepted",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="cathode", answer_type="OPEN", system="custom system prompt",
            turns=[{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
        ))
        s.add(QA(
            id="qa2", doc_ids=["d1"], chunk_ids=["d1#c0"], status="generated",  # not accepted - must be excluded
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="cathode", answer_type="OPEN",
            turns=[{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
        ))
        s.add(Negative(
            id="neg1", doc_ids=["d2"], chunk_ids=["d2#c0"], status="generated",
            task_format="false_premise", polarity="negative", kind="false_premise",
            prompt="False premise question", flawed_element="x", expert_response="Corrected answer",
        ))
        s.add(DPOPair(
            id="dpo1", doc_ids=["d1"], chunk_ids=["d1#c0"], status="generated", error_type="wrong_mechanism",
            prompt=[{"role": "user", "content": "Q"}], chosen=[{"role": "assistant", "content": "right"}],
            rejected=[{"role": "assistant", "content": "wrong"}],
        ))
        s.add(DPOPair(
            id="dpo2", doc_ids=["d3"], chunk_ids=["d3#c0"], status="generated", error_type="overclaiming",
            prompt=[], chosen=[], rejected=[],  # doc d3 has no split - must be excluded
        ))


def test_export_cpt_splits_by_document_and_skips_blank_and_unsplit(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        counts = export_cpt(s, tmp_path, include_ontology=False)
    assert counts == {"train": 1, "eval": 1}
    train_rows = read_jsonl(tmp_path / "cpt_train.jsonl")
    eval_rows = read_jsonl(tmp_path / "cpt_eval.jsonl")
    assert train_rows == [{"text": "Train chunk text."}]
    assert eval_rows == [{"text": "Eval chunk text."}]


def test_export_cpt_appends_ontology_rows_to_train_only_when_requested(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        counts = export_cpt(s, tmp_path, include_ontology=True)

    # 1 real chunk row (train/eval as in the base test) plus every generated ontology row, all in train.
    assert counts["eval"] == 1
    assert counts["train"] > 1
    train_rows = read_jsonl(tmp_path / "cpt_train.jsonl")
    assert train_rows[0] == {"text": "Train chunk text."}  # real chunk row still comes first, unaffected
    assert all("text" in r and "images" not in r for r in train_rows[1:])

    manifest = read_attribution(tmp_path / "cpt_train.attribution.json")
    ontology_keys = [k for k in manifest if k.startswith("ontology:")]
    assert ontology_keys  # at least one ontology domain attributed
    assert all(manifest[k]["license"] == "CC-BY-4.0" for k in ontology_keys)


def test_export_sft_includes_accepted_qa_and_generated_negatives_with_system_prompt(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        counts = export_sft(s, tmp_path)
    assert counts == {"train": 1, "eval": 1}
    train_rows = read_jsonl(tmp_path / "sft_train.jsonl")
    eval_rows = read_jsonl(tmp_path / "sft_eval.jsonl")
    assert train_rows[0]["messages"] == [
        {"role": "system", "content": "custom system prompt"},
        {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
    ]
    assert eval_rows[0]["messages"][1] == {"role": "user", "content": "False premise question"}
    assert eval_rows[0]["messages"][2] == {"role": "assistant", "content": "Corrected answer"}


def test_export_dpo_writes_prompt_chosen_rejected_and_skips_unsplit_docs(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        counts = export_dpo(s, tmp_path)
    assert counts == {"train": 1, "eval": 0}
    train_rows = read_jsonl(tmp_path / "dpo_train.jsonl")
    assert train_rows == [{"prompt": [{"role": "user", "content": "Q"}],
                           "chosen": [{"role": "assistant", "content": "right"}],
                           "rejected": [{"role": "assistant", "content": "wrong"}]}]


def test_images_are_omitted_by_default(engine, tmp_path):
    """The default export carries no `images` field at all, even for a row whose chunk has a downloaded
    image - include_images must be explicitly opted into."""
    seed(engine)
    with get_session(engine) as s:
        export_cpt(s, tmp_path, include_ontology=False)
        export_sft(s, tmp_path)
        export_dpo(s, tmp_path)
    for path in ("cpt_train.jsonl", "sft_train.jsonl", "dpo_train.jsonl"):
        for row in read_jsonl(tmp_path / path):
            assert "images" not in row


def test_include_images_links_each_row_to_its_chunks_downloaded_figures(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        export_cpt(s, tmp_path, include_images=True, include_ontology=False)
        export_sft(s, tmp_path, include_images=True)
        export_dpo(s, tmp_path, include_images=True)

    cpt_row = read_jsonl(tmp_path / "cpt_train.jsonl")[0]
    assert cpt_row["images"] == ["/data/images/t/1/fig1.jpg"]  # missing.jpg silently dropped, not downloaded

    qa_row = read_jsonl(tmp_path / "sft_train.jsonl")[0]  # qa1, sourced from d1#c0
    assert qa_row["images"] == ["/data/images/t/1/fig1.jpg"]

    neg_row = read_jsonl(tmp_path / "sft_eval.jsonl")[0]  # neg1, sourced from d2#c0 - no images at all
    assert "images" not in neg_row

    dpo_row = read_jsonl(tmp_path / "dpo_train.jsonl")[0]  # dpo1, sourced from d1#c0
    assert dpo_row["images"] == ["/data/images/t/1/fig1.jpg"]


def test_export_all_writes_every_file(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        result = export_all(s, tmp_path, include_ontology=False)
    assert set(result) == {"cpt", "sft", "dpo"}
    for stage in ("cpt", "sft", "dpo"):
        assert (tmp_path / f"{stage}_train.jsonl").exists()
        assert (tmp_path / f"{stage}_eval.jsonl").exists()


def test_exported_cpt_jsonl_is_loadable_by_load_cpt_dataset(engine, tmp_path):
    pytest.importorskip("datasets")  # optional `train` extra
    from batterygemma.train.data import load_cpt_dataset

    seed(engine)
    with get_session(engine) as s:
        export_cpt(s, tmp_path, include_ontology=False)
    dataset = load_cpt_dataset(tmp_path / "cpt_train.jsonl")
    assert dataset[0]["text"] == "Train chunk text."


def test_exported_sft_jsonl_is_loadable_by_load_sft_dataset(engine, tmp_path):
    pytest.importorskip("datasets")  # optional `train` extra
    from batterygemma.train.data import load_sft_dataset

    seed(engine)
    with get_session(engine) as s:
        export_sft(s, tmp_path)
    dataset = load_sft_dataset(tmp_path / "sft_train.jsonl")
    assert dataset[0]["messages"][-1]["role"] == "assistant"


def test_export_cpt_packs_chunks_across_the_whole_document_up_to_the_token_budget(engine, tmp_path):
    """Packing is document-scoped only, not section-scoped: a real corpus measurement showed
    stopping at each section (even after merging subsections into their parent) left 94.2% of
    sections short of a 2000-token budget on their own, so packing now continues across a genuine
    section change too, as long as it's still the same document."""
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", split="train"))
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", section_path=["Intro"], order=0, tokens=40,
                    overlap_prev_tokens=0, text="First part of the introduction."))
        s.add(Chunk(chunk_id="d1#c1", doc_id="d1", section_path=["Intro"], order=1, tokens=40,
                    overlap_prev_tokens=0, text="Second part of the introduction."))
        # A different section within the SAME document now packs right along with Intro.
        s.add(Chunk(chunk_id="d1#m0", doc_id="d1", section_path=["Methods"], order=2, tokens=10,
                    overlap_prev_tokens=0, text="Methods section text."))
        s.add(Chunk(chunk_id="d1#m1", doc_id="d1", section_path=["Methods"], order=3, tokens=40,
                    overlap_prev_tokens=0, text="A chunk that overflows the 100-token budget."))
        # A different DOCUMENT must never be packed in, budget or not.
        s.add(Chunk(chunk_id="d2#c0", doc_id="d2", section_path=["Intro"], order=0, tokens=10,
                    overlap_prev_tokens=0, text="Unrelated other paper."))

    with get_session(engine) as s:
        export_cpt(s, tmp_path, pack_tokens=100, include_ontology=False)

    rows = read_jsonl(tmp_path / "cpt_train.jsonl")
    texts = {r["text"] for r in rows}
    assert ("First part of the introduction.\n\nSecond part of the introduction.\n\n"
            "Methods section text.") in texts  # 40+40+10=90 <= 100, crosses the Intro->Methods boundary
    assert "A chunk that overflows the 100-token budget." in texts  # own row: 90+40 > 100
    assert "Unrelated other paper." in texts  # never merged across a document boundary
    assert len(rows) == 3


def test_export_cpt_without_pack_tokens_keeps_one_row_per_chunk(engine, tmp_path):
    """Default behavior (pack_tokens unset) is unchanged: one row per chunk."""
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", section_path=["Intro"], order=0, tokens=10, text="A."))
        s.add(Chunk(chunk_id="d1#c1", doc_id="d1", section_path=["Intro"], order=1, tokens=10, text="B."))

    with get_session(engine) as s:
        counts = export_cpt(s, tmp_path, include_ontology=False)

    assert counts == {"train": 2, "eval": 0}


def test_export_cpt_strips_the_overlap_prefix_shared_with_the_previous_chunk(engine, tmp_path):
    """parse/chunk.py prepends up to overlap_tokens worth of the previous chunk's trailing sentences,
    joined by a space, followed by a blank line, then the chunk's own body - see its module docstring.
    CPT training must use the body only: that overlap exists for retrieval-style grounding, not for a
    language-modeling objective, where it would just be the same tokens getting extra gradient updates."""
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=5, overlap_prev_tokens=0,
                    text="First chunk, no overlap."))
        s.add(Chunk(chunk_id="d1#c1", doc_id="d1", order=1, tokens=12, overlap_prev_tokens=4,
                    text="Trailing sentence from chunk zero.\n\nSecond chunk's own new body text."))

    with get_session(engine) as s:
        export_cpt(s, tmp_path, include_ontology=False)

    rows = read_jsonl(tmp_path / "cpt_train.jsonl")
    assert rows[0]["text"] == "First chunk, no overlap."  # unchanged: nothing to strip
    assert rows[1]["text"] == "Second chunk's own new body text."  # overlap prefix dropped


def test_blacklisted_documents_are_excluded_from_every_export(engine, tmp_path):
    """Blacklisting must retroactively drop already-generated content, not just stop future
    generation - otherwise a document blacklisted for copyright (or any other) reasons after its
    chunks/QA/DPO rows were already produced would still leak into the export."""
    seed(engine)
    with get_session(engine) as s:
        s.get(Document, "d1").blacklisted = True
    with get_session(engine) as s:
        cpt_counts = export_cpt(s, tmp_path, include_ontology=False)
        sft_counts = export_sft(s, tmp_path)
        dpo_counts = export_dpo(s, tmp_path)
    assert cpt_counts == {"train": 0, "eval": 1}  # d1's chunk dropped, d2's kept
    assert sft_counts == {"train": 0, "eval": 1}  # qa1 (d1) dropped, neg1 (d2) kept
    assert dpo_counts == {"train": 0, "eval": 0}  # dpo1 (d1) dropped


def test_attribution_manifest_traces_every_row_back_to_its_source_document(engine, tmp_path):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="Paper One", norm_title="t",
                        split="train", doi="10.1/one", license="CC-BY"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="Paper Two", norm_title="t",
                        split="train", license="all-rights-reserved"))  # no DOI - falls back to doc_id
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=10, text="First chunk."))
        s.add(Chunk(chunk_id="d1#c1", doc_id="d1", order=1, tokens=10, text="Second chunk."))
        s.add(Chunk(chunk_id="d2#c0", doc_id="d2", order=0, tokens=10, text="Third chunk."))

    with get_session(engine) as s:
        export_cpt(s, tmp_path, include_ontology=False)

    manifest = read_attribution(tmp_path / "cpt_train.attribution.json")
    assert manifest["10.1/one"]["license"] == "CC-BY"
    assert manifest["10.1/one"]["title"] == "Paper One"
    assert sorted(manifest["10.1/one"]["chunks"]) == [0, 1]
    assert manifest["d2"]["license"] == "all-rights-reserved"  # keyed by doc_id: no DOI
    assert manifest["d2"]["chunks"] == [2]
