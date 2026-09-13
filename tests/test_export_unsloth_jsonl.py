import json

from batterygemma.db.models import Chunk, DPOPair, Document, Negative, QA
from batterygemma.db.session import get_session
from batterygemma.export.unsloth_jsonl import export_all, export_cpt, export_dpo, export_sft


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="train"))
        s.add(Document(doc_id="d2", source="t", external_id="2", title="t", norm_title="t", split="eval"))
        s.add(Document(doc_id="d3", source="t", external_id="3", title="t", norm_title="t", split=None))  # not split yet
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=10, text="Train chunk text."))
        s.add(Chunk(chunk_id="d2#c0", doc_id="d2", order=0, tokens=10, text="Eval chunk text."))
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
        counts = export_cpt(s, tmp_path)
    assert counts == {"train": 1, "eval": 1}
    train_rows = read_jsonl(tmp_path / "cpt_train.jsonl")
    eval_rows = read_jsonl(tmp_path / "cpt_eval.jsonl")
    assert train_rows == [{"text": "Train chunk text."}]
    assert eval_rows == [{"text": "Eval chunk text."}]


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


def test_export_all_writes_every_file(engine, tmp_path):
    seed(engine)
    with get_session(engine) as s:
        result = export_all(s, tmp_path)
    assert set(result) == {"cpt", "sft", "dpo"}
    for stage in ("cpt", "sft", "dpo"):
        assert (tmp_path / f"{stage}_train.jsonl").exists()
        assert (tmp_path / f"{stage}_eval.jsonl").exists()


def test_exported_cpt_jsonl_is_loadable_by_load_cpt_dataset(engine, tmp_path):
    from batterygemma.train.data import load_cpt_dataset

    seed(engine)
    with get_session(engine) as s:
        export_cpt(s, tmp_path)
    dataset = load_cpt_dataset(tmp_path / "cpt_train.jsonl")
    assert dataset[0]["text"] == "Train chunk text."


def test_exported_sft_jsonl_is_loadable_by_load_sft_dataset(engine, tmp_path):
    from batterygemma.train.data import load_sft_dataset

    seed(engine)
    with get_session(engine) as s:
        export_sft(s, tmp_path)
    dataset = load_sft_dataset(tmp_path / "sft_train.jsonl")
    assert dataset[0]["messages"][-1]["role"] == "assistant"
