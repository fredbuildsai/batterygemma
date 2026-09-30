import json
from types import SimpleNamespace

import pytest
from corpusforge.models import Chunk, Document, GenTask
from llmrouter_free import LLMRouter
from sqlalchemy import select

from batterygemma.db.models import QA, DPOPair
from batterygemma.db.session import get_session
from batterygemma.generate.dpo import annotate_qa_dpo, generate_dpo_pair

QA_ID = "doc:1#s00-c00#qa0"

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"negatives": ["gen"]},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")


def make_router(engine, text_by_call):
    calls = []

    def completion(**kw):
        calls.append(kw)
        text = text_by_call[len(calls) - 1] if len(calls) <= len(text_by_call) else text_by_call[-1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

    return LLMRouter(CONFIG, engine=engine, completion_fn=completion), calls


def seed_qa(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0"))
        s.add(Chunk(chunk_id="doc:1#s00-c00", doc_id="doc:1", order=0, tokens=50, text="LiPF6 hydrolyzes."))
        s.add(QA(
            id=QA_ID, doc_ids=["doc:1"], chunk_ids=["doc:1#s00-c00"], license="CC-BY-4.0",
            generator_model="p/gen", tier="silver", status="generated",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="electrolyte", answer_type="OPEN",
            turns=[{"role": "user", "content": "Why does LiPF6 form HF?"},
                   {"role": "assistant", "content": "Trace water hydrolyzes LiPF6, producing HF."}],
        ))


GOOD_REJECTION = json.dumps({"rejected": "HF forms because LiPF6 reduces at the anode.", "error_type": "wrong_mechanism"})


def test_generate_dpo_pair_persists_chosen_and_rejected(engine):
    seed_qa(engine)
    router, calls = make_router(engine, [GOOD_REJECTION])
    with get_session(engine) as s:
        row = generate_dpo_pair(s, router, s.get(QA, QA_ID))
    assert row.error_type == "wrong_mechanism"
    with get_session(engine) as s:
        stored = s.get(DPOPair, f"{QA_ID}#dpo")
    assert stored.source_qa_id == QA_ID and stored.component == "electrolyte"
    assert stored.chosen[0]["content"] == "Trace water hydrolyzes LiPF6, producing HF."
    assert stored.rejected[0]["content"] == "HF forms because LiPF6 reduces at the anode."
    assert stored.prompt[0]["content"] == "Why does LiPF6 form HF?"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "DPORejectionOut"


def test_annotate_qa_dpo_is_resumable(engine):
    seed_qa(engine)
    router, calls = make_router(engine, [GOOD_REJECTION, GOOD_REJECTION])
    assert annotate_qa_dpo(engine, router, QA_ID) == "done"
    assert annotate_qa_dpo(engine, router, QA_ID) == "skipped"
    assert len(calls) == 1
    assert annotate_qa_dpo(engine, router, QA_ID, force=True) == "done"
    assert len(calls) == 2


def test_annotate_qa_dpo_records_failure(engine):
    seed_qa(engine)
    router, _ = make_router(engine, ["not json"])
    assert annotate_qa_dpo(engine, router, QA_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"generate_dpo:{QA_ID}")).one()
        assert task.status == "failed"


def test_rerun_replaces_previous_dpo_pair(engine):
    seed_qa(engine)
    other = json.dumps({"rejected": "different rejection", "error_type": "overclaiming"})
    router, _ = make_router(engine, [GOOD_REJECTION, other])
    with get_session(engine) as s:
        generate_dpo_pair(s, router, s.get(QA, QA_ID))
    with get_session(engine) as s:
        generate_dpo_pair(s, router, s.get(QA, QA_ID), use_cache=False)
    with get_session(engine) as s:
        rows = s.scalars(select(DPOPair)).all()
    assert len(rows) == 1 and rows[0].error_type == "overclaiming"
