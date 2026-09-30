import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.db.models import Chunk, Document, GenTask, QA
from batterygemma.db.session import get_session
from batterygemma.generate.qa import annotate_chunk_qa, generate_qa
from batterygemma.llm.router import LLMRouter

CHUNK_ID = "doc:1#s00-c00"
CHUNK_TEXT = (
    "Above 4.2 V vs Li/Li+, NMC811 undergoes the H2->H3 phase transition, accompanied by an abrupt "
    "contraction of the c lattice parameter, which drives intergranular cracking."
)

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"qa": ["gen"]},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")


GOOD_RESPONSE = json.dumps({
    "items": [{
        "question": "Why does capacity fade accelerate above 4.2 V in NMC811?",
        "answer": "Above 4.2 V, NMC811 undergoes the H2->H3 phase transition, causing abrupt c-axis "
                  "contraction that drives intergranular cracking.",
        "reasoning": "H2->H3 above 4.2V -> c-axis collapse -> anisotropic strain -> intergranular cracks.",
        "question_type": "mechanism", "answer_type": "OPEN", "component": "cathode", "chemistry": "NMC811",
    }]
})

UNGROUNDED_RESPONSE = json.dumps({
    "items": [{
        "question": "What causes graphite exfoliation?", "answer": "Solvent co-intercalation, unrelated here.",
        "reasoning": "unrelated", "question_type": "mechanism", "answer_type": "OPEN",
        "component": "anode", "chemistry": None,
    }]
})


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


def seed_chunk(engine, chunk_id=CHUNK_ID, text=CHUNK_TEXT):
    with get_session(engine) as s:
        doc = Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0")
        s.add(doc)
        s.add(Chunk(chunk_id=chunk_id, doc_id="doc:1", order=0, tokens=100, text=text))


def test_generate_qa_persists_grounded_items(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE])
    with get_session(engine) as s:
        rows = generate_qa(s, router, s.get(Chunk, CHUNK_ID))
    assert len(rows) == 1
    with get_session(engine) as s:
        qa = s.get(QA, f"{CHUNK_ID}#qa0")
    assert qa.status == "generated" and qa.question_type == "mechanism" and qa.component == "cathode"
    assert qa.turns[0]["role"] == "user" and qa.turns[1]["role"] == "assistant"
    assert qa.doc_ids == ["doc:1"] and qa.license == "CC-BY-4.0"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "QAGenerationOut"


def test_ungrounded_answer_is_rejected(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        rows = generate_qa(s, router, s.get(Chunk, CHUNK_ID))
    assert rows[0].status == "rejected" and rows[0].reject_reason == "ungrounded_answer"


def test_annotate_chunk_qa_is_resumable(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE, GOOD_RESPONSE])
    assert annotate_chunk_qa(engine, router, CHUNK_ID) == "done"
    assert len(calls) == 1
    assert annotate_chunk_qa(engine, router, CHUNK_ID) == "skipped"
    assert len(calls) == 1
    assert annotate_chunk_qa(engine, router, CHUNK_ID, force=True) == "done"
    assert len(calls) == 2
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"generate_qa:{CHUNK_ID}")).one()
        assert task.status == "done" and task.attempts == 2


def test_annotate_chunk_qa_records_failure(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, ["not json"])
    assert annotate_chunk_qa(engine, router, CHUNK_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"generate_qa:{CHUNK_ID}")).one()
        assert task.status == "failed"


def test_rerun_replaces_previous_qa_for_the_same_chunk(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [GOOD_RESPONSE, UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        generate_qa(s, router, s.get(Chunk, CHUNK_ID))
    with get_session(engine) as s:
        generate_qa(s, router, s.get(Chunk, CHUNK_ID), use_cache=False)
    with get_session(engine) as s:
        rows = s.scalars(select(QA)).all()
    assert len(rows) == 1 and rows[0].reject_reason == "ungrounded_answer"
