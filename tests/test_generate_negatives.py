import json
from types import SimpleNamespace

import pytest
from corpusforge.models import Chunk, Document, GenTask
from llmrouter_free import LLMRouter
from sqlalchemy import select

from batterygemma.db.models import ClaimPair, Negative
from batterygemma.db.session import get_session
from batterygemma.generate.negatives import (
    annotate_chunk_false_premise,
    derive_contradiction_negatives,
    generate_false_premise,
)

CHUNK_ID = "doc:1#s00-c00"
CHUNK_TEXT = "LiFePO4 stores charge on a flat two-phase Fe2+/Fe3+ plateau near 3.45 V vs Li/Li+."

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


def seed_chunk(engine, chunk_id=CHUNK_ID, text=CHUNK_TEXT):
    with get_session(engine) as s:
        doc = Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0")
        s.add(doc)
        s.add(Chunk(chunk_id=chunk_id, doc_id="doc:1", order=0, tokens=100, text=text))


GOOD_FP = json.dumps({
    "prompt": "Why does raising LiFePO4's charge cutoff to 4.5V give much higher capacity?",
    "flawed_element": "Assumes extra voltage above the plateau adds capacity.",
    "expert_response": "LiFePO4 stores charge on a flat two-phase Fe2+/Fe3+ plateau near 3.45 V; there is no "
                       "further redox to access above it, so raising the cutoff adds no real capacity.",
    "reasoning": "The plateau reaction is complete once all Fe2+ converts; extra voltage just oxidizes electrolyte.",
})

EMPTY_FP = json.dumps({"prompt": None, "flawed_element": None, "expert_response": None, "reasoning": None})


def test_generate_false_premise_persists_a_grounded_expert_response(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_FP])
    with get_session(engine) as s:
        row = generate_false_premise(s, router, s.get(Chunk, CHUNK_ID))
    assert row is not None
    with get_session(engine) as s:
        stored = s.get(Negative, f"{CHUNK_ID}#negfp")
    assert stored.kind == "false_premise" and stored.status == "generated"
    assert "3.45" in stored.expert_response
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "FalsePremiseOut"


def test_generate_false_premise_returns_none_when_model_finds_nothing(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [EMPTY_FP])
    with get_session(engine) as s:
        row = generate_false_premise(s, router, s.get(Chunk, CHUNK_ID))
    assert row is None
    with get_session(engine) as s:
        assert s.get(Negative, f"{CHUNK_ID}#negfp") is None


def test_annotate_chunk_false_premise_is_resumable(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_FP, GOOD_FP])
    assert annotate_chunk_false_premise(engine, router, CHUNK_ID) == "done"
    assert annotate_chunk_false_premise(engine, router, CHUNK_ID) == "skipped"
    assert len(calls) == 1
    assert annotate_chunk_false_premise(engine, router, CHUNK_ID, force=True) == "done"
    assert len(calls) == 2


def test_annotate_chunk_false_premise_records_failure(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, ["not json"])
    assert annotate_chunk_false_premise(engine, router, CHUNK_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"generate_false_premise:{CHUNK_ID}")).one()
        assert task.status == "failed"


def test_derive_contradiction_negatives_from_claim_pairs_no_llm_call(engine):
    seed_chunk(engine)
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:2", source="test", external_id="2", title="t2", norm_title="t2",
                       license="CC-BY-4.0"))
        s.add(ClaimPair(
            id=f"{CHUNK_ID}#pair0", doc_ids=["doc:1"], chunk_ids=[CHUNK_ID], license="CC-BY-4.0",
            generator_model="p/gen", tier="silver", status="generated",
            sentence_1="LiFePO4 plateaus near 3.45 V.", sentence_2="LiFePO4 plateaus near 4.45 V.",
            category="contradiction", subset_name="numeric",
        ))
        # a rejected pair must NOT be converted
        s.add(ClaimPair(
            id=f"{CHUNK_ID}#pair1", doc_ids=["doc:1"], chunk_ids=[CHUNK_ID], license="CC-BY-4.0",
            generator_model="p/gen", tier="silver", status="rejected", reject_reason="ungrounded_sentence_1",
            sentence_1="unrelated", sentence_2="also unrelated", category="contradiction", subset_name="entity",
        ))
        # a paraphrase pair must NOT be converted
        s.add(ClaimPair(
            id=f"{CHUNK_ID}#pair2", doc_ids=["doc:1"], chunk_ids=[CHUNK_ID], license="CC-BY-4.0",
            generator_model="p/gen", tier="silver", status="generated",
            sentence_1="a", sentence_2="a restated", category="paraphrase", subset_name="entity",
        ))

    with get_session(engine) as s:
        rows = derive_contradiction_negatives(s, "doc:1")
    assert len(rows) == 1
    with get_session(engine) as s:
        neg = s.get(Negative, f"{CHUNK_ID}#pair0#neg")
    assert neg.kind == "contradiction_detection" and neg.polarity == "negative"
    assert "4.45" in neg.flawed_element and "3.45" in neg.expert_response
    assert neg.linked_positive_id == f"{CHUNK_ID}#pair0"
    assert neg.license == "CC-BY-4.0"


def test_derive_contradiction_negatives_is_idempotent(engine):
    seed_chunk(engine)
    with get_session(engine) as s:
        s.add(ClaimPair(
            id=f"{CHUNK_ID}#pair0", doc_ids=["doc:1"], chunk_ids=[CHUNK_ID], license="CC-BY-4.0",
            generator_model="p/gen", tier="silver", status="generated",
            sentence_1="a", sentence_2="not a", category="contradiction", subset_name="swap",
        ))
    with get_session(engine) as s:
        derive_contradiction_negatives(s, "doc:1")
    with get_session(engine) as s:
        derive_contradiction_negatives(s, "doc:1")
    with get_session(engine) as s:
        assert len(s.scalars(select(Negative)).all()) == 1
