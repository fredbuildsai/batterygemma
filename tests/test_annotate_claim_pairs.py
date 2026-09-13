import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.annotate.claim_pairs import annotate_chunk_claims, extract_claim_pairs
from batterygemma.db.models import Chunk, ClaimPair, Document, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import LLMRouter

CHUNK_ID = "doc:2#s00-c00"
CHUNK_TEXT = (
    "Single-crystal NMC811 retains 10% more capacity than polycrystalline NMC811 after 300 cycles at 45C, "
    "because the absence of internal grain boundaries prevents intergranular microcracking."
)

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"extract": ["gen"]},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")


GOOD_RESPONSE = json.dumps({
    "pairs": [
        {"sentence_1": "Single-crystal NMC811 retains 10% more capacity than polycrystalline NMC811 after 300 cycles.",
         "sentence_2": "After 300 cycles, single-crystal NMC811 keeps 10% more of its capacity than the polycrystalline form.",
         "category": "paraphrase", "subset_name": "entity"},
        {"sentence_1": "Single-crystal NMC811 retains 10% more capacity than polycrystalline NMC811 after 300 cycles.",
         "sentence_2": "Polycrystalline NMC811 retains 10% more capacity than single-crystal NMC811 after 300 cycles.",
         "category": "contradiction", "subset_name": "swap"},
    ]
})

UNGROUNDED_RESPONSE = json.dumps({
    "pairs": [
        {"sentence_1": "Graphite anodes swell during sodiation, unrelated to this excerpt entirely.",
         "sentence_2": "Graphite anodes shrink during sodiation.", "category": "contradiction", "subset_name": "swap"},
    ]
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
        doc = Document(doc_id="doc:2", source="test", external_id="2", title="t", norm_title="t", license="CC-BY-4.0")
        s.add(doc)
        s.add(Chunk(chunk_id=chunk_id, doc_id="doc:2", order=0, tokens=100, text=text))


def test_extract_persists_a_mix_of_paraphrase_and_contradiction(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE])
    with get_session(engine) as s:
        pairs = extract_claim_pairs(s, router, s.get(Chunk, CHUNK_ID))
    assert len(pairs) == 2
    assert {p.category for p in pairs} == {"paraphrase", "contradiction"}

    with get_session(engine) as s:
        pair0 = s.get(ClaimPair, f"{CHUNK_ID}#pair0")
        pair1 = s.get(ClaimPair, f"{CHUNK_ID}#pair1")
    assert pair0.status == "generated" and pair0.doc_ids == ["doc:2"] and pair0.license == "CC-BY-4.0"
    assert pair1.category == "contradiction" and pair1.subset_name == "swap"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_ungrounded_sentence_1_is_rejected(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        pairs = extract_claim_pairs(s, router, s.get(Chunk, CHUNK_ID))
    assert pairs[0].status == "rejected" and pairs[0].reject_reason == "ungrounded_sentence_1"


def test_annotate_chunk_claims_is_resumable(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE, GOOD_RESPONSE])
    assert annotate_chunk_claims(engine, router, CHUNK_ID) == "done"
    assert len(calls) == 1
    assert annotate_chunk_claims(engine, router, CHUNK_ID) == "skipped"
    assert len(calls) == 1
    assert annotate_chunk_claims(engine, router, CHUNK_ID, force=True) == "done"
    assert len(calls) == 2

    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_claims:{CHUNK_ID}")).one()
        assert task.status == "done" and task.attempts == 2 and task.payload == {"pairs": 2}


def test_annotate_chunk_claims_records_failure(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, ["not json"])
    assert annotate_chunk_claims(engine, router, CHUNK_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_claims:{CHUNK_ID}")).one()
        assert task.status == "failed" and task.last_error
