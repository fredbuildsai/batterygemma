import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.annotate.facts import annotate_chunk_facts, extract_facts_and_comparisons
from batterygemma.db.models import Chunk, Comparison, Document, Fact, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import LLMRouter

CHUNK_ID = "doc:1#s00-c00"
CHUNK_TEXT = (
    "Above 4.2 V vs Li/Li+, NMC811 undergoes the H2->H3 phase transition, accompanied by an abrupt "
    "contraction of the c lattice parameter. Single-crystal NMC811 retained 10% more capacity than "
    "polycrystalline NMC811 after 300 cycles."
)

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"extract": ["gen"]},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")


GOOD_RESPONSE = json.dumps({
    "facts": [{
        "material": {"name": "NMC811", "formula": None, "class": "layered oxide", "component": "cathode"},
        "property": "H2->H3 transition onset", "value": "4.2", "unit": "V vs Li/Li+", "conditions": {},
        "category": "structure", "polarity": "positive", "triple": ["NMC811", "undergoes_above", "H2->H3 transition"],
        "evidence_sentence": "NMC811 undergoes the H2->H3 phase transition.",
    }],
    "comparisons": [{
        "baseline": "polycrystalline NMC811", "modification": "single-crystal NMC811",
        "metric": "capacity retention", "direction": "improves", "magnitude": "10% at 300 cycles",
        "conditions": {}, "component": "cathode",
        "evidence_sentence": "Single-crystal NMC811 retained 10% more capacity than polycrystalline NMC811.",
    }],
})

UNGROUNDED_RESPONSE = json.dumps({
    "facts": [{
        "material": {"name": "Silicon", "class": None, "component": "anode"},
        "property": "volume expansion", "value": "300", "unit": "%", "conditions": {}, "category": "mechanical",
        "polarity": "negative", "triple": [],
        "evidence_sentence": "Silicon anodes expand by 300% during lithiation, unrelated to this excerpt.",
    }],
    "comparisons": [],
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
    """Create and commit a document+chunk in their own transaction, as if left by a prior parse run -
    mirrors real usage, where annotate always operates on already-committed chunks."""
    with get_session(engine) as s:
        doc = Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0")
        s.add(doc)
        s.add(Chunk(chunk_id=chunk_id, doc_id="doc:1", order=0, tokens=100, text=text))


def test_extract_persists_grounded_facts_and_comparisons(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE])
    with get_session(engine) as s:
        chunk = s.get(Chunk, CHUNK_ID)
        facts, comparisons = extract_facts_and_comparisons(s, router, chunk)
        assert len(facts) == 1 and len(comparisons) == 1

    with get_session(engine) as s:
        fact = s.get(Fact, f"{CHUNK_ID}#fact0")
        comp = s.get(Comparison, f"{CHUNK_ID}#cmp0")
    assert fact.status == "generated" and fact.doc_ids == ["doc:1"] and fact.chunk_ids == [CHUNK_ID]
    assert fact.license == "CC-BY-4.0" and fact.component == "cathode" and fact.category == "structure"
    assert fact.material["name"] == "NMC811"
    assert comp.status == "generated" and comp.direction == "improves"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_ungrounded_evidence_is_rejected_but_still_stored(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        chunk = s.get(Chunk, CHUNK_ID)
        facts, _ = extract_facts_and_comparisons(s, router, chunk)
    assert facts[0].status == "rejected" and facts[0].reject_reason == "ungrounded_evidence"


def test_a_failed_chunk_is_retried_and_can_succeed_on_a_later_run(engine):
    # Single deployment configured, so one bad response exhausts that call outright; the retry happens by
    # re-running annotate_chunk_facts later (a "failed" task is not skipped, unlike a "done" one).
    seed_chunk(engine)
    router, calls = make_router(engine, ["not json at all", GOOD_RESPONSE])
    assert annotate_chunk_facts(engine, router, CHUNK_ID) == "failed"
    assert annotate_chunk_facts(engine, router, CHUNK_ID) == "done"
    assert len(calls) == 2


def test_annotate_chunk_facts_is_resumable(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE, GOOD_RESPONSE])
    assert annotate_chunk_facts(engine, router, CHUNK_ID) == "done"
    assert len(calls) == 1

    assert annotate_chunk_facts(engine, router, CHUNK_ID) == "skipped"
    assert len(calls) == 1  # no new LLM call

    assert annotate_chunk_facts(engine, router, CHUNK_ID, force=True) == "done"
    assert len(calls) == 2  # force also bypasses the response cache, so this is a genuinely fresh call

    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_facts:{CHUNK_ID}")).one()
        assert task.status == "done" and task.attempts == 2 and task.payload == {"facts": 1, "comparisons": 1}


def test_annotate_chunk_facts_records_failure_when_all_deployments_exhausted(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, ["not json", "still not json"])
    assert annotate_chunk_facts(engine, router, CHUNK_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_facts:{CHUNK_ID}")).one()
        assert task.status == "failed" and task.last_error


def test_rerun_replaces_previous_facts_for_the_same_chunk(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [GOOD_RESPONSE, UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        extract_facts_and_comparisons(s, router, s.get(Chunk, CHUNK_ID))
    with get_session(engine) as s:
        extract_facts_and_comparisons(s, router, s.get(Chunk, CHUNK_ID), use_cache=False)  # ungrounded this time
    with get_session(engine) as s:
        all_facts = s.scalars(select(Fact)).all()
    assert len(all_facts) == 1 and all_facts[0].reject_reason == "ungrounded_evidence"
