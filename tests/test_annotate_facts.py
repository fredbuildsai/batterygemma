import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.annotate.facts import annotate_chunks_facts, extract_facts_and_comparisons
from batterygemma.db.models import Chunk, Comparison, Document, Fact, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import LLMRouter

CHUNK_ID = "doc:1#s00-c00"
CHUNK_TEXT = (
    "Above 4.2 V vs Li/Li+, NMC811 undergoes the H2->H3 phase transition, accompanied by an abrupt "
    "contraction of the c lattice parameter. Single-crystal NMC811 retained 10% more capacity than "
    "polycrystalline NMC811 after 300 cycles."
)
CHUNK_ID_2 = "doc:1#s00-c01"
CHUNK_TEXT_2 = "Silicon anodes expand by roughly 300% in volume during lithiation."

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"extract": ["gen"]},
}


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("KEY_A", "x")


def extraction_result(chunk_index, facts=None, comparisons=None):
    return {"chunk_index": chunk_index, "facts": facts or [], "comparisons": comparisons or []}


GOOD_FACT = {
    "material": {"name": "NMC811", "formula": None, "class": "layered oxide", "component": "cathode"},
    "property": "H2->H3 transition onset", "value": "4.2", "unit": "V vs Li/Li+", "conditions": {},
    "category": "structure", "polarity": "positive", "triple": ["NMC811", "undergoes_above", "H2->H3 transition"],
    "evidence_sentence": "NMC811 undergoes the H2->H3 phase transition.",
}
GOOD_COMPARISON = {
    "baseline": "polycrystalline NMC811", "modification": "single-crystal NMC811",
    "metric": "capacity retention", "direction": "improves", "magnitude": "10% at 300 cycles",
    "conditions": {}, "component": "cathode",
    "evidence_sentence": "Single-crystal NMC811 retained 10% more capacity than polycrystalline NMC811.",
}
UNGROUNDED_FACT = {
    "material": {"name": "Silicon", "class": None, "component": "anode"},
    "property": "volume expansion", "value": "300", "unit": "%", "conditions": {}, "category": "mechanical",
    "polarity": "negative", "triple": [],
    "evidence_sentence": "Silicon anodes expand by 300% during lithiation, unrelated to this excerpt.",
}

GOOD_RESPONSE = json.dumps({"results": [extraction_result(0, [GOOD_FACT], [GOOD_COMPARISON])]})
UNGROUNDED_RESPONSE = json.dumps({"results": [extraction_result(0, [UNGROUNDED_FACT])]})


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
        doc = s.get(Document, "doc:1")
        if doc is None:
            doc = Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t",
                            license="CC-BY-4.0")
            s.add(doc)
        s.add(Chunk(chunk_id=chunk_id, doc_id="doc:1", order=0, tokens=100, text=text))


def test_extract_persists_grounded_facts_and_comparisons(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE])
    with get_session(engine) as s:
        chunk = s.get(Chunk, CHUNK_ID)
        results = extract_facts_and_comparisons(s, router, [chunk])
        facts, comparisons = results[CHUNK_ID]
        assert len(facts) == 1 and len(comparisons) == 1

    with get_session(engine) as s:
        fact = s.get(Fact, f"{CHUNK_ID}#fact0")
        comp = s.get(Comparison, f"{CHUNK_ID}#cmp0")
    assert fact.status == "generated" and fact.doc_ids == ["doc:1"] and fact.chunk_ids == [CHUNK_ID]
    assert fact.license == "CC-BY-4.0" and fact.component == "cathode" and fact.category == "structure"
    assert fact.material["name"] == "NMC811"
    assert comp.status == "generated" and comp.direction == "improves"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "BatchExtractionOut"


def test_a_batch_of_several_chunks_uses_one_llm_call(engine):
    """The whole point of batching: one call for both chunks, both persisted correctly by chunk_index."""
    seed_chunk(engine, CHUNK_ID, CHUNK_TEXT)
    seed_chunk(engine, CHUNK_ID_2, CHUNK_TEXT_2)
    response = json.dumps({"results": [
        extraction_result(0, [GOOD_FACT], [GOOD_COMPARISON]),
        extraction_result(1, [UNGROUNDED_FACT]),
    ]})
    router, calls = make_router(engine, [response])
    with get_session(engine) as s:
        chunks = [s.get(Chunk, CHUNK_ID), s.get(Chunk, CHUNK_ID_2)]
        results = extract_facts_and_comparisons(s, router, chunks)
    assert len(calls) == 1
    assert len(results[CHUNK_ID][0]) == 1
    assert len(results[CHUNK_ID_2][0]) == 1


def test_ungrounded_evidence_is_rejected_but_still_stored(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        chunk = s.get(Chunk, CHUNK_ID)
        facts, _ = extract_facts_and_comparisons(s, router, [chunk])[CHUNK_ID]
    assert facts[0].status == "rejected" and facts[0].reject_reason == "ungrounded_evidence"


def test_a_failed_chunk_is_retried_and_can_succeed_on_a_later_run(engine):
    # Single deployment configured, so one bad response exhausts that call outright; the retry happens by
    # re-running annotate_chunks_facts later (a "failed" task is not skipped, unlike a "done" one).
    seed_chunk(engine)
    router, calls = make_router(engine, ["not json at all", GOOD_RESPONSE])
    assert annotate_chunks_facts(engine, router, [CHUNK_ID]) == {CHUNK_ID: "failed"}
    assert annotate_chunks_facts(engine, router, [CHUNK_ID]) == {CHUNK_ID: "done"}
    assert len(calls) == 2


def test_annotate_chunks_facts_is_resumable(engine):
    seed_chunk(engine)
    router, calls = make_router(engine, [GOOD_RESPONSE, GOOD_RESPONSE])
    assert annotate_chunks_facts(engine, router, [CHUNK_ID]) == {CHUNK_ID: "done"}
    assert len(calls) == 1

    assert annotate_chunks_facts(engine, router, [CHUNK_ID]) == {CHUNK_ID: "skipped"}
    assert len(calls) == 1  # no new LLM call

    assert annotate_chunks_facts(engine, router, [CHUNK_ID], force=True) == {CHUNK_ID: "done"}
    assert len(calls) == 2  # force also bypasses the response cache, so this is a genuinely fresh call

    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_facts:{CHUNK_ID}")).one()
        assert task.status == "done" and task.attempts == 2 and task.payload == {"facts": 1, "comparisons": 1}


def test_annotate_chunks_facts_records_failure_when_all_deployments_exhausted(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, ["not json", "still not json"])
    assert annotate_chunks_facts(engine, router, [CHUNK_ID]) == {CHUNK_ID: "failed"}
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"extract_facts:{CHUNK_ID}")).one()
        assert task.status == "failed" and task.last_error


def test_a_chunk_missing_from_the_batch_response_is_retried_as_a_singleton_and_can_still_succeed(engine):
    seed_chunk(engine, CHUNK_ID, CHUNK_TEXT)
    seed_chunk(engine, CHUNK_ID_2, CHUNK_TEXT_2)
    # First call's response only covers chunk_index 0 - chunk_id 2 is silently dropped from the batch.
    # The retry re-sends it as a singleton list, so its own chunk_index-0 response now matches it.
    dropped = json.dumps({"results": [extraction_result(0, [GOOD_FACT])]})
    router, calls = make_router(engine, [dropped, dropped])
    outcomes = annotate_chunks_facts(engine, router, [CHUNK_ID, CHUNK_ID_2])
    assert outcomes[CHUNK_ID] == "done"
    assert outcomes[CHUNK_ID_2] == "done"  # succeeded via the singleton retry - same mechanism, not a fallback
    assert len(calls) == 2  # one batch call, one singleton retry for the dropped chunk


def test_a_chunk_still_missing_after_the_retry_is_marked_failed(engine):
    seed_chunk(engine, CHUNK_ID, CHUNK_TEXT)
    seed_chunk(engine, CHUNK_ID_2, CHUNK_TEXT_2)
    # chunk_index 1 never appears, even in the singleton retry (an empty "results" list both times).
    dropped = json.dumps({"results": [extraction_result(0, [GOOD_FACT])]})
    empty = json.dumps({"results": []})
    router, calls = make_router(engine, [dropped, empty])
    outcomes = annotate_chunks_facts(engine, router, [CHUNK_ID, CHUNK_ID_2])
    assert outcomes[CHUNK_ID] == "done"
    assert outcomes[CHUNK_ID_2] == "failed"
    assert len(calls) == 2  # one batch call, one singleton retry that still comes back empty


def test_rerun_replaces_previous_facts_for_the_same_chunk(engine):
    seed_chunk(engine)
    router, _ = make_router(engine, [GOOD_RESPONSE, UNGROUNDED_RESPONSE])
    with get_session(engine) as s:
        extract_facts_and_comparisons(s, router, [s.get(Chunk, CHUNK_ID)])
    with get_session(engine) as s:
        extract_facts_and_comparisons(s, router, [s.get(Chunk, CHUNK_ID)], use_cache=False)  # ungrounded this time
    with get_session(engine) as s:
        all_facts = s.scalars(select(Fact)).all()
    assert len(all_facts) == 1 and all_facts[0].reject_reason == "ungrounded_evidence"
