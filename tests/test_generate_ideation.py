import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.db.models import Chunk, Comparison, Document, Fact, GenTask, Ideation
from batterygemma.db.session import get_session
from batterygemma.generate.ideation import annotate_chunk_ideation, generate_ideation, has_enough_facts
from batterygemma.llm.router import LLMRouter

CHUNK_ID = "doc:1#s00-c00"
CHUNK_TEXT = "Silicon anodes expand by 300% during lithiation, repeatedly fracturing the SEI."

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"ideation": ["gen"]},
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


def seed_chunk_with_facts(engine, n_facts=2):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0"))
        s.add(Chunk(chunk_id=CHUNK_ID, doc_id="doc:1", order=0, tokens=50, text=CHUNK_TEXT))
        for i in range(n_facts):
            s.add(Fact(id=f"{CHUNK_ID}#fact{i}", chunk_ids=[CHUNK_ID], status="generated",
                      property="p", evidence_sentence="e", category="mechanical", material={"name": "Si"}))


GOOD_IDEATION = json.dumps({
    "problem": "Silicon anodes fade quickly because ~300% volume change repeatedly fractures the SEI.",
    "constraints": ["aqueous slurry compatible"],
    "reasoning": "A more elastic, LiF-rich SEI should tolerate repeated fracture better.",
    "ideas": [{
        "hypothesis": "A crosslinked PAA/CMC binder plus FEC-rich electrolyte keeps particles connected.",
        "mechanism": "Crosslinked binder absorbs strain; FEC forms a thin, elastic SEI.",
        "risks": "Binder swelling could raise impedance.",
        "validation_experiments": ["300-cycle retention vs baseline binder"],
        "success_metrics": [">=80% retention at 300 cycles"],
    }],
})

NO_IDEAS = json.dumps({"problem": "x", "constraints": [], "reasoning": "y", "ideas": []})


def test_has_enough_facts_counts_facts_and_comparisons(engine):
    seed_chunk_with_facts(engine, n_facts=1)
    with get_session(engine) as s:
        assert not has_enough_facts(s, CHUNK_ID)
        s.add(Comparison(id=f"{CHUNK_ID}#cmp0", chunk_ids=[CHUNK_ID], status="generated", baseline="a",
                         modification="b", metric="m", direction="improves", evidence_sentence="e"))
    with get_session(engine) as s:
        assert has_enough_facts(s, CHUNK_ID)


def test_has_enough_facts_ignores_rejected_facts(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0"))
        s.add(Chunk(chunk_id=CHUNK_ID, doc_id="doc:1", order=0, tokens=50, text=CHUNK_TEXT))
        for i in range(3):
            s.add(Fact(id=f"{CHUNK_ID}#fact{i}", chunk_ids=[CHUNK_ID], status="rejected",
                      property="p", evidence_sentence="e", category="mechanical", material={"name": "Si"}))
    with get_session(engine) as s:
        assert not has_enough_facts(s, CHUNK_ID)


def test_generate_ideation_persists_ideas(engine):
    seed_chunk_with_facts(engine)
    router, calls = make_router(engine, [GOOD_IDEATION])
    with get_session(engine) as s:
        row = generate_ideation(s, router, s.get(Chunk, CHUNK_ID))
    assert row is not None and len(row.ideas) == 1
    with get_session(engine) as s:
        stored = s.get(Ideation, f"{CHUNK_ID}#idea")
    assert stored.problem.startswith("Silicon anodes")
    assert stored.ideas[0]["hypothesis"].startswith("A crosslinked")
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "IdeationOut"


def test_generate_ideation_returns_none_when_no_ideas_proposed(engine):
    seed_chunk_with_facts(engine)
    router, _ = make_router(engine, [NO_IDEAS])
    with get_session(engine) as s:
        row = generate_ideation(s, router, s.get(Chunk, CHUNK_ID))
    assert row is None
    with get_session(engine) as s:
        assert s.get(Ideation, f"{CHUNK_ID}#idea") is None


def test_annotate_chunk_ideation_skips_chunks_with_too_few_facts(engine):
    seed_chunk_with_facts(engine, n_facts=1)
    router, calls = make_router(engine, [GOOD_IDEATION])
    assert annotate_chunk_ideation(engine, router, CHUNK_ID) == "too_few_facts"
    assert calls == []


def test_annotate_chunk_ideation_is_resumable(engine):
    seed_chunk_with_facts(engine)
    router, calls = make_router(engine, [GOOD_IDEATION, GOOD_IDEATION])
    assert annotate_chunk_ideation(engine, router, CHUNK_ID) == "done"
    assert annotate_chunk_ideation(engine, router, CHUNK_ID) == "skipped"
    assert len(calls) == 1
    assert annotate_chunk_ideation(engine, router, CHUNK_ID, force=True) == "done"
    assert len(calls) == 2


def test_annotate_chunk_ideation_records_failure(engine):
    seed_chunk_with_facts(engine)
    router, _ = make_router(engine, ["not json"])
    assert annotate_chunk_ideation(engine, router, CHUNK_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"generate_ideation:{CHUNK_ID}")).one()
        assert task.status == "failed"
