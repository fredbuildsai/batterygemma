import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from batterygemma.db.models import Chunk, Document, GenTask, Ideation, QA
from batterygemma.db.session import get_session
from batterygemma.llm.router import LLMRouter
from batterygemma.verify.judge import _family_of, judge_ideation, judge_one_ideation, judge_one_qa, judge_qa

QA_ID = "doc:1#s00-c00#qa0"
CHUNK_ID = "doc:1#s00-c00"

CONFIG = {
    "deployments": [
        {"name": "gen", "model": "nvidia_nim/deepseek-ai/deepseek-v4-flash-0731", "api_key_env": "KEY_A",
         "family": "deepseek"},
        {"name": "judge", "model": "mistral/magistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
    ],
    "routes": {"judge": ["gen", "judge"]},
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


def seed_qa(engine, generator_model="nvidia_nim/deepseek-ai/deepseek-v4-flash-0731"):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:1", source="test", external_id="1", title="t", norm_title="t", license="CC-BY-4.0"))
        s.add(Chunk(chunk_id=CHUNK_ID, doc_id="doc:1", order=0, tokens=50,
                   text="Trace water hydrolyzes LiPF6, producing HF that attacks the cathode."))
        s.add(QA(
            id=QA_ID, doc_ids=["doc:1"], chunk_ids=[CHUNK_ID], license="CC-BY-4.0",
            generator_model=generator_model, tier="silver", status="generated",
            task_format="instruction_response", polarity="positive", question_type="mechanism",
            component="electrolyte", answer_type="OPEN",
            turns=[{"role": "user", "content": "Why does LiPF6 form HF?"},
                   {"role": "assistant", "content": "Trace water hydrolyzes LiPF6, producing HF."}],
        ))


def seed_ideation(engine):
    with get_session(engine) as s:
        s.add(Document(doc_id="doc:2", source="test", external_id="2", title="t", norm_title="t", license="CC-BY-4.0"))
        s.add(Chunk(chunk_id="doc:2#s00-c00", doc_id="doc:2", order=0, tokens=50,
                   text="Silicon anodes expand by 300% during lithiation, fracturing the SEI."))
        s.add(Ideation(
            id="doc:2#s00-c00#idea", doc_ids=["doc:2"], chunk_ids=["doc:2#s00-c00"], license="CC-BY-4.0",
            generator_model="nvidia_nim/deepseek-ai/deepseek-v4-flash-0731", tier="silver", status="generated",
            task_format="grounded_ideation", polarity="positive",
            problem="Silicon anodes fade due to volume change.", constraints=[], reasoning="...",
            ideas=[{"hypothesis": "Use an elastic binder.", "mechanism": "...", "risks": "...",
                   "validation_experiments": [], "success_metrics": []}],
        ))


GOOD_QA_SCORE = json.dumps({"faithfulness": 5, "correctness": 5, "specificity": 4, "rationale": "well grounded"})
BAD_QA_SCORE = json.dumps({"faithfulness": 2, "correctness": 3, "specificity": 2, "rationale": "too vague"})
GOOD_IDEA_SCORE = json.dumps({"groundedness": 5, "correctness": 4, "novelty": 4, "feasibility": 4, "rationale": "solid"})


def test_family_of_handles_two_and_three_part_model_ids():
    assert _family_of("nvidia_nim/deepseek-ai/deepseek-v4-flash-0731") == "deepseek"
    assert _family_of("nvidia_nim/meta/llama-3.3-70b-instruct") == "llama"
    assert _family_of("mistral/mistral-medium-latest") == "mistral"
    assert _family_of("anthropic/claude-sonnet-5") == "claude"
    assert _family_of(None) is None


def test_judge_qa_accepts_when_scores_meet_the_floor(engine):
    seed_qa(engine)
    router, calls = make_router(engine, [GOOD_QA_SCORE])
    with get_session(engine) as s:
        judge_qa(s, router, s.get(QA, QA_ID))
    with get_session(engine) as s:
        qa = s.get(QA, QA_ID)
    assert qa.status == "accepted" and qa.judge_scores["faithfulness"] == 5
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_judge_qa_uses_a_different_family_from_the_generator(engine):
    seed_qa(engine)  # generator family is "deepseek"
    router, calls = make_router(engine, [GOOD_QA_SCORE])
    with get_session(engine) as s:
        judge_qa(s, router, s.get(QA, QA_ID))
    # only the "judge" deployment (family "mistral") should have been called, not "gen" (deepseek, the generator)
    assert calls[0]["model"] == "mistral/magistral-medium-latest"


def test_judge_qa_rejects_below_floor_and_keeps_rationale(engine):
    seed_qa(engine)
    router, _ = make_router(engine, [BAD_QA_SCORE])
    with get_session(engine) as s:
        judge_qa(s, router, s.get(QA, QA_ID))
    with get_session(engine) as s:
        qa = s.get(QA, QA_ID)
    assert qa.status == "rejected" and "too vague" in qa.reject_reason


def test_judge_ideation_accepts_when_scores_meet_the_floor(engine):
    seed_ideation(engine)
    router, _ = make_router(engine, [GOOD_IDEA_SCORE])
    with get_session(engine) as s:
        judge_ideation(s, router, s.get(Ideation, "doc:2#s00-c00#idea"))
    with get_session(engine) as s:
        ideation = s.get(Ideation, "doc:2#s00-c00#idea")
    assert ideation.status == "accepted" and ideation.judge_scores["novelty"] == 4


def test_judge_one_qa_is_resumable(engine):
    seed_qa(engine)
    router, calls = make_router(engine, [GOOD_QA_SCORE, GOOD_QA_SCORE])
    assert judge_one_qa(engine, router, QA_ID) == "done"
    assert judge_one_qa(engine, router, QA_ID) == "skipped"
    assert len(calls) == 1
    assert judge_one_qa(engine, router, QA_ID, force=True) == "done"
    assert len(calls) == 2


def test_judge_one_qa_records_failure(engine):
    seed_qa(engine)
    router, _ = make_router(engine, ["not json", "still not json"])
    assert judge_one_qa(engine, router, QA_ID) == "failed"
    with get_session(engine) as s:
        task = s.scalars(select(GenTask).where(GenTask.key == f"judge_qa:{QA_ID}")).one()
        assert task.status == "failed"


def test_judge_one_ideation_is_resumable(engine):
    seed_ideation(engine)
    router, calls = make_router(engine, [GOOD_IDEA_SCORE, GOOD_IDEA_SCORE])
    assert judge_one_ideation(engine, router, "doc:2#s00-c00#idea") == "done"
    assert judge_one_ideation(engine, router, "doc:2#s00-c00#idea") == "skipped"
    assert len(calls) == 1
