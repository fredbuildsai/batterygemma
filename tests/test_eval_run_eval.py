import json
from types import SimpleNamespace

import pytest
from corpusforge.models import Chunk, Document
from llmrouter_free import LLMRouter

from batterygemma.db.session import get_session
from batterygemma.eval.run_eval import run_eval, score_closed_qa

CONFIG = {
    "deployments": [
        {"name": "judge", "model": "mistral/magistral-medium-latest", "api_key_env": "KEY_A", "family": "mistral"},
    ],
    "routes": {"judge": ["judge"]},
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


def test_score_closed_qa_matches_yes_no_in_free_text():
    item = {"gold_id": "g1", "closed_label": "yes"}
    assert score_closed_qa(item, "Yes, that is correct because of X.")["correct"] is True
    assert score_closed_qa(item, "No, that's not right.")["correct"] is False


def test_run_eval_scores_closed_open_negative_and_ideation_items(engine, tmp_path):
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="t", norm_title="t", split="eval"))
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=10, text="NMC811 undergoes H2->H3 above 4.2V."))

    gold_items = [
        {"gold_id": "g-closed", "category": "closed_qa", "prompt": [{"role": "user", "content": "Does X happen?"}],
         "closed_label": "yes"},
        {"gold_id": "g-open", "category": "open_qa", "prompt": [{"role": "user", "content": "Why does X happen?"}],
         "reference_answer": "Because of Y."},
        {"gold_id": "g-neg", "category": "negative_detection",
         "prompt": [{"role": "user", "content": "Flawed premise question"}], "reference_answer": "Corrected."},
        {"gold_id": "g-idea", "category": "ideation", "prompt": [{"role": "user", "content": "Problem statement"}],
         "source_chunk_ids": ["d1#c0"]},
    ]

    answers = iter(["Yes, definitely.", "model open answer", "model negative answer", "model idea answer"])

    def generate_fn(messages):
        return next(answers)

    judge_responses = [
        json.dumps({"correct": True, "score": 5, "rationale": "matches"}),
        json.dumps({"caught_flaw": True, "rationale": "caught it"}),
        json.dumps({"groundedness": 4, "correctness": 5, "novelty": 3, "feasibility": 4, "rationale": "ok"}),
    ]
    router, calls = make_router(engine, judge_responses)

    with get_session(engine) as s:
        results = run_eval(s, router, gold_items, generate_fn, system_prompt="You are an expert.")

    assert len(results) == 4
    assert len(calls) == 3  # closed_qa needs no LLM call

    by_id = {r["gold_id"]: r for r in results}
    assert by_id["g-closed"]["correct"] is True
    assert by_id["g-open"]["correct"] is True and by_id["g-open"]["score"] == 5
    assert by_id["g-neg"]["correct"] is True
    assert by_id["g-idea"]["scores"]["groundedness"] == 4
    assert all(r["answer"] for r in results)
