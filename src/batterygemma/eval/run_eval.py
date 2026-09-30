"""Stage 11 (Eval): run a model against the gold benchmark and score its answers.

Scoring:
- closed_qa: exact-match against `closed_label` after extracting the model's yes/no (or first character) -
  no LLM call needed, since the answer space is small and unambiguous.
- open_qa / negative_detection: an LLM judge compares the model's answer against the gold reference answer
  (the reference came from stage 8's generator, not the model under test, so this is not the "same model
  grading its own homework" case `verify.judge` guards against - no family exclusion is needed here).
- ideation: an LLM judge scores the model's own ideas against the same source excerpt used at generation
  time, with the same rubric `verify.judge.judge_ideation` uses.

Generation is injected as `generate_fn(messages) -> str` so this module never has to import Unsloth/MLX
itself; `cli.py` wires up the real model (via `mlx_lm.generate`) and tests inject a fake.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corpusforge.models import Chunk
from llmrouter_free import LLMRouter, json_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from batterygemma.llm.schemas import IdeationJudgeOut, NegativeDetectionJudgeOut, OpenAnswerJudgeOut

GenerateFn = Callable[[list[dict[str, str]]], str]

OPEN_QA_JUDGE_SYSTEM = (
    "You are grading a lithium-ion battery materials expert model's answer against a reference answer. "
    "Judge whether the candidate answer is scientifically consistent with the reference - it does not need "
    "to match wording, only substance."
)
OPEN_QA_JUDGE_TEMPLATE = """Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

Score as JSON: {{"correct": true|false, "score": 1-5, "rationale": str}}
Return JSON only, no other text."""

NEGATIVE_JUDGE_SYSTEM = (
    "You are grading whether a lithium-ion battery materials expert model caught a planted flaw in the "
    "question and corrected it, the way the reference answer does."
)
NEGATIVE_JUDGE_TEMPLATE = """Prompt (contains a planted flaw): {prompt}

Reference correction: {reference}

Candidate answer: {candidate}

Did the candidate identify and correct the flaw? Score as JSON:
{{"caught_flaw": true|false, "rationale": str}}
Return JSON only, no other text."""

IDEATION_JUDGE_SYSTEM = (
    "You are grading a grounded-ideation example for a lithium-ion battery materials expert model. Score "
    "1-5 on: groundedness (is the problem/reasoning actually supported by the excerpt?), correctness (is the "
    "battery science right?), novelty (is this more than a restatement of the excerpt?), feasibility (are "
    "the validation experiments concrete and doable?)."
)
IDEATION_JUDGE_TEMPLATE = """Source excerpt:
\"\"\"
{chunk_text}
\"\"\"

Problem: {problem}
Candidate response: {candidate}

Score as JSON matching this exact shape:
{{"groundedness": 1-5, "correctness": 1-5, "novelty": 1-5, "feasibility": 1-5, "rationale": str}}
Return JSON only, no other text."""

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def load_gold(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunk_text(session: Session, chunk_ids: list[str]) -> str:
    if not chunk_ids:
        return ""
    return "\n\n".join(c.text for c in session.scalars(select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))).all())


def score_closed_qa(item: dict[str, Any], answer: str) -> dict[str, Any]:
    """Exact-match scoring: extract the model's yes/no (or leading option letter) and compare to `closed_label`."""
    label = (item.get("closed_label") or "").strip().lower()
    match = _YES_NO.search(answer)
    predicted = match.group(1).lower() if match else answer.strip().lower()[:1]
    correct = bool(label) and predicted == label
    return {"gold_id": item["gold_id"], "category": "closed_qa", "predicted": predicted, "label": label,
            "correct": correct}


def score_open_qa(
    router: LLMRouter, item: dict[str, Any], answer: str, *, route: str = "judge", use_cache: bool = True
) -> dict[str, Any]:
    question = item["prompt"][0]["content"]
    messages = [
        {"role": "system", "content": OPEN_QA_JUDGE_SYSTEM},
        {"role": "user", "content": OPEN_QA_JUDGE_TEMPLATE.format(
            question=question, reference=item["reference_answer"], candidate=answer)},
    ]
    result = router.complete(route, messages, validate=json_validator(OpenAnswerJudgeOut), json_mode=True,
                              max_tokens=500, use_cache=use_cache)
    out: OpenAnswerJudgeOut = result.parsed
    return {"gold_id": item["gold_id"], "category": "open_qa", "correct": out.correct, "score": out.score,
            "rationale": out.rationale}


def score_negative_detection(
    router: LLMRouter, item: dict[str, Any], answer: str, *, route: str = "judge", use_cache: bool = True
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": NEGATIVE_JUDGE_SYSTEM},
        {"role": "user", "content": NEGATIVE_JUDGE_TEMPLATE.format(
            prompt=item["prompt"][0]["content"], reference=item["reference_answer"], candidate=answer)},
    ]
    result = router.complete(route, messages, validate=json_validator(NegativeDetectionJudgeOut), json_mode=True,
                              max_tokens=500, use_cache=use_cache)
    out: NegativeDetectionJudgeOut = result.parsed
    return {"gold_id": item["gold_id"], "category": "negative_detection", "correct": out.caught_flaw,
            "rationale": out.rationale}


def score_ideation(
    session: Session, router: LLMRouter, item: dict[str, Any], answer: str, *, route: str = "judge",
    use_cache: bool = True,
) -> dict[str, Any]:
    chunk_text = _chunk_text(session, item.get("source_chunk_ids") or [])
    messages = [
        {"role": "system", "content": IDEATION_JUDGE_SYSTEM},
        {"role": "user", "content": IDEATION_JUDGE_TEMPLATE.format(
            chunk_text=chunk_text, problem=item["prompt"][0]["content"], candidate=answer)},
    ]
    result = router.complete(route, messages, validate=json_validator(IdeationJudgeOut), json_mode=True,
                              max_tokens=800, use_cache=use_cache)
    out: IdeationJudgeOut = result.parsed
    return {"gold_id": item["gold_id"], "category": "ideation", "scores": out.model_dump()}


def run_eval(
    session: Session, router: LLMRouter, gold_items: list[dict[str, Any]], generate_fn: GenerateFn, *,
    system_prompt: str, use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Generate an answer for every gold item and score it. Returns one result row per item."""
    results = []
    for item in gold_items:
        messages = [{"role": "system", "content": system_prompt}, *item["prompt"]]
        answer = generate_fn(messages)
        if item["category"] == "closed_qa":
            result = score_closed_qa(item, answer)
        elif item["category"] == "open_qa":
            result = score_open_qa(router, item, answer, use_cache=use_cache)
        elif item["category"] == "negative_detection":
            result = score_negative_detection(router, item, answer, use_cache=use_cache)
        elif item["category"] == "ideation":
            result = score_ideation(session, router, item, answer, use_cache=use_cache)
        else:
            raise ValueError(f"unknown gold item category: {item['category']!r}")
        result["answer"] = answer
        results.append(result)
    return results
