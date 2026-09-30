"""Stage 9 (Verify): judge-model scoring of generated QA and ideation.

The judge route excludes the generator's model family (see `configs/llm_routes.yaml`'s judge route and
`llm.router`'s `exclude_families`), so a different model family grades the work - not the same model marking
its own homework. A row passes when every score meets `configs/generation.yaml`'s `verify.min_judge_scores`
floor; otherwise it is marked rejected with the judge's rationale kept for inspection.

See `annotate.facts` module docstring for why the resumable wrappers manage their own short-lived sessions
around the LLM call (SQLite deadlock risk).
"""

from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import GenTask, Ideation, QA
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import IdeationJudgeOut, QAJudgeOut, json_schema_response_format, json_validator

QA_JUDGE_SYSTEM_PROMPT = (
    "You are grading a training example for a lithium-ion battery materials expert model. Score the answer "
    "against the source excerpt on a 1-5 scale for each dimension: faithfulness (is every claim actually "
    "supported by the excerpt, no fabrication?), correctness (is the battery science right?), specificity "
    "(does it use the excerpt's actual numbers/mechanisms rather than generic statements?)."
)

QA_JUDGE_USER_TEMPLATE = """Source excerpt:
\"\"\"
{chunk_text}
\"\"\"

Question: {question}
Answer: {answer}

Score as JSON matching this exact shape:
{{"faithfulness": 1-5, "correctness": 1-5, "specificity": 1-5, "rationale": str}}
Return JSON only, no other text."""

IDEATION_JUDGE_SYSTEM_PROMPT = (
    "You are grading a grounded-ideation example for a lithium-ion battery materials expert model. Score "
    "1-5 on: groundedness (is the problem/reasoning actually supported by the excerpt?), correctness (is the "
    "battery science right?), novelty (is this more than a restatement of the excerpt?), feasibility (are "
    "the validation experiments concrete and doable?)."
)

IDEATION_JUDGE_USER_TEMPLATE = """Source excerpt:
\"\"\"
{chunk_text}
\"\"\"

Problem: {problem}
Reasoning: {reasoning}
Ideas: {ideas}

Score as JSON matching this exact shape:
{{"groundedness": 1-5, "correctness": 1-5, "novelty": 1-5, "feasibility": 1-5, "rationale": str}}
Return JSON only, no other text."""

QA_MIN_SCORES = {"faithfulness": 4, "correctness": 4, "specificity": 3}
IDEATION_MIN_SCORES = {"groundedness": 4, "correctness": 4, "novelty": 3, "feasibility": 3}


def _passes(scores: dict[str, Any], floor: dict[str, int]) -> bool:
    return all(scores.get(dim, 0) >= minimum for dim, minimum in floor.items())


def judge_qa(
    session: Session, router: LLMRouter, qa: QA, *, route: str = "judge", use_cache: bool = True
) -> QAJudgeOut:
    """One LLM call scoring `qa`, applied in place. `qa.status` becomes accepted/rejected accordingly."""
    from batterygemma.db.models import Chunk

    chunk_text = "\n\n".join(
        c.text for c in session.scalars(select(Chunk).where(Chunk.chunk_id.in_(qa.chunk_ids))).all()
    )
    question, answer = qa.turns[0]["content"], qa.turns[1]["content"]
    messages = [
        {"role": "system", "content": QA_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": QA_JUDGE_USER_TEMPLATE.format(chunk_text=chunk_text, question=question, answer=answer)},
    ]
    result = router.complete(
        route, messages, validate=json_validator(QAJudgeOut),
        response_format=json_schema_response_format(QAJudgeOut, strict=False), max_tokens=800,
        use_cache=use_cache, exclude_families=[f for f in [_family_of(qa.generator_model)] if f],
        allow_same_family_fallback=True,
    )
    scores: QAJudgeOut = result.parsed
    qa.judge_scores = scores.model_dump()
    qa.status = "accepted" if _passes(qa.judge_scores, QA_MIN_SCORES) else "rejected"
    if qa.status == "rejected":
        qa.reject_reason = f"judge:{scores.rationale}"[:2000] if scores.rationale else "judge:below_threshold"
    return scores


def judge_ideation(
    session: Session, router: LLMRouter, ideation: Ideation, *, route: str = "judge", use_cache: bool = True
) -> IdeationJudgeOut:
    """One LLM call scoring `ideation`, applied in place."""
    from batterygemma.db.models import Chunk

    chunk_text = "\n\n".join(
        c.text for c in session.scalars(select(Chunk).where(Chunk.chunk_id.in_(ideation.chunk_ids))).all()
    )
    ideas_text = "; ".join(idea.get("hypothesis", "") for idea in ideation.ideas)
    messages = [
        {"role": "system", "content": IDEATION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": IDEATION_JUDGE_USER_TEMPLATE.format(
            chunk_text=chunk_text, problem=ideation.problem, reasoning=ideation.reasoning, ideas=ideas_text,
        )},
    ]
    result = router.complete(
        route, messages, validate=json_validator(IdeationJudgeOut),
        response_format=json_schema_response_format(IdeationJudgeOut, strict=False), max_tokens=800,
        use_cache=use_cache, exclude_families=[f for f in [_family_of(ideation.generator_model)] if f],
        allow_same_family_fallback=True,
    )
    scores: IdeationJudgeOut = result.parsed
    ideation.judge_scores = scores.model_dump()
    ideation.status = "accepted" if _passes(ideation.judge_scores, IDEATION_MIN_SCORES) else "rejected"
    if ideation.status == "rejected":
        ideation.reject_reason = f"judge:{scores.rationale}"[:2000] if scores.rationale else "judge:below_threshold"
    return scores


def _family_of(generator_model: str | None) -> str | None:
    """Best-effort family guess from a stored model id (e.g. 'nvidia_nim/meta/llama-3.3-70b-instruct' -> 'llama').

    A heuristic, not exact: it takes the last '/'-segment and its first '-'-token, which happens to match
    every family currently configured in configs/llm_routes.yaml. Provenance rows only store the model id
    string, not the deployment's configured `family`; storing that explicitly (a small migration) would make
    this exact instead of inferred, and is a reasonable follow-up if a new deployment's id doesn't fit the
    pattern.
    """
    if not generator_model:
        return None
    return generator_model.rsplit("/", 1)[-1].split("-")[0]


def judge_one_qa(engine: Engine, router: LLMRouter, qa_id: str, *, force: bool = False) -> str:
    """Resumable single-QA judging, by qa_id. Returns: done | skipped | failed."""
    task_key = f"judge_qa:{qa_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "judge_qa", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            qa = s.get(QA, qa_id)
            scores = judge_qa(s, router, qa, use_cache=not force)
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), scores.model_dump())
    return "done"


def judge_one_ideation(engine: Engine, router: LLMRouter, ideation_id: str, *, force: bool = False) -> str:
    """Resumable single-ideation judging, by ideation_id. Returns: done | skipped | failed."""
    task_key = f"judge_ideation:{ideation_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "judge_ideation", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            ideation = s.get(Ideation, ideation_id)
            scores = judge_ideation(s, router, ideation, use_cache=not force)
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), scores.model_dump())
    return "done"
