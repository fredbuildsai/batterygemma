"""Stage 8 (Generate): DPO preference pairs, derived from accepted QA rows.

`chosen` is the QA row's own grounded answer; `rejected` is one LLM call asking for the same answer with one
specific, named error injected (see `configs/taxonomy.yaml` dpo_error_types). This matches the plan: DPO
pairs sampled from verified QA, each with an injected error type.

Note: DPO itself is not trainable on this project's local Unsloth/MLX backend (see `train.sft` module
docstring) - these pairs are exported for a future CUDA-based DPOTrainer run. Building and exporting them
now means that step needs no new data work later, only a different machine.
"""

from corpusforge.annotate.tasks import get_or_create_task, mark_done, mark_failed
from corpusforge.models import GenTask
from llmrouter_free import AllDeploymentsExhausted, LLMRouter, json_schema_response_format, json_validator
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.db.models import QA, DPOPair
from batterygemma.db.session import get_session
from batterygemma.llm.schemas import DPORejectionOut

SYSTEM_PROMPT = (
    "You are helping build a preference-tuning dataset for a lithium-ion battery materials model. Given a "
    "correct expert answer, rewrite it with exactly one specific error injected, so it reads as plausible "
    "but is factually wrong in that one respect. Keep the same length and voice - only the one error should "
    "give it away, not obviously bad writing."
)

USER_TEMPLATE = """Question: {question}

Correct answer: {answer}

Rewrite the answer with exactly one injected error as JSON matching this exact shape:
{{"rejected": str, "error_type": one of ["wrong_mechanism","wrong_magnitude_or_units","ignored_tradeoff",\
"fabricated_evidence","overclaiming","thermodynamically_impossible"]}}

Pick whichever error_type fits the answer's content best. Return JSON only, no other text."""


def build_dpo_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, answer=answer)},
    ]


def generate_dpo_pair(
    session: Session, router: LLMRouter, qa: QA, *, route: str = "negatives", use_cache: bool = True
) -> DPOPair:
    """One LLM call for `qa`'s answer, persisted as a DPOPair row. Replaces any previous row for this QA."""
    question, answer = qa.turns[0]["content"], qa.turns[1]["content"]
    result = router.complete(
        route, build_dpo_messages(question, answer), validate=json_validator(DPORejectionOut),
        response_format=json_schema_response_format(DPORejectionOut), max_tokens=1500, use_cache=use_cache,
    )
    parsed: DPORejectionOut = result.parsed

    row_id = f"{qa.id}#dpo"
    session.execute(delete(DPOPair).where(DPOPair.id == row_id))
    row = DPOPair(
        id=row_id, doc_ids=qa.doc_ids, chunk_ids=qa.chunk_ids, license=qa.license,
        generator_model=result.model, tier="silver", status="generated",
        source_qa_id=qa.id, component=qa.component, error_type=parsed.error_type,
        prompt=[{"role": "user", "content": question}],
        chosen=[{"role": "assistant", "content": answer}],
        rejected=[{"role": "assistant", "content": parsed.rejected}],
    )
    session.add(row)
    return row


def annotate_qa_dpo(engine: Engine, router: LLMRouter, qa_id: str, *, force: bool = False) -> str:
    """Resumable single-QA DPO generation, by qa_id. Returns: done | skipped | failed."""
    task_key = f"generate_dpo:{qa_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "generate_dpo", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            qa = s.get(QA, qa_id)
            generate_dpo_pair(s, router, qa, use_cache=not force)
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), {"created": True})
    return "done"
