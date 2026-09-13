"""Stage 8 (Generate): expert Q&A pairs from one chunk via the teacher LLM.

Each item's `answer` is checked against the source chunk for grounding (see `annotate.grounding`) before
being accepted - the same rule stage 7 applies to extracted facts. See `annotate.facts` module docstring for
why `annotate_chunk_qa` manages its own short-lived sessions around the LLM call rather than accepting one
long-lived session from the caller (SQLite deadlock risk when the router logs to llm_calls mid-transaction).
"""

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.grounding import is_grounded
from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, GenTask, QA
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import QAGenerationOut, QAItemOut, json_validator

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist writing training questions. Every answer "
    "must be fully supported by the given excerpt - use its specific numbers, mechanisms and conclusions, "
    "never outside knowledge that isn't in the text. Write in the voice of an expert explaining to a "
    "colleague: precise, mechanistic, using correct units."
)

USER_TEMPLATE = """From this excerpt, write 2-4 expert Q&A pairs as JSON matching this exact shape:
{{"items": [{{"question": str, "answer": str, "reasoning": str, "question_type": one of ["mechanism",\
"structure_property","trade_off","characterization_interpretation","synthesis_processing",\
"failure_analysis","quantitative","safety_cost","comparison","other"], "answer_type": "OPEN" or "CLOSED", \
"component": one of ["cathode","anode","electrolyte","interphase","separator_binder","cell"]|null, \
"chemistry": str|null}}]}}

"reasoning" is the step-by-step derivation from the excerpt to the answer (mechanism chain, not just a
restatement). Use a mix of question_types where the excerpt supports it. A CLOSED question has a short
factual/yes-no answer; OPEN needs explanation. Only ask what this excerpt can actually support - do not pad
with generic questions. If the excerpt has no clear, specific, checkable content, return an empty list.
Return JSON only, no other text.

Excerpt:
\"\"\"
{text}
\"\"\""""


def build_qa_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=chunk_text)},
    ]


def _qa_row(chunk: Chunk, index: int, item: QAItemOut, *, generator_model: str) -> QA:
    doc = chunk.document
    grounded = is_grounded(item.answer, chunk.text)
    return QA(
        id=f"{chunk.chunk_id}#qa{index}",
        doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=generator_model, tier="silver",
        status="generated" if grounded else "rejected",
        reject_reason=None if grounded else "ungrounded_answer",
        task_format="instruction_response", polarity="positive",
        question_type=item.question_type, component=item.component, chemistry=item.chemistry,
        grounding="source", answer_type=item.answer_type, phrase_type="free-form",
        system=SYSTEM_PROMPT, reasoning=item.reasoning,
        turns=[{"role": "user", "content": item.question}, {"role": "assistant", "content": item.answer}],
    )


def generate_qa(session: Session, router: LLMRouter, chunk: Chunk, *, route: str = "qa", use_cache: bool = True) -> list[QA]:
    """One LLM call for `chunk`, persisted as QA rows. Replaces any previous QA rows for this chunk.

    Caller's responsibility: `session` must not already hold pending writes from earlier in the same
    transaction (see `annotate.facts` module docstring for why).
    """
    result = router.complete(
        route, build_qa_messages(chunk.text), validate=json_validator(QAGenerationOut),
        json_mode=True, max_tokens=3000, use_cache=use_cache,
    )
    parsed: QAGenerationOut = result.parsed

    session.execute(delete(QA).where(QA.id.like(f"{chunk.chunk_id}#qa%")))
    rows = [_qa_row(chunk, i, item, generator_model=result.model) for i, item in enumerate(parsed.items)]
    session.add_all(rows)
    return rows


def annotate_chunk_qa(engine: Engine, router: LLMRouter, chunk_id: str, *, force: bool = False) -> str:
    """Resumable single-chunk Q&A generation, by chunk_id. Returns: done | skipped | failed."""
    task_key = f"generate_qa:{chunk_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "generate_qa", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            chunk = s.get(Chunk, chunk_id)
            rows = generate_qa(s, router, chunk, use_cache=not force)
            payload = {"items": len(rows)}
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), payload)
    return "done"
