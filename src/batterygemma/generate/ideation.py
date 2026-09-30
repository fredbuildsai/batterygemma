"""Stage 8 (Generate): grounded ideation from one chunk's extracted facts/comparisons.

v1 simplification: the plan describes ideation from topic clusters spanning 3-5 papers; this generates one
ideation record per chunk that already has >= `MIN_FACTS_FOR_IDEATION` accepted facts/comparisons, using
just that chunk's own text and evidence. Cross-paper clustering (grouping chunks by shared component +
chemistry + failure mode) is a real follow-up, not implemented here - flagged rather than half-built.

No automatic grounding check here (unlike facts/QA/negatives): an idea's `hypothesis` is meant to go beyond
what's stated, not restate it, so word-overlap grounding doesn't apply. Quality control instead comes from
stage 9's judge scoring (`verify.judge`), matching the plan's "2 judges + agreement" design for ideation.
"""

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, Comparison, Fact, GenTask, Ideation
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import IdeationOut, json_schema_response_format, json_validator

MIN_FACTS_FOR_IDEATION = 2

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist ideating on a research problem. Reason from "
    "the mechanisms and evidence in the excerpt to a specific problem statement, then propose testable ideas "
    "with concrete validation experiments - grounded in the science, not generic advice."
)

USER_TEMPLATE = """From this excerpt, identify one specific materials problem and propose 1-2 ideas to \
address it, as JSON matching this exact shape:
{{"problem": str, "constraints": [str], "reasoning": str, "ideas": [{{"hypothesis": str, "mechanism": str, \
"risks": str, "validation_experiments": [str], "success_metrics": [str]}}]}}

"reasoning" traces from the excerpt's mechanisms/evidence to why the problem matters. Each idea's \
"mechanism" explains *why* it should work, not just what it is. "validation_experiments" must be concrete \
and doable (e.g. "cycle at C/3 for 300 cycles and compare capacity retention to baseline"), not vague.
Return JSON only, no other text.

Excerpt:
\"\"\"
{text}
\"\"\""""


def build_ideation_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=chunk_text)},
    ]


def has_enough_facts(session: Session, chunk_id: str) -> bool:
    fact_count = session.scalar(
        select(func.count()).select_from(Fact).where(Fact.chunk_ids.contains([chunk_id]), Fact.status == "generated")
    ) or 0
    comparison_count = session.scalar(
        select(func.count()).select_from(Comparison).where(
            Comparison.chunk_ids.contains([chunk_id]), Comparison.status == "generated"
        )
    ) or 0
    return (fact_count + comparison_count) >= MIN_FACTS_FOR_IDEATION


def generate_ideation(
    session: Session, router: LLMRouter, chunk: Chunk, *, route: str = "ideation", use_cache: bool = True
) -> Ideation | None:
    """One LLM call for `chunk`, persisted as an Ideation row. Returns None if the model proposed no ideas."""
    result = router.complete(
        route, build_ideation_messages(chunk.text), validate=json_validator(IdeationOut),
        response_format=json_schema_response_format(IdeationOut), max_tokens=3000, use_cache=use_cache,
    )
    parsed: IdeationOut = result.parsed

    row_id = f"{chunk.chunk_id}#idea"
    session.execute(delete(Ideation).where(Ideation.id == row_id))
    if not parsed.ideas:
        return None
    doc = chunk.document
    row = Ideation(
        id=row_id, doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=result.model, tier="silver", status="generated",
        task_format="grounded_ideation", polarity="positive", component=None, grounding="source+background",
        problem=parsed.problem, constraints=parsed.constraints, reasoning=parsed.reasoning,
        ideas=[idea.model_dump() for idea in parsed.ideas],
    )
    session.add(row)
    return row


def annotate_chunk_ideation(engine: Engine, router: LLMRouter, chunk_id: str, *, force: bool = False) -> str:
    """Resumable single-chunk ideation, by chunk_id. Returns: done | skipped | too_few_facts | failed."""
    task_key = f"generate_ideation:{chunk_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "generate_ideation", task_key)
        if task.status == "done" and not force:
            return "skipped"
        if not has_enough_facts(s, chunk_id):
            return "too_few_facts"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            chunk = s.get(Chunk, chunk_id)
            row = generate_ideation(s, router, chunk, use_cache=not force)
            payload = {"created": row is not None}
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), payload)
    return "done"
