"""Stage 8 (Generate): negative training examples (SFT).

Two sources, matching the plan's negative-example kinds:

- `contradiction_detection`: derived directly from stage 7's `claim_pairs` with category="contradiction" -
  no extra LLM call needed, since the pair already states a plausible-but-wrong claim and the true one it
  contradicts. This is the cheapest and most reliable negative source.
- `false_premise`: one LLM call per chunk, asking for a question that embeds a specific factual error the
  excerpt disproves, plus the correction an expert would give.

See `annotate.facts` module docstring for why the resumable wrappers manage their own short-lived sessions
around the LLM call (SQLite deadlock risk).
"""

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import ClaimPair, Document, GenTask, Negative
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import FalsePremiseOut, json_validator

FALSE_PREMISE_SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist. Write one question that contains a specific, "
    "plausible-sounding factual error the given excerpt disproves or contradicts, then the correction an "
    "expert would give. Do not use a strawman error - it should sound like something a non-expert might "
    "genuinely believe."
)

FALSE_PREMISE_USER_TEMPLATE = """From this excerpt, write one false-premise example as JSON matching this \
exact shape:
{{"prompt": str, "flawed_element": str, "expert_response": str, "reasoning": str}}

"prompt" is a question that embeds the false premise (e.g. "Why does X do Y?" when X does not do Y).
"flawed_element" names the specific false claim embedded in the prompt.
"expert_response" corrects it and explains why, grounded in this excerpt.
"reasoning" is the mechanism chain supporting the correction.
If the excerpt has no specific, checkable claim to build a false premise from, return \
{{"prompt": null, "flawed_element": null, "expert_response": null, "reasoning": null}}.
Return JSON only, no other text.

Excerpt:
\"\"\"
{text}
\"\"\""""


def build_false_premise_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FALSE_PREMISE_SYSTEM_PROMPT},
        {"role": "user", "content": FALSE_PREMISE_USER_TEMPLATE.format(text=chunk_text)},
    ]


def derive_contradiction_negatives(session: Session, doc_id: str) -> list[Negative]:
    """Converts accepted contradiction claim_pairs for `doc_id`'s chunks into Negative rows. No LLM call."""
    doc = session.get(Document, doc_id)
    pairs = session.scalars(
        select(ClaimPair).where(
            ClaimPair.id.like(f"{doc_id}%"), ClaimPair.category == "contradiction", ClaimPair.status == "generated"
        )
    ).all()
    rows = []
    for pair in pairs:
        negative_id = f"{pair.id}#neg"
        session.execute(delete(Negative).where(Negative.id == negative_id))
        rows.append(
            Negative(
                id=negative_id, doc_ids=pair.doc_ids, chunk_ids=pair.chunk_ids, license=pair.license,
                generator_model=pair.generator_model, tier=pair.tier, status="generated",
                task_format="contradiction_detection", polarity="negative",
                kind="contradiction_detection",
                prompt=f"Does this claim hold, given the source material? \"{pair.sentence_2}\"",
                flawed_element=pair.sentence_2,
                expert_response=(
                    f"No - this contradicts the source, which states: \"{pair.sentence_1}\" "
                    f"(altered via {pair.subset_name})."
                ),
                reasoning=f"The source states: \"{pair.sentence_1}\". The claim above reverses/alters this.",
                linked_positive_id=pair.id,
            )
        )
    if doc is not None:
        for row in rows:
            row.license = row.license or doc.license
    session.add_all(rows)
    return rows


def generate_false_premise(
    session: Session, router: LLMRouter, chunk, *, route: str = "negatives", use_cache: bool = True
) -> Negative | None:
    """One LLM call for `chunk`; returns the persisted Negative row, or None if the model found nothing to use."""
    result = router.complete(
        route, build_false_premise_messages(chunk.text), validate=json_validator(FalsePremiseOut),
        json_mode=True, max_tokens=1500, use_cache=use_cache,
    )
    parsed: FalsePremiseOut = result.parsed
    row_id = f"{chunk.chunk_id}#negfp"
    session.execute(delete(Negative).where(Negative.id == row_id))
    if parsed.prompt is None:
        return None
    doc = chunk.document
    row = Negative(
        id=row_id, doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=result.model, tier="silver", status="generated",
        task_format="false_premise", polarity="negative", kind="false_premise",
        prompt=parsed.prompt, flawed_element=parsed.flawed_element, expert_response=parsed.expert_response,
        reasoning=parsed.reasoning,
    )
    session.add(row)
    return row


def annotate_chunk_false_premise(engine: Engine, router: LLMRouter, chunk_id: str, *, force: bool = False) -> str:
    """Resumable single-chunk false-premise generation, by chunk_id. Returns: done | skipped | failed."""
    from batterygemma.db.models import Chunk

    task_key = f"generate_false_premise:{chunk_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "generate_false_premise", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            chunk = s.get(Chunk, chunk_id)
            row = generate_false_premise(s, router, chunk, use_cache=not force)
            payload = {"created": row is not None}
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), payload)
    return "done"
