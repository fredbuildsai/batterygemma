"""Stage 7 (Annotate): extract facts and comparisons from one chunk via the teacher LLM.

Facts (material -> property/value/unit/conditions) and comparisons (baseline vs. modification -> metric,
direction) are the base layer the plan describes; the derived layer (Q&A, ideation, negatives, DPO — stage 8)
is built from these plus the raw chunks. Every fact/comparison carries an `evidence_sentence`, which is
checked against the source chunk (see `annotate.grounding`) before being accepted.

`annotate_chunk_facts` (the resumable, engine-level entry point used by `bg annotate facts`) deliberately
never holds a database transaction open across the LLM call: `router.complete()` writes its own `llm_calls`
row through an independent session on the same SQLite file, and on file-based SQLite two overlapping write
transactions from the same process will deadlock (each waits for the other's commit) rather than merely
queue. Task bookkeeping (before) and result persistence (after) each get their own short, fully-committed
session; only the request/response round trip happens with no open transaction at all.
"""

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.grounding import is_grounded
from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, Comparison, Fact, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import ComparisonOut, ExtractionOut, FactOut, json_validator

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist extracting structured facts from a paper "
    "excerpt. Only extract facts and comparisons that are explicitly stated in the given text - do not use "
    "outside knowledge, and do not infer numbers that are not written down. Each item must carry an "
    "evidence_sentence copied (or lightly trimmed) verbatim from the text that supports it. If the text "
    "contains no clear, checkable facts or comparisons, return empty lists."
)

USER_TEMPLATE = """Extract facts and comparisons from this excerpt as JSON matching this exact shape:
{{"facts": [{{"material": {{"name": str, "formula": str|null, "class": str|null, "component": one of \
["cathode","anode","electrolyte","interphase","separator_binder","cell"]|null}}, "property": str, "value": \
str|null, "unit": str|null, "conditions": {{}}, "category": one of ["structure","electrochemical","thermal",\
"mechanical","synthesis","characterization","degradation","cost_safety"], "polarity": one of \
["positive","negative","uncertain"], "triple": [subject, relation, object] or [], "evidence_sentence": str}}],
"comparisons": [{{"baseline": str, "modification": str, "metric": str, "direction": one of \
["improves","unchanged","worsens"], "magnitude": str|null, "conditions": {{}}, "component": str|null, \
"evidence_sentence": str}}]}}

A "fact" is a single material's measured or stated property. A "comparison" is baseline-vs-modification \
(e.g. "single-crystal NMC811 retained 10% more capacity than polycrystalline"). Extract at most 8 facts and \
4 comparisons - only the clearest, most specific ones. Return JSON only, no other text.

Excerpt:
\"\"\"
{text}
\"\"\""""


def build_extraction_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=chunk_text)},
    ]


def _fact_row(chunk: Chunk, index: int, fact: FactOut, *, generator_model: str) -> Fact:
    doc = chunk.document
    grounded = is_grounded(fact.evidence_sentence, chunk.text)
    return Fact(
        id=f"{chunk.chunk_id}#fact{index}",
        doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=generator_model, tier="silver",
        status="generated" if grounded else "rejected",
        reject_reason=None if grounded else "ungrounded_evidence",
        material=fact.material.model_dump(by_alias=True, exclude_none=True),
        component=fact.material.component, property=fact.property, value=fact.value, unit=fact.unit,
        conditions=fact.conditions, category=fact.category, polarity=fact.polarity, triple=fact.triple,
        evidence_sentence=fact.evidence_sentence,
    )


def _comparison_row(chunk: Chunk, index: int, comp: ComparisonOut, *, generator_model: str) -> Comparison:
    doc = chunk.document
    grounded = is_grounded(comp.evidence_sentence, chunk.text)
    return Comparison(
        id=f"{chunk.chunk_id}#cmp{index}",
        doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=generator_model, tier="silver",
        status="generated" if grounded else "rejected",
        reject_reason=None if grounded else "ungrounded_evidence",
        baseline=comp.baseline, modification=comp.modification, metric=comp.metric, direction=comp.direction,
        magnitude=comp.magnitude, conditions=comp.conditions, component=comp.component,
        evidence_sentence=comp.evidence_sentence,
    )


def extract_facts_and_comparisons(
    session: Session, router: LLMRouter, chunk: Chunk, *, route: str = "extract", use_cache: bool = True
) -> tuple[list[Fact], list[Comparison]]:
    """One LLM call for `chunk`, persisted as Fact/Comparison rows. Replaces any previous rows for this chunk.

    Caller's responsibility: `session` must not already hold pending writes from earlier in the same
    transaction (see the module docstring for why) - open a fresh session and call this first if unsure.
    """
    result = router.complete(
        route, build_extraction_messages(chunk.text), validate=json_validator(ExtractionOut),
        json_mode=True, max_tokens=3000, use_cache=use_cache,
    )
    parsed: ExtractionOut = result.parsed

    session.execute(delete(Fact).where(Fact.id.like(f"{chunk.chunk_id}#fact%")))
    session.execute(delete(Comparison).where(Comparison.id.like(f"{chunk.chunk_id}#cmp%")))

    facts = [_fact_row(chunk, i, f, generator_model=result.model) for i, f in enumerate(parsed.facts)]
    comparisons = [_comparison_row(chunk, i, c, generator_model=result.model) for i, c in enumerate(parsed.comparisons)]
    session.add_all([*facts, *comparisons])
    return facts, comparisons


def annotate_chunk_facts(engine: Engine, router: LLMRouter, chunk_id: str, *, force: bool = False) -> str:
    """Resumable single-chunk extraction, by chunk_id. Returns: done | skipped | failed.

    `force=True` also bypasses the router's response cache - otherwise re-running with the identical prompt
    would just replay the same cached answer, defeating the point of forcing a re-run.
    """
    task_key = f"extract_facts:{chunk_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "extract_facts", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1
    # committed and closed above: no open transaction while the router call (and its own DB writes) runs

    try:
        with get_session(engine) as s:
            chunk = s.get(Chunk, chunk_id)
            facts, comparisons = extract_facts_and_comparisons(s, router, chunk, use_cache=not force)
            payload = {"facts": len(facts), "comparisons": len(comparisons)}
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), payload)
    return "done"
