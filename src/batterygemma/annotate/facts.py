"""Stage 7 (Annotate): extract facts and comparisons from one or more chunks via the teacher LLM.

Facts (material -> property/value/unit/conditions) and comparisons (baseline vs. modification -> metric,
direction) are the base layer the plan describes; the derived layer (Q&A, ideation, negatives, DPO — stage 8)
is built from these plus the raw chunks. Every fact/comparison carries an `evidence_sentence`, which is
checked against the source chunk (see `annotate.grounding`) before being accepted.

There is exactly one extraction mechanism for 1 chunk or many: `extract_facts_and_comparisons` always takes
a list of chunks and always uses the batch-shaped prompt/schema (`BatchExtractionOut`), even for a list of
length 1 - bundling several chunks' excerpts into one LLM call is what lets `bg annotate facts` cut the
number of API calls roughly in proportion to the batch size. A single-chunk call is just the batch mechanism
applied to a list of one; there is deliberately no separate single-chunk code path.

`annotate_chunks_facts` (the resumable, engine-level entry point used by `bg annotate facts`) deliberately
never holds a database transaction open across the LLM call: `router.complete()` writes its own `llm_calls`
row through an independent session on the same SQLite file, and on file-based SQLite two overlapping write
transactions from the same process will deadlock (each waits for the other's commit) rather than merely
queue. Task bookkeeping (before) and result persistence (after) each get their own short, fully-committed
session; only the request/response round trip happens with no open transaction at all.
"""

import logging
from datetime import datetime

from rich.console import Console
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.grounding import is_grounded
from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, Comparison, Fact, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import (
    BatchExtractionOut,
    ComparisonOut,
    FactOut,
    json_schema_response_format,
    json_validator,
)

logger = logging.getLogger(__name__)
_console = Console()

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist extracting structured facts from paper "
    "excerpts. Only extract facts and comparisons that are explicitly stated in the given text - do not use "
    "outside knowledge, and do not infer numbers that are not written down. Each item must carry an "
    "evidence_sentence copied (or lightly trimmed) verbatim from the text that supports it. If an excerpt "
    "contains no clear, checkable facts or comparisons, return empty lists for it."
)

USER_TEMPLATE = """Extract facts and comparisons from each of the following excerpts as JSON matching this exact shape:
{{"results": [{{"chunk_index": int, \
"facts": [{{"material": {{"name": str, "formula": str|null, "class": str|null, "component": one of \
["cathode","anode","electrolyte","interphase","separator_binder","cell","other"]|null}}, "property": str, \
"value": str|null, "unit": str|null, "conditions": {{}}, "category": one of ["structure","electrochemical",\
"thermal","mechanical","synthesis","characterization","degradation","cost_safety"], "polarity": one of \
["positive","negative","uncertain"], "triple": [subject, relation, object] or [], "evidence_sentence": str}}],
"comparisons": [{{"baseline": str, "modification": str, "metric": str, "direction": one of \
["improves","unchanged","worsens"], "magnitude": str|null, "conditions": {{}}, "component": str|null, \
"evidence_sentence": str}}]}}]}}

There is exactly one entry in "results" per excerpt below, in order, and each entry's "chunk_index" must \
equal the excerpt's number. An excerpt with nothing worth extracting still gets an entry with empty lists - \
never omit an excerpt from "results".

A "fact" is a single material's measured or stated property: `material` must be the actual subject that \
`property`/`value` describes in the text, not a different nearby material mentioned in the same sentence - \
check that `material.name` and `triple[0]` (the subject) refer to the same thing before finalizing each fact.

A "comparison" is ANY baseline-vs-modification \
claim in the text - not only two named materials. This includes one approach/technology vs another (e.g. \
"solid-state electrolytes vs liquid electrolytes"), a design change (e.g. "single-crystal vs polycrystalline"), \
or a process condition change. Actively scan for comparison language such as "compared with", "vs", \
"instead of", "substituting X for Y", "higher/lower than", "improved/worse than" - do not restrict \
comparisons to only two named materials.

Only report a comparison when the text states the actual outcome (which one is better/worse, or that they \
are equivalent) - not merely that a comparison, measurement or analysis was performed. "Results were \
compared with prior measurements" or "an approach was developed to compare X and Y" describes a comparison \
*method*, not a result, and must NOT be turned into a "direction" - if the text doesn't say which side won, \
skip that comparison entirely rather than guessing improves/unchanged/worsens.

Counter-example - do NOT extract a comparison from a sentence like this one: "An approach has been developed \
to estimate the HRRs from TR triggered fires and results compared with previous HRR measurements for type \
18650 cylindrical cells with a similar cathode composition." This states that a comparison was *made*, not \
which cell type had the higher or lower HRR - the correct output for this sentence is to emit NO comparison \
at all, even though it mentions two cell types and the word "compared".

Classifying `material.component`: this is standard domain classification, not "outside knowledge" - always \
classify a named material even if the text doesn't use the word itself. Examples: graphite, silicon/Si, \
lithium metal, hard carbon -> "anode". LFP/LiFePO4, NMC (also written NCM), NCA, LCO, LMO -> "cathode". \
Liquid/solid/polymer electrolytes, separator membranes -> "electrolyte" (or "interphase" for SEI/CEI). \
Binder (PVDF, CMC), conductive carbon/carbon black, current collector foils -> "other".

Material identity for doped/coated/compositional-family materials: use the material's standard family name \
even when the exact reported composition differs slightly from the nominal ratio (e.g. a reported \
Li(Ni0.8Mn0.1Co0.1)O2 or a closely related Ni-rich NMC composition is still "NMC811"; use the specific ratio \
given, e.g. "NMC622", only when the text states a materially different composition). Always note any stated \
dopant or surface coating directly in `material.name` (e.g. "Al-doped NMC811", "Al2O3-coated NMC811") rather \
than dropping that detail - a doped/coated variant is a distinct fact-worthy material, not the same as the \
undoped/uncoated parent.

Worked example of a populated comparison (for format only, not content):
{{"baseline": "liquid electrolyte cells", "modification": "solid-state electrolyte cells", "metric": \
"safety", "direction": "improves", "magnitude": "significantly", "conditions": {{}}, "component": "cell", \
"evidence_sentence": "..."}}

Extract at most 8 facts and 4 comparisons per excerpt - only the clearest, most specific ones. Return JSON \
only, no other text.

{excerpts}"""


def _format_excerpts(chunk_texts: list[str]) -> str:
    return "\n\n".join(f'Excerpt {i}:\n"""\n{text}\n"""' for i, text in enumerate(chunk_texts))


def build_extraction_messages(chunk_texts: list[str]) -> list[dict[str, str]]:
    """`chunk_texts` may hold 1 or more excerpts - the prompt is always batch-shaped (see module docstring)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(excerpts=_format_excerpts(chunk_texts))},
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
    session: Session, router: LLMRouter, chunks: list[Chunk], *, route: str = "extract", use_cache: bool = True
) -> dict[str, tuple[list[Fact], list[Comparison]]]:
    """One LLM call covering all of `chunks` (1 or more), persisted as Fact/Comparison rows keyed by
    chunk_id. Replaces any previous rows for each chunk in `chunks`. A chunk_index missing from the LLM's
    response (it's expected to return one result per excerpt, but isn't guaranteed to) is simply absent
    from the returned dict - the caller retries it by calling this function again with just that chunk.

    Caller's responsibility: `session` must not already hold pending writes from earlier in the same
    transaction (see the module docstring for why) - open a fresh session and call this first if unsure.
    """
    result = router.complete(
        route, build_extraction_messages([c.text for c in chunks]), validate=json_validator(BatchExtractionOut),
        response_format=json_schema_response_format(BatchExtractionOut),
        max_tokens=max(3000, 3000 * len(chunks)), use_cache=use_cache,
        temperature=0,  # schema-constrained structured extraction: verified live (2026-09-15) that temperature=0
                        # gives the most consistent, fully-populated JSON for both Nemotron and local gemma4:e4b
    )
    parsed: BatchExtractionOut = result.parsed

    logger.info(
        f"extract_facts call via {result.deployment} ({len(chunks)} chunks): "
        f"tokens_in={result.tokens_in}, tokens_out={result.tokens_out}",
        extra={"context": {"deployment": result.deployment, "chunks": len(chunks),
                            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cached": result.cached}},
    )
    _console.print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] extract_facts via {result.deployment}: "
                   f"tokens_in={result.tokens_in}, tokens_out={result.tokens_out} ({len(chunks)} chunks)"
                   + (" [cached]" if result.cached else ""))

    out: dict[str, tuple[list[Fact], list[Comparison]]] = {}
    for chunk_result in parsed.results:
        if not (0 <= chunk_result.chunk_index < len(chunks)):
            continue
        chunk = chunks[chunk_result.chunk_index]
        session.execute(delete(Fact).where(Fact.id.like(f"{chunk.chunk_id}#fact%")))
        session.execute(delete(Comparison).where(Comparison.id.like(f"{chunk.chunk_id}#cmp%")))
        facts = [_fact_row(chunk, i, f, generator_model=result.model) for i, f in enumerate(chunk_result.facts)]
        comparisons = [_comparison_row(chunk, i, c, generator_model=result.model)
                       for i, c in enumerate(chunk_result.comparisons)]
        session.add_all([*facts, *comparisons])
        out[chunk.chunk_id] = (facts, comparisons)
    return out


def annotate_chunks_facts(
    engine: Engine, router: LLMRouter, chunk_ids: list[str], *, force: bool = False, _retry: bool = True
) -> dict[str, str]:
    """Resumable extraction over 1 or more chunk_ids in a single batched LLM call. Returns a
    {chunk_id: "done" | "skipped" | "failed"} map, one entry per input chunk_id.

    `force=True` also bypasses the router's response cache - otherwise re-running with the identical prompt
    would just replay the same cached answer, defeating the point of forcing a re-run.
    """
    task_keys = {chunk_id: f"extract_facts:{chunk_id}" for chunk_id in chunk_ids}
    pending: list[str] = []
    outcomes: dict[str, str] = {}
    with get_session(engine) as s:
        for chunk_id in chunk_ids:
            task = get_or_create_task(s, "extract_facts", task_keys[chunk_id])
            if task.status == "done" and not force:
                outcomes[chunk_id] = "skipped"
                continue
            task.attempts += 1
            pending.append(chunk_id)
    if not pending:
        return outcomes
    # committed and closed above: no open transaction while the router call (and its own DB writes) runs

    try:
        with get_session(engine) as s:
            chunks = [s.get(Chunk, chunk_id) for chunk_id in pending]
            results = extract_facts_and_comparisons(s, router, chunks, use_cache=not force)
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            for chunk_id in pending:
                mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_keys[chunk_id])).one(), str(exc))
        outcomes.update({chunk_id: "failed" for chunk_id in pending})
        return outcomes

    missing = [chunk_id for chunk_id in pending if chunk_id not in results]
    with get_session(engine) as s:
        for chunk_id in pending:
            if chunk_id not in results:
                continue
            facts, comparisons = results[chunk_id]
            payload = {"facts": len(facts), "comparisons": len(comparisons)}
            mark_done(s.scalars(select(GenTask).where(GenTask.key == task_keys[chunk_id])).one(), payload)
            outcomes[chunk_id] = "done"

    if missing and _retry:
        # Retry, once, any chunk the LLM silently dropped from the batch response - same mechanism, applied
        # to just the missing ids (a singleton list when only one was dropped). `_retry=False` on the
        # recursive call caps this at a single extra attempt per chunk, so a chunk the model keeps refusing
        # to return still terminates as "failed" instead of looping forever.
        retried = annotate_chunks_facts(engine, router, missing, force=force, _retry=False)
        outcomes.update(retried)
    elif missing:
        outcomes.update({chunk_id: "failed" for chunk_id in missing})
    return outcomes
