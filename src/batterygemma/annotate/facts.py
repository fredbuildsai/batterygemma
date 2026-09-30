"""Stage 7 (Annotate): extract facts and comparisons from one or more chunks via the teacher LLM.

Facts (material -> property/value/unit/conditions) and comparisons (baseline vs. modification -> metric,
direction) are the base layer the plan describes; the derived layer (Q&A, ideation, negatives, DPO — stage 8)
is built from these plus the raw chunks. Every fact/comparison carries an `evidence_sentence`, which is
checked against the source chunk (see `annotate.grounding`) before being accepted.

This module is only the *domain* half of the stage: the prompt (`build_extraction_messages`), the response
schema (`BatchExtractionOut`) and how one chunk's result becomes `Fact`/`Comparison` rows (`persist_facts`),
bundled as `FACTS_SPEC`. Everything mechanical - one router call for 1 or N chunks, `gen_tasks` bookkeeping,
retrying chunks the model dropped, back-off, concurrency, never holding a transaction across the LLM call - is
`corpusforge.runner`, shared with every other stage. A single chunk is just a batch of one; there is
deliberately no separate single-chunk code path.
"""

from corpusforge.annotate.grounding import is_grounded
from corpusforge.models import Chunk
from corpusforge.runner import ChunkTaskSpec, run_batch
from llmrouter_free import LLMRouter
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from batterygemma.db.models import Comparison, Fact
from batterygemma.llm.schemas import BatchExtractionOut, ChunkExtractionOut, ComparisonOut, FactOut

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


def persist_facts(session: Session, chunk: Chunk, chunk_result: ChunkExtractionOut, model: str) -> dict[str, int]:
    """Replace any earlier Fact/Comparison rows for `chunk` with rows built from `chunk_result`."""
    session.execute(delete(Fact).where(Fact.id.like(f"{chunk.chunk_id}#fact%")))
    session.execute(delete(Comparison).where(Comparison.id.like(f"{chunk.chunk_id}#cmp%")))
    facts = [_fact_row(chunk, i, f, generator_model=model) for i, f in enumerate(chunk_result.facts)]
    comparisons = [_comparison_row(chunk, i, c, generator_model=model) for i, c in enumerate(chunk_result.comparisons)]
    session.add_all([*facts, *comparisons])
    return {"facts": len(facts), "comparisons": len(comparisons)}


FACTS_SPEC = ChunkTaskSpec(
    task_type="extract_facts", build_messages=build_extraction_messages, response_schema=BatchExtractionOut,
    persist_result=persist_facts, output_tokens_per_chunk=3000,  # keep in sync with llm.context_budget
    route="extract",  # temperature 0 (the runner's default): verified live (2026-09-15) to give the most
                      # consistent, fully-populated JSON for both Nemotron and local gemma4:e4b
)


def annotate_chunks_facts(
    engine: Engine, router: LLMRouter, chunk_ids: list[str], *, force: bool = False
) -> dict[str, str]:
    """Resumable extraction over 1 or more chunk_ids in a single batched LLM call. Returns a
    {chunk_id: "done" | "skipped" | "failed"} map, one entry per input chunk_id."""
    return run_batch(engine, router, FACTS_SPEC, chunk_ids, force=force)
