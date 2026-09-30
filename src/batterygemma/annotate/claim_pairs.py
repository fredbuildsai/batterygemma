"""Stage 7 (Annotate): generate claim pairs (paraphrase / contradiction) from one or more chunks.

Modeled on MS-CXR-T's temporal sentence-similarity set: each pair starts from a real claim in the chunk,
then either paraphrases it (same meaning, different wording) or contradicts it (a plausible-sounding but
factually altered version - a swapped direction, a changed number, an incorrect mechanism, or the wrong
entity). Paraphrase/contradiction pairs are the main source of negative examples (contradiction detection)
in stage 8, and both feed `bat-claim-pairs`. `sentence_1` is checked against the source chunk for grounding
(see `annotate.grounding`); `sentence_2` for a contradiction is *supposed* to diverge from the source, so it
is not checked.

Like `annotate.facts`, this module is only the domain half of the stage (prompt, schema `BatchClaimPairsOut`,
row builder, `CLAIMS_SPEC`); the batching, bookkeeping, retry and back-off mechanics are `corpusforge.runner`,
and a single chunk is a batch of one.
"""

from corpusforge.annotate.grounding import is_grounded
from corpusforge.models import Chunk
from corpusforge.runner import ChunkTaskSpec, run_batch
from llmrouter_free import LLMRouter
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from batterygemma.db.models import ClaimPair
from batterygemma.llm.schemas import BatchClaimPairsOut, ChunkClaimPairsOut, ClaimPairOut

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist writing claim pairs for a fact-checking "
    "dataset. Every sentence_1 must be a specific, checkable claim taken (lightly trimmed is fine) from the "
    "given text - never invent a claim that isn't there."
)

USER_TEMPLATE = """From each of the following excerpts, write claim pairs as JSON matching this exact shape:
{{"results": [{{"chunk_index": int, "pairs": [{{"sentence_1": str, "sentence_2": str, "category": \
"paraphrase" or "contradiction", "subset_name": one of ["swap","numeric","mechanism","entity"]}}]}}]}}

There is exactly one entry in "results" per excerpt below, in order, and each entry's "chunk_index" must \
equal the excerpt's number. An excerpt with no claim worth pairing still gets an entry with an empty \
"pairs" list - never omit an excerpt from "results".

For "paraphrase" pairs: sentence_2 restates sentence_1 with different wording but the identical meaning.
For "contradiction" pairs: sentence_2 alters sentence_1 in one specific, plausible-sounding way that makes \
it factually wrong - use subset_name to say how: "swap" (reversed direction/outcome, e.g. improves<->worsens), \
"numeric" (a changed number, unit or magnitude), "mechanism" (a different, incorrect causal mechanism), or \
"entity" (swapped material/component, e.g. cathode<->anode).

Write up to 4 pairs per excerpt, aiming for a mix of paraphrase and contradiction, only from claims that are \
explicit and specific enough to be verified as true or false. Return JSON only, no other text.

{excerpts}"""


def _format_excerpts(chunk_texts: list[str]) -> str:
    return "\n\n".join(f'Excerpt {i}:\n"""\n{text}\n"""' for i, text in enumerate(chunk_texts))


def build_claim_pair_messages(chunk_texts: list[str]) -> list[dict[str, str]]:
    """`chunk_texts` may hold 1 or more excerpts - the prompt is always batch-shaped (see module docstring)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(excerpts=_format_excerpts(chunk_texts))},
    ]


def _pair_row(chunk: Chunk, index: int, pair: ClaimPairOut, *, generator_model: str) -> ClaimPair:
    doc = chunk.document
    grounded = is_grounded(pair.sentence_1, chunk.text)  # sentence_2 may deliberately diverge; not checked
    return ClaimPair(
        id=f"{chunk.chunk_id}#pair{index}",
        doc_ids=[doc.doc_id], chunk_ids=[chunk.chunk_id], license=doc.license,
        generator_model=generator_model, tier="silver",
        status="generated" if grounded else "rejected",
        reject_reason=None if grounded else "ungrounded_sentence_1",
        sentence_1=pair.sentence_1, sentence_2=pair.sentence_2, category=pair.category,
        subset_name=pair.subset_name,
    )


def persist_claim_pairs(session: Session, chunk: Chunk, chunk_result: ChunkClaimPairsOut, model: str) -> dict[str, int]:
    """Replace any earlier ClaimPair rows for `chunk` with rows built from `chunk_result`."""
    session.execute(delete(ClaimPair).where(ClaimPair.id.like(f"{chunk.chunk_id}#pair%")))
    pairs = [_pair_row(chunk, i, p, generator_model=model) for i, p in enumerate(chunk_result.pairs)]
    session.add_all(pairs)
    return {"pairs": len(pairs)}


CLAIMS_SPEC = ChunkTaskSpec(
    task_type="extract_claims", build_messages=build_claim_pair_messages, response_schema=BatchClaimPairsOut,
    persist_result=persist_claim_pairs, output_tokens_per_chunk=2000,  # keep in sync with llm.context_budget
    route="extract",
)


def annotate_chunks_claims(
    engine: Engine, router: LLMRouter, chunk_ids: list[str], *, force: bool = False
) -> dict[str, str]:
    """Resumable claim-pair generation over 1 or more chunk_ids in a single batched LLM call. Returns a
    {chunk_id: "done" | "skipped" | "failed"} map, one entry per input chunk_id."""
    return run_batch(engine, router, CLAIMS_SPEC, chunk_ids, force=force)
