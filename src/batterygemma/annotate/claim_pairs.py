"""Stage 7 (Annotate): generate claim pairs (paraphrase / contradiction) from one or more chunks.

Modeled on MS-CXR-T's temporal sentence-similarity set: each pair starts from a real claim in the chunk,
then either paraphrases it (same meaning, different wording) or contradicts it (a plausible-sounding but
factually altered version - a swapped direction, a changed number, an incorrect mechanism, or the wrong
entity). Paraphrase/contradiction pairs are the main source of negative examples (contradiction detection)
in stage 8, and both feed `bat-claim-pairs`. `sentence_1` is checked against the source chunk for grounding
(see `annotate.grounding`); `sentence_2` for a contradiction is *supposed* to diverge from the source, so it
is not checked.

There is exactly one mechanism for 1 chunk or many, mirroring `annotate.facts`: `extract_claim_pairs`
always takes a list of chunks and always uses the batch-shaped prompt/schema (`BatchClaimPairsOut`), even
for a list of length 1.

See `annotate.facts` module docstring for why `annotate_chunks_claims` manages its own short-lived sessions
around the LLM call rather than accepting one long-lived session from the caller (SQLite deadlock risk).
"""

import logging
from datetime import datetime

from rich.console import Console
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.grounding import is_grounded
from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, ClaimPair, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import BatchClaimPairsOut, ClaimPairOut, json_schema_response_format, json_validator

logger = logging.getLogger(__name__)
_console = Console()

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


def extract_claim_pairs(
    session: Session, router: LLMRouter, chunks: list[Chunk], *, route: str = "extract", use_cache: bool = True
) -> dict[str, list[ClaimPair]]:
    """One LLM call covering all of `chunks` (1 or more), persisted as ClaimPair rows keyed by chunk_id.
    Replaces any previous rows for each chunk in `chunks`. A chunk_index missing from the LLM's response is
    simply absent from the returned dict - the caller retries it by calling this function again with just
    that chunk.

    Caller's responsibility: `session` must not already hold pending writes from earlier in the same
    transaction (see `annotate.facts` module docstring for why).
    """
    result = router.complete(
        route, build_claim_pair_messages([c.text for c in chunks]), validate=json_validator(BatchClaimPairsOut),
        response_format=json_schema_response_format(BatchClaimPairsOut),
        max_tokens=max(2000, 2000 * len(chunks)), use_cache=use_cache,
        temperature=0,  # schema-constrained structured extraction: see annotate.facts for why
    )
    parsed: BatchClaimPairsOut = result.parsed

    logger.info(
        f"extract_claims call via {result.deployment} ({len(chunks)} chunks): "
        f"tokens_in={result.tokens_in}, tokens_out={result.tokens_out}",
        extra={"context": {"deployment": result.deployment, "chunks": len(chunks),
                            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cached": result.cached}},
    )
    _console.print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] extract_claims via {result.deployment}: "
                   f"tokens_in={result.tokens_in}, tokens_out={result.tokens_out} ({len(chunks)} chunks)"
                   + (" [cached]" if result.cached else ""))

    out: dict[str, list[ClaimPair]] = {}
    for chunk_result in parsed.results:
        if not (0 <= chunk_result.chunk_index < len(chunks)):
            continue
        chunk = chunks[chunk_result.chunk_index]
        session.execute(delete(ClaimPair).where(ClaimPair.id.like(f"{chunk.chunk_id}#pair%")))
        pairs = [_pair_row(chunk, i, p, generator_model=result.model) for i, p in enumerate(chunk_result.pairs)]
        session.add_all(pairs)
        out[chunk.chunk_id] = pairs
    return out


def annotate_chunks_claims(
    engine: Engine, router: LLMRouter, chunk_ids: list[str], *, force: bool = False, _retry: bool = True
) -> dict[str, str]:
    """Resumable claim-pair generation over 1 or more chunk_ids in a single batched LLM call. Returns a
    {chunk_id: "done" | "skipped" | "failed"} map, one entry per input chunk_id."""
    task_keys = {chunk_id: f"extract_claims:{chunk_id}" for chunk_id in chunk_ids}
    pending: list[str] = []
    outcomes: dict[str, str] = {}
    with get_session(engine) as s:
        for chunk_id in chunk_ids:
            task = get_or_create_task(s, "extract_claims", task_keys[chunk_id])
            if task.status == "done" and not force:
                outcomes[chunk_id] = "skipped"
                continue
            task.attempts += 1
            pending.append(chunk_id)
    if not pending:
        return outcomes

    try:
        with get_session(engine) as s:
            chunks = [s.get(Chunk, chunk_id) for chunk_id in pending]
            results = extract_claim_pairs(s, router, chunks, use_cache=not force)
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
            payload = {"pairs": len(results[chunk_id])}
            mark_done(s.scalars(select(GenTask).where(GenTask.key == task_keys[chunk_id])).one(), payload)
            outcomes[chunk_id] = "done"

    if missing and _retry:
        # Retry, once, any chunk the LLM silently dropped from the batch response - same mechanism, applied
        # to just the missing ids. `_retry=False` on the recursive call caps this at one extra attempt.
        outcomes.update(annotate_chunks_claims(engine, router, missing, force=force, _retry=False))
    elif missing:
        outcomes.update({chunk_id: "failed" for chunk_id in missing})
    return outcomes
