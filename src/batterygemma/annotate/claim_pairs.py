"""Stage 7 (Annotate): generate claim pairs (paraphrase / contradiction) from one chunk.

Modeled on MS-CXR-T's temporal sentence-similarity set: each pair starts from a real claim in the chunk,
then either paraphrases it (same meaning, different wording) or contradicts it (a plausible-sounding but
factually altered version - a swapped direction, a changed number, an incorrect mechanism, or the wrong
entity). Paraphrase/contradiction pairs are the main source of negative examples (contradiction detection)
in stage 8, and both feed `bat-claim-pairs`. `sentence_1` is checked against the source chunk for grounding
(see `annotate.grounding`); `sentence_2` for a contradiction is *supposed* to diverge from the source, so it
is not checked.

See `annotate.facts` module docstring for why `annotate_chunk_claims` manages its own short-lived sessions
around the LLM call rather than accepting one long-lived session from the caller (SQLite deadlock risk).
"""

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from batterygemma.annotate.grounding import is_grounded
from batterygemma.annotate.tasks import get_or_create_task, mark_done, mark_failed
from batterygemma.db.models import Chunk, ClaimPair, GenTask
from batterygemma.db.session import get_session
from batterygemma.llm.router import AllDeploymentsExhausted, LLMRouter
from batterygemma.llm.schemas import ClaimPairOut, ClaimPairsOut, json_validator

SYSTEM_PROMPT = (
    "You are an expert lithium-ion battery materials scientist writing claim pairs for a fact-checking "
    "dataset. Every sentence_1 must be a specific, checkable claim taken (lightly trimmed is fine) from the "
    "given text - never invent a claim that isn't there."
)

USER_TEMPLATE = """From this excerpt, write claim pairs as JSON matching this exact shape:
{{"pairs": [{{"sentence_1": str, "sentence_2": str, "category": "paraphrase" or "contradiction", \
"subset_name": one of ["swap","numeric","mechanism","entity"]}}]}}

For "paraphrase" pairs: sentence_2 restates sentence_1 with different wording but the identical meaning.
For "contradiction" pairs: sentence_2 alters sentence_1 in one specific, plausible-sounding way that makes \
it factually wrong - use subset_name to say how: "swap" (reversed direction/outcome, e.g. improves<->worsens), \
"numeric" (a changed number, unit or magnitude), "mechanism" (a different, incorrect causal mechanism), or \
"entity" (swapped material/component, e.g. cathode<->anode).

Write up to 4 pairs total, aiming for a mix of paraphrase and contradiction, only from claims that are \
explicit and specific enough to be verified as true or false. If no such claims exist, return an empty list. \
Return JSON only, no other text.

Excerpt:
\"\"\"
{text}
\"\"\""""


def build_claim_pair_messages(chunk_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=chunk_text)},
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
    session: Session, router: LLMRouter, chunk: Chunk, *, route: str = "extract", use_cache: bool = True
) -> list[ClaimPair]:
    """One LLM call for `chunk`, persisted as ClaimPair rows. Replaces any previous rows for this chunk.

    Caller's responsibility: `session` must not already hold pending writes from earlier in the same
    transaction (see `annotate.facts` module docstring for why).
    """
    result = router.complete(
        route, build_claim_pair_messages(chunk.text), validate=json_validator(ClaimPairsOut),
        json_mode=True, max_tokens=2000, use_cache=use_cache,
    )
    parsed: ClaimPairsOut = result.parsed

    session.execute(delete(ClaimPair).where(ClaimPair.id.like(f"{chunk.chunk_id}#pair%")))
    pairs = [_pair_row(chunk, i, p, generator_model=result.model) for i, p in enumerate(parsed.pairs)]
    session.add_all(pairs)
    return pairs


def annotate_chunk_claims(engine: Engine, router: LLMRouter, chunk_id: str, *, force: bool = False) -> str:
    """Resumable single-chunk claim-pair generation, by chunk_id. Returns: done | skipped | failed."""
    task_key = f"extract_claims:{chunk_id}"
    with get_session(engine) as s:
        task = get_or_create_task(s, "extract_claims", task_key)
        if task.status == "done" and not force:
            return "skipped"
        task.attempts += 1

    try:
        with get_session(engine) as s:
            chunk = s.get(Chunk, chunk_id)
            pairs = extract_claim_pairs(s, router, chunk, use_cache=not force)
            payload = {"pairs": len(pairs)}
    except AllDeploymentsExhausted as exc:
        with get_session(engine) as s:
            mark_failed(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), str(exc))
        return "failed"

    with get_session(engine) as s:
        mark_done(s.scalars(select(GenTask).where(GenTask.key == task_key)).one(), payload)
    return "done"
