# 8. Cross-Cutting Concepts

### 8.1 Licensing and provenance

Every `Document` carries `license` + `license_evidence` (the concrete signal that produced the license
classification, e.g. "chemrxiv API: license.name"). `corpusforge.screen.license` evaluates it against an allow/flag
list (`configs/sources.yaml`: `license_allow: [CC0, CC-BY, public-domain]`, `license_flag: [CC-BY-SA]`) before
a document can be accepted. Every derived row inherits `license` from its source document(s) so the export
stage can filter by license without re-deriving it.

### 8.2 The resumable-task pattern

Formalized in `corpusforge.annotate.tasks`: `get_or_create_task(session, task_type, key)`, `mark_done`, `mark_failed`,
backed by the `gen_tasks` table (`key` is unique — `"<task_type>:<id>"`). Every annotate/generate/judge
operation checks this before doing any LLM work and updates it after, so `bg annotate facts`, `bg generate
qa`, etc. can be re-run at any time (a partially completed pipeline is a normal state, not an error state).

### 8.3 The LLM router's failure taxonomy

`_classify_error()` in `router.py` maps every provider exception into one of: `auth`/`unavailable`
(disabling — stop trying this deployment this session), `rate_limited`/`quota` (cooldown, shared across a
`rate_group`), or `error` (short cooldown, e.g. transient 5xx/timeout). This taxonomy is what lets the router
distinguish "try again later" from "never try this one again" without hardcoding provider-specific logic at
every call site.

### 8.4 Grounding checks

`corpusforge.annotate.grounding::overlap_ratio`/`is_grounded` — a word-overlap heuristic (not exact match) between
generated text and its source chunk, used as a cheap sanity filter before a more expensive judge call, and
as the sole quality gate for `Negative` rows (which have no judge step — see §8.6).

### 8.5 Judge independence

`verify/judge.py` always excludes the generator's model family from the judge route
(`exclude_families=[_family_of(qa.generator_model)]`), so a model never grades its own output. When only one
family is actually available, `allow_same_family_fallback=True` permits a same-family fallback but marks the
result `relaxed_family=True` so the weaker independence is recorded, not hidden. `_family_of()` is a
best-effort heuristic on the stored model-id string (`rsplit("/", 1)[-1].split("-")[0]`) rather than an exact
stored field — a known, documented simplification.

### 8.6 Two-tier acceptance: judged vs. grounding-only

Not every generated row goes through the judge:

| Row type | Quality gate |
|---|---|
| `QA` | LLM judge (`faithfulness≥4, correctness≥4, specificity≥3`) → `status=accepted/rejected` |
| `Ideation` | LLM judge (`groundedness≥4, correctness≥4, novelty≥3, feasibility≥3`) |
| `Negative` | Grounding check only at generation time; stays `status=generated` forever (no judge step exists) — `export_sft` deliberately filters on `"generated"`, not `"accepted"`, for this table (a real bug once filtered on `"accepted"` and silently exported zero negatives). |
| `DPOPair` | Built only from already-`accepted` `QA` rows, so its `chosen` side is always judge-verified by construction. |

### 8.7 Local-model fallback and its own failure modes

Ollama-hosted Gemma models are *thinking* models. Two real failure modes were found and fixed (ADR-006):
leaving `think` enabled either burns the token budget on `reasoning_content` and returns empty `content` (at
a small `max_tokens`), or takes 150-180s+ per call (at a larger budget) — both broke the pipeline differently.
`extra_body: {think: false}` disables reasoning outright, fixing both at once (~30-70s/call, confirmed live).

### 8.8 Testing philosophy

Every module pairs with a test file that encodes a *specific, previously real* failure — not just a happy-path
smoke test. Examples: the SQLite deadlock reproduction, the `FalsePremiseOut` empty-string-vs-null validation
gap, the `export_sft` negatives-status filter bug, the `_family_of()` three-segment model-id bug, the fetch
`httpx.TransportError` crash. New code is expected to follow the same pattern: find the real bug via a live
call or a close reading, then write the test that would have caught it.
