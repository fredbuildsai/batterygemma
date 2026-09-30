# 6. Runtime View

### 6.1 Scenario: resumable chunk annotation (the SQLite-deadlock-avoidance pattern)

This is the most consequential runtime pattern in the codebase (see ADR-003). Every `annotate_chunk_*` /
`generate_*` / `judge_one_*` function follows the same three-phase shape, taking an `Engine` rather than a
`Session` so it can open and close short-lived sessions around the one long-running step (the LLM call):

```mermaid
sequenceDiagram
    participant CLI
    participant TaskFn as annotate_chunk_facts(engine, ...)
    participant DB as SQLite (via short sessions)
    participant Router as LLMRouter
    participant LLMCallsTbl as llm_calls table

    CLI->>TaskFn: chunk_id
    TaskFn->>DB: session 1: get-or-create GenTask, increment attempts, COMMIT
    TaskFn->>DB: session 2 (fresh): fetch chunk
    TaskFn->>Router: complete(route="extract", messages)
    Router->>LLMCallsTbl: session (router's own): log attempt, COMMIT
    Router-->>TaskFn: LLMResult
    TaskFn->>DB: session 2 continued: persist Fact/Comparison rows, COMMIT
    TaskFn->>DB: session 3: mark_done(task), COMMIT
    TaskFn-->>CLI: "done" | "skipped" | "failed"
```

No session is ever left open with pending writes while the LLM call is in flight — the router logs to
`llm_calls` via its *own* independent session concurrently, and two overlapping SQLite writers on the same
file without this discipline deadlock rather than queue (confirmed by reproduction before the fix).

### 6.2 Scenario: teacher-LLM call with failover

```mermaid
sequenceDiagram
    participant Caller
    participant Router as LLMRouter.complete()
    participant D1 as nvidia-deepseek
    participant D2 as mistral-small
    participant D3 as ollama-gemma4-12b (local)

    Caller->>Router: complete(route="extract", messages, validate=...)
    Router->>D1: chat completion
    D1-->>Router: 429 rate_limited
    Router->>Router: cool down nvidia-workspace group (60s)
    Router->>D2: chat completion
    D2-->>Router: 429 rate_limited
    Router->>Router: cool down mistral-workspace group (60s)
    Router->>D3: chat completion (extra_body: think=false)
    D3-->>Router: 200 OK, valid JSON
    Router-->>Caller: LLMResult(deployment="ollama-gemma4-12b", ...)
```

If every deployment is only *temporarily* blocked, the router sleeps in bounded increments up to
`max_wait_seconds` (default 120s) before giving up with `AllDeploymentsExhausted`; a deployment that returns
an auth or "not in your tier" error is disabled outright rather than retried.

### 6.3 Scenario: export and train/eval split integrity

```mermaid
flowchart LR
    D[Document] -->|verify.split assigns split by paper| S{split}
    S -->|train ~90%| TrainChunks[Chunks, accepted QA, Negatives, DPOPairs]
    S -->|eval ~10%, held out| EvalChunks[Chunks, accepted QA, Negatives, DPOPairs]
    TrainChunks --> CPTTrain[cpt_train.jsonl]
    TrainChunks --> SFTTrain[sft_train.jsonl]
    TrainChunks --> DPOTrain[dpo_train.jsonl]
    EvalChunks --> CPTEval[cpt_eval.jsonl]
    EvalChunks --> SFTEval[sft_eval.jsonl]
    EvalChunks --> DPOEval[dpo_eval.jsonl]
    EvalChunks -.eval split only, never trained on.-> GoldSet[bg eval build-gold:<br/>gold_v1.jsonl]
```

Every exported row's split is inherited from its source `Document.split`, never assigned independently — this
is what prevents one paper's content from leaking across train/eval, and it is also why the stage 11 gold
benchmark (`eval/build_gold.py`) only draws from `split == "eval"` documents: it is drawing from the same
papers already excluded from training, not a separate mechanism.
