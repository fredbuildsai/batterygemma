# 9. Architecture Decisions

Recorded as lightweight ADRs — decision, context, consequence — in chronological order.

### ADR-001: Move the repository out of iCloud Drive
- **Context**: `import batterygemma` intermittently failed after `uv sync`.
- **Decision**: relocate the repo from `~/Documents/...` (iCloud-synced) to `/Users/fred/code/batterygemma`.
- **Reason**: iCloud sets the BSD `hidden` flag on synced `.venv/.../*.pth` files, and CPython 3.11.15 skips
  hidden `.pth` files when building `sys.path`, silently breaking the editable install.
- **Consequence**: any future clone must live outside an iCloud-synced folder.

### ADR-002: Always pass the full extra set to `uv sync`
- **Context**: `uv sync --extra train` silently uninstalled `docling`/`pytest` installed by a previous
  `--extra parse` call — `uv sync` defines the *complete* extra set for that invocation, it does not add to
  the existing one.
- **Decision**: `train/environment.py::ALL_EXTRAS` lists every extra together; `ensure_unsloth()` always syncs
  all of them at once.
- **Consequence**: adding a new optional extra means adding it to `ALL_EXTRAS`, not just `pyproject.toml`.

### ADR-003: Resumable task wrappers take an `Engine`, never a caller's open `Session`
- **Context**: the original design let a caller's session (with pending writes) call into
  `annotate_chunk_facts`, which internally called `router.complete()` — which opens its *own* session to log
  to `llm_calls`. Two overlapping SQLite writers on the same file deadlock rather than queue; reproduced with
  the literal "database is locked" error, then a full hang once `busy_timeout` alone was added (proving
  genuine deadlock, not simple contention).
- **Decision**: every resumable wrapper (`annotate_chunk_facts`, `annotate_chunk_claims`, `annotate_chunk_qa`,
  `annotate_chunk_false_premise`, `annotate_chunk_ideation`, `annotate_qa_dpo`, `judge_one_qa`,
  `judge_one_ideation`) takes an `Engine` and manages 2-3 short-lived, fully-committed sessions with no
  transaction left open across the LLM call.
- **Consequence**: `register_sqlite_pragmas()` (WAL + `foreign_keys` + `busy_timeout=30000`) was extracted into
  `corpusforge.db.session` (`register_sqlite_pragmas`, shared with llmrouter-free) and is applied by both `get_engine()` and the test fixture — the test fixture previously
  built a bare engine with none of these pragmas, a related bug.

### ADR-004: Two dataset-licensing tracks, default CC/open-only
- **Context**: the plan called for both a public-releasable dataset and a broader one including
  commercially-licensed or non-redistributable sources (e.g. the locally-supplied Handbook of Batteries PDF).
- **Decision**: default export path is CC0/CC-BY/public-domain only (for the public HF release); a second
  path may include everything, for local fine-tuning only, never public export. **[the second path's
  dedicated CLI flag is planned, not yet implemented — export currently filters by license implicitly via
  what documents were accepted at screen time, not via an explicit `--include-commercial` switch]**
- **Consequence**: `bg add-local` accepts any local PDF regardless of license, but license-based export
  filtering must be revisited before that flag is built.

### ADR-005: Router failover taxonomy and shared rate groups
- **Context**: providers return heterogeneous error shapes for "try again later" vs. "never usable" vs.
  "daily quota gone," and several models from one provider (e.g. every Mistral model) draw on one shared
  workspace budget.
- **Decision**: `_classify_error()` normalizes provider exceptions into `auth`/`unavailable` (disable),
  `rate_limited`/`quota` (cooldown, shared per `rate_group`), `error` (short cooldown, per-deployment).
- **Consequence**: adding a new provider only requires getting its error shapes right in this one function.

### ADR-006: Local Ollama as a rate-limit-proof fallback, with `think: false`
- **Context**: at full-corpus scale (10k+ chunks), all free-tier cloud deployments in a route were observed
  hitting rate limits simultaneously, exhausting the router's retry budget (~30-70% task failure in bursts on
  `annotate claims`/`annotate facts` — confirmed in `data/logs/full-run-20260914-073925.log`).
- **Decision**: added `ollama-gemma4-12b`, `ollama-gemma4-e4b`, `ollama-gemma3-4b` (via `ollama_chat/` and
  `http://localhost:11434`, no API key) to the end of every route, tried only after cloud options are
  exhausted.
- **Complication found and fixed**: these are Ollama "thinking" models. With thinking on, they either return
  empty `content` (small `max_tokens` — the whole budget goes to `reasoning_content`) or take 150-180s+ per
  call (larger budget) — both broke the pipeline differently. `extra_body: {think: false}` fixed both
  (confirmed live: ~30-70s/call, populated `content`).
- **Consequence**: local inference is slower per-call than an unthrottled cloud call, but has no rate limit;
  it is deliberately ordered last so cloud is preferred when available.

### ADR-007: Judge-route family exclusion survives local fallback
- **Context**: `ollama-gemma4-e4b` shares a model family (`gemma4`) with `ollama-gemma4-12b`, which can also
  serve as the *generator's* fallback — naively adding both to the judge route risked the same-model-grades-
  itself problem exactly when both routes fall back to local models simultaneously.
- **Decision**: both share `family: gemma4` in config; the router's existing `exclude_families` mechanism
  correctly skips `ollama-gemma4-e4b` as a judge when the generator used `ollama-gemma4-12b`, falling through
  to `ollama-gemma3-4b` (a genuinely different family) instead.
- **Consequence**: no special-case code was needed — the existing exclusion mechanism generalized correctly,
  confirmed by reasoning through the config rather than by asserting it.

### ADR-008: Fetch and parse network errors must not crash the whole stage
- **Context**: `_try_pdf_url()` only caught `BlockedByBotProtection` and `httpx.HTTPStatusError`; a genuine
  network error (`httpx.ReadTimeout`, confirmed live) propagated uncaught and crashed the entire `bg fetch`
  loop after only 4 of 452 documents, rather than failing just that one document.
- **Decision**: catch `httpx.TransportError` at every network call site in `corpusforge.fetch`, returning/recording a
  `network_error` outcome and continuing to the next document/URL.
- **Consequence**: `bg fetch` is now robust to transient connectivity issues mid-run; regression tests added
  (`test_network_error_on_pdf_url_does_not_crash_and_falls_back_to_unpaywall`, and the no-fallback case).

### ADR-009: MLX model selection through three real failures
- **Context**: Unsloth's documentation is CUDA-first; following its obvious recipe on Apple Silicon degrades
  silently into memory thrashing rather than a clean error. Three model repos were tried, live, in order:
  `unsloth/gemma-4-E2B-it` (full precision) hung reproducibly 3 times at the identical point in loading — `top`
  showed state `stuck`, ~19GB combined resident+compressed memory on a 16GB machine, confirmed independent of
  `max_seq_length` (2048 vs 1024 made no difference, proving the spike is load-time, not sequence-length).
  `unsloth/gemma-4-E2B-it-GGUF` looked like the fix but, per direct inspection of the installed `unsloth_zoo`
  source, this Unsloth version's MLX backend only *exports* to GGUF — no load path exists for training.
- **Decision**: use `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (Unsloth's own MLX-native, pre-quantized checkpoint).
  Confirmed live: no runtime quantization pass at load (`'...' is already quantized` in the log), 574MB
  resident after LoRA setup vs. multi-gigabyte for the full-precision path at the same point.
- **Related finding**: on this backend, `FastLanguageModel` (Unsloth's CUDA-oriented API name) is not a
  different code path — `unsloth/__init__.py` aliases `FastLanguageModel = FastModel = FastTextModel`, all
  delegating to the same `FastMLXModel`. There is no CUDA/bitsandbytes path to fall into on Apple Silicon.
- **Consequence**: `configs/train_cpt.yaml`/`train_sft.yaml`'s `model_name` carries this entire finding as an
  inline comment, so nobody re-discovers it by re-running the same three failures. Full narrative and a
  diagram: [docs/training.md](../training.md#the-model-matters-more-than-the-code).

### ADR-010: License gate — the hard block is at publish, not at GGUF export
- **Context**: two commercial, copyrighted books had been added via `bg add-local` (which accepts any local
  PDF for personal research use) and had already contributed thousands of rows to the exported training data
  — invisible until the attribution manifest (below) was built and read. An earlier version of this gate
  refused to even *produce* a GGUF file from an adapter trained on such material; explicit user feedback
  mid-build corrected this: local artifacts for personal use are legitimate, only *redistribution* isn't.
- **Decision**: three checkpoints warn and offer to fix it (`bg export`, `bg train cpt|sft` before training
  starts, and an informational warning in `bg train export-gguf`); exactly one, `bg train push-to-hub`,
  refuses outright. Every export and training run writes `LICENSE_STATUS.json` recording `shareable` plus
  enough `training_provenance` (stage, config, dataset path) to retrain clean automatically when blocked.
  Classification reuses `corpusforge.screen.license::evaluate_license` against the same allow/flag lists ingestion
  uses, so a document's status is judged identically at every checkpoint.
- **Consequence**: an adapter with a missing `LICENSE_STATUS.json` (e.g. one trained before this feature
  existed) is treated as unverified, not as safe-by-default — `push-to-hub` refuses it the same as a
  known-restricted one, until retrained under the current gate. Full walkthrough with the real incident:
  [docs/licensing.md](../licensing.md).
- **Related fix found in the same pass**: `bg blacklist`'s own docstring says it never touches already-
  generated content — but the export functions never checked `Document.blacklisted` at all, so a document
  blacklisted for *any* reason (not just this) still leaked its content into every export. Fixed by the same
  change: every export now excludes blacklisted documents unconditionally.

### ADR-011: Self-contained per-run training folders
- **Context**: `outputs/cpt/`'s only record of what produced it was "whatever `configs/train_cpt.yaml`
  happened to contain at the time" — a file that gets edited for the next run. There was no way to know, six
  months later, what config or exact dataset a given adapter actually came from.
- **Decision**: every `bg train cpt|sft` run copies its exact resolved config (`config.yaml`) and exact
  training data (`dataset.jsonl`, `dataset.attribution.json`) into its own output directory, and
  `LICENSE_STATUS.json`'s `training_provenance.dataset` is updated to point at this local copy from that
  point on, not the shared `data/export/` tree.
- **Consequence**: a run folder is reproducible and portable on its own — move it to another machine and
  everything needed to explain or redo that run travels with it. As a side effect, the license gate's
  retrain-clean flow also became self-contained: its filtered dataset now lands inside the same run folder
  rather than polluting the shared export tree. Diagram: [docs/training.md](../training.md#self-contained-run-folders).

### ADR-013: Split into three packages
- **Context**: the repository held three different things - a quota-aware LLM failover router, a domain-agnostic
  paper-to-corpus pipeline, and the battery-specific annotation/generation/training work. Other domains were
  expected, and the first two had proven general in use.
- **Options**: (a) keep the monolith and copy it per domain; (b) one shared package with everything generic; (c) two
  packages split along the existing seams, chain `llmrouter-free` <- `corpusforge` <- `batterygemma`.
- **Decision**: (c), three public repositories. Boundary "generic core + mechanisms": corpusforge owns the pipeline
  stages *and* the generic batched-chunk runner; batterygemma supplies prompts, schemas, row builders and tables.
  Installed from GitHub tags (PyPI later); `[tool.uv.sources]` editable overrides for cross-repository development.
- **Consequences**: no domain text in the packages (the model-card tags, User-Agent, tokenizer id and config templates
  became parameters); the `bg` CLI registers corpusforge's commands so `scripts/*.sh` and existing habits keep working;
  three suites (about 500 tests in total) instead of one; a change spanning layers is a coordinated tag bump.
  Verified against real data: after `bg db adopt-split` every table count matched and all six exported JSONL files and
  their attribution manifests were byte-identical to the pre-split export.

### ADR-014: Independent schemas and migration histories; `bg db adopt-split`
- **Context**: one database must hold corpusforge's tables, batterygemma's tables and the router's `llm_calls` ledger,
  each evolving at its own pace, and the existing production-like database (1,010 documents, 9,030 chunks, ~25k facts,
  a large `llm_calls` cache) must survive the split untouched.
- **Decision**: each package has its own declarative base and Alembic history *inside the package* (a repo-root
  `alembic/` would not exist in a git install) with a private version table (`corpusforge_alembic_version`,
  `batterygemma_alembic_version`); the ledger is created idempotently by the router. Battery tables reference corpus
  rows by string id only - no cross-schema foreign keys (enforced by a test). `bg db adopt-split` verifies that every
  expected table and column already exists, takes a consistent backup (SQLite online-backup API), records both
  baselines and drops the old `alembic_version` table - no DDL, no row touched.
- **Consequence**: a fresh install and an adopted legacy database end up in the identical state; refused (with
  advice) if the schema does not match, rather than guessing.

### ADR-015: Annotation stages are `ChunkTaskSpec`s run by one shared runner
- **Context**: `annotate/facts.py`, `annotate/claim_pairs.py` and the `bg annotate` loop each carried their own copy of
  task bookkeeping, dropped-chunk retry, back-off and concurrency.
- **Decision**: extract the mechanism into `corpusforge.runner` (`run_batch`, `run_backlog`); a stage supplies only its
  prompt builder, batch-shaped schema and `persist_result`. One code path for 1 or N chunks (the earlier user
  requirement: "the same mechanism for one or several chunks, not two separate routes").
- **Consequence**: new stages are ~40 lines; the resilience behavior is tested once, in corpusforge. `generate/*` still
  use their own per-chunk functions and are the next candidates to move onto the runner.

### ADR-012: `PROJECT_ROOT` is the working directory, not the package install location
- **Context**: `PROJECT_ROOT` was `Path(__file__).resolve().parents[2]` — the batterygemma *package's own*
  source location. Every config, the database, `data/`, `outputs/` were pinned to wherever the package
  happened to be installed, regardless of the shell's current directory. A `bg init` command intended to
  scaffold "a project in the current folder" would, under this definition, always re-touch the one repo
  checkout the code lives in — never a different folder.
- **Decision**: split into two constants (`settings.py`). `PACKAGE_ROOT` keeps the old meaning, used only for
  packaged assets that ship with the code and never move (migration scripts - since ADR-014 shipped inside each package -
  `.env.example`, the `uv sync` working directory). `PROJECT_ROOT` becomes `Path.cwd()`, used for everything
  project-specific (`configs/`, `data/`, `outputs/`, `.env`) — the same model `git`/`npm`/`cargo init` use.
- **Consequence**: `bg init`, run in any directory, scaffolds config YAMLs (from bundled templates in
  `src/batterygemma/templates/`), `.env`, data directories, and a migrated database there — verified live in
  an empty `/tmp` directory, completely independent of this repo, followed by a working `bg export` against
  it. Existing usage (always running `bg` from within this checkout) is unaffected, since CWD and package
  location happen to coincide there.
