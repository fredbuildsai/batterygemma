# Appendices

## Appendix A — CLI Reference

All commands are namespaced under `bg` (installed via `pyproject.toml`'s `[project.scripts]`). `discover`, `screen`, `fetch`, `images`, `add-local`, `parse`, `fetch-failures`, `pipeline-failures`, `blacklist*` and `logs` are [corpusforge](https://github.com/fredbuildsai/corpusforge)'s commands, registered on `bg` unchanged (they are also available as `corpusforge <command>` for a project with no battery code); their help texts say `corpusforge`.

| Command | Purpose |
|---|---|
| `bg init [--force]` | Scaffold a fresh project in the current directory: config YAMLs (from bundled templates), `.env`, data/output directories, and a migrated database. Idempotent. See [ADR-012](09-architecture-decisions.md#adr-012-project_root-is-the-working-directory-not-the-package-install-location). |
| `bg db init` | Apply the migrations of all three schemas (corpusforge, batterygemma, router ledger) - a subset of `bg init`. |
| `bg db adopt-split [--yes]` | One-time: adopt a database created before the split (verify schema, back up, record baselines; no DDL). See [ADR-014](09-architecture-decisions.md#adr-014-independent-schemas-and-migration-histories-bg-db-adopt-split). |
| `bg stats` | Row counts per table; breakdowns by document status/source/license. |
| `bg llm status` | Per-deployment quota usage, cooldown state, enabled flag. |
| `bg llm test` | Ad hoc single-call test of the router. |
| `bg llm context-budget [--batch-size N]` | Show the local-model context sizing (`num_ctx`) for every battery task. |
| `bg discover` | Query configured sources, dedupe cross-source, store as `discovered`. |
| `bg screen` | License gate + relevance scoring → `accepted`/`rejected`/`borderline`. |
| `bg fetch [--limit N]` | Download full text for up to `N` accepted documents (default 50). |
| `bg add-local <path>...` | Register local PDF(s) directly as fetched documents (bypasses the license screen — see [docs/licensing.md](../licensing.md)). |
| `bg parse [--limit N] [--reparse]` | Parse + chunk fetched documents (default limit 100; `--reparse` also re-chunks already-`chunked` docs). |
| `bg annotate facts \| claims [--limit N] [--force]` | Stage 7 extraction. |
| `bg generate qa \| negatives \| dpo \| ideation [--limit N] [--force]` | Stage 8 generation. |
| `bg judge qa \| ideation [--limit N] [--force]` | Stage 9 judging. |
| `bg export --version vX.Y [--output DIR] [--yes]` | Assigns splits, dedupes accepted QA, writes JSONL + attribution manifests; warns and offers to blacklist+re-export if any source isn't open-licensed. |
| `bg logs tail \| query \| stats \| clear` | Query the structured application log (`data/logs.db`). |
| `bg blacklist [--threshold N]` / `bg blacklist-list` / `bg blacklist-remove <doc_id>` | Permanently exclude documents from every future export (repeat pipeline failures, or a license-gate decision). |
| `bg train status` | Installs Unsloth if missing; reports package/accelerator versions. |
| `bg train cpt \| sft <dataset.jsonl> [--output DIR] [--config NAME] [--gguf] [--from-adapter DIR] [--yes]` | Runs LoRA training; see [docs/training.md](../training.md). |
| `bg train export-gguf <adapter_dir> [--output DIR] [--config NAME] [--quantization TYPE]` | Merge + quantize a trained adapter to GGUF. Never blocked by licensing. |
| `bg train push-to-hub <gguf_dir> --repo-id OWNER/NAME [--private/--no-private] [--yes]` | The license-gated Hugging Face publish step. |
| `bg eval build-gold [--output PATH] [--closed N] [--open N] [--negative N] [--ideation N]` | Builds the held-out gold benchmark. |
| `bg eval run <gold.jsonl> --model NAME_OR_PATH [--output PATH] [--limit N]` | Generates + scores a model against the benchmark. |

## Appendix B — Configuration files (`configs/`)

Every file below is generated from a bundled template by `bg init` if missing (`src/batterygemma/templates/`).

| File | Purpose |
|---|---|
| `sources.yaml` | Discovery query terms, license allow/flag lists. |
| `llm_routes.yaml` | Deployment definitions (provider, model, rate limits, family, tags) and ordered route chains per task; cooldown durations; max attempts per call. |
| `taxonomy.yaml` | Controlled vocabularies (components, fact categories, question types, etc.) mirrored as pydantic `Literal`s in `llm/schemas.py`. |
| `generation.yaml` | Chunking parameters, per-chunk/per-paper/per-cluster generation quotas, balance ratios, judge score floors, audit sample sizes, gold-eval-set sizing, system prompt. |
| `train_cpt.yaml` / `train_sft.yaml` | Model name, `text_only`, LoRA rank/alpha, training hyperparameters matching `MLXTrainingConfig` field names. See [docs/training.md](../training.md). |

## Appendix C — Test suite shape

The suite is split with the code: `llmrouter-free` (router, validation, context arithmetic, ledger/cache, config, CLI,
docs), `corpusforge` (sources, screening, fetch, parse, images, runner, export, migrations, CLI, docs) and this repository
(annotate/generate/judge/ontology/export/train/eval, the three-schema migrations, `bg db adopt-split`, the `bg` CLI
including an end-to-end subprocess test that catches wiring mistakes between the packages, the setup script and the docs).
Each test is anchored on a real, previously-encountered bug rather than only happy-path coverage (see §8.8). Run with
`uv run pytest`; the docs are tested too (`tests/test_docs.py`: all twelve arc42 sections exist, links resolve, cited tests
exist). The suite must be green before every commit.
