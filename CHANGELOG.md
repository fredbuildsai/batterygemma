# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] - branch `split-packages`

### Changed
- **Split into three packages.** The LLM router is now [`llmrouter-free`](https://github.com/fredbuildsai/llmrouter-free)
  and the literature pipeline is now [`corpusforge`](https://github.com/fredbuildsai/corpusforge); batterygemma
  depends on both (installed from GitHub tags) and keeps only the battery-specific code. See ADR-013.
- Battery tables live on their own SQLAlchemy base with an in-package Alembic history
  (`batterygemma_alembic_version`); no foreign keys into corpusforge's tables. The repo-root `alembic/` was removed.
- `bg annotate facts|claims` run through `corpusforge.runner` (`ChunkTaskSpec`, `run_backlog`); behavior, flags and
  resumability are unchanged.
- `bg` registers corpusforge's pipeline commands and `logs` group unchanged.
- CPT export delegates to `corpusforge.export_cpt`, with the ontology rows passed as `extra_rows`; exports are
  byte-identical to the pre-split output.
- The Hugging Face model card's domain text moved to `export/model_card.py`.
- `docs/architecture.md` became `docs/architecture/` (arc42, one file per section) and gained ADR-013..015.

### Added
- `bg db adopt-split`: adopt a database created before the split without changing its data (verifies schema, backs up).
- `scripts/setup.sh`: one-command setup (uv, venv, both packages from GitHub, `bg init`), with `--dev`, `--with-train`,
  `--no-parse`, `--dry-run`.
- GitHub Actions CI (pytest + ruff + shellcheck on Python 3.11/3.12); tests for the docs, the setup script, the
  three-schema migrations and an end-to-end subprocess test of the package wiring.

### Removed
- `uv.lock` from version control (it pins local sibling paths used for development).
