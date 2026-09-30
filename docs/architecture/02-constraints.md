# 2. Constraints

### 2.1 Technical constraints

| Constraint | Detail |
|---|---|
| Local hardware | Apple Silicon (M1, 16GB unified memory). No CUDA. Training runs through Unsloth's MLX backend (`mlx-lm`), not the standard CUDA/bitsandbytes path. |
| Model | Gemma 4 E2B, specifically `unsloth/gemma-4-E2B-it-UD-MLX-4bit` — Unsloth's own pre-quantized MLX checkpoint. This exact repo was chosen only after two other variants reproducibly failed on this hardware; see [ADR-009](09-architecture-decisions.md#adr-009-mlx-model-selection-through-three-real-failures) and [docs/training.md](../training.md#the-model-matters-more-than-the-code). A small multimodal model, loaded via `text_only: true` (a config key, not a hardcoded constant — see ADR-009) for the plain-text path. |
| Budget | Free-tier LLM APIs only by default (`BG_ALLOW_PAID=false`); a small paid budget (`BG_MAX_USD_PER_DAY`) is available but unused so far. |
| Storage | SQLite today; the ORM uses only portable SQLAlchemy types so the same schema can later point at Postgres/Supabase via `BG_DATABASE_URL` with no code change. |
| Licensing | Default dataset export must be restricted to CC0/CC-BY/public-domain (with CC-BY-SA flagged), because the intent is a public Hugging Face release. `bg add-local` still accepts any local PDF regardless of license for personal research use, but a document's license is enforced at three further checkpoints (export, train, and a hard block at publish) — see [ADR-010](09-architecture-decisions.md#adr-010-license-gate--the-hard-block-is-at-publish-not-at-gguf-export) and [docs/licensing.md](../licensing.md). |
| Filesystem | The repository must live outside iCloud Drive sync (see ADR-001) — this is an environment constraint discovered the hard way, not a design choice. |
| Portability | `bg` resolves `configs/`/`data/`/`outputs/`/`.env` relative to the current working directory (`PROJECT_ROOT = Path.cwd()`), not to wherever the package is installed — `bg init` scaffolds a fresh project in any folder. See [ADR-012](09-architecture-decisions.md#adr-012-project_root-is-the-working-directory-not-the-package-install-location). |

### 2.2 Organizational constraints

- Single-developer project. CI (GitHub Actions: pytest + ruff on Python 3.11/3.12) runs on all three repositories; each
  suite must be green before every commit.
- The dependency chain is one-way: `llmrouter-free` <- `corpusforge` <- `batterygemma`. Neither package may import
  batterygemma or contain battery-specific text (enforced by tests in the packages).
- Both packages are installed from GitHub tags (pinned in `pyproject.toml`) until they are published on PyPI.
- No dedicated ML infra — training, inference, and the whole data pipeline run on the same laptop.
