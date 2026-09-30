# 12. Glossary

| Term | Meaning |
|---|---|
| CPT | Continued pretraining — plain-text fine-tuning with no chat template or response masking. |
| SFT | Supervised fine-tuning — instruction/response pairs, chat-templated. |
| DPO | Direct Preference Optimization — trains on (prompt, chosen, rejected) triples. |
| Silver / gold tier | `silver` = LLM-extracted, unvalidated by a human; `gold` = human-verified; `database` = sourced directly from a structured database (e.g. Materials Project) rather than extracted. |
| Grounding | How well a generated statement is supported by its cited source chunk; measured cheaply via word-overlap ratio (`corpusforge.annotate.grounding`), and more rigorously via LLM judge scoring for `QA`/`Ideation`. |
| Rate group | A set of deployments sharing one provider's rate/quota budget (e.g. every Mistral model shares `mistral-workspace`); a 429 on one cools down the whole group. |
| Relaxed family | A judge result whose model shares a family with the generator, used only as a fallback when no independent family was available; recorded explicitly rather than hidden. |
| GenTask | The resumable-job-queue row (`gen_tasks` table) backing every annotate/generate/judge operation; uniqueness on `key` is what makes re-running a stage idempotent. |
| Split | `train` or `eval`, assigned once per `Document` (never per generated row) so no paper's content appears on both sides of the split. |
| Attribution manifest | A `*.attribution.json` file next to each exported JSONL, mapping every source document (by DOI, or `doc_id` when no DOI exists) to the row indices derived from it — the mechanism behind the whole license gate. See [docs/licensing.md](../licensing.md). |
| Shareable | Whether an export or trained adapter is safe to publish: every contributing source document's license evaluates to `ALLOWED` (not `FLAGGED`, `REJECTED`, or `UNKNOWN`) via `corpusforge.screen.license::evaluate_license`. Recorded in `LICENSE_STATUS.json`. |
| GGUF | A quantized single-file model format (llama.cpp/Ollama/LM Studio compatible), produced by merging a LoRA adapter into base weights and re-quantizing — see [docs/training.md](../training.md#gguf-export). |
| llmrouter-free | The extracted LLM failover router package (`import llmrouter_free`). |
| corpusforge | The extracted domain-agnostic literature-to-corpus pipeline package. |
| ChunkTaskSpec | corpusforge's description of one chunk-level LLM stage (prompt, batch schema, persistence); `FACTS_SPEC` and `CLAIMS_SPEC` are batterygemma's. |
| Adopting (`bg db adopt-split`) | Recording the per-package migration baselines on a database created before the split, without changing it. |
| PACKAGE_ROOT / PROJECT_ROOT | Two distinct path roots (`settings.py`, ADR-012): PACKAGE_ROOT is where batterygemma itself is installed (fixed); PROJECT_ROOT is `Path.cwd()` — the project `bg` currently operates on. |
