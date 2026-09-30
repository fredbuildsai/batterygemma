# 1. Introduction and Goals

> **Status.** Reflects the codebase on branch `split-packages`: BatteryGemma is now the *domain* project that sits on
> two reusable packages it was extracted from - [`llmrouter-free`](https://github.com/fredbuildsai/llmrouter-free)
> (the quota-aware LLM failover router) and [`corpusforge`](https://github.com/fredbuildsai/corpusforge) (the
> domain-agnostic literature-to-corpus pipeline). See [ADR-013](09-architecture-decisions.md#adr-013-split-into-three-packages)
> and the [architecture index](README.md). Where this documentation names a module that moved, the package is stated.

### 1.1 Requirements overview

BatteryGemma turns openly-licensed lithium-ion battery literature into fine-tuning datasets, and fine-tunes
Gemma 4 E2B (via Unsloth) to act as an expert battery materials scientist: able to answer technical questions
grounded in the literature, correct false premises, and propose testable research ideas.

Concretely, the system must:

1. Discover and collect open-access (and, optionally, licensed-but-non-redistributable) battery-science
   documents from multiple sources.
2. Extract clean, section-aware text chunks with full provenance (document, license, section).
3. Derive structured annotations (facts, comparisons, contradiction/paraphrase claim pairs) from those chunks
   using teacher LLMs.
4. Generate training data in multiple formats — continued-pretraining text, instruction Q&A, multi-turn,
   negatives (false-premise/contradiction/insufficient-evidence), DPO preference pairs, and grounded ideation
   — at a volume large enough to be statistically meaningful, with an enforced positive/negative balance.
5. Verify generated data with an independent judge model, deduplicate it, and split it by source paper into
   train/eval so no paper leaks across the split.
6. Export Unsloth-ready JSONL and fine-tune Gemma 4 E2B locally.
7. Evaluate the fine-tuned model against a held-out gold benchmark with real accuracy metrics, not just
   training loss.

### 1.2 Quality goals

| # | Quality goal | Motivation / scenario |
|---|---|---|
| 1 | **Correctness of licensing** | Every document's license is evaluated before its text can reach an export; the default HF-releasable export must never include a paper whose license disallows redistribution. |
| 2 | **Statistical adequacy of generated data** | Early manual review found "4 or 5 examples" is not remotely enough to draw conclusions; every dataset type needs both a generation quota (`configs/generation.yaml`) and a human-audit sample size large enough for a defensible confidence interval (≥385 items, ±5% at 95%). |
| 3 | **Resumability / no lost work** | The pipeline processes tens of thousands of chunks against free-tier, rate-limited APIs; any stage must be interruptible and resumable without reprocessing or duplicating completed work. |
| 4 | **No silent data leakage across train/eval** | Splits are assigned per source document, not per generated row, so a paper's chunks, facts, Q&A, and eval-benchmark items never appear on both sides. |
| 5 | **No bypassing of bot protection** | A hard policy: fetch failures caused by Cloudflare/Radware/WAF challenges are recorded and left alone, never circumvented. |
| 6 | **Reliability of the teacher-LLM layer** | At full-corpus scale, free-tier cloud rate limits are real (confirmed: up to ~70% task failure in bursts); the router must fail over automatically, including to a local model, rather than stall the pipeline. |
| 7 | **Honesty of "done"** | Every stage's correctness is checked with unit tests against real bugs (not just written and assumed), and, where feasible, with a live run against real data before being reported as working. |

### 1.3 Stakeholders

| Role | Concern |
|---|---|
| Project owner (sole user/operator) | Wants a working pipeline they can run end-to-end on their own machine (Apple Silicon, 16GB) without paid infrastructure, and a fine-tuned model that is actually useful. |
| Future maintainer (could be the same person, later) | Needs to understand *why* the code is shaped as it is — several parts exist specifically because an earlier, simpler version had a real bug. |
