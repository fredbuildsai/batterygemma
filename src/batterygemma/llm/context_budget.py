"""Compute the local Ollama context window (`num_ctx`) once, from known configuration bounds.

Chunking has no LLM in the loop and is hard-capped by `configs/generation.yaml`'s `chunking.max_tokens`, and
every call site's output budget (`max_tokens` passed to `router.complete()`) is a fixed constant in code - so
the true worst-case prompt+output size for any given task is knowable in advance, not something that needs
measuring per call. This matters because Ollama reloads the model whenever `num_ctx` changes between
consecutive requests (confirmed live: ~6-7s, not a full cold start, but real cost paid on every single call if
`num_ctx` is recomputed per prompt) - so the goal here is one static value good for every task, computed once
at startup, rather than a per-call estimate that would cause an internal reload each time the estimate moves
between buckets.

The same "prompt + output would need N tokens" arithmetic is not Ollama-specific: it is also the right way to
sanity-check that a cloud deployment's own context limit isn't exceeded (`Deployment` doesn't currently record
one, but `fits_within` is here for when it does), rather than discovering that as a truncated response.
"""

from collections.abc import Callable
from typing import Any

TokenCounter = Callable[[str], int]

# Every `router.complete(..., max_tokens=...)` call site's output budget, kept in sync by hand - see
# annotate/facts.py, annotate/claim_pairs.py, generate/{qa,negatives,dpo,ideation}.py, verify/judge.py.
TASK_OUTPUT_TOKENS: dict[str, int] = {
    "extract_facts": 3000,
    "extract_claims": 2000,
    "generate_qa": 3000,
    "generate_negatives": 1500,
    "generate_dpo": 1500,
    "generate_ideation": 3000,
    "judge": 800,
}

# Tasks whose prompt bundles `batch_size` chunks in one call (see annotate/facts.py, annotate/claim_pairs.py)
# instead of exactly one - their chunk-text and output budgets both scale with the batch size actually
# passed to `bg annotate --batch-size`; every other task is always exactly one chunk/item per call.
BATCHABLE_TASKS = {"extract_facts", "extract_claims"}

DEFAULT_BUFFER = 0.20  # the requested safety margin on top of the computed worst case
CONTEXT_BUCKET = 1024  # round up to a clean number; irrelevant to correctness, just tidier config/logs


def round_up_to_bucket(value: int, bucket: int = CONTEXT_BUCKET) -> int:
    return -(-value // bucket) * bucket


def measure_template_overhead(system_prompt: str, user_template: str, count_tokens: TokenCounter) -> int:
    """Token cost of a task's fixed prompt scaffolding alone, i.e. with the chunk text/variable content
    removed. `user_template` should be the raw template string with its `{text}`-style placeholder already
    substituted with an empty string, so only the literal instructions are counted."""
    return count_tokens(system_prompt) + count_tokens(user_template)


def compute_num_ctx(
    *, chunk_max_tokens: int, template_overhead_tokens: int, output_tokens: int, buffer: float = DEFAULT_BUFFER,
    bucket: int = CONTEXT_BUCKET,
) -> int:
    """The static context window one task actually needs, worst case, with `buffer` extra headroom."""
    worst_case = chunk_max_tokens + template_overhead_tokens + output_tokens
    return round_up_to_bucket(round(worst_case * (1 + buffer)), bucket)


def global_num_ctx(
    *, chunk_max_tokens: int, template_overhead_tokens: int, tasks: dict[str, int] = TASK_OUTPUT_TOKENS,
    buffer: float = DEFAULT_BUFFER, bucket: int = CONTEXT_BUCKET,
) -> int:
    """One context window sized for the largest task, so every local Ollama deployment can share a single
    `num_ctx` value across every route it serves - the point being that it then never has to change between
    calls, however the pipeline moves between stages, and so never triggers a reload for that reason."""
    return compute_num_ctx(
        chunk_max_tokens=chunk_max_tokens, template_overhead_tokens=template_overhead_tokens,
        output_tokens=max(tasks.values()), buffer=buffer, bucket=bucket,
    )


def worst_case_batch_size(batch_size: int, tasks: dict[str, int] = TASK_OUTPUT_TOKENS) -> dict[str, int]:
    """Per-task chunk multiplier: `batch_size` for a batchable extract task, 1 for everything else - a
    single-chunk `bg annotate --batch-size 1` run degenerates back to the original, non-batched sizing."""
    return {task: batch_size if task in BATCHABLE_TASKS else 1 for task in tasks}


def fits_within(context_limit: int, *, chunk_max_tokens: int, template_overhead_tokens: int, output_tokens: int) -> bool:
    """Whether a deployment's real context limit (e.g. a cloud model's documented window) comfortably covers
    the worst case for one task - useful to validate a cloud deployment's configured limit rather than
    discover a silent truncation in production."""
    return context_limit >= chunk_max_tokens + template_overhead_tokens + output_tokens


def measured_overheads(count_tokens: TokenCounter) -> dict[str, int]:
    """Measure every registered task's real template overhead from the actual prompt-building code, so this
    stays correct if a prompt template is edited - never hand-copy a number here."""
    from batterygemma.annotate.claim_pairs import SYSTEM_PROMPT as CLAIMS_SYS
    from batterygemma.annotate.claim_pairs import USER_TEMPLATE as CLAIMS_TMPL
    from batterygemma.annotate.facts import SYSTEM_PROMPT as FACTS_SYS
    from batterygemma.annotate.facts import USER_TEMPLATE as FACTS_TMPL
    from batterygemma.generate.dpo import SYSTEM_PROMPT as DPO_SYS
    from batterygemma.generate.dpo import USER_TEMPLATE as DPO_TMPL
    from batterygemma.generate.ideation import SYSTEM_PROMPT as IDEATION_SYS
    from batterygemma.generate.ideation import USER_TEMPLATE as IDEATION_TMPL
    from batterygemma.generate.negatives import FALSE_PREMISE_SYSTEM_PROMPT as NEG_SYS
    from batterygemma.generate.negatives import FALSE_PREMISE_USER_TEMPLATE as NEG_TMPL
    from batterygemma.generate.qa import SYSTEM_PROMPT as QA_SYS
    from batterygemma.generate.qa import USER_TEMPLATE as QA_TMPL
    from batterygemma.verify.judge import IDEATION_JUDGE_SYSTEM_PROMPT, QA_JUDGE_SYSTEM_PROMPT

    def overhead_text(system: str, rendered_template: str) -> int:
        """`rendered_template` must already have every `{placeholder}` substituted - the literal `{...}`
        JSON-shape examples in some templates would otherwise be mistaken for more format placeholders."""
        return measure_template_overhead(system, rendered_template, count_tokens)

    return {
        "extract_facts": overhead_text(FACTS_SYS, FACTS_TMPL.format(excerpts="")),
        "extract_claims": overhead_text(CLAIMS_SYS, CLAIMS_TMPL.format(excerpts="")),
        "generate_qa": overhead_text(QA_SYS, QA_TMPL.format(text="")),
        "generate_negatives": overhead_text(NEG_SYS, NEG_TMPL.format(text="")),
        "generate_dpo": overhead_text(DPO_SYS, DPO_TMPL.format(question="", answer="")),
        "generate_ideation": overhead_text(IDEATION_SYS, IDEATION_TMPL.format(text="")),
        "judge": max(count_tokens(QA_JUDGE_SYSTEM_PROMPT), count_tokens(IDEATION_JUDGE_SYSTEM_PROMPT)),
    }


def recommend_num_ctx(*, chunk_max_tokens: int, count_tokens: TokenCounter, batch_size: int = 1) -> dict[str, Any]:
    """The full sizing report: measured overhead and recommended `num_ctx` per task, plus the single global
    value (sized for the largest task) that every local deployment should actually use in practice, so
    `num_ctx` never has to change between calls regardless of which stage the pipeline is currently in.

    `batch_size` is `bg annotate --batch-size` (default 1, i.e. today's non-batched sizing): only
    `extract_facts`/`extract_claims` bundle that many chunks' text and output budget into one call (see
    `BATCHABLE_TASKS`) - every other task is always exactly one chunk/item per call regardless of this value.
    """
    overheads = measured_overheads(count_tokens)
    multipliers = worst_case_batch_size(batch_size, overheads)
    per_task = {
        task: compute_num_ctx(chunk_max_tokens=chunk_max_tokens * multipliers[task], template_overhead_tokens=overhead,
                              output_tokens=TASK_OUTPUT_TOKENS[task] * multipliers[task])
        for task, overhead in overheads.items()
    }
    return {
        "chunk_max_tokens": chunk_max_tokens, "batch_size": batch_size, "overheads": overheads,
        "per_task_num_ctx": per_task, "global_num_ctx": max(per_task.values()),
    }


def apply_global_num_ctx(config: dict[str, Any], num_ctx: int, *, model_prefix: str = "ollama_chat/") -> dict[str, Any]:
    """Return a copy of a router config with `num_ctx` set on every deployment whose model uses `model_prefix`
    (i.e. every local Ollama deployment). This is the dedicated pipeline step: `bg llm context-budget` reports
    the numbers, and `cli.build_router()` calls this once at startup, before any deployment is used - so
    `num_ctx` is fixed for the life of the process rather than guessed once in a config comment or recomputed
    per call (see the module docstring for why per-call recomputation is actively worse: it forces an Ollama
    reload whenever the estimate moves between buckets)."""
    patched = dict(config)
    patched["deployments"] = [
        {**d, "extra_body": {**d.get("extra_body", {}), "options": {**d.get("extra_body", {}).get("options", {}), "num_ctx": num_ctx}}}
        if d.get("model", "").startswith(model_prefix) else d
        for d in config["deployments"]
    ]
    return patched
