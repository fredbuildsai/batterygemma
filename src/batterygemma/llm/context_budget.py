"""Battery task budgets for local-model context sizing.

The sizing arithmetic itself (`compute_num_ctx`, `recommend_num_ctx`, `apply_global_num_ctx`, ...) is generic and
lives in `llmrouter_free.context_budget`; see its module docstring for why `num_ctx` is computed once, from known
configuration bounds, rather than per call. This module only supplies what is specific to batterygemma: which
tasks exist, their fixed output budgets, which of them bundle several chunks per call, and the real prompt
templates whose overhead gets measured.
"""

from typing import Any

from llmrouter_free import TaskBudget, recommend_num_ctx
from llmrouter_free.context_budget import TokenCounter

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


def task_budgets() -> dict[str, TaskBudget]:
    """Every registered task's prompt scaffolding, taken from the actual prompt-building code so it stays
    correct if a template is edited - never hand-copy a number here."""
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

    # Each template is rendered with every `{placeholder}` replaced by "" - the literal `{...}` JSON-shape
    # examples in some templates would otherwise be mistaken for more format placeholders.
    rendered = {
        "extract_facts": (FACTS_SYS, FACTS_TMPL.format(excerpts="")),
        "extract_claims": (CLAIMS_SYS, CLAIMS_TMPL.format(excerpts="")),
        "generate_qa": (QA_SYS, QA_TMPL.format(text="")),
        "generate_negatives": (NEG_SYS, NEG_TMPL.format(text="")),
        "generate_dpo": (DPO_SYS, DPO_TMPL.format(question="", answer="")),
        "generate_ideation": (IDEATION_SYS, IDEATION_TMPL.format(text="")),
        # The judge has two system prompts of similar size and no per-call template; budget the larger one.
        "judge": (max(QA_JUDGE_SYSTEM_PROMPT, IDEATION_JUDGE_SYSTEM_PROMPT, key=len), ""),
    }
    return {
        task: TaskBudget(system_prompt=system, rendered_template=template, output_tokens=TASK_OUTPUT_TOKENS[task],
                         batchable=task in BATCHABLE_TASKS)
        for task, (system, template) in rendered.items()
    }




def recommend(*, chunk_max_tokens: int, count_tokens: TokenCounter, batch_size: int = 1) -> dict[str, Any]:
    """The full sizing report for batterygemma's tasks (see `llmrouter_free.recommend_num_ctx`).

    `batch_size` is `bg annotate --batch-size` (default 1, i.e. non-batched sizing): only the `BATCHABLE_TASKS`
    bundle that many chunks' text and output budget into one call."""
    return recommend_num_ctx(
        tasks=task_budgets(), chunk_max_tokens=chunk_max_tokens, count_tokens=count_tokens, batch_size=batch_size
    )
