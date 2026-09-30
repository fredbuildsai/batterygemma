"""batterygemma's task registry for local-model context sizing (the arithmetic itself is tested in llmrouter-free)."""

from llmrouter_free import TaskBudget

from batterygemma.annotate.claim_pairs import CLAIMS_SPEC
from batterygemma.annotate.facts import FACTS_SPEC
from batterygemma.llm.context_budget import BATCHABLE_TASKS, TASK_OUTPUT_TOKENS, recommend, task_budgets


def word_counter(text: str) -> int:
    return len(text.split())


def test_every_registered_task_has_a_measured_budget_from_the_real_prompts():
    budgets = task_budgets()
    assert set(budgets) == set(TASK_OUTPUT_TOKENS)
    assert all(isinstance(b, TaskBudget) and b.system_prompt for b in budgets.values())
    assert {name for name, b in budgets.items() if b.batchable} == BATCHABLE_TASKS


def test_output_budgets_in_the_registry_match_the_stage_specs():
    """The registry and the specs are kept in sync by hand; this fails if one is edited without the other."""
    assert FACTS_SPEC.output_tokens_per_chunk == TASK_OUTPUT_TOKENS["extract_facts"]
    assert CLAIMS_SPEC.output_tokens_per_chunk == TASK_OUTPUT_TOKENS["extract_claims"]
    assert FACTS_SPEC.task_type in TASK_OUTPUT_TOKENS and CLAIMS_SPEC.task_type in TASK_OUTPUT_TOKENS


def test_recommend_reproduces_the_pre_split_numbers_exactly():
    """Golden values computed with the monolithic pre-split implementation (word counter, 900-token chunks)."""
    solo = recommend(chunk_max_tokens=900, count_tokens=word_counter, batch_size=1)
    assert solo["overheads"] == {
        "extract_facts": 681, "extract_claims": 211, "generate_qa": 163, "generate_negatives": 150,
        "generate_dpo": 95, "generate_ideation": 135, "judge": 56,
    }
    assert solo["per_task_num_ctx"] == {
        "extract_facts": 6144, "extract_claims": 4096, "generate_qa": 5120, "generate_negatives": 3072,
        "generate_dpo": 3072, "generate_ideation": 5120, "judge": 3072,
    }
    assert solo["global_num_ctx"] == 6144

    batched = recommend(chunk_max_tokens=900, count_tokens=word_counter, batch_size=5)
    assert batched["per_task_num_ctx"]["extract_facts"] == 24576
    assert batched["per_task_num_ctx"]["extract_claims"] == 18432
    assert batched["per_task_num_ctx"]["generate_qa"] == 5120  # never batched
    assert batched["global_num_ctx"] == 24576


def test_only_the_batchable_extract_tasks_scale_with_batch_size():
    solo = recommend(chunk_max_tokens=900, count_tokens=word_counter, batch_size=1)
    batched = recommend(chunk_max_tokens=900, count_tokens=word_counter, batch_size=5)
    for task in BATCHABLE_TASKS:
        assert batched["per_task_num_ctx"][task] > solo["per_task_num_ctx"][task]
    for task in set(TASK_OUTPUT_TOKENS) - BATCHABLE_TASKS:
        assert batched["per_task_num_ctx"][task] == solo["per_task_num_ctx"][task]
