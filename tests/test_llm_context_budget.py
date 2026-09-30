from batterygemma.llm.context_budget import (
    apply_global_num_ctx,
    compute_num_ctx,
    fits_within,
    global_num_ctx,
    measured_overheads,
    measure_template_overhead,
    recommend_num_ctx,
    round_up_to_bucket,
)


def word_counter(text: str) -> int:
    return len(text.split())


def test_round_up_to_bucket():
    assert round_up_to_bucket(1, 1024) == 1024
    assert round_up_to_bucket(1024, 1024) == 1024
    assert round_up_to_bucket(1025, 1024) == 2048


def test_measure_template_overhead_sums_system_and_template():
    assert measure_template_overhead("one two", "three four five", word_counter) == 5


def test_compute_num_ctx_applies_buffer_and_rounds_up():
    # worst_case = 900 + 380 + 3000 = 4280; *1.2 = 5136; rounded up to 1024 -> 6144
    assert compute_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000) == 6144


def test_compute_num_ctx_respects_custom_buffer_and_bucket():
    # worst_case = 100; *1.5 = 150; rounded up to 100 -> 200
    assert compute_num_ctx(chunk_max_tokens=50, template_overhead_tokens=30, output_tokens=20,
                           buffer=0.5, bucket=100) == 200


def test_global_num_ctx_uses_the_largest_task():
    tasks = {"small": 100, "big": 3000}
    result = global_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380, tasks=tasks)
    assert result == compute_num_ctx(chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)


def test_fits_within():
    assert fits_within(8192, chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)
    assert not fits_within(2048, chunk_max_tokens=900, template_overhead_tokens=380, output_tokens=3000)


def test_measured_overheads_covers_every_task_with_positive_real_values():
    overheads = measured_overheads(word_counter)
    from batterygemma.llm.context_budget import TASK_OUTPUT_TOKENS

    assert set(overheads) == set(TASK_OUTPUT_TOKENS)
    assert all(v > 0 for v in overheads.values())


def test_recommend_num_ctx_report_shape_and_consistency():
    report = recommend_num_ctx(chunk_max_tokens=900, count_tokens=word_counter)
    assert report["chunk_max_tokens"] == 900
    assert report["batch_size"] == 1
    assert set(report["overheads"]) == set(report["per_task_num_ctx"])
    assert report["global_num_ctx"] == max(report["per_task_num_ctx"].values())


def test_recommend_num_ctx_scales_only_the_batchable_extract_tasks_with_batch_size():
    solo = recommend_num_ctx(chunk_max_tokens=900, count_tokens=word_counter, batch_size=1)
    batched = recommend_num_ctx(chunk_max_tokens=900, count_tokens=word_counter, batch_size=5)

    # extract_facts/extract_claims bundle batch_size chunks' text and output into one call, so their
    # num_ctx must grow with batch_size...
    assert batched["per_task_num_ctx"]["extract_facts"] > solo["per_task_num_ctx"]["extract_facts"]
    assert batched["per_task_num_ctx"]["extract_claims"] > solo["per_task_num_ctx"]["extract_claims"]
    # ...but every other task is always exactly one chunk/item per call, batch_size or not.
    assert batched["per_task_num_ctx"]["generate_qa"] == solo["per_task_num_ctx"]["generate_qa"]
    assert batched["per_task_num_ctx"]["judge"] == solo["per_task_num_ctx"]["judge"]


def test_apply_global_num_ctx_only_touches_ollama_deployments_and_preserves_other_extra_body():
    config = {
        "deployments": [
            {"name": "cloud", "model": "nvidia_nim/deepseek", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
            {"name": "local", "model": "ollama_chat/gemma4:e4b", "extra_body": {"think": False}},
            {"name": "local-bare", "model": "ollama_chat/gemma3:4b"},
        ],
        "routes": {},
    }
    patched = apply_global_num_ctx(config, 6144)

    by_name = {d["name"]: d for d in patched["deployments"]}
    assert by_name["cloud"]["extra_body"] == {"chat_template_kwargs": {"thinking": True}}  # untouched
    assert by_name["local"]["extra_body"] == {"think": False, "options": {"num_ctx": 6144}}
    assert by_name["local-bare"]["extra_body"] == {"options": {"num_ctx": 6144}}
    # original config is not mutated
    assert "options" not in config["deployments"][1]["extra_body"]
