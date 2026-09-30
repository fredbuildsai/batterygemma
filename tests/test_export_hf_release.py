import json

from batterygemma.export.hf_release import attribution_summary, build_model_card


def write_attribution(tmp_path, name, manifest):
    (tmp_path / f"{name}.attribution.json").write_text(json.dumps(manifest))


def test_attribution_summary_unions_sources_across_files_and_sums_rows(tmp_path):
    write_attribution(tmp_path, "cpt_train", {
        "10.1/a": {"doc_id": "d1", "license": "CC-BY", "title": "Paper A", "chunks": [0, 1]},
    })
    write_attribution(tmp_path, "sft_train", {
        "10.1/a": {"doc_id": "d1", "license": "CC-BY", "title": "Paper A", "chunks": [0]},
        "10.1/b": {"doc_id": "d2", "license": "CC0", "title": "Paper B", "chunks": [1, 2]},
    })
    summary = attribution_summary(tmp_path)
    assert summary["10.1/a"]["rows"] == 3  # 2 from cpt + 1 from sft, same source
    assert summary["10.1/b"]["rows"] == 2
    assert summary["10.1/a"]["license"] == "CC-BY"


def test_build_model_card_includes_base_model_license_and_source_count():
    attribution = {
        "10.1/a": {"doc_id": "d1", "license": "CC-BY", "title": "Paper A", "rows": 5},
        "10.1/b": {"doc_id": "d2", "license": "CC0", "title": "Paper B", "rows": 3},
    }
    card = build_model_card(
        repo_name="batterygemma-cpt", base_model="unsloth/gemma-4-E2B-it-UD-MLX-4bit",
        license_id="cc-by-4.0", quantization="Q4_K_M",
        training_provenance={"stage": "cpt", "dataset": "data/export/v0.1/cpt_train.jsonl",
                              "config": "train_cpt", "from_adapter": None},
        attribution=attribution,
    )
    assert "base_model: unsloth/gemma-4-E2B-it-UD-MLX-4bit" in card
    assert "license: cc-by-4.0" in card
    assert "8 example(s)" in card  # 5 + 3 rows
    assert "2 open-licensed source document(s)" in card
    assert "CPT" in card


def test_build_model_card_notes_chained_stage_when_continued_from_an_adapter():
    card = build_model_card(
        repo_name="r", base_model="m", license_id="cc-by-4.0", quantization="Q4_K_M",
        training_provenance={"stage": "sft", "dataset": "d.jsonl", "config": "train_sft",
                              "from_adapter": "outputs/cpt"},
        attribution={},
    )
    assert "continued from a prior outputs/cpt adapter" in card
