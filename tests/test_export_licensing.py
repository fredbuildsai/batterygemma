import json

from batterygemma.export.licensing import (
    classify_export, read_license_status, restricted_doc_ids_for_file, restricted_row_indices_for_file,
    write_filtered_jsonl, write_license_status,
)

ALLOW = ["CC0", "CC-BY", "public-domain"]
FLAG = ["CC-BY-SA"]


def write_attribution(tmp_path, name, manifest):
    (tmp_path / f"{name}.attribution.json").write_text(json.dumps(manifest))


def test_classify_export_flags_non_open_licenses_and_leaves_open_ones_alone(tmp_path):
    write_attribution(tmp_path, "cpt_train", {
        "10.1/open": {"doc_id": "d1", "license": "CC-BY", "title": "Open paper", "chunks": [0, 1]},
        "d2": {"doc_id": "d2", "license": "all-rights-reserved", "title": "Copyrighted book", "chunks": [2]},
        "d3": {"doc_id": "d3", "license": None, "title": "Unknown license doc", "chunks": [3]},
    })
    status = classify_export(tmp_path, ALLOW, FLAG)
    assert status["shareable"] is False
    assert set(status["restricted_doc_ids"]) == {"d2", "d3"}
    assert status["restricted_sources"]["d2"]["status"] == "rejected"
    assert status["restricted_sources"]["d3"]["status"] == "unknown"
    assert "10.1/open" not in status["restricted_sources"]
    assert status["by_file"]["cpt_train.jsonl"] == {
        "shareable": False, "restricted_doc_ids": ["d2", "d3"], "restricted_rows": 2,
    }


def test_classify_export_is_shareable_when_every_source_is_open(tmp_path):
    write_attribution(tmp_path, "sft_train", {
        "10.1/a": {"doc_id": "d1", "license": "CC-BY", "title": "A", "chunks": [0]},
        "10.1/b": {"doc_id": "d2", "license": "CC0", "title": "B", "chunks": [1]},
    })
    status = classify_export(tmp_path, ALLOW, FLAG)
    assert status["shareable"] is True
    assert status["restricted_doc_ids"] == []
    assert status["by_file"]["sft_train.jsonl"]["shareable"] is True


def test_classify_export_aggregates_across_multiple_files(tmp_path):
    write_attribution(tmp_path, "cpt_train", {
        "d1": {"doc_id": "d1", "license": "CC-BY", "title": "A", "chunks": [0]},
    })
    write_attribution(tmp_path, "sft_train", {
        "d2": {"doc_id": "d2", "license": "all-rights-reserved", "title": "B", "chunks": [0]},
    })
    status = classify_export(tmp_path, ALLOW, FLAG)
    assert status["shareable"] is False  # one bad file makes the whole export non-shareable
    assert status["by_file"]["cpt_train.jsonl"]["shareable"] is True  # but per-file is still granular
    assert status["by_file"]["sft_train.jsonl"]["shareable"] is False


def test_flagged_license_is_restricted_not_silently_allowed(tmp_path):
    """CC-BY-SA is FLAGGED (needs manual review per the project's own license policy), not
    ALLOWED - it must not be treated as safe to auto-include in a public release."""
    write_attribution(tmp_path, "cpt_train", {
        "d1": {"doc_id": "d1", "license": "CC-BY-SA", "title": "A", "chunks": [0]},
    })
    status = classify_export(tmp_path, ALLOW, FLAG)
    assert status["shareable"] is False
    assert status["restricted_sources"]["d1"]["status"] == "flagged"


def test_restricted_doc_ids_for_file_reads_the_matching_attribution_file(tmp_path):
    write_attribution(tmp_path, "cpt_train", {
        "d1": {"doc_id": "d1", "license": "CC-BY", "title": "A", "chunks": [0]},
        "d2": {"doc_id": "d2", "license": "all-rights-reserved", "title": "B", "chunks": [1]},
    })
    assert restricted_doc_ids_for_file(tmp_path, "cpt_train.jsonl", ALLOW, FLAG) == {"d2"}


def test_restricted_doc_ids_for_file_is_empty_when_no_attribution_file_exists(tmp_path):
    assert restricted_doc_ids_for_file(tmp_path, "cpt_train.jsonl", ALLOW, FLAG) == set()


def test_restricted_row_indices_for_file_maps_doc_ids_to_row_numbers(tmp_path):
    write_attribution(tmp_path, "cpt_train", {
        "d1": {"doc_id": "d1", "license": "CC-BY", "title": "A", "chunks": [0, 2]},
        "d2": {"doc_id": "d2", "license": "all-rights-reserved", "title": "B", "chunks": [1, 3]},
    })
    assert restricted_row_indices_for_file(tmp_path, "cpt_train.jsonl", {"d2"}) == {1, 3}


def test_write_filtered_jsonl_drops_only_the_given_row_indices(tmp_path):
    source = tmp_path / "cpt_train.jsonl"
    source.write_text("\n".join(json.dumps({"text": f"row{i}"}) for i in range(4)) + "\n")
    dest = tmp_path / "cpt_train.open.jsonl"

    kept = write_filtered_jsonl(source, {1, 3}, dest)

    assert kept == 2
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert rows == [{"text": "row0"}, {"text": "row2"}]


def test_license_status_round_trips_through_json(tmp_path):
    path = tmp_path / "LICENSE_STATUS.json"
    summary = {"shareable": False, "restricted_doc_ids": ["d1"], "training_provenance": {"stage": "cpt"}}
    write_license_status(path, summary)
    assert read_license_status(path) == summary


def test_read_license_status_returns_none_when_missing(tmp_path):
    assert read_license_status(tmp_path / "LICENSE_STATUS.json") is None
