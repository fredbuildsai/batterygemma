import json

from batterygemma.cli import _materialize_run_folder


def test_materialize_run_folder_copies_config_and_dataset(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "train_cpt.yaml").write_text("model_name: some/model\n")
    monkeypatch.setattr("batterygemma.cli.get_settings", lambda: type("S", (), {"configs_dir": configs_dir})())

    dataset = tmp_path / "export" / "cpt_train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"text": "a"}\n{"text": "b"}\n')

    output = tmp_path / "outputs" / "cpt"
    local_dataset = _materialize_run_folder(output, "train_cpt", dataset)

    assert local_dataset == output / "dataset.jsonl"
    assert local_dataset.read_text() == dataset.read_text()
    assert (output / "config.yaml").read_text() == "model_name: some/model\n"


def test_materialize_run_folder_copies_sibling_attribution_file_when_present(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "train_cpt.yaml").write_text("model_name: m\n")
    monkeypatch.setattr("batterygemma.cli.get_settings", lambda: type("S", (), {"configs_dir": configs_dir})())

    dataset = tmp_path / "export" / "cpt_train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"text": "a"}\n')
    attribution = {"10.1/x": {"doc_id": "d1", "license": "CC-BY", "title": "T", "chunks": [0]}}
    (tmp_path / "export" / "cpt_train.attribution.json").write_text(json.dumps(attribution))

    output = tmp_path / "outputs" / "cpt"
    _materialize_run_folder(output, "train_cpt", dataset)

    assert json.loads((output / "dataset.attribution.json").read_text()) == attribution


def test_materialize_run_folder_writes_explicit_attribution_when_given(tmp_path, monkeypatch):
    """The CPT->drop-restricted-rows path passes a pre-filtered attribution dict explicitly,
    which must win over any (stale, unfiltered) sibling attribution.json next to the dataset."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "train_cpt.yaml").write_text("model_name: m\n")
    monkeypatch.setattr("batterygemma.cli.get_settings", lambda: type("S", (), {"configs_dir": configs_dir})())

    dataset = tmp_path / "export" / "cpt_train.open.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"text": "a"}\n')
    # a stale, unrelated attribution file that happens to share the stem - must be ignored
    (tmp_path / "export" / "cpt_train.open.attribution.json").write_text(json.dumps({"stale": {}}))

    explicit = {"10.1/clean": {"doc_id": "d2", "license": "CC0", "title": "Clean", "chunks": [0]}}
    output = tmp_path / "outputs" / "cpt-open"
    _materialize_run_folder(output, "train_cpt", dataset, attribution=explicit)

    assert json.loads((output / "dataset.attribution.json").read_text()) == explicit


def test_materialize_run_folder_skips_config_copy_when_source_config_missing(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()  # train_cpt.yaml deliberately not created
    monkeypatch.setattr("batterygemma.cli.get_settings", lambda: type("S", (), {"configs_dir": configs_dir})())

    dataset = tmp_path / "cpt_train.jsonl"
    dataset.write_text('{"text": "a"}\n')
    output = tmp_path / "outputs" / "cpt"

    local_dataset = _materialize_run_folder(output, "train_cpt", dataset)

    assert local_dataset.exists()
    assert not (output / "config.yaml").exists()
