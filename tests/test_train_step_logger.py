import csv
import json

from batterygemma.train.sft import StepLogger, write_baseline_metrics


def test_step_logger_diffs_cumulative_elapsed_and_tokens_into_per_step_values():
    logger = StepLogger(run_start=1_000_000.0)
    logger(step=1, total_steps=3, loss=8.5, lr=0.0, tokens_sec=50.0, peak_gb=6.0,
           elapsed=10.0, num_tokens=500)
    logger(step=2, total_steps=3, loss=8.0, lr=1e-5, tokens_sec=60.0, peak_gb=6.5,
           elapsed=18.0, num_tokens=980)

    assert logger.rows[0]["duration_seconds"] == 10.0
    assert logger.rows[0]["tokens_this_step"] == 500
    assert logger.rows[0]["cumulative_tokens"] == 500
    assert logger.rows[0]["start_time"] == "1970-01-12T13:46:40+00:00"  # run_start + 0 prior elapsed

    assert logger.rows[1]["duration_seconds"] == 8.0  # 18.0 - 10.0
    assert logger.rows[1]["tokens_this_step"] == 480  # 980 - 500
    assert logger.rows[1]["cumulative_tokens"] == 980


def test_step_logger_records_loss_lr_memory_and_grad_norm():
    logger = StepLogger(run_start=0.0)
    logger(step=1, total_steps=1, loss=7.234, lr=1e-4, tokens_sec=93.0, peak_gb=9.401,
           elapsed=5.0, num_tokens=465, grad_norm=1.23)

    row = logger.rows[0]
    assert row["loss"] == 7.234
    assert row["learning_rate"] == 1e-4
    assert row["tokens_per_second"] == 93.0
    assert row["peak_memory_gb"] == 9.401
    assert row["grad_norm"] == 1.23


def test_step_logger_grad_norm_defaults_to_none_when_not_reported():
    logger = StepLogger(run_start=0.0)
    logger(step=1, total_steps=1, loss=1.0, lr=1e-4, tokens_sec=10.0, peak_gb=1.0,
           elapsed=1.0, num_tokens=10)
    assert logger.rows[0]["grad_norm"] is None


def test_write_csv_writes_a_header_and_one_row_per_step(tmp_path):
    logger = StepLogger(run_start=0.0)
    logger(step=1, total_steps=2, loss=8.0, lr=0.0, tokens_sec=50.0, peak_gb=6.0, elapsed=10.0, num_tokens=500)
    logger(step=2, total_steps=2, loss=7.5, lr=1e-5, tokens_sec=55.0, peak_gb=6.2, elapsed=20.0, num_tokens=1000)

    path = tmp_path / "step_log.csv"
    logger.write_csv(path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["step"] == "1"
    assert rows[0]["tokens_this_step"] == "500"
    assert rows[1]["step"] == "2"
    assert rows[1]["tokens_this_step"] == "500"


def test_write_csv_is_a_no_op_when_no_steps_were_logged(tmp_path):
    logger = StepLogger(run_start=0.0)
    path = tmp_path / "step_log.csv"
    logger.write_csv(path)
    assert not path.exists()


def test_step_logger_merges_in_a_live_system_snapshot(monkeypatch):
    """Every step should carry system/GPU telemetry alongside training metrics - see telemetry.py.
    system_snapshot is mocked here for a deterministic value, not to avoid calling the real one
    (which works fine locally), but so this test doesn't depend on this machine's live state."""
    import batterygemma.train.sft as sft

    monkeypatch.setattr(sft, "system_snapshot", lambda: {"system_cpu_percent": 42.0, "battery_percent": 80})

    logger = StepLogger(run_start=0.0)
    logger(step=1, total_steps=1, loss=1.0, lr=1e-4, tokens_sec=10.0, peak_gb=1.0, elapsed=1.0, num_tokens=10)

    assert logger.rows[0]["system_cpu_percent"] == 42.0
    assert logger.rows[0]["battery_percent"] == 80


def test_write_baseline_metrics_writes_device_and_system_snapshot_before_anything_else(tmp_path, monkeypatch):
    import batterygemma.train.sft as sft

    monkeypatch.setattr(sft, "device_snapshot", lambda: {"device_name": "Apple M1"})
    monkeypatch.setattr(sft, "system_snapshot", lambda: {"system_cpu_percent": 5.0})

    baseline = write_baseline_metrics(tmp_path)

    assert baseline["device_name"] == "Apple M1"
    assert baseline["system_cpu_percent"] == 5.0
    assert "captured_at" in baseline

    with open(tmp_path / "baseline_metrics.json") as f:
        on_disk = json.load(f)
    assert on_disk == baseline


def test_write_baseline_metrics_creates_the_output_directory_if_missing(tmp_path, monkeypatch):
    import batterygemma.train.sft as sft

    monkeypatch.setattr(sft, "device_snapshot", lambda: {})
    monkeypatch.setattr(sft, "system_snapshot", lambda: {})

    output_dir = tmp_path / "nested" / "outputs"
    write_baseline_metrics(output_dir)

    assert (output_dir / "baseline_metrics.json").exists()
