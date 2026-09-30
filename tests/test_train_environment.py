from unittest.mock import patch

import pytest

from batterygemma.train.environment import UnslothInstallError, ensure_unsloth


def fake_run(returncode: int, stdout: str = "", stderr: str = ""):
    def run(args, **kwargs):
        from subprocess import CompletedProcess

        return CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    return run


def test_ensure_unsloth_skips_install_when_already_present():
    with (
        patch("batterygemma.train.environment.is_unsloth_installed", return_value=True),
        patch("subprocess.run") as run,
    ):
        ensure_unsloth()
    run.assert_not_called()


def test_ensure_unsloth_installs_with_every_extra_in_one_call_when_missing():
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        from subprocess import CompletedProcess

        return CompletedProcess(args, 0)

    with (
        patch("batterygemma.train.environment.is_unsloth_installed", side_effect=[False, True]),
        patch("subprocess.run", side_effect=run),
    ):
        ensure_unsloth(extras=("dev", "parse", "train"))

    assert len(calls) == 1  # every extra passed in ONE `uv sync` call, never split across separate calls
    args = calls[0]
    assert args[:2] == ["uv", "sync"]
    assert args.count("--extra") == 3
    for extra in ("dev", "parse", "train"):
        assert extra in args


def test_ensure_unsloth_raises_with_captured_output_when_install_fails():
    with (
        patch("batterygemma.train.environment.is_unsloth_installed", return_value=False),
        patch("subprocess.run", side_effect=fake_run(1, stderr="no wheel found for this platform")),
        pytest.raises(UnslothInstallError, match="no wheel found for this platform"),
    ):
        ensure_unsloth()


def test_ensure_unsloth_raises_if_command_succeeds_but_import_still_fails():
    with (
        patch("batterygemma.train.environment.is_unsloth_installed", side_effect=[False, False]),
        patch("subprocess.run", side_effect=fake_run(0)),
        pytest.raises(UnslothInstallError),
    ):
        ensure_unsloth()
