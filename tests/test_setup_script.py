"""scripts/setup.sh is tested end to end in --dry-run mode (which prints every command it would run) and linted."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "setup.sh"


def run_script(*args, script=SCRIPT, env=None, cwd=None):
    return subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True, env={**os.environ, **(env or {})},
        cwd=cwd, timeout=60,
    )


@pytest.fixture
def checkout(tmp_path):
    """A copy of the script in a throwaway `<parent>/batterygemma/scripts/` tree: no sibling clones exist there."""
    repo = tmp_path / "batterygemma"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "setup.sh")
    return repo / "scripts" / "setup.sh"


def test_script_is_valid_bash_and_executable():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_script_passes_shellcheck():
    result = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_default_dry_run_installs_both_packages_from_pinned_github_tags(checkout):
    result = run_script("--dry-run", script=checkout)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "llmrouter-free @ git+https://github.com/fredbuildsai/llmrouter-free.git@v0.1.0" in out.replace("\\", "")
    assert "corpusforge[parse] @ git+https://github.com/fredbuildsai/corpusforge.git@v0.1.0" in out.replace("\\", "")
    assert "-e .[dev,parse,dedupe]" in out.replace("\\", "")
    assert "bg init" in out
    assert "Done" in out


def test_dry_run_changes_nothing(checkout):
    before = sorted(p.relative_to(checkout.parent.parent) for p in checkout.parent.parent.rglob("*"))
    run_script("--dry-run", "--dev", script=checkout)
    after = sorted(p.relative_to(checkout.parent.parent) for p in checkout.parent.parent.rglob("*"))
    assert before == after
    assert not (checkout.parent.parent / ".venv").exists()


def test_dev_mode_clones_missing_siblings_and_installs_all_three_editable(checkout):
    result = run_script("--dry-run", "--dev", script=checkout)
    assert result.returncode == 0, result.stderr
    out = result.stdout.replace("\\", "")
    parent = checkout.parent.parent.parent
    assert f"git clone https://github.com/fredbuildsai/llmrouter-free.git {parent}/llmrouter-free" in out
    assert f"git clone https://github.com/fredbuildsai/corpusforge.git {parent}/corpusforge" in out
    assert f"-e {parent}/llmrouter-free" in out and f"-e {parent}/corpusforge[parse]" in out and "-e .[dev" in out
    assert "git+https" not in out  # dev mode never installs the packages from GitHub


def test_dev_mode_leaves_existing_sibling_checkouts_alone(checkout):
    for name in ("llmrouter-free", "corpusforge"):
        (checkout.parent.parent.parent / name / ".git").mkdir(parents=True)
    result = run_script("--dry-run", "--dev", script=checkout)
    assert "already cloned - leaving it as is" in result.stdout
    assert "git clone" not in result.stdout


def test_flags_select_extras(checkout):
    out = run_script("--dry-run", "--with-train", "--no-parse", "--skip-init", script=checkout).stdout.replace("\\", "")
    assert "-e .[dev,train]" in out and "corpusforge @ git+" in out and "corpusforge[parse]" not in out
    assert not [line for line in out.splitlines() if line.startswith("+") and "bg init" in line]
    assert "Skipping 'bg init'" in out


def test_python_version_and_ref_overrides(checkout):
    out = run_script("--dry-run", "--python", "3.11", script=checkout,
                     env={"LLMROUTER_REF": "v9.9.9", "CORPUSFORGE_REF": "main"}).stdout.replace("\\", "")
    assert "--python 3.11" in out or "Python 3.11" in out
    assert "llmrouter-free.git@v9.9.9" in out and "corpusforge.git@main" in out


def test_missing_uv_is_installed_with_the_official_installer(checkout, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").symlink_to(shutil.which("git"))
    path = f"{bin_dir}:/usr/bin:/bin"
    if shutil.which("uv", path=path):
        pytest.skip("uv is installed in a system directory on this machine")
    result = run_script("--dry-run", script=checkout, env={"PATH": path})
    assert result.returncode == 0, result.stderr
    assert "uv not found" in result.stdout and "astral.sh/uv/install.sh" in result.stdout


def test_missing_git_is_a_clear_error(checkout, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for tool in ("dirname", "bash"):
        (empty / tool).symlink_to(shutil.which(tool))
    result = run_script("--dry-run", script=checkout, env={"PATH": str(empty)})
    assert result.returncode == 1 and "git is required" in result.stderr


def test_unsupported_python_and_unknown_flags_are_rejected(checkout):
    bad_python = run_script("--python", "3.9", "--dry-run", script=checkout)
    assert bad_python.returncode == 2 and "unsupported Python" in bad_python.stderr
    unknown = run_script("--nonsense", script=checkout)
    assert unknown.returncode == 2 and "unknown option" in unknown.stderr
    missing_value = run_script("--python", script=checkout)
    assert missing_value.returncode == 2


def test_help_lists_every_option():
    out = run_script("--help").stdout
    for flag in ("--dev", "--with-train", "--no-parse", "--skip-init", "--python", "--dry-run"):
        assert flag in out


def test_readme_documents_every_setup_option():
    readme = (ROOT / "README.md").read_text()
    for flag in ("--dev", "--with-train", "--no-parse", "--dry-run"):
        assert flag in readme, f"README does not mention setup.sh {flag}"
