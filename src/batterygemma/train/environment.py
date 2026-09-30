"""Make sure Unsloth (and its extras) are installed before anything that needs it runs.

Unsloth is declared as the optional `train` extra in pyproject.toml rather than a base dependency, since it
pulls in torch/trl/peft/etc — a multi-GB install nobody wants just to run the ingestion pipeline. But it is
the basis of every training stage (M3+), so callers there should not have to remember to install it by hand;
`ensure_unsloth()` checks first and installs on demand.

`uv sync --extra train` alone REPLACES the active extra set rather than adding to it — it will silently
uninstall unrelated extras already in place (docling, dev/test tools), which is exactly the mistake to avoid
here. This module always syncs the full known extra set together.
"""

import importlib
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from batterygemma.settings import PACKAGE_ROOT

# Every optional extra this project defines, kept in sync with pyproject.toml's [project.optional-dependencies].
# Each `uv sync` call defines the *complete* extra set for that call - passing only `--extra train` uninstalls
# any extras from a previous call (e.g. `--extra parse`) that aren't repeated here. Listing every extra in one
# call is what keeps the venv's existing installs (docling, pytest, ...) rather than clobbering them.
ALL_EXTRAS = ("dev", "parse", "dedupe", "train")


class UnslothInstallError(RuntimeError):
    """Installing or importing unsloth failed; see the wrapped output for the underlying `uv sync` error."""


def is_unsloth_installed() -> bool:
    try:
        version("unsloth")
        return True
    except PackageNotFoundError:
        return False


def ensure_unsloth(*, extras: tuple[str, ...] = ALL_EXTRAS) -> None:
    """Install unsloth (and the project's other extras) if it isn't already importable, then verify it is.

    Raises `UnslothInstallError` with the `uv sync` output if the install fails (e.g. no matching wheel for
    this platform) — this is not swallowed, since a broken training environment should fail loudly rather
    than proceed silently without the library it's supposed to guarantee.
    """
    if is_unsloth_installed():
        return
    args = ["uv", "sync", *[flag for extra in extras for flag in ("--extra", extra)]]
    result = subprocess.run(args, cwd=PACKAGE_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not is_unsloth_installed():
        raise UnslothInstallError(
            f"`{' '.join(args)}` did not produce an importable `unsloth`.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    importlib.invalidate_caches()  # the running interpreter's sys.path entries were populated before install


def environment_report() -> dict[str, str | bool | None]:
    """Human-readable snapshot of training readiness: package versions and accelerator availability."""
    report: dict[str, str | bool | None] = {"python": sys.version.split()[0]}
    for package in ("unsloth", "unsloth_zoo", "torch", "trl", "peft", "datasets"):
        try:
            report[package] = version(package)
        except PackageNotFoundError:
            report[package] = None
    try:
        import torch

        report["mps_available"] = bool(torch.backends.mps.is_available())
        report["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        report["mps_available"] = report["cuda_available"] = None
    return report
