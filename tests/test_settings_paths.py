from pathlib import Path

from batterygemma.settings import PACKAGE_ROOT, PROJECT_ROOT, Settings


def test_package_root_is_the_installed_package_location_not_the_cwd():
    """PACKAGE_ROOT locates packaged assets (alembic scripts, .env.example) that ship with the
    code and never move; PROJECT_ROOT (used for configs/data/.env) is CWD-based instead, so
    `bg init` can scaffold a fresh project directory anywhere - see settings.py's module docstring
    comment on the split."""
    assert (PACKAGE_ROOT / "alembic.ini").exists()
    assert (PACKAGE_ROOT / "pyproject.toml").exists()
    # PROJECT_ROOT is wherever bg is invoked from (Path.cwd() at import time) - independent of
    # where the package itself lives, unless the tool happens to be run from its own checkout.
    assert isinstance(PROJECT_ROOT, Path)


def test_relative_sqlite_url_and_dirs_resolve_against_project_root():
    settings = Settings(_env_file=None, database_url="sqlite:///data/x.db", data_dir="data", configs_dir="configs")
    assert settings.database_url == f"sqlite:///{PROJECT_ROOT / 'data' / 'x.db'}"
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.configs_dir == PROJECT_ROOT / "configs"


def test_absolute_and_non_sqlite_urls_are_untouched():
    assert Settings(_env_file=None, database_url="sqlite:////tmp/a.db").database_url == "sqlite:////tmp/a.db"
    url = "postgresql+psycopg://u:p@host/db"
    assert Settings(_env_file=None, database_url=url).database_url == url
    assert Settings(_env_file=None, data_dir=Path("/tmp/bg")).data_dir == Path("/tmp/bg")
