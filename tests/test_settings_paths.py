from pathlib import Path

from batterygemma.settings import PROJECT_ROOT, Settings


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
