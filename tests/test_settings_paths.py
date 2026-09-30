from pathlib import Path

import corpusforge.settings as corpus_settings
from corpusforge.settings import Settings as CorpusSettings

from batterygemma.settings import PACKAGE_ROOT, PROJECT_ROOT, Settings, get_settings


def test_package_root_is_the_installed_package_location_not_the_cwd():
    """PACKAGE_ROOT locates packaged assets (.env.example, the packaged config templates) that ship with the
    code and never move; PROJECT_ROOT (configs/data/.env) is CWD-based instead, so `bg init` can scaffold a
    fresh project directory anywhere."""
    assert (PACKAGE_ROOT / ".env.example").exists()
    assert (PACKAGE_ROOT / "pyproject.toml").exists()
    assert (PACKAGE_ROOT / "src" / "batterygemma" / "templates" / "sources.yaml").exists()
    assert isinstance(PROJECT_ROOT, Path)


def test_the_migrations_ship_inside_the_package():
    package = Path(__file__).resolve().parents[1] / "src" / "batterygemma"
    assert (package / "migrations" / "env.py").exists()
    assert list((package / "migrations" / "versions").glob("*.py"))


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


def test_existing_bg_prefixed_environment_variables_still_configure_the_project(monkeypatch):
    monkeypatch.setenv("BG_CONTACT_EMAIL", "who@example.org")
    monkeypatch.setenv("BG_MAX_USD_PER_DAY", "2.5")
    settings = Settings(_env_file=None)
    assert settings.contact_email == "who@example.org" and settings.max_usd_per_day == 2.5
    assert settings.database_url.endswith("batterygemma.db")  # battery default, not corpusforge's corpus.db


def test_batterygemma_settings_are_a_corpusforge_settings_and_identify_the_app():
    assert issubclass(Settings, CorpusSettings)
    settings = Settings(_env_file=None)
    assert settings.app_name == "batterygemma" and "batterygemma" in settings.app_url


def test_get_settings_installs_itself_into_corpusforge(monkeypatch):
    """The embedded pipeline (fetch, parse, logging, migrations) must follow batterygemma's settings."""
    get_settings.cache_clear()
    try:
        installed = get_settings()
        assert corpus_settings.get_settings() is installed
    finally:
        corpus_settings.set_settings(None)
        get_settings.cache_clear()
