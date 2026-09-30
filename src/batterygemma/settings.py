"""BatteryGemma settings: corpusforge's `Settings` with the `BG_` prefix and battery-specific defaults.

`get_settings()` also installs the result into corpusforge (`set_settings`), so the embedded pipeline (discovery,
fetch, parse, logging, migrations) follows the same database, directories and contact e-mail as the rest of
batterygemma, and existing `BG_*` variables in `.env` keep working.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from corpusforge.settings import PROJECT_ROOT, load_env, set_settings
from corpusforge.settings import Settings as CorpusSettings
from pydantic_settings import SettingsConfigDict

# Where the batterygemma package's own assets live (`.env.example`, the packaged config templates).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "Settings", "get_settings", "load_config", "load_env"]


class Settings(CorpusSettings):
    model_config = SettingsConfigDict(env_prefix="BG_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'batterygemma.db'}"
    app_name: str = "batterygemma"
    app_url: str = "https://github.com/fredbuildsai/batterygemma"


@lru_cache
def get_settings() -> Settings:
    load_env()
    settings = Settings()
    set_settings(settings)
    return settings


def load_config(name: str, settings: Settings | None = None):
    """Load configs/<name>.yaml."""
    settings = settings or get_settings()
    with open(settings.configs_dir / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
