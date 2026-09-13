from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(env_file: Path = PROJECT_ROOT / ".env") -> None:
    """Export .env into os.environ so LiteLLM and the router see provider keys.

    pydantic-settings only reads BG_-prefixed fields into Settings; provider keys such as
    GEMINI_API_KEY must be real environment variables. Variables already set in the shell win.
    """
    load_dotenv(env_file, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BG_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'batterygemma.db'}"
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"
    contact_email: str = ""
    allow_paid: bool = False
    max_usd_per_day: float = 5.0

    @field_validator("database_url")
    @classmethod
    def _anchor_sqlite_path(cls, url: str) -> str:
        """Resolve a relative SQLite path against the project root, not the shell's working directory."""
        prefix = "sqlite:///"
        if url.startswith(prefix) and not Path(url.removeprefix(prefix)).is_absolute():
            return f"{prefix}{PROJECT_ROOT / url.removeprefix(prefix)}"
        return url

    @field_validator("data_dir", "configs_dir")
    @classmethod
    def _anchor_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "export"


@lru_cache
def get_settings() -> Settings:
    load_env()
    return Settings()


def load_config(name: str, settings: Settings | None = None) -> dict[str, Any]:
    """Load configs/<name>.yaml."""
    settings = settings or get_settings()
    with open(settings.configs_dir / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
