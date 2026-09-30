"""Engine, session and migration helpers for batterygemma's database.

One database holds three independently-versioned schemas:

- corpusforge: documents, files, chunks, gen_tasks, releases      (Alembic table `corpusforge_alembic_version`)
- batterygemma: facts, comparisons, claim_pairs, qa, ideation, ... (Alembic table `batterygemma_alembic_version`)
- llmrouter-free: the `llm_calls` ledger/cache                     (idempotent `create_all`, no migrations)

`migrate()` brings all three up to date; `init_db()` (tests, throwaway databases) creates every table directly.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from corpusforge.db.session import get_engine as get_corpus_engine
from corpusforge.db.session import get_session as get_corpus_session
from corpusforge.db.session import init_db as init_corpus_db
from corpusforge.db.session import migrate as migrate_corpus
from corpusforge.db.session import reenable_loggers, register_sqlite_pragmas
from llmrouter_free.store import init_ledger
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from batterygemma.db.models import Base
from batterygemma.settings import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
VERSION_TABLE = "batterygemma_alembic_version"
LOGGER_ROOTS = ("batterygemma", "corpusforge", "llmrouter_free")

__all__ = [
    "MIGRATIONS_DIR", "VERSION_TABLE", "alembic_config", "get_engine", "get_session", "init_db", "migrate",
    "register_sqlite_pragmas", "stamp_battery",
]


def get_engine(database_url: str | None = None) -> Engine:
    """The project's engine. Calling `get_settings()` first installs batterygemma's settings into corpusforge, so
    every embedded corpusforge module (and the `bg` commands registered from it) uses the same database."""
    get_settings()
    return get_corpus_engine(database_url)


get_engine.cache_clear = get_corpus_engine.cache_clear  # type: ignore[attr-defined]


@contextmanager
def get_session(engine: Engine | None = None) -> Iterator[Session]:
    with get_corpus_session(engine or get_engine()) as session:
        yield session


def init_db(engine: Engine | None = None) -> None:
    """Create every table directly from the models. For tests and throwaway databases only."""
    engine = engine or get_engine()
    init_corpus_db(engine)
    Base.metadata.create_all(engine)
    init_ledger(engine)


def alembic_config(database_url: str | None = None):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", (database_url or get_settings().database_url).replace("%", "%%"))
    config.set_main_option("version_table", VERSION_TABLE)
    return config


def migrate(database_url: str | None = None) -> None:
    """Bring a real database to the latest revision of every schema (use this instead of `init_db`).

    Order matters only in that the corpus tables come first; the schemas do not reference each other.
    """
    from alembic import command

    url = database_url or get_settings().database_url
    migrate_corpus(url)
    command.upgrade(alembic_config(url), "head")
    init_ledger(get_engine(url))
    reenable_loggers(LOGGER_ROOTS)


def stamp_battery(database_url: str | None = None, revision: str = "head") -> None:
    """Record batterygemma's schema revision as applied WITHOUT running any DDL."""
    from alembic import command

    command.stamp(alembic_config(database_url), revision)
