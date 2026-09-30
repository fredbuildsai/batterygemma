from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from batterygemma.db.models import Base
from batterygemma.settings import get_settings


def register_sqlite_pragmas(engine: Engine) -> Engine:
    """Apply this project's required SQLite settings to any engine (used by get_engine() and by tests).

    A caller can hold an open write transaction on one session (e.g. building Fact/Comparison rows) while
    code it calls opens a second, independent session on the same engine (e.g. the LLM router logging to
    llm_calls) - two concurrent SQLite writers otherwise fail immediately with "database is locked" rather
    than waiting. 30s comfortably covers a slow request+commit.
    """
    if engine.dialect.name != "sqlite":
        return engine

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return register_sqlite_pragmas(create_engine(url, future=True))


def init_db(engine: Engine | None = None) -> None:
    """Create tables directly from the models. For tests and throwaway databases only."""
    Base.metadata.create_all(engine or get_engine())


def migrate(database_url: str | None = None) -> None:
    """Bring a real database to the latest Alembic revision (use this instead of init_db).

    Alembic applies `alembic.ini`'s `[loggers]` section via `logging.config.fileConfig`, which defaults to
    `disable_existing_loggers=True` - that silently disables every *already-created* logger not listed there
    (root/sqlalchemy/alembic only), `batterygemma`'s whole tree included, since those loggers are created at
    module-import time, before this function ever runs. A disabled logger drops every record before even
    checking its handlers, so this broke `bg logs` app-wide with no error anywhere - confirmed live: `.info()`
    calls, an attached handler, everything looked right except `Logger.disabled` was `True`. Re-enabling the
    `batterygemma` tree straight after is the fix, since Alembic's own fileConfig call is not itself
    something this project's code controls the parameters of.
    """
    from alembic import command
    from alembic.config import Config

    from batterygemma.settings import PACKAGE_ROOT

    url = database_url or get_settings().database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    config = Config(str(PACKAGE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PACKAGE_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    _reenable_batterygemma_loggers()


def _reenable_batterygemma_loggers() -> None:
    import logging

    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(obj, logging.Logger) and (name == "batterygemma" or name.startswith("batterygemma.")):
            obj.disabled = False


@contextmanager
def get_session(engine: Engine | None = None) -> Iterator[Session]:
    factory = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
