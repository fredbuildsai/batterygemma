"""Structured application logging, persisted to its own SQLite database (separate from `data/batterygemma.db`
so a growing log history never competes with the pipeline's own data for that file's write lock).

Every module logs through the standard library (`logging.getLogger(__name__)`), the normal Python idiom -
nothing here requires call sites to learn a bespoke API. `configure_logging()` (called once, at CLI startup)
attaches a `DBLogHandler` to the `batterygemma` logger tree, which writes each record as a `LogEntry` row:
`ts`, `level`, `component` (the logger name, e.g. `batterygemma.fetch`), `message`, and an optional structured
`context` dict passed as `logger.info("...", extra={"context": {...}})`.

This is deliberately not the same thing as `llm_calls` (the per-request LLM ledger/cache in the main DB,
already fine-grained) - this is the higher-level, human-readable trail: task outcomes, stage summaries,
warnings and errors, queryable from the CLI (`bg logs tail|query|stats`) instead of grepped out of a shell
redirect file.

`LLMCallMetric` (below) is a third thing again: a per-call analytics record - every field the provider
response and this machine actually expose, for later offline analysis (cost/throughput/hardware
comparisons), as opposed to `llm_calls`' narrower operational ledger (used for the response cache and
routing decisions) or this module's human-readable event trail. It lives in the same logs.db file (a
deliberate choice - see `record_llm_call_metric`'s docstring for what's captured and why some fields are
always null for a given provider).
"""

import logging
import os
import platform
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, Index, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from batterygemma.db.session import register_sqlite_pragmas
from batterygemma.settings import get_settings

LOGGER_ROOT = "batterygemma"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LogBase(DeclarativeBase):
    """A separate declarative base (and so a separate metadata/schema) from `db.models.Base` - this really is
    a different database file, not just a different table in the main one."""

    type_annotation_map = {dict[str, Any]: JSON}


class LogEntry(LogBase):
    __tablename__ = "log_entries"
    __table_args__ = (
        Index("ix_log_entries_ts_level", "ts", "level"),
        Index("ix_log_entries_component_ts", "component", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String(8), index=True)
    component: Mapped[str] = mapped_column(String(128), index=True)
    message: Mapped[str] = mapped_column(Text)
    context: Mapped[dict[str, Any]] = mapped_column(default=dict)


class LLMCallMetric(LogBase):
    """One row per LLM call attempt (cloud or local), for offline analytics - not the operational cache/
    routing ledger (`llm_calls`, in the main DB) and not a human-readable event (`LogEntry`, above)."""

    __tablename__ = "llm_call_metrics"
    __table_args__ = (Index("ix_llm_call_metrics_deployment_started", "deployment", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    route: Mapped[str] = mapped_column(String(32), index=True)
    deployment: Mapped[str] = mapped_column(String(64), index=True)
    requested_model: Mapped[str] = mapped_column(String(128))  # the model string we asked for
    is_local: Mapped[bool] = mapped_column(default=False, index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # ok | rate_limited | quota | error | invalid_output

    # Populated on both success and failure, where measurable:
    latency_ms: Mapped[int | None] = mapped_column(Integer)  # our own wall-clock measurement around the call

    # Populated on success only (response.usage / response fields - see `record_llm_call_metric`):
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    tokens_total: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    finish_reason: Mapped[str | None] = mapped_column(String(32))
    response_id: Mapped[str | None] = mapped_column(String(128))
    response_model: Mapped[str | None] = mapped_column(String(128))  # the model the provider says it used -
                                                                      # can differ from requested_model, e.g.
                                                                      # OpenRouter serving a :free alias
    system_fingerprint: Mapped[str | None] = mapped_column(String(128))
    litellm_response_ms: Mapped[float | None] = mapped_column(Float)  # LiteLLM's own internal timing, as a
                                                                       # cross-check against our latency_ms
    cost_usd: Mapped[float | None] = mapped_column(Float)
    api_base: Mapped[str | None] = mapped_column(String(256))

    error: Mapped[str | None] = mapped_column(Text)

    # Machine info, denormalized onto every row (this project runs on one machine at a time, so the
    # duplication cost is negligible and it keeps every row independently analyzable with no join):
    hostname: Mapped[str | None] = mapped_column(String(128))
    hw_model: Mapped[str | None] = mapped_column(String(64))  # e.g. "MacBookAir10,1" - see `machine_info()`
    ram_bytes: Mapped[int | None] = mapped_column(Integer)
    cpu_count: Mapped[int | None] = mapped_column(Integer)
    machine_arch: Mapped[str | None] = mapped_column(String(32))  # e.g. "arm64"
    platform_str: Mapped[str | None] = mapped_column(String(256))  # platform.platform()
    python_version: Mapped[str | None] = mapped_column(String(32))


@lru_cache
def machine_info() -> dict[str, Any]:
    """Static facts about the machine this process is running on. Cached for the process lifetime - these
    don't change mid-run, so we don't want to shell out to `sysctl` on every single LLM call.

    `hw.model`/`hw.memsize` are macOS-specific (`sysctl`, not exposed by the stdlib `platform` module -
    Apple Silicon in particular has no CPU "brand string" the way Intel does, so `platform.processor()`
    returns an empty string there); on any other OS those two fields are left None rather than guessed.
    """
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "machine_arch": platform.machine(),
        "platform_str": platform.platform(),
        "python_version": platform.python_version(),
        "hw_model": None,
        "ram_bytes": None,
    }
    if platform.system() == "Darwin":
        for key, field in (("hw.model", "hw_model"), ("hw.memsize", "ram_bytes")):
            try:
                value = subprocess.run(
                    ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
                ).stdout.strip()
                info[field] = int(value) if field == "ram_bytes" else value
            except (subprocess.SubprocessError, OSError, ValueError):
                pass  # best-effort - never let hardware introspection break the pipeline
    return info


def record_llm_call_metric(
    engine: Engine | None = None,
    *,
    started_at: datetime,
    route: str,
    deployment: str,
    requested_model: str,
    is_local: bool,
    status: str,
    latency_ms: int | None = None,
    response: Any = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> None:
    """Record one LLM call attempt for later analytics. Called from `LLMRouter._attempt` on every attempt,
    success or failure - `response` is the raw LiteLLM `ModelResponse` on success, None on failure.

    What's actually available varies by provider - see the module docstring's research notes: LiteLLM
    normalizes every provider (including local Ollama, via its `ollama_chat` wrapper) to the same
    OpenAI-style `usage` block and hidden params, but does NOT surface Ollama's own richer native timing
    breakdown (`total_duration`/`load_duration`/`prompt_eval_duration`/`eval_duration`, all in the native
    `/api/chat` response) - getting those would mean bypassing LiteLLM for local calls specifically, which
    this project's router deliberately does not do (LiteLLM is its one uniform call path). So for local
    calls this record's `litellm_response_ms` is the best available internal-timing cross-check, same as
    for cloud calls - not a gap specific to local, just a real limit of the abstraction layer in use.
    """
    usage = getattr(response, "usage", None)
    hidden = getattr(response, "_hidden_params", None) or {}
    reasoning_tokens = None
    if usage is not None and getattr(usage, "completion_tokens_details", None):
        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", None)

    with get_log_session(engine) as s:
        s.add(LLMCallMetric(
            started_at=started_at, route=route, deployment=deployment, requested_model=requested_model,
            is_local=is_local, status=status, latency_ms=latency_ms,
            tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
            tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            tokens_total=getattr(usage, "total_tokens", None) if usage else None,
            reasoning_tokens=reasoning_tokens,
            finish_reason=getattr(response.choices[0], "finish_reason", None) if response else None,
            response_id=getattr(response, "id", None) if response else None,
            response_model=getattr(response, "model", None) if response else None,
            system_fingerprint=getattr(response, "system_fingerprint", None) if response else None,
            litellm_response_ms=hidden.get("_response_ms"),
            cost_usd=cost_usd if cost_usd is not None else hidden.get("response_cost"),
            api_base=hidden.get("api_base"),
            error=error,
            **machine_info(),
        ))


@lru_cache
def get_log_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().log_database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return register_sqlite_pragmas(create_engine(url, future=True))


def init_log_db(engine: Engine | None = None) -> None:
    LogBase.metadata.create_all(engine or get_log_engine())


@contextmanager
def get_log_session(engine: Engine | None = None) -> Iterator[Session]:
    factory = sessionmaker(bind=engine or get_log_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DBLogHandler(logging.Handler):
    """Writes each emitted record as one `LogEntry` row. Failure to log must never crash the caller - a
    handler exception is swallowed (via `handleError`, the standard logging contract) rather than propagated.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        super().__init__()
        self.engine = engine or get_log_engine()
        init_log_db(self.engine)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            context = getattr(record, "context", None) or {}
            with get_log_session(self.engine) as s:
                s.add(LogEntry(
                    ts=datetime.fromtimestamp(record.created, tz=timezone.utc), level=record.levelname,
                    component=record.name, message=record.getMessage(), context=context,
                ))
        except Exception:  # noqa: BLE001 - logging must never be the thing that crashes the pipeline
            self.handleError(record)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        # Local time, not UTC: this is what a human watching the terminal live wants to compare against a
        # wall clock (e.g. to make sense of "waiting 300s" messages) - LogEntry.ts in the DB stays UTC.
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def configure_logging(
    level: int = logging.INFO, *, engine: Engine | None = None, console_level: int = logging.WARNING
) -> DBLogHandler:
    """Attach (once) a `DBLogHandler` (everything at `level`+) and a console `StreamHandler` (only
    `console_level`+, so routine per-task INFO entries stay in the DB without flooding the terminal, while
    things worth seeing live - a cloud route falling back to local, a task failing, a route exhausting every
    deployment - print immediately. Safe to call more than once: a second call is a no-op if a `DBLogHandler`
    is already attached, so every CLI command can call this defensively without double-logging.
    """
    logger = logging.getLogger(LOGGER_ROOT)
    logger.setLevel(level)
    # Don't bubble up to the stdlib root logger: Alembic's alembic.ini attaches its own console handler
    # there (see db.session.migrate's docstring for the related disabled-logger issue) - our own console
    # handler below is the deliberate replacement for that, not a gap.
    logger.propagate = False
    for h in logger.handlers:
        if isinstance(h, DBLogHandler):
            return h
    handler = DBLogHandler(engine)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)
    return handler


def query_logs(
    session: Session, *, level: str | None = None, component: str | None = None, contains: str | None = None,
    since: datetime | None = None, after_id: int | None = None, limit: int = 100,
) -> list[LogEntry]:
    """Most-recent-first, unless `after_id` is given (then ascending by id - the `tail --follow` case)."""
    stmt = select(LogEntry)
    if level:
        stmt = stmt.where(LogEntry.level == level.upper())
    if component:
        stmt = stmt.where(LogEntry.component.contains(component))
    if contains:
        stmt = stmt.where(LogEntry.message.contains(contains))
    if since:
        stmt = stmt.where(LogEntry.ts >= since)
    if after_id is not None:
        stmt = stmt.where(LogEntry.id > after_id).order_by(LogEntry.id.asc())
    else:
        stmt = stmt.order_by(LogEntry.id.desc())
    return list(session.scalars(stmt.limit(limit)).all())


def log_stats(session: Session) -> dict[str, dict[str, int]]:
    by_level = dict(session.execute(select(LogEntry.level, func.count()).group_by(LogEntry.level)).all())
    by_component = dict(session.execute(select(LogEntry.component, func.count()).group_by(LogEntry.component)).all())
    return {"by_level": by_level, "by_component": by_component}


def clear_logs(session: Session, *, older_than: datetime | None = None) -> int:
    from sqlalchemy import delete

    stmt = delete(LogEntry)
    if older_than is not None:
        stmt = stmt.where(LogEntry.ts < older_than)
    result = session.execute(stmt)
    return result.rowcount or 0
