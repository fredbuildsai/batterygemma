import logging

import pytest

from batterygemma.logs import (
    DBLogHandler,
    LOGGER_ROOT,
    clear_logs,
    configure_logging,
    get_log_engine,
    get_log_session,
    init_log_db,
    log_stats,
    query_logs,
)


@pytest.fixture
def log_engine(tmp_path):
    engine = get_log_engine(f"sqlite:///{tmp_path / 'logs.db'}")
    init_log_db(engine)
    yield engine
    get_log_engine.cache_clear()


@pytest.fixture
def logger(log_engine):
    log = logging.getLogger("test.logs")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    handler = DBLogHandler(log_engine)
    log.addHandler(handler)
    yield log
    log.removeHandler(handler)


def test_emitted_record_is_persisted_with_level_component_message_and_context(logger, log_engine):
    logger.warning("chunk failed: %s", "c1", extra={"context": {"chunk_id": "c1", "attempts": 2}})

    with get_log_session(log_engine) as s:
        entries = query_logs(s, limit=10)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.level == "WARNING" and entry.component == "test.logs"
    assert entry.message == "chunk failed: c1"
    assert entry.context == {"chunk_id": "c1", "attempts": 2}


def test_record_without_context_stores_an_empty_dict(logger, log_engine):
    logger.info("plain message")
    with get_log_session(log_engine) as s:
        entries = query_logs(s, limit=10)
    assert entries[0].context == {}


def test_query_logs_filters_by_level_component_and_contains(logger, log_engine):
    logger.info("alpha message")
    logger.error("beta failure")
    logger.warning("gamma alpha thing")

    with get_log_session(log_engine) as s:
        assert len(query_logs(s, level="ERROR")) == 1
        assert len(query_logs(s, contains="alpha")) == 2
        assert len(query_logs(s, level="WARNING", contains="alpha")) == 1
        assert len(query_logs(s, component="test.logs")) == 3
        assert len(query_logs(s, component="nonexistent")) == 0


def test_query_logs_after_id_returns_ascending_for_tail_follow(logger, log_engine):
    logger.info("one")
    logger.info("two")
    with get_log_session(log_engine) as s:
        first = query_logs(s, limit=10)[-1]  # oldest of the desc-ordered results = "one"
        logger.info("three")
        newer = query_logs(s, after_id=first.id)
    assert [e.message for e in newer] == ["two", "three"]


def test_log_stats_groups_by_level_and_component(logger, log_engine):
    logger.info("a")
    logger.info("b")
    logger.error("c")
    with get_log_session(log_engine) as s:
        stats = log_stats(s)
    assert stats["by_level"] == {"INFO": 2, "ERROR": 1}
    assert stats["by_component"] == {"test.logs": 3}


def test_clear_logs_removes_everything_by_default(logger, log_engine):
    logger.info("a")
    logger.info("b")
    with get_log_session(log_engine) as s:
        n = clear_logs(s)
    assert n == 2
    with get_log_session(log_engine) as s:
        assert query_logs(s, limit=10) == []


def test_clear_logs_respects_older_than(logger, log_engine):
    from datetime import datetime, timedelta, timezone

    logger.info("old one")
    future_cutoff = datetime.now(timezone.utc) + timedelta(days=1)
    with get_log_session(log_engine) as s:
        n = clear_logs(s, older_than=future_cutoff)
    assert n == 1


def test_configure_logging_is_idempotent(log_engine):
    root = logging.getLogger(LOGGER_ROOT)
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        h1 = configure_logging(engine=log_engine)
        h2 = configure_logging(engine=log_engine)
        assert h1 is h2
        assert sum(isinstance(h, DBLogHandler) for h in root.handlers) == 1
        assert sum(isinstance(h, logging.StreamHandler) and not isinstance(h, DBLogHandler)
                   for h in root.handlers) == 1  # console handler added once too, not duplicated
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)


def test_configure_logging_disables_propagation_to_the_stdlib_root_logger(log_engine):
    root = logging.getLogger(LOGGER_ROOT)
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        configure_logging(engine=log_engine)
        assert root.propagate is False
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)


def test_configure_logging_console_handler_only_shows_warning_and_above(log_engine, capsys):
    root = logging.getLogger(LOGGER_ROOT)
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        configure_logging(engine=log_engine)
        logging.getLogger("batterygemma.somemodule").info("routine progress, DB only")
        logging.getLogger("batterygemma.somemodule").warning("worth seeing live")
        err = capsys.readouterr().err
        assert "routine progress" not in err
        assert "worth seeing live" in err
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)


def test_handler_failure_does_not_raise(logger, log_engine, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("db is gone")

    monkeypatch.setattr("batterygemma.logs.get_log_session", boom)
    monkeypatch.setattr(logging.Handler, "handleError", lambda self, record: None)
    logger.info("this must not raise even though persistence fails")  # no exception propagates
