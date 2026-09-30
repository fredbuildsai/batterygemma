import pytest
from sqlalchemy import create_engine

from batterygemma.db.session import init_db, register_sqlite_pragmas


@pytest.fixture
def engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True))
    init_db(engine)
    return engine


@pytest.fixture(autouse=True)
def _isolated_log_db(tmp_path, monkeypatch):
    """Without this, LLMRouter's per-call analytics recording (`logs.record_llm_call_metric`) would write
    real rows into the actual project data/logs.db on every test run, since that module's default engine
    (unlike the main-DB `engine` fixture above) isn't otherwise test-isolated."""
    from batterygemma import logs

    log_engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'test_logs.db'}", future=True))
    logs.init_log_db(log_engine)
    monkeypatch.setattr(logs, "get_log_engine", lambda *a, **kw: log_engine)
