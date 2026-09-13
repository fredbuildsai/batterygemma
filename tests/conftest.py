import pytest
from sqlalchemy import create_engine

from batterygemma.db.session import init_db, register_sqlite_pragmas


@pytest.fixture
def engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True))
    init_db(engine)
    return engine
