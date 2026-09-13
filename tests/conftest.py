import pytest
from sqlalchemy import create_engine

from batterygemma.db.session import init_db


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    init_db(engine)
    return engine
