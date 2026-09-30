import importlib.util

import pytest
from sqlalchemy import create_engine

from batterygemma.db.session import init_db, register_sqlite_pragmas


@pytest.fixture
def engine(tmp_path):
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True))
    init_db(engine)
    return engine


# These modules exercise the training code, which needs the optional `train` extra (Hugging Face `datasets`).
collect_ignore = [] if importlib.util.find_spec("datasets") else ["test_train_data.py", "test_train_step_logger.py"]
