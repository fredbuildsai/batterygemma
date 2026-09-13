from sqlalchemy import create_engine, inspect, text

from batterygemma.db.models import Base
from batterygemma.db.session import migrate


def test_migrate_creates_every_model_table_and_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'nested' / 'migrated.db'}"

    migrate(url)
    migrate(url)  # second run must be a no-op, not a "table already exists" failure

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) <= tables
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM alembic_version")).scalar() == 1
