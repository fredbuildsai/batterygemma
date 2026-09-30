from corpusforge.db.session import VERSION_TABLE as CORPUS_VERSION_TABLE
from corpusforge.models import Base as CorpusBase
from llmrouter_free.store import Base as LedgerBase
from sqlalchemy import create_engine, inspect, text

from batterygemma.db.models import Base
from batterygemma.db.session import VERSION_TABLE, init_db, migrate


def test_migrate_creates_every_table_of_all_three_packages_and_is_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'nested' / 'migrated.db'}"

    migrate(url)
    migrate(url)  # second run must be a no-op, not a "table already exists" failure

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) | set(CorpusBase.metadata.tables) | set(LedgerBase.metadata.tables) <= tables
    with engine.connect() as conn:
        for table in (VERSION_TABLE, CORPUS_VERSION_TABLE):
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar() == 1


def test_each_package_has_its_own_version_history_and_no_shared_alembic_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    migrate(str(engine.url))
    assert VERSION_TABLE != CORPUS_VERSION_TABLE
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_migrated_schema_matches_the_models_column_for_column(tmp_path):
    """The baselines must not drift from the models (a model change needs a new revision)."""
    migrated = create_engine(f"sqlite:///{tmp_path / 'migrated.db'}")
    created = create_engine(f"sqlite:///{tmp_path / 'created.db'}")
    migrate(str(migrated.url))
    init_db(created)
    tables = set(Base.metadata.tables) | set(CorpusBase.metadata.tables) | set(LedgerBase.metadata.tables)
    for table in tables:
        want = {c["name"]: str(c["type"]) for c in inspect(created).get_columns(table)}
        got = {c["name"]: str(c["type"]) for c in inspect(migrated).get_columns(table)}
        assert got == want, table


def test_battery_tables_have_no_foreign_keys_into_the_corpus_schema():
    """The two schemas must stay independently migratable: battery rows refer to corpus rows by string id."""
    corpus_tables = set(CorpusBase.metadata.tables)
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.column.table.name not in corpus_tables, f"{table.name} -> {fk.column.table.name}"
