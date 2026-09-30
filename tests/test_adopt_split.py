"""`bg db adopt-split`: adopt a database created by the pre-split monolith without touching its data."""

import pytest
from corpusforge.db.session import VERSION_TABLE as CORPUS_VERSION_TABLE
from corpusforge.models import Chunk, Document
from sqlalchemy import create_engine, inspect, text
from typer.testing import CliRunner

from batterygemma import cli
from batterygemma.db.adopt import AdoptError, adopt_split, expected_schema
from batterygemma.db.models import QA, Fact
from batterygemma.db.session import VERSION_TABLE, get_session, init_db, migrate

OLD_HEAD = "e8faa8057f51"


@pytest.fixture
def old_database(tmp_path):
    """What a pre-split database looks like: every table present, one shared `alembic_version` table, real rows."""
    url = f"sqlite:///{tmp_path / 'data' / 'batterygemma.db'}"
    (tmp_path / "data").mkdir()
    engine = create_engine(url, future=True)
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        conn.execute(text("INSERT INTO alembic_version VALUES (:v)"), {"v": OLD_HEAD})
    with get_session(engine) as s:
        s.add(Document(doc_id="d1", source="t", external_id="1", title="Paper", norm_title="paper", license="CC-BY"))
        s.flush()
        s.add(Chunk(chunk_id="d1#c0", doc_id="d1", order=0, tokens=5, text="x"))
        s.add(Fact(id="f1", doc_ids=["d1"], chunk_ids=["d1#c0"], property="p", evidence_sentence="e"))
        s.add(QA(id="q1", doc_ids=["d1"], chunk_ids=["d1#c0"], task_format="t", polarity="positive",
                 answer_type="OPEN", turns=[]))
    engine.dispose()
    return url


def row_counts(url):
    engine = create_engine(url)
    with engine.connect() as conn:
        counts = {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in expected_schema()}
    engine.dispose()
    return counts


def test_adopt_records_both_baselines_retires_the_old_table_and_keeps_every_row(old_database, tmp_path):
    before = row_counts(old_database)
    assert before["documents"] == 1 and before["facts"] == 1

    report = adopt_split(old_database, backup_dir=tmp_path / "backups")

    engine = create_engine(old_database)
    tables = set(inspect(engine).get_table_names())
    assert {VERSION_TABLE, CORPUS_VERSION_TABLE} <= tables and "alembic_version" not in tables
    assert report.previous_revision == OLD_HEAD and report.tables_checked == len(expected_schema())
    assert row_counts(old_database) == before  # not a single row moved, added or removed


def test_adopt_makes_a_consistent_backup_first(old_database, tmp_path):
    report = adopt_split(old_database, backup_dir=tmp_path / "backups")
    assert report.backup_path is not None and report.backup_path.parent == tmp_path / "backups"
    backup = create_engine(f"sqlite:///{report.backup_path}")
    with backup.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM documents")).scalar() == 1
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == OLD_HEAD  # pre-adopt state


def test_after_adopting_migrate_is_a_no_op_and_new_revisions_would_apply_on_top(old_database, tmp_path):
    adopt_split(old_database, backup_dir=tmp_path / "b")
    before = row_counts(old_database)
    migrate(old_database)
    assert row_counts(old_database) == before
    assert "alembic_version" not in inspect(create_engine(old_database)).get_table_names()


def test_adopting_twice_is_harmless(old_database, tmp_path):
    adopt_split(old_database, backup_dir=tmp_path / "b")
    again = adopt_split(old_database, backup_dir=tmp_path / "b")
    assert again.already_adopted and again.lines() == ["already adopted: nothing to do"]
    assert len(list((tmp_path / "b").glob("*.db"))) == 1  # no second backup


def test_a_database_missing_columns_is_refused_and_left_untouched(old_database, tmp_path):
    engine = create_engine(old_database)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE chunks DROP COLUMN images"))
    with pytest.raises(AdoptError, match="missing columns.*chunks"):
        adopt_split(old_database, backup_dir=tmp_path / "b")
    assert "alembic_version" in inspect(create_engine(old_database)).get_table_names()  # nothing changed
    assert not (tmp_path / "b").exists()


def test_a_database_missing_tables_is_refused_with_advice(tmp_path):
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    create_engine(url).connect().close()
    with pytest.raises(AdoptError, match="bg db init"):
        adopt_split(url, backup_dir=tmp_path / "b")


def test_cli_adopt_split_reports_and_respects_declining(old_database, tmp_path, monkeypatch):
    from batterygemma.settings import Settings

    settings = Settings(_env_file=None, database_url=old_database)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    runner = CliRunner()

    declined = runner.invoke(cli.app, ["db", "adopt-split"], input="n\n")
    assert declined.exit_code == 1 and "alembic_version" in inspect(create_engine(old_database)).get_table_names()

    accepted = runner.invoke(cli.app, ["db", "adopt-split", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert "recorded corpusforge and batterygemma baselines" in accepted.output
    assert "alembic_version" not in inspect(create_engine(old_database)).get_table_names()
