"""Adopt a database created before batterygemma was split into llmrouter-free + corpusforge + batterygemma.

Before the split, one Alembic history (table `alembic_version`) covered every table. Afterwards each package owns
its own history (`corpusforge_alembic_version`, `batterygemma_alembic_version`) and the router ledger is created
idempotently. The table definitions are identical, so adopting an existing database needs NO schema change: this
module verifies that claim (every expected table and column is present), takes a backup, records the new
baselines, and retires the old version table. Nothing is altered and no row is touched.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import corpusforge.db.session as corpus_session
from corpusforge.models import Base as CorpusBase
from llmrouter_free.store import Base as LedgerBase
from sqlalchemy import create_engine, inspect, text

from batterygemma.db import session as battery_session
from batterygemma.db.models import Base as BatteryBase

OLD_VERSION_TABLE = "alembic_version"


class AdoptError(Exception):
    """The database cannot be adopted as-is; the message says what to do."""


@dataclass
class AdoptReport:
    backup_path: Path | None = None
    previous_revision: str | None = None
    already_adopted: bool = False
    tables_checked: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        if self.already_adopted:
            return ["already adopted: nothing to do"]
        return [
            f"backup: {self.backup_path}" if self.backup_path else "backup: (none)",
            f"previous single-history revision: {self.previous_revision}",
            f"verified {self.tables_checked} tables",
            "recorded corpusforge and batterygemma baselines; retired the old alembic_version table",
            *self.notes,
        ]


def expected_schema() -> dict[str, set[str]]:
    """{table: column names} that the three packages together expect."""
    expected: dict[str, set[str]] = {}
    for base in (CorpusBase, BatteryBase, LedgerBase):
        for name, table in base.metadata.tables.items():
            expected[name] = {c.name for c in table.columns}
    return expected


def _backup_sqlite(database_url: str, backup_dir: Path | None) -> Path:
    path = Path(database_url.removeprefix("sqlite:///"))
    target_dir = backup_dir or path.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"pre_adopt_split_{datetime.now():%Y%m%d_%H%M%S}.db"
    source = sqlite3.connect(path)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)  # consistent even if the database is in WAL mode / being written
        finally:
            dest.close()
    finally:
        source.close()
    return target


def adopt_split(database_url: str, *, backup_dir: Path | None = None) -> AdoptReport:
    engine = create_engine(database_url, future=True)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    report = AdoptReport()

    has_new = {corpus_session.VERSION_TABLE, battery_session.VERSION_TABLE} <= tables
    if has_new and OLD_VERSION_TABLE not in tables:
        report.already_adopted = True
        return report

    missing_tables: list[str] = []
    missing_columns: list[str] = []
    expected = expected_schema()
    for table, columns in expected.items():
        if table not in tables:
            missing_tables.append(table)
            continue
        actual = {c["name"] for c in inspector.get_columns(table)}
        if columns - actual:
            missing_columns.append(f"{table}: {sorted(columns - actual)}")
    if missing_tables or missing_columns:
        raise AdoptError(
            "this database does not match the expected schema, so it cannot be adopted without changes.\n"
            + (f"  missing tables: {missing_tables}\n" if missing_tables else "")
            + (f"  missing columns: {missing_columns}\n" if missing_columns else "")
            + "For a fresh database use `bg db init`. For an old one, first bring it to the last pre-split "
              "revision (e8faa8057f51) with the pre-split release, then run this command again."
        )
    report.tables_checked = len(expected)

    if OLD_VERSION_TABLE in tables:
        with engine.connect() as conn:
            report.previous_revision = conn.execute(text(f"SELECT version_num FROM {OLD_VERSION_TABLE}")).scalar()

    if engine.dialect.name == "sqlite":
        report.backup_path = _backup_sqlite(database_url, backup_dir)
    else:
        report.notes.append("no automatic backup for this database type - make sure you have one")

    corpus_session.stamp(database_url)
    battery_session.stamp_battery(database_url)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {OLD_VERSION_TABLE}"))
    engine.dispose()
    return report


__all__ = ["AdoptError", "AdoptReport", "adopt_split", "expected_schema"]
