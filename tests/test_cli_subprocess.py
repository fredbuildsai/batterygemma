"""End-to-end through a real process, with NO settings monkeypatching.

This is the test that catches wiring mistakes between the three packages: a corpusforge command registered on `bg`
must talk to *batterygemma's* database (BG_DATABASE_URL), not corpusforge's own default.
"""

import os
import subprocess
import sys

from sqlalchemy import create_engine, text


def bg(tmp_path, *args):
    env = {**os.environ, "BG_DATABASE_URL": f"sqlite:///{tmp_path / 'project.db'}", "COLUMNS": "200"}
    env.pop("CF_DATABASE_URL", None)
    return subprocess.run(
        [sys.executable, "-m", "batterygemma.cli", *args], cwd=tmp_path, env=env, capture_output=True, text=True,
        timeout=120,
    )


def test_registered_corpusforge_commands_use_batterygemmas_database(tmp_path):
    assert bg(tmp_path, "db", "init").returncode == 0
    engine = create_engine(f"sqlite:///{tmp_path / 'project.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO documents (doc_id, source, external_id, title, norm_title, authors, topic_tags, status, "
            "license_flagged, blacklisted, raw_metadata, created_at, updated_at) VALUES "
            "('x:1', 'x', '1', 'Bad paper', 'bad paper', '[]', '[]', 'chunked', 0, 1, '{}', '2026-01-01', '2026-01-01')"
        ))
        conn.execute(text("UPDATE documents SET blacklist_reason='keeps failing' WHERE doc_id='x:1'"))

    result = bg(tmp_path, "blacklist-list")  # a corpusforge command, registered on bg

    assert result.returncode == 0, result.stderr
    assert "x:1" in result.stdout and "keeps failing" in result.stdout
    assert not (tmp_path / "data" / "corpus.db").exists()  # corpusforge's own default DB was never touched


def test_bg_stats_reads_the_configured_database(tmp_path):
    bg(tmp_path, "db", "init")
    result = bg(tmp_path, "stats")
    assert result.returncode == 0, result.stderr
    assert "documents" in result.stdout and "facts" in result.stdout and "llm_calls" in result.stdout
