"""`bg` after the split: corpusforge's commands are registered, and `bg annotate` runs through the shared runner."""

import json
from types import SimpleNamespace

import pytest
from corpusforge.logs import LOGGER_ROOTS, get_log_engine
from corpusforge.models import Chunk, Document, GenTask
from llmrouter_free import LLMRouter
from sqlalchemy import select
from typer.testing import CliRunner

import batterygemma.settings as bg_settings
from batterygemma import cli
from batterygemma.db.models import ClaimPair, Fact
from batterygemma.db.session import get_engine, get_session, migrate
from batterygemma.settings import Settings

runner = CliRunner()
ENV = {"COLUMNS": "250"}

CONFIG = {
    "deployments": [{"name": "gen", "model": "p/gen", "api_key_env": "KEY_A", "family": "fam1"}],
    "routes": {"extract": ["gen"]},
    "cooldown": {"rate_limit_seconds": 0, "daily_quota_seconds": 0, "error_seconds": 0},
}
TEXT = "Single-crystal NMC811 retained 10% more capacity than polycrystalline NMC811 after 300 cycles."


@pytest.fixture
def project(tmp_path, monkeypatch):
    import logging

    monkeypatch.setenv("KEY_A", "x")
    monkeypatch.setattr("corpusforge.settings.PROJECT_ROOT", tmp_path)
    settings = Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path / 'data' / 'bg.db'}",
        log_database_url=f"sqlite:///{tmp_path / 'data' / 'logs.db'}", data_dir=tmp_path / "data",
        configs_dir=tmp_path / "configs",
    )
    bg_settings.get_settings.cache_clear()
    monkeypatch.setattr(bg_settings, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("batterygemma.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("corpusforge.settings.get_settings", lambda: settings)
    monkeypatch.setattr("corpusforge.db.session.get_settings", lambda: settings)
    monkeypatch.setattr("corpusforge.logs.get_settings", lambda: settings)
    get_engine.cache_clear()
    get_log_engine.cache_clear()
    yield tmp_path
    for name in LOGGER_ROOTS + ("batterygemma",):
        for handler in list(logging.getLogger(name).handlers):
            logging.getLogger(name).removeHandler(handler)
    get_engine.cache_clear()
    get_log_engine.cache_clear()


def scripted_router(engine, respond):
    calls = []

    def completion(**kw):
        calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=respond(kw)))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    return LLMRouter(CONFIG, engine=engine, completion_fn=completion), calls


def seed_chunks(n):
    with get_session() as s:
        s.add(Document(doc_id="doc:1", source="t", external_id="1", title="t", norm_title="t",
                       license="CC-BY-4.0", status="chunked"))
        s.flush()
        s.add_all([Chunk(chunk_id=f"doc:1#s00-c0{i}", doc_id="doc:1", order=i, tokens=20, text=TEXT) for i in range(n)])


def facts_response(kw):
    n = kw["messages"][1]["content"].count("Excerpt ")
    fact = {"material": {"name": "NMC811", "component": "cathode"}, "property": "capacity retention",
            "value": "10", "unit": "%", "conditions": {}, "category": "electrochemical", "polarity": "positive",
            "triple": [], "evidence_sentence": "Single-crystal NMC811 retained 10% more capacity"}
    return json.dumps({"results": [{"chunk_index": i, "facts": [fact], "comparisons": []} for i in range(n)]})


def test_bg_exposes_every_corpusforge_pipeline_command_and_the_logs_group():
    from typer.main import get_command

    commands = get_command(cli.app).commands
    corpusforge_commands = {"discover", "screen", "fetch", "images", "add-local", "parse", "fetch-failures",
                            "pipeline-failures", "blacklist", "blacklist-list", "blacklist-remove"}
    battery_commands = {"init", "stats", "annotate", "generate", "judge", "export"}
    assert (corpusforge_commands | battery_commands | {"db", "llm", "train", "eval", "logs"}) <= set(commands)
    assert {"tail", "query", "stats", "clear"} <= set(commands["logs"].commands)
    assert {"init", "adopt-split"} <= set(commands["db"].commands)


def test_bg_annotate_facts_runs_through_the_corpusforge_runner(project, monkeypatch):
    migrate()
    seed_chunks(3)
    router, calls = scripted_router(get_engine(), facts_response)
    monkeypatch.setattr(cli, "build_router", lambda batch_size=1: router)

    result = runner.invoke(cli.app, ["annotate", "facts", "--limit", "10", "--batch-size", "2"], env=ENV)

    assert result.exit_code == 0, result.output
    assert len(calls) == 2  # ceil(3 / 2) batched calls
    with get_session() as s:
        assert len(list(s.scalars(select(Fact)))) == 3
        tasks = list(s.scalars(select(GenTask).where(GenTask.task_type == "extract_facts")))
    assert len(tasks) == 3 and {t.status for t in tasks} == {"done"} and tasks[0].payload == {"facts": 1, "comparisons": 0}

    # resumable: the second run finds nothing left to do and makes no LLM call
    again = runner.invoke(cli.app, ["annotate", "facts", "--limit", "10"], env=ENV)
    assert again.exit_code == 0 and len(calls) == 2


def test_bg_annotate_claims_uses_its_own_task_type(project, monkeypatch):
    migrate()
    seed_chunks(1)

    def claims_response(kw):
        pair = {"sentence_1": "Single-crystal NMC811 retained 10% more capacity", "sentence_2": "It kept 10% more capacity",
                "category": "paraphrase", "subset_name": "swap"}
        return json.dumps({"results": [{"chunk_index": 0, "pairs": [pair]}]})

    router, _ = scripted_router(get_engine(), claims_response)
    monkeypatch.setattr(cli, "build_router", lambda batch_size=1: router)
    result = runner.invoke(cli.app, ["annotate", "claims", "--limit", "5"], env=ENV)
    assert result.exit_code == 0, result.output
    with get_session() as s:
        assert len(list(s.scalars(select(ClaimPair)))) == 1
        assert list(s.scalars(select(GenTask.task_type))) == ["extract_claims"]


def test_bg_annotate_stops_with_exit_code_1_when_the_route_stays_down(project, monkeypatch):
    migrate()
    seed_chunks(2)

    def down(kw):
        raise RuntimeError("503 service unavailable")

    router, _ = scripted_router(get_engine(), down)
    monkeypatch.setattr(cli, "build_router", lambda batch_size=1: router)
    result = runner.invoke(cli.app, ["annotate", "facts", "--limit", "5", "--retry-wait-seconds", "0",
                                     "--max-consecutive-failures", "2"], env=ENV)
    assert result.exit_code == 1
    assert "re-run this exact command" in result.output.replace("\n", " ") or "stopping" in result.output


def test_bg_annotate_rejects_an_unknown_kind(project):
    assert runner.invoke(cli.app, ["annotate", "nonsense"], env=ENV).exit_code != 0


def test_bg_stats_counts_tables_from_all_three_packages(project):
    migrate()
    seed_chunks(1)
    result = runner.invoke(cli.app, ["stats"], env=ENV)
    assert result.exit_code == 0, result.output
    for table in ("documents", "chunks", "facts", "qa", "llm_calls", "gen_tasks"):
        assert table in result.output


def test_bg_db_init_migrates_everything(project):
    result = runner.invoke(cli.app, ["db", "init"], env=ENV)
    assert result.exit_code == 0
    from sqlalchemy import inspect

    tables = set(inspect(get_engine()).get_table_names())
    assert {"documents", "facts", "llm_calls", "corpusforge_alembic_version", "batterygemma_alembic_version"} <= tables
