from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from batterygemma.db import get_session
from batterygemma.db import models as m
from batterygemma.db.session import get_engine, migrate
from batterygemma.llm import AllDeploymentsExhausted, LLMRouter
from batterygemma.settings import get_settings, load_config

app = typer.Typer(help="BatteryGemma data pipeline", no_args_is_help=True)
db_app = typer.Typer(help="Database commands", no_args_is_help=True)
llm_app = typer.Typer(help="Teacher-LLM router commands", no_args_is_help=True)
train_app = typer.Typer(help="Training commands (CPT/SFT via Unsloth's MLX backend; DPO not supported there)",
                        no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(train_app, name="train")
console = Console()


def build_router() -> LLMRouter:
    settings = get_settings()
    return LLMRouter(
        load_config("llm_routes"), allow_paid=settings.allow_paid, max_usd_per_day=settings.max_usd_per_day
    )


@db_app.command("init")
def db_init() -> None:
    """Apply database migrations up to the latest revision (idempotent)."""
    migrate()
    console.print(f"Database ready: {get_settings().database_url}")


@app.command()
def stats() -> None:
    """Row counts per table and document breakdowns by status, source and license."""
    tables = [m.Document, m.File, m.Chunk, m.Material, m.Fact, m.Comparison, m.ClaimPair, m.QA, m.Ideation,
              m.Negative, m.DPOPair, m.LLMCall, m.GenTask]
    with get_session() as s:
        counts = Table("table", "rows")
        for model in tables:
            counts.add_row(model.__tablename__, str(s.scalar(select(func.count()).select_from(model))))
        console.print(counts)
        for column in (m.Document.status, m.Document.source, m.Document.license):
            breakdown = Table(f"documents.{column.key}", "rows")
            for value, n in s.execute(select(column, func.count()).group_by(column).order_by(func.count().desc())):
                breakdown.add_row(str(value), str(n))
            console.print(breakdown)


@llm_app.command("status")
def llm_status() -> None:
    """Show each deployment's availability, requests used today and cooldown."""
    migrate()
    router = build_router()
    table = Table("name", "model", "tier", "family", "enabled", "used today", "rpd", "cooldown s")
    for row in router.status():
        table.add_row(row["name"], row["model"], row["tier"], row["family"], "yes" if row["enabled"] else "no",
                      str(row["used_today"]), str(row["rpd"] or "∞"), str(row["cooldown_s"]))
    console.print(table)
    console.print(f"Paid spend today: ${router.spent_today_usd():.4f} (allow_paid={router.allow_paid})")


@llm_app.command("test")
def llm_test(
    route: str = typer.Option("qa", help="Route name from llm_routes.yaml"),
    prompt: str = typer.Option("In one sentence, why does LiPF6 generate HF in the presence of water?"),
) -> None:
    """Send one prompt through the router and show which deployment answered."""
    migrate()
    try:
        result = build_router().complete(route, [{"role": "user", "content": prompt}], use_cache=False)
    except AllDeploymentsExhausted as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]{result.deployment}[/bold] ({result.model})\n{result.text}")


@app.command()
def discover(
    source: str = typer.Option("all", help="openalex | chemrxiv | arxiv | all"),
    limit: int = typer.Option(500, help="Maximum records per source"),
    from_date: str = typer.Option("2020-01-01", help="arXiv OAI-PMH harvest start date (YYYY-MM-DD)"),
    batch_size: int = typer.Option(200, help="Records per database commit (progress survives interruption)"),
) -> None:
    """Discover candidate papers and store them with cross-source deduplication."""
    from collections import Counter
    from itertools import islice

    from batterygemma.sources.arxiv import ArxivOaiSource
    from batterygemma.sources.base import BlockedByBotProtection, PoliteClient
    from batterygemma.sources.crossref import CrossrefChemRxivSource
    from batterygemma.sources.openalex import OpenAlexSource
    from batterygemma.sources.store import upsert_records

    migrate()
    email = get_settings().contact_email
    cfg = load_config("sources")
    source_cfg = cfg["sources"]
    builders = {
        "openalex": lambda: (
            OpenAlexSource(PoliteClient(contact_email=email, min_interval=0.2), contact_email=email),
            source_cfg["openalex"]["terms"],
        ),
        "chemrxiv": lambda: (
            CrossrefChemRxivSource(PoliteClient(contact_email=email, min_interval=0.5), contact_email=email),
            source_cfg["chemrxiv"]["terms"],
        ),
        "arxiv": lambda: (
            ArxivOaiSource(
                PoliteClient(contact_email=email, min_interval=3.0),  # arXiv asks for >= 3 s between requests
                sets=source_cfg["arxiv"]["oai_sets"], categories=source_cfg["arxiv"]["categories"],
                from_date=from_date,
            ),
            cfg["scope"]["must"][0],
        ),
    }
    names = list(builders) if source == "all" else [source]
    if unknown := set(names) - builders.keys():
        raise typer.BadParameter(f"Unknown source(s): {sorted(unknown)}")

    for name in names:
        if not source_cfg.get(name, {}).get("enabled", False):
            console.print(f"{name}: disabled in sources.yaml, skipped")
            continue
        adapter, terms = builders[name]()
        totals: Counter[str] = Counter()
        records = adapter.discover(terms, limit)
        try:
            while batch := list(islice(records, batch_size)):
                with get_session() as s:
                    totals += upsert_records(s, batch, allow=cfg["license_allow"], flag=cfg["license_flag"])
                console.print(f"  {name}: {sum(totals.values())} records stored so far")
        except BlockedByBotProtection as exc:
            console.print(f"[yellow]{name}: {exc}. Stopped this source (bot protection is not bypassed).[/yellow]")
        finally:
            adapter.client.close()
        console.print(f"[bold]{name}[/bold]: {dict(totals)}")


@app.command()
def screen(
    rescreen: bool = typer.Option(False, help="Also re-evaluate documents already accepted or rejected"),
) -> None:
    """Apply the license gate and keyword relevance scoring to discovered documents."""
    from batterygemma.screen.screening import screen_documents

    migrate()
    with get_session() as s:
        counts = screen_documents(s, load_config("sources"), rescreen=rescreen)
    console.print(dict(counts))


@app.command()
def fetch(
    limit: int = typer.Option(50, help="Maximum accepted documents to fetch in this run"),
) -> None:
    """Download full text: Europe PMC XML, else the licensed PDF, else an Unpaywall repository mirror."""
    from collections import Counter

    from batterygemma.fetch import fetch_document
    from batterygemma.sources.base import PoliteClient

    migrate()
    settings = get_settings()
    cfg = load_config("sources")
    with get_session() as s:
        doc_ids = s.scalars(
            select(m.Document.doc_id)
            .where(m.Document.status == "accepted", ~m.Document.files.any())
            .order_by(m.Document.relevance.desc())
            .limit(limit)
        ).all()
    client = PoliteClient(contact_email=settings.contact_email, min_interval=1.0)
    outcomes: Counter[str] = Counter()
    try:
        for doc_id in doc_ids:
            with get_session() as s:  # one transaction per document so progress survives interruption
                doc = s.get(m.Document, doc_id)
                outcome = fetch_document(s, doc, client, settings.raw_dir, allow=cfg["license_allow"],
                                         flag=cfg["license_flag"], contact_email=settings.contact_email)
            outcomes[outcome] += 1
            console.print(f"  {doc_id}: {outcome}")
    finally:
        client.close()
    console.print(dict(outcomes))


@app.command("add-local")
def add_local(
    paths: list[Path] = typer.Argument(..., help="Local PDF file(s) to add to the corpus"),
    license: str = typer.Option(
        "all-rights-reserved",
        help="License to record. Leave the default unless you actually hold the rights to release this "
        "file's content publicly — the default keeps it out of the open/CC-only track (the default track "
        "for dataset export) while still being usable for your own local fine-tuning.",
    ),
) -> None:
    """Add local PDF file(s) directly into the corpus as already-fetched documents, ready for `bg parse`."""
    from batterygemma.screen.license import ALLOWED, evaluate_license
    from batterygemma.sources.local import add_local_pdf

    migrate()
    settings = get_settings()
    cfg = load_config("sources")
    for path in paths:
        with get_session() as s:
            doc = add_local_pdf(s, path, settings.raw_dir, license=license)
        decision = evaluate_license(doc.license, cfg["license_allow"], cfg["license_flag"])
        track = "open/CC-only" if decision == ALLOWED else "all-sources only, excluded from public export"
        console.print(f'{doc.doc_id}: "{doc.title}" — license={doc.license} ({track})')


@app.command()
def parse(
    limit: int = typer.Option(100, help="Maximum fetched documents to parse in this run"),
    reparse: bool = typer.Option(False, help="Also re-chunk documents that were already chunked"),
) -> None:
    """Parse fetched full text (JATS XML, else PDF via Docling) into section-aware, quality-flagged chunks."""
    from batterygemma.parse.chunk import load_token_counter
    from batterygemma.parse.pdf_docling import build_converter, parse_pdf
    from batterygemma.parse.pipeline import parse_and_chunk

    migrate()
    chunking = load_config("generation")["chunking"]
    count_tokens = load_token_counter()
    statuses = ["fetched", "chunked"] if reparse else ["fetched"]
    with get_session() as s:
        doc_ids = s.scalars(
            select(m.Document.doc_id)
            .where(m.Document.status.in_(statuses), m.Document.files.any())
            .limit(limit)
        ).all()

    converter = None

    def pdf_parser(path):  # the Docling converter loads layout models, so build it only if a PDF needs it
        nonlocal converter
        converter = converter or build_converter()
        return parse_pdf(path, converter)

    total = 0
    for doc_id in doc_ids:
        with get_session() as s:
            n = parse_and_chunk(s, s.get(m.Document, doc_id), count_tokens, chunking, pdf_parser=pdf_parser)
        total += n
        console.print(f"  {doc_id}: {n} chunks")
    console.print(f"Parsed {len(doc_ids)} documents into {total} chunks")


@app.command()
def annotate(
    kind: str = typer.Argument(..., help="facts | claims"),
    limit: int = typer.Option(50, help="Maximum chunks to annotate in this run"),
    force: bool = typer.Option(False, help="Re-annotate chunks already done, bypassing the response cache"),
) -> None:
    """Stage 7: extract facts/comparisons (kind=facts) or claim pairs (kind=claims) from chunks via the LLM."""
    from collections import Counter

    from batterygemma.annotate.facts import annotate_chunk_facts
    from batterygemma.annotate.claim_pairs import annotate_chunk_claims

    runners = {"facts": annotate_chunk_facts, "claims": annotate_chunk_claims}
    if kind not in runners:
        raise typer.BadParameter(f"kind must be one of {sorted(runners)}", param_hint="kind")
    task_type = {"facts": "extract_facts", "claims": "extract_claims"}[kind]

    migrate()
    router = build_router()
    engine = get_engine()
    with get_session(engine) as s:
        done_keys = {
            row for row in s.scalars(
                select(m.GenTask.key).where(m.GenTask.task_type == task_type, m.GenTask.status == "done")
            )
        } if not force else set()
        chunk_ids = s.scalars(
            select(m.Chunk.chunk_id)
            .join(m.Document, m.Chunk.doc_id == m.Document.doc_id)
            .where(m.Document.status == "chunked")
            .order_by(m.Chunk.doc_id, m.Chunk.order)
            .limit(limit + len(done_keys))
        ).all()
    pending = [c for c in chunk_ids if force or f"{task_type}:{c}" not in done_keys][:limit]

    outcomes: Counter[str] = Counter()
    for chunk_id in pending:
        outcome = runners[kind](engine, router, chunk_id, force=force)
        outcomes[outcome] += 1
        console.print(f"  {chunk_id}: {outcome}")
    console.print(dict(outcomes))


def _pending_chunk_ids(engine, task_type: str, limit: int, force: bool) -> list[str]:
    with get_session(engine) as s:
        done_keys = set() if force else {
            row for row in s.scalars(
                select(m.GenTask.key).where(m.GenTask.task_type == task_type, m.GenTask.status == "done")
            )
        }
        chunk_ids = s.scalars(
            select(m.Chunk.chunk_id)
            .join(m.Document, m.Chunk.doc_id == m.Document.doc_id)
            .where(m.Document.status == "chunked")
            .order_by(m.Chunk.doc_id, m.Chunk.order)
            .limit(limit + len(done_keys))
        ).all()
    return [c for c in chunk_ids if force or f"{task_type}:{c}" not in done_keys][:limit]


@app.command()
def generate(
    kind: str = typer.Argument(..., help="qa | negatives | dpo | ideation"),
    limit: int = typer.Option(50, help="Maximum chunks (or QA rows, for dpo) to process in this run"),
    force: bool = typer.Option(False, help="Re-generate items already done, bypassing the response cache"),
) -> None:
    """Stage 8: generate Q&A, negatives, DPO pairs or ideation from stage 7's chunks/facts."""
    from collections import Counter

    from batterygemma.generate.dpo import annotate_qa_dpo
    from batterygemma.generate.ideation import annotate_chunk_ideation
    from batterygemma.generate.negatives import annotate_chunk_false_premise, derive_contradiction_negatives
    from batterygemma.generate.qa import annotate_chunk_qa

    if kind not in ("qa", "negatives", "dpo", "ideation"):
        raise typer.BadParameter("kind must be one of ['qa', 'negatives', 'dpo', 'ideation']", param_hint="kind")

    migrate()
    router = build_router()
    engine = get_engine()
    outcomes: Counter[str] = Counter()

    if kind == "qa":
        for chunk_id in _pending_chunk_ids(engine, "generate_qa", limit, force):
            outcome = annotate_chunk_qa(engine, router, chunk_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {chunk_id}: {outcome}")

    elif kind == "ideation":
        for chunk_id in _pending_chunk_ids(engine, "generate_ideation", limit, force):
            outcome = annotate_chunk_ideation(engine, router, chunk_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {chunk_id}: {outcome}")

    elif kind == "negatives":
        with get_session(engine) as s:
            doc_ids = s.scalars(select(m.Document.doc_id).where(m.Document.status == "chunked")).all()
        for doc_id in doc_ids:  # cheap and idempotent (delete-then-insert): safe to re-run over every document
            with get_session(engine) as s:
                derived = derive_contradiction_negatives(s, doc_id)
            outcomes["contradiction_derived"] += len(derived)
        for chunk_id in _pending_chunk_ids(engine, "generate_false_premise", limit, force):
            outcome = annotate_chunk_false_premise(engine, router, chunk_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {chunk_id}: {outcome}")

    elif kind == "dpo":
        with get_session(engine) as s:
            done_keys = set() if force else {
                row for row in s.scalars(
                    select(m.GenTask.key).where(m.GenTask.task_type == "generate_dpo", m.GenTask.status == "done")
                )
            }
            qa_ids = s.scalars(
                select(m.QA.id).where(m.QA.status == "accepted")  # judged and passed - see export module docstring
                .order_by(m.QA.id).limit(limit + len(done_keys))
            ).all()
        pending = [q for q in qa_ids if force or f"generate_dpo:{q}" not in done_keys][:limit]
        for qa_id in pending:
            outcome = annotate_qa_dpo(engine, router, qa_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {qa_id}: {outcome}")

    console.print(dict(outcomes))


@app.command()
def judge(
    kind: str = typer.Argument(..., help="qa | ideation"),
    limit: int = typer.Option(50, help="Maximum rows to judge in this run"),
    force: bool = typer.Option(False, help="Re-judge rows already done, bypassing the response cache"),
) -> None:
    """Stage 9: score generated Q&A or ideation with a judge model (different family from the generator)."""
    from collections import Counter

    from batterygemma.verify.judge import judge_one_ideation, judge_one_qa

    if kind not in ("qa", "ideation"):
        raise typer.BadParameter("kind must be one of ['qa', 'ideation']", param_hint="kind")

    migrate()
    router = build_router()
    engine = get_engine()
    model_cls, task_type, runner = {
        "qa": (m.QA, "judge_qa", judge_one_qa),
        "ideation": (m.Ideation, "judge_ideation", judge_one_ideation),
    }[kind]

    with get_session(engine) as s:
        done_keys = set() if force else {
            row for row in s.scalars(
                select(m.GenTask.key).where(m.GenTask.task_type == task_type, m.GenTask.status == "done")
            )
        }
        row_ids = s.scalars(
            select(model_cls.id).where(model_cls.status == "generated")
            .order_by(model_cls.id).limit(limit + len(done_keys))
        ).all()
    pending = [r for r in row_ids if force or f"{task_type}:{r}" not in done_keys][:limit]

    outcomes: Counter[str] = Counter()
    for row_id in pending:
        outcome = runner(engine, router, row_id, force=force)
        outcomes[outcome] += 1
        console.print(f"  {row_id}: {outcome}")
    console.print(dict(outcomes))


@app.command()
def export(
    version: str = typer.Option(..., help="Release version tag, e.g. v0.1"),
    output: Path | None = typer.Option(None, help="Output directory (default: data/export/<version>)"),
) -> None:
    """Stage 9: assign train/eval splits, dedupe, then export CPT/SFT/DPO JSONL for `bg train`."""
    from batterygemma.export.unsloth_jsonl import export_all
    from batterygemma.verify.dedupe import mark_duplicates
    from batterygemma.verify.split import assign_document_splits

    migrate()
    settings = get_settings()
    output_dir = output or (settings.export_dir / version)

    with get_session() as s:
        split_counts = assign_document_splits(s)
    console.print(f"newly split documents: {split_counts}")

    with get_session() as s:
        n = mark_duplicates(s, m.QA, bucket_columns=("question_type", "component"),
                            text_fn=lambda row: row.turns[0]["content"], status_filter="accepted")
    console.print(f"near-duplicate QA rows rejected: {n}")

    with get_session() as s:
        result = export_all(s, output_dir)
    for stage_name, counts in result.items():
        console.print(f"{stage_name}: {counts}")
    console.print(f"exported to {output_dir}")


@train_app.command("status")
def train_status() -> None:
    """Install Unsloth if it's missing, then report package versions and accelerator availability."""
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth, environment_report

    try:
        ensure_unsloth()
    except UnslothInstallError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    for key, value in environment_report().items():
        console.print(f"{key:16} {value if value is not None else '[dim]not installed[/dim]'}")


def _run_training(stage: str, dataset: Path, output: Path, config: str) -> None:
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth
    from batterygemma.train.sft import run_cpt, run_sft

    try:
        ensure_unsloth()
    except UnslothInstallError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg = load_config(config)
    runner = {"cpt": run_cpt, "sft": run_sft}[stage]
    result = runner(dataset, output, cfg)
    console.print(f"{stage.upper()} training complete -> {result.output_dir}")
    if result.log_history:
        console.print(f"final logged step: {result.log_history[-1]}")


@train_app.command("cpt")
def train_cpt_cmd(
    dataset: Path = typer.Argument(..., help="Path to a CPT JSONL export ({\"text\": ...} per line)"),
    output: Path = typer.Option(Path("outputs/cpt"), help="Directory to save the trained LoRA adapter"),
    config: str = typer.Option("train_cpt", help="Config name under configs/ (without .yaml)"),
) -> None:
    """Continued pretraining (LoRA) on Apple Silicon via Unsloth's MLX backend."""
    _run_training("cpt", dataset, output, config)


@train_app.command("sft")
def train_sft_cmd(
    dataset: Path = typer.Argument(..., help="Path to an SFT JSONL export ({\"messages\": [...]} per line)"),
    output: Path = typer.Option(Path("outputs/sft"), help="Directory to save the trained LoRA adapter"),
    config: str = typer.Option("train_sft", help="Config name under configs/ (without .yaml)"),
) -> None:
    """Supervised fine-tuning (LoRA) on Apple Silicon via Unsloth's MLX backend.

    DPO is not supported on this backend - see src/batterygemma/train/sft.py module docstring.
    """
    _run_training("sft", dataset, output, config)


if __name__ == "__main__":
    app()
