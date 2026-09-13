from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from batterygemma.db import get_session
from batterygemma.db import models as m
from batterygemma.db.session import migrate
from batterygemma.llm import AllDeploymentsExhausted, LLMRouter
from batterygemma.settings import get_settings, load_config

app = typer.Typer(help="BatteryGemma data pipeline", no_args_is_help=True)
db_app = typer.Typer(help="Database commands", no_args_is_help=True)
llm_app = typer.Typer(help="Teacher-LLM router commands", no_args_is_help=True)
train_app = typer.Typer(help="Training environment commands (M3 training stages are not yet implemented)",
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


if __name__ == "__main__":
    app()
