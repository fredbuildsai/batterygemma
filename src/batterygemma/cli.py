import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

from batterygemma.db import get_session
from batterygemma.db import models as m
from batterygemma.db.session import get_engine, migrate
from batterygemma.llm import AllDeploymentsExhausted, LLMRouter
from batterygemma.logs import configure_logging
from batterygemma.settings import get_settings, load_config

app = typer.Typer(help="BatteryGemma data pipeline", no_args_is_help=True)
db_app = typer.Typer(help="Database commands", no_args_is_help=True)
llm_app = typer.Typer(help="Teacher-LLM router commands", no_args_is_help=True)
train_app = typer.Typer(help="Training commands (CPT/SFT via Unsloth's MLX backend; DPO not supported there)",
                        no_args_is_help=True)
eval_app = typer.Typer(help="Stage 11: build the gold benchmark and score a model against it",
                       no_args_is_help=True)
logs_app = typer.Typer(help="Query the structured application log (data/logs.db)", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(logs_app, name="logs")
console = Console()
fetch_logger = logging.getLogger("batterygemma.fetch")
annotate_logger = logging.getLogger("batterygemma.annotate")


def _confirm(prompt: str, auto: bool) -> bool:
    """`auto=True` (the `--yes` flag) always takes the safe/compliant branch of a license-gate
    prompt (drop restricted material and proceed) without pausing for input - it never skips the
    check itself, only the interactive confirmation of an already-safe action."""
    if auto:
        console.print(f"{prompt} [auto-confirmed via --yes]")
        return True
    return typer.confirm(prompt)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", help="Log at DEBUG instead of INFO")) -> None:
    """Runs before every command: attaches the structured DB log handler (see `bg logs`)."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)


def build_router(batch_size: int = 1) -> LLMRouter:
    """Construct the router, running the context-budget sizing step first (see `llm/context_budget.py`):
    local Ollama deployments get a single, precomputed `num_ctx` baked into their config so it never changes
    across a run - computed here, once, rather than guessed in a YAML comment or recomputed per call.

    `batch_size` is `bg annotate --batch-size` (default 1 for every other command, which is today's
    non-batched sizing): a batched `extract_facts`/`extract_claims` call bundles that many chunks' text and
    output budget into one request, so both `num_ctx` and the per-attempt request timeout need to scale
    with it - a fixed single-chunk timeout/context budget silently starves a 5-chunk local Ollama call
    (confirmed live: a 150s single-chunk timeout was too short once the prompt+output grew 5x for a
    batch-of-5 call that fell over to the local fallback; cloud's own context/output limits are large
    enough - 262k tokens for openrouter-nemotron-super - that only the local path actually needed this).
    """
    from batterygemma.llm.context_budget import apply_global_num_ctx, recommend_num_ctx
    from batterygemma.parse.chunk import load_token_counter

    settings = get_settings()
    config = load_config("llm_routes")
    chunk_max_tokens = load_config("generation")["chunking"]["max_tokens"]
    report = recommend_num_ctx(chunk_max_tokens=chunk_max_tokens, count_tokens=load_token_counter(),
                                batch_size=batch_size)
    config = apply_global_num_ctx(config, report["global_num_ctx"])
    config = {**config, "request_timeout_seconds": config.get("request_timeout_seconds", 150) * batch_size}
    return LLMRouter(config, allow_paid=settings.allow_paid, max_usd_per_day=settings.max_usd_per_day)


@app.command()
def init(
    force: bool = typer.Option(
        False, help="Overwrite existing configs/.env with the bundled defaults - existing "
        "customizations are lost. Off by default: init is idempotent and never touches what's already there.",
    ),
) -> None:
    """Scaffold a fresh checkout: config YAML files, .env, data/output directories, and the database.

    Every config in `configs/` (sources, llm_routes, taxonomy, generation, train_cpt, train_sft) is
    generated from a bundled template if missing, for maximum portability - clone this repo (or copy
    just `src/`) onto a new machine and `bg init` gets it running without hand-authoring any YAML.
    Safe to re-run any time: existing files are reported and left alone unless `--force` is passed.
    """
    import shutil

    from batterygemma.settings import PACKAGE_ROOT, PROJECT_ROOT

    settings = get_settings()
    templates_dir = Path(__file__).parent / "templates"

    settings.configs_dir.mkdir(parents=True, exist_ok=True)
    for template in sorted(templates_dir.glob("*.yaml")):
        target = settings.configs_dir / template.name
        existed = target.exists()
        if existed and not force:
            console.print(f"configs/{template.name}: already exists, left alone")
            continue
        shutil.copy2(template, target)
        console.print(f"configs/{template.name}: {'overwritten from template' if existed else 'created'}")

    env_example, env_file = PACKAGE_ROOT / ".env.example", PROJECT_ROOT / ".env"
    env_existed = env_file.exists()
    if env_existed and not force:
        console.print(".env: already exists, left alone")
    elif env_example.exists():
        shutil.copy2(env_example, env_file)
        console.print(f".env: {'overwritten' if env_existed else 'created'} from .env.example - "
                       "fill in your API keys before running `bg discover`/`bg annotate` etc.")
    else:
        console.print("[yellow].env.example not found - skipping .env creation[/yellow]")

    for directory in (settings.data_dir, settings.raw_dir, settings.export_dir, settings.images_dir,
                      settings.data_dir / "eval", settings.data_dir / "review", PROJECT_ROOT / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)
    console.print("data/output directories ready")

    migrate()
    console.print(f"database ready: {settings.database_url}")
    console.print("\n[green]init complete[/green] - edit .env with your provider API keys, then `bg discover --help` to start.")


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


@llm_app.command("context-budget")
def llm_context_budget(
    batch_size: int = typer.Option(
        1, "--batch-size", help="Simulate the sizing `bg annotate --batch-size` would use for the "
        "batchable extract_facts/extract_claims tasks (every other task is always 1 chunk/item per call)."
    ),
) -> None:
    """Show the context-window sizing computed for local Ollama deployments (see llm/context_budget.py).

    Also useful for cloud deployments even though they manage their own context automatically: the same
    per-task worst-case numbers are what you'd check a cloud model's documented context limit against.
    """
    from batterygemma.llm.context_budget import recommend_num_ctx
    from batterygemma.parse.chunk import load_token_counter

    chunk_max_tokens = load_config("generation")["chunking"]["max_tokens"]
    report = recommend_num_ctx(chunk_max_tokens=chunk_max_tokens, count_tokens=load_token_counter(),
                                batch_size=batch_size)
    console.print(f"chunk_max_tokens (configs/generation.yaml): {report['chunk_max_tokens']}, batch_size: {batch_size}")
    table = Table("task", "template overhead (tokens)", "output budget", "recommended num_ctx")
    from batterygemma.llm.context_budget import TASK_OUTPUT_TOKENS

    for task, overhead in report["overheads"].items():
        table.add_row(task, str(overhead), str(TASK_OUTPUT_TOKENS[task]), str(report["per_task_num_ctx"][task]))
    console.print(table)
    console.print(f"[bold]global num_ctx applied to every local deployment: {report['global_num_ctx']}[/bold]")


@app.command()
def discover(
    source: str = typer.Option("all", help="openalex | chemrxiv | arxiv | all"),
    limit: int = typer.Option(500, help="Maximum records per source"),
    from_date: str = typer.Option("2020-01-01", help="arXiv OAI-PMH harvest start date (YYYY-MM-DD)"),
    batch_size: int = typer.Option(200, help="Records per database commit (progress survives interruption)"),
    fields: str = typer.Option(
        "default", "--fields", help="Fields to request from sources that support field selection "
        "(openalex, chemrxiv): 'default' for the curated minimal set, 'all' for the full unrestricted "
        "record (kept on the stored document's raw_metadata), or a comma-separated custom list. "
        "No effect on arxiv - its OAI-PMH feed has no field-selection concept, every record is fixed."
    ),
) -> None:
    """Discover candidate papers and store them with cross-source deduplication."""
    from collections import Counter
    from itertools import islice

    from batterygemma.sources.arxiv import ArxivOaiSource
    from batterygemma.sources.base import BlockedByBotProtection, PoliteClient
    from batterygemma.sources.crossref import DEFAULT_FIELDS as CHEMRXIV_DEFAULT_FIELDS
    from batterygemma.sources.crossref import CrossrefChemRxivSource
    from batterygemma.sources.openalex import DEFAULT_FIELDS as OPENALEX_DEFAULT_FIELDS
    from batterygemma.sources.openalex import OpenAlexSource
    from batterygemma.sources.store import upsert_records

    if fields == "default":
        custom_fields: list[str] | None = None
    elif fields == "all":
        custom_fields = []  # distinguished from "default" below via `use_default`
    else:
        custom_fields = [f.strip() for f in fields.split(",") if f.strip()]
    use_default = fields == "default"

    def resolve_fields(default: list[str]) -> list[str] | None:
        if use_default:
            return default
        return custom_fields or None  # "all" (empty list) or a real custom list both fall through correctly

    migrate()
    email = get_settings().contact_email
    cfg = load_config("sources")
    source_cfg = cfg["sources"]
    builders = {
        "openalex": lambda: (
            OpenAlexSource(PoliteClient(contact_email=email, min_interval=0.2), contact_email=email,
                          fields=resolve_fields(OPENALEX_DEFAULT_FIELDS)),
            source_cfg["openalex"]["terms"],
        ),
        "chemrxiv": lambda: (
            CrossrefChemRxivSource(PoliteClient(contact_email=email, min_interval=0.5), contact_email=email,
                                   fields=resolve_fields(CHEMRXIV_DEFAULT_FIELDS)),
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
    """Download full text: Europe PMC XML, else the licensed PDF, else CORE, else an Unpaywall repository mirror."""
    import os
    from collections import Counter

    from batterygemma.fetch import fetch_document
    from batterygemma.sources.base import PoliteClient

    migrate()
    settings = get_settings()
    cfg = load_config("sources")
    core_api_key = os.environ.get("CORE_API_KEY", "")
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
                                         flag=cfg["license_flag"], contact_email=settings.contact_email,
                                         core_api_key=core_api_key)
            outcomes[outcome] += 1
            console.print(f"  {doc_id}: {outcome}")
            level = logging.INFO if outcome in ("xml", "pdf", "pdf_core", "pdf_unpaywall") else logging.WARNING
            fetch_logger.log(level, "fetch %s: %s", doc_id, outcome,
                             extra={"context": {"doc_id": doc_id, "outcome": outcome}})
    finally:
        client.close()
    console.print(dict(outcomes))


@app.command()
def images(
    limit: int = typer.Option(1000, help="Maximum documents to process in this run"),
) -> None:
    """Resolve <graphic>/<inline-graphic> hrefs in stored XML full text and download the actual figures.

    Only documents fetched as JATS XML (Europe PMC route) carry resolvable figure references; images come
    from the official PMC Open Access S3 bucket (see images.py for why). Saved under data/images/<source>/
    <external_id>/ and linked to the article via a File(kind="image") row, same as the XML/PDF files.
    """
    from collections import Counter

    from batterygemma.images import download_images_for_document
    from batterygemma.sources.base import PoliteClient

    from sqlalchemy.orm import aliased

    migrate()
    settings = get_settings()
    ImageFile = aliased(m.File)
    with get_session() as s:
        rows = s.execute(
            select(m.File.doc_id, m.File.path)
            .where(m.File.kind == "xml", ~m.File.doc_id.in_(
                select(ImageFile.doc_id).where(ImageFile.kind == "image")
            ))
            .limit(limit)
        ).all()
    client = PoliteClient(contact_email=settings.contact_email, min_interval=0.2)
    counts: Counter[str] = Counter()
    try:
        for doc_id, xml_path in rows:
            with get_session() as s:
                doc = s.get(m.Document, doc_id)
                new_files = download_images_for_document(s, doc, Path(xml_path), client, settings.images_dir)
            if new_files:
                counts["documents_with_images"] += 1
                counts["images_downloaded"] += len(new_files)
                console.print(f"  {doc_id}: {len(new_files)} image(s) -> {new_files[0].path.rsplit('/', 1)[0]}")
            else:
                counts["no_images_found"] += 1
    finally:
        client.close()
    console.print(dict(counts))
    console.print(f"[bold]Images saved under {settings.images_dir}[/bold]")


@app.command("add-local")
def add_local(
    paths: list[Path] = typer.Argument(..., help="Local PDF file(s) to add to the corpus"),
    license: str = typer.Option(
        "all-rights-reserved",
        help="License to record. Leave the default unless you actually hold the rights to release this "
        "file's content publicly — the default keeps it out of the open/CC-only track (the default track "
        "for dataset export) while still being usable for your own local fine-tuning.",
    ),
    doc_id: str | None = typer.Option(
        None,
        help="Attach this single file to an EXISTING document (e.g. one from `bg fetch-failures`) instead of "
        "adding it as a new one - preserves that document's original DOI/license/title. Only valid with "
        "exactly one path.",
    ),
) -> None:
    """Add local PDF file(s) directly into the corpus as already-fetched documents, ready for `bg parse`."""
    from batterygemma.screen.license import ALLOWED, evaluate_license
    from batterygemma.sources.local import add_local_pdf, attach_local_pdf

    migrate()
    settings = get_settings()
    cfg = load_config("sources")

    if doc_id is not None:
        if len(paths) != 1:
            console.print("[red]--doc-id only accepts a single path.[/red]")
            raise typer.Exit(1)
        try:
            with get_session() as s:
                doc = attach_local_pdf(s, paths[0], settings.raw_dir, doc_id)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(f'{doc.doc_id}: "{doc.title}" — attached, status={doc.status} (license unchanged: {doc.license})')
        return

    for path in paths:
        with get_session() as s:
            doc = add_local_pdf(s, path, settings.raw_dir, license=license)
        decision = evaluate_license(doc.license, cfg["license_allow"], cfg["license_flag"])
        track = "open/CC-only" if decision == ALLOWED else "all-sources only, excluded from public export"
        console.print(f'{doc.doc_id}: "{doc.title}" — license={doc.license} ({track})')


@app.command("fetch-failures")
def fetch_failures_cmd(
    output: Path = typer.Option(
        Path("data/review/fetch_failures.csv"), help="Where to write the CSV for manual review"
    ),
) -> None:
    """List accepted documents `bg fetch` never got full text for, with a link and the reason, for manual retrieval.

    Every automated, legitimate channel (Europe PMC XML, the licensed PDF, an Unpaywall mirror) has already
    been tried - see the reason column. This never attempts to bypass bot protection or paywalls; it exists
    so you can retrieve a paper by hand (library access, contacting the author, etc.) and add it back with
    `bg add-local --doc-id <doc_id> <file>`, which reattaches it to this same document (keeping its DOI and
    license) instead of creating a disconnected duplicate.
    """
    import csv

    from batterygemma.fetch import best_effort_url

    with get_session() as s:
        docs = s.scalars(
            select(m.Document)
            .where(m.Document.status == "accepted", ~m.Document.files.any())
            .order_by(m.Document.relevance.desc())
        ).all()

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "title", "doi", "year", "source", "reason", "url"])
        for doc in docs:
            writer.writerow([doc.doc_id, doc.title, doc.doi or "", doc.year or "", doc.source,
                             doc.status_reason or "not_yet_attempted", best_effort_url(doc)])

    console.print(f"{len(docs)} documents need manual retrieval -> {output}")
    for doc in docs[:20]:
        console.print(f"  {doc.doc_id}: {doc.status_reason or 'not_yet_attempted'} — {best_effort_url(doc)}")
    if len(docs) > 20:
        console.print(f"  ... and {len(docs) - 20} more, see {output}")


@app.command("pipeline-failures")
def pipeline_failures_cmd(
    output: Path = typer.Option(
        Path("data/review/pipeline_failures.csv"), help="Where to write the CSV for review"
    ),
) -> None:
    """List documents with at least one failed annotate/generate/judge task, across every stage.

    `bg fetch-failures` covers documents that never got full text; this covers everything downstream of
    that - a chunk that kept failing extraction, a Q&A that failed judging, etc. Most failures here are
    transient (a rate-limited or timed-out LLM call) and will simply succeed on the next `bg annotate`/
    `bg generate`/`bg judge` re-run (the task queue is resumable) - this command is for spotting a document
    that fails *every* time, which usually means something about that specific chunk (garbled text, an
    unusual structure) rather than bad luck.
    """
    import csv

    from batterygemma.pipeline_failures import failed_documents

    with get_session() as s:
        results = failed_documents(s)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "title", "failure_count", "stages", "sample_error"])
        for r in results:
            sample_error = r["failures"][0]["error"][:300] if r["failures"] else ""
            writer.writerow([r["doc_id"], r["title"], r["failure_count"], ",".join(r["stages"]), sample_error])

    console.print(f"{len(results)} documents have at least one failed task -> {output}")
    for r in results[:20]:
        console.print(f"  {r['doc_id']} ({r['failure_count']} failures, stages: {', '.join(r['stages'])}): {r['title'][:70]}")
    if len(results) > 20:
        console.print(f"  ... and {len(results) - 20} more, see {output}")


@app.command("blacklist")
def blacklist_cmd(
    threshold: int = typer.Option(20, help="Blacklist any document with at least this many failed tasks"),
) -> None:
    """Blacklist documents that keep failing every time (see `bg pipeline-failures`), so future `bg annotate`/
    `bg generate` runs stop retrying their chunks - it does not touch or delete anything already generated.
    """
    from batterygemma.pipeline_failures import blacklist_repeat_failures

    with get_session() as s:
        newly = blacklist_repeat_failures(s, threshold=threshold)

    if not newly:
        console.print(f"no documents reached the failure threshold ({threshold}) - nothing blacklisted")
        return
    console.print(f"blacklisted {len(newly)} document(s):")
    for r in newly:
        console.print(f"  {r['doc_id']} ({r['failure_count']} failures, stages: {', '.join(r['stages'])}): {r['title'][:70]}")


@app.command("blacklist-list")
def blacklist_list_cmd() -> None:
    """Show every currently blacklisted document and why."""
    with get_session() as s:
        docs = s.scalars(select(m.Document).where(m.Document.blacklisted).order_by(m.Document.doc_id)).all()
    if not docs:
        console.print("no documents are blacklisted")
        return
    for doc in docs:
        console.print(f"  {doc.doc_id}: {doc.blacklist_reason} — {doc.title[:70]}")


@app.command("blacklist-remove")
def blacklist_remove_cmd(doc_id: str = typer.Argument(..., help="Document id to un-blacklist")) -> None:
    """Remove a document from the blacklist (e.g. after fixing whatever made it keep failing)."""
    with get_session() as s:
        doc = s.get(m.Document, doc_id)
        if doc is None:
            console.print(f"[red]no such document: {doc_id}[/red]")
            raise typer.Exit(1)
        was_blacklisted = doc.blacklisted
        doc.blacklisted, doc.blacklist_reason = False, None
    console.print(f"{doc_id}: removed from blacklist" if was_blacklisted else f"{doc_id}: was not blacklisted")


@app.command()
def parse(
    limit: int = typer.Option(100, help="Maximum fetched documents to parse in this run"),
    reparse: bool = typer.Option(False, help="Also re-chunk documents that were already chunked"),
    from_existing_chunks: bool = typer.Option(
        False, "--from-existing-chunks",
        help="Re-chunk at the current configs/generation.yaml chunking settings using each "
        "document's EXISTING chunks as source text, instead of re-parsing its original PDF/XML "
        "(skips Docling entirely - use this after only `target_tokens`/`max_tokens` changed, not "
        "after a genuine re-extraction is needed). Implies --reparse; only affects already-chunked "
        "documents, since there's no existing chunk text to reconstruct from otherwise.",
    ),
) -> None:
    """Parse fetched full text (JATS XML, else PDF via Docling) into section-aware, quality-flagged chunks."""
    from batterygemma.parse.chunk import load_token_counter
    from batterygemma.parse.pdf_docling import build_converter, parse_pdf
    from batterygemma.parse.pipeline import parse_and_chunk, rechunk_from_existing_chunks

    migrate()
    chunking = load_config("generation")["chunking"]
    count_tokens = load_token_counter()
    statuses = ["chunked"] if from_existing_chunks else (["fetched", "chunked"] if reparse else ["fetched"])
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
            doc = s.get(m.Document, doc_id)
            if from_existing_chunks:
                n = rechunk_from_existing_chunks(s, doc, count_tokens, chunking)
            else:
                n = parse_and_chunk(s, doc, count_tokens, chunking, pdf_parser=pdf_parser)
        total += n
        console.print(f"  {doc_id}: {n} chunks")
    console.print(f"Parsed {len(doc_ids)} documents into {total} chunks")


def _load_chunk_ids_filter(path: Path | None) -> set[str] | None:
    """Chunk ids from a one-per-line file, or None if no file was given. Used to pin every pipeline stage
    to the exact same fixed batch of chunks, instead of each stage independently diversifying/truncating
    its own pending pool - which can otherwise pick different chunks at each stage and never converge on a
    batch that has gone through facts, claims, qa, negatives *and* ideation."""
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


@app.command()
def annotate(
    kind: str = typer.Argument(..., help="facts | claims"),
    limit: int = typer.Option(50, help="Maximum chunks to annotate in this run"),
    force: bool = typer.Option(False, help="Re-annotate chunks already done, bypassing the response cache"),
    chunks_file: Path | None = typer.Option(
        None, "--chunks-file", help="Restrict to the chunk ids listed in this file (one per line), instead "
        "of diversifying across the whole corpus - use this to drive one fixed batch through every stage."
    ),
    batch_size: int = typer.Option(
        5, "--batch-size", help="Chunks bundled into a single LLM call - the same annotation mechanism "
        "handles any batch size, including 1."
    ),
    report_every: int = typer.Option(
        100, "--report-every", help="Print a progress table to the terminal every N API calls (batches)."
    ),
    retry_wait_seconds: int = typer.Option(
        300, "--retry-wait-seconds", help="Pause this long before retrying a batch when every deployment "
        "on the route was exhausted (all chunks in the batch came back failed)."
    ),
    max_consecutive_failures: int = typer.Option(
        10, "--max-consecutive-failures", help="Stop the run (without losing progress) after this many "
        "back-to-back fully-failed batches - re-run the same command later to pick up where it left off."
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", help="Run this many batches' LLM calls at once from worker threads, sharing one "
        "router - most of the extract route's deployments have no rpm cap of our own, so this can noticeably "
        "speed up a large backlog. 1 (default) is the original fully-sequential behavior."
    ),
) -> None:
    """Stage 7: extract facts/comparisons (kind=facts) or claim pairs (kind=claims) from chunks via the LLM."""
    from collections import Counter

    from batterygemma.annotate.facts import annotate_chunks_facts
    from batterygemma.annotate.claim_pairs import annotate_chunks_claims

    runners = {"facts": annotate_chunks_facts, "claims": annotate_chunks_claims}
    if kind not in runners:
        raise typer.BadParameter(f"kind must be one of {sorted(runners)}", param_hint="kind")
    task_type = {"facts": "extract_facts", "claims": "extract_claims"}[kind]

    migrate()
    router = build_router(batch_size=batch_size)
    engine = get_engine()
    restrict_to = _load_chunk_ids_filter(chunks_file)
    with get_session(engine) as s:
        done_keys = {
            row for row in s.scalars(
                select(m.GenTask.key).where(m.GenTask.task_type == task_type, m.GenTask.status == "done")
            )
        } if not force else set()
        chunk_ids = s.scalars(
            select(m.Chunk.chunk_id)
            .join(m.Document, m.Chunk.doc_id == m.Document.doc_id)
            .where(m.Document.status == "chunked", ~m.Document.blacklisted)
            .order_by(m.Chunk.doc_id, m.Chunk.order)
        ).all()
    if restrict_to is not None:
        chunk_ids = [c for c in chunk_ids if c in restrict_to]
    pending = _diversify_by_doc([c for c in chunk_ids if force or f"{task_type}:{c}" not in done_keys])[:limit]

    import time
    from datetime import datetime, timedelta

    def _ts() -> str:
        """Local wall-clock timestamp, for console lines that need to be readable against `date`/a
        clock while watching the terminal live - `elapsed`/`waiting Ns` alone don't say when they happened."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_batches = (len(pending) + batch_size - 1) // batch_size
    console.print(f"[{_ts()}] {kind}: {len(pending)} pending chunks, batch_size={batch_size} "
                  f"({total_batches} API calls planned, before any retries)")

    def render_progress(call_count: int, elapsed_s: float) -> Table:
        table = Table(f"{kind} annotation progress", "value", title=f"[{_ts()}] after {call_count} API calls")
        table.add_row("elapsed", f"{elapsed_s / 60:.1f} min")
        table.add_row("api calls / min", f"{call_count / (elapsed_s / 60):.2f}" if elapsed_s > 0 else "n/a")
        table.add_row("chunks done", str(outcomes["done"]))
        table.add_row("chunks skipped", str(outcomes["skipped"]))
        table.add_row("chunks failed", str(outcomes["failed"]))
        table.add_row("chunks remaining", str(len(pending) - sum(outcomes.values())))
        return table

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
    from concurrent.futures import wait as futures_wait

    outcomes: Counter[str] = Counter()
    call_count = 0
    consecutive_full_failures = 0
    started = time.monotonic()
    stopped = False

    def handle_batch_result(batch: list[str], batch_outcomes: dict[str, str]) -> None:
        """Process one completed batch's outcome. Mutates the enclosing `outcomes`/`call_count`/
        `consecutive_full_failures` - only ever called from the main thread (as futures complete via
        `futures_wait`), never from a worker thread, so it needs no lock of its own."""
        nonlocal call_count, consecutive_full_failures, stopped
        call_count += 1
        # Every chunk in the batch failed: almost certainly every deployment on the route is down/exhausted
        # right now (rather than a per-chunk issue), so retrying the exact same batch immediately would just
        # burn another round of the same failures - pause first, per the requested backoff behavior. A GenTask
        # marked "failed" (not "done") is still picked up as pending on the next call, batch or standalone -
        # this is the same resumability that already lets a killed/interrupted run be re-run from scratch
        # without redoing anything already "done".
        batch_all_failed = bool(batch_outcomes) and all(v == "failed" for v in batch_outcomes.values())
        if batch_all_failed:
            consecutive_full_failures += 1
            annotate_logger.warning(
                f"{kind} batch fully failed ({consecutive_full_failures}/{max_consecutive_failures} "
                f"consecutive) - every deployment on route 'extract' appears exhausted; "
                f"waiting {retry_wait_seconds}s before retrying the same batch",
                extra={"context": {"batch": batch, "consecutive_full_failures": consecutive_full_failures}},
            )
            resume_at = (datetime.now() + timedelta(seconds=retry_wait_seconds)).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"[yellow][{_ts()}] all deployments exhausted (failure {consecutive_full_failures}/"
                          f"{max_consecutive_failures}) - waiting {retry_wait_seconds}s, retrying at "
                          f"{resume_at}[/yellow]")
            if consecutive_full_failures >= max_consecutive_failures:
                console.print(render_progress(call_count, time.monotonic() - started))
                console.print(f"[red][{_ts()}] {max_consecutive_failures} consecutive fully-failed batches - "
                              f"stopping. Nothing already marked done was lost; re-run this exact command "
                              f"later to resume from where it stopped.[/red]")
                stopped = True
                return
            work.append({"batch": batch, "not_before": time.monotonic() + retry_wait_seconds})
            return
        consecutive_full_failures = 0
        for chunk_id in batch:
            outcome = batch_outcomes.get(chunk_id, "failed")
            outcomes[outcome] += 1
        annotate_logger.info(
            f"{kind} batch {call_count}/{total_batches}: {dict(Counter(batch_outcomes.values()))}",
            extra={"context": {"batch": batch, "outcomes": batch_outcomes}},
        )
        if call_count % report_every == 0:
            console.print(render_progress(call_count, time.monotonic() - started))

    # Each work item is one batch plus the earliest time it may be (re)submitted - a fresh batch is always
    # ready (0.0); a fully-failed batch gets pushed back with `not_before` set `retry_wait_seconds` out.
    work: list[dict[str, Any]] = [
        {"batch": pending[i:i + batch_size], "not_before": 0.0} for i in range(0, len(pending), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        in_flight: dict[Any, list[str]] = {}
        while (work or in_flight) and not stopped:
            now = time.monotonic()
            while work and len(in_flight) < concurrency:
                ready_index = next((i for i, w in enumerate(work) if w["not_before"] <= now), None)
                if ready_index is None:
                    break
                item = work.pop(ready_index)
                future = pool.submit(runners[kind], engine, router, item["batch"], force=force)
                in_flight[future] = item["batch"]
            if not in_flight:
                # Nothing submittable and nothing running: every remaining item is still waiting out a
                # retry backoff - sleep until the earliest one is ready rather than busy-polling.
                time.sleep(max(0.1, min(w["not_before"] for w in work) - now))
                continue
            done, _ = futures_wait(in_flight.keys(), timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                batch = in_flight.pop(future)
                handle_batch_result(batch, future.result())
                if stopped:
                    break
        # `stopped=True` only stops submitting NEW work - any batches already in flight when we stopped are
        # left to finish naturally as the `with` block exits (their GenTask rows are written by the worker
        # itself regardless of whether we're still here to read the future's result), rather than cancelled
        # mid-call and leaving a chunk's task row in a half-attempted state.
    console.print(render_progress(call_count, time.monotonic() - started))
    if stopped:
        raise typer.Exit(1)
    console.print(dict(outcomes))


def _diversify_by_doc(chunk_ids: list[str]) -> list[str]:
    """Round-robin `chunk_ids` (in "<doc_id>#..." form) across their documents, preserving each document's
    own chunk order. Without this, a query ordered by (doc_id, order) and then truncated to `limit` lets one
    large document (e.g. a long local PDF) consume an entire run's whole budget by itself - confirmed live
    2026-09-16: a single 706-chunk local textbook ate all 2000 slots of an `annotate facts` run before any
    of the ~500 other documents got a single chunk processed."""
    from collections import defaultdict, deque

    by_doc: dict[str, deque[str]] = defaultdict(deque)
    doc_order: list[str] = []
    for chunk_id in chunk_ids:
        doc_id = chunk_id.split("#", 1)[0]
        if doc_id not in by_doc:
            doc_order.append(doc_id)
        by_doc[doc_id].append(chunk_id)

    result: list[str] = []
    while doc_order:
        for doc_id in list(doc_order):
            result.append(by_doc[doc_id].popleft())
            if not by_doc[doc_id]:
                doc_order.remove(doc_id)
    return result


def _pending_chunk_ids(engine, task_type: str, limit: int, force: bool, restrict_to: set[str] | None = None) -> list[str]:
    with get_session(engine) as s:
        done_keys = set() if force else {
            row for row in s.scalars(
                select(m.GenTask.key).where(m.GenTask.task_type == task_type, m.GenTask.status == "done")
            )
        }
        chunk_ids = s.scalars(
            select(m.Chunk.chunk_id)
            .join(m.Document, m.Chunk.doc_id == m.Document.doc_id)
            .where(m.Document.status == "chunked", ~m.Document.blacklisted)
            .order_by(m.Chunk.doc_id, m.Chunk.order)
        ).all()
    if restrict_to is not None:
        chunk_ids = [c for c in chunk_ids if c in restrict_to]
    pending = [c for c in chunk_ids if force or f"{task_type}:{c}" not in done_keys]
    return _diversify_by_doc(pending)[:limit]


@app.command()
def generate(
    kind: str = typer.Argument(..., help="qa | negatives | dpo | ideation"),
    limit: int = typer.Option(50, help="Maximum chunks (or QA rows, for dpo) to process in this run"),
    force: bool = typer.Option(False, help="Re-generate items already done, bypassing the response cache"),
    chunks_file: Path | None = typer.Option(
        None, "--chunks-file", help="Restrict to the chunk ids listed in this file (one per line) - ignored "
        "for kind=dpo, which operates on already-accepted QA rows instead of chunks directly."
    ),
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
    restrict_to = _load_chunk_ids_filter(chunks_file)
    outcomes: Counter[str] = Counter()

    if kind == "qa":
        for chunk_id in _pending_chunk_ids(engine, "generate_qa", limit, force, restrict_to):
            outcome = annotate_chunk_qa(engine, router, chunk_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {chunk_id}: {outcome}")

    elif kind == "ideation":
        for chunk_id in _pending_chunk_ids(engine, "generate_ideation", limit, force, restrict_to):
            outcome = annotate_chunk_ideation(engine, router, chunk_id, force=force)
            outcomes[outcome] += 1
            console.print(f"  {chunk_id}: {outcome}")

    elif kind == "negatives":
        with get_session(engine) as s:
            doc_ids = s.scalars(
                select(m.Document.doc_id).where(m.Document.status == "chunked", ~m.Document.blacklisted)
            ).all()
        if restrict_to is not None:
            doc_ids = [d for d in doc_ids if any(c.startswith(f"{d}#") for c in restrict_to)]
        for doc_id in doc_ids:  # cheap and idempotent (delete-then-insert): safe to re-run over every document
            with get_session(engine) as s:
                derived = derive_contradiction_negatives(s, doc_id)
            outcomes["contradiction_derived"] += len(derived)
        for chunk_id in _pending_chunk_ids(engine, "generate_false_premise", limit, force, restrict_to):
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
    include_images: bool = typer.Option(
        False,
        help="Attach each row's linked figure path(s) as an `images` field (see images.py). Off by default: "
        "Gemma 4 E2B's text-only training doesn't read this field - turn it on to audit which figures back "
        "an example, or to prepare a multimodal export.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Non-interactive: auto-confirm blacklisting and re-exporting around any license-gate warning",
    ),
    cpt_pack: bool = typer.Option(
        True,
        help="Pack consecutive same-section CPT chunks into rows up to train_cpt.yaml's max_seq_length, "
        "instead of one row per chunk. CPT chunks average ~400 tokens against a 2048 max_seq_length "
        "default - packing means each training step actually uses the sequence-length budget instead "
        "of processing a chunk far shorter than it, with a large fraction of steps spent on fixed "
        "per-step overhead rather than the model. Never crosses a section boundary. See docs/training.md.",
    ),
    include_ontology: bool = typer.Option(
        True,
        help="Append lithium-ion-relevant EMMO/BattINFO ontology definition rows (materials, "
        "electrodes/electrolytes, battery types) to the CPT train split - see ontology/lithium_cpt.py.",
    ),
) -> None:
    """Stage 9: assign train/eval splits, dedupe, then export CPT/SFT/DPO JSONL for `bg train`."""
    from batterygemma.export.licensing import (
        LICENSE_STATUS_FILENAME, classify_export, write_license_status,
    )
    from batterygemma.export.unsloth_jsonl import export_all
    from batterygemma.verify.dedupe import mark_duplicates
    from batterygemma.verify.split import assign_document_splits

    migrate()
    settings = get_settings()
    output_dir = output or (settings.export_dir / version)
    license_cfg = load_config("sources")
    cpt_pack_tokens = None
    if cpt_pack:
        cpt_pack_tokens = max(1, load_config("train_cpt")["max_seq_length"] - 64)  # small safety margin
        console.print(f"packing CPT rows up to {cpt_pack_tokens} tokens (train_cpt.yaml max_seq_length - 64)")

    with get_session() as s:
        split_counts = assign_document_splits(s)
    console.print(f"newly split documents: {split_counts}")

    with get_session() as s:
        n = mark_duplicates(s, m.QA, bucket_columns=("question_type", "component"),
                            text_fn=lambda row: row.turns[0]["content"], status_filter="accepted")
    console.print(f"near-duplicate QA rows rejected: {n}")

    with get_session() as s:
        result = export_all(s, output_dir, include_images=include_images, cpt_pack_tokens=cpt_pack_tokens,
                            include_ontology=include_ontology)
    for stage_name, counts in result.items():
        console.print(f"{stage_name}: {counts}")
    console.print(f"exported to {output_dir}")

    status = classify_export(output_dir, license_cfg["license_allow"], license_cfg["license_flag"])
    if status["shareable"]:
        write_license_status(output_dir / LICENSE_STATUS_FILENAME, status)
        console.print(f"[green]license check passed: every source is open-licensed - safe for public "
                       f"release. Saved locally at {output_dir}.[/green]")
        return

    console.print(
        f"\n[red]WARNING: {len(status['restricted_doc_ids'])} source document(s) in this export are NOT "
        "safely redistributable[/red] (missing/unclear license, or a license outside "
        f"{license_cfg['license_allow']} / flagged {license_cfg['license_flag']}):"
    )
    for key, entry in list(status["restricted_sources"].items())[:8]:
        console.print(f"  [{entry['status']}] {entry['doc_id']} ({entry['license']}): {entry['title'][:70]}")
    if len(status["restricted_sources"]) > 8:
        console.print(f"  ... and {len(status['restricted_sources']) - 8} more")
    console.print(
        "[yellow]Consequence: a model or dataset trained on this export must NOT be uploaded to Hugging "
        "Face or any public platform while these are included - `bg train push-to-hub` will refuse to "
        "publish a GGUF built from it.[/yellow]"
    )
    if not _confirm("Blacklist these documents and re-export a clean, shareable version now?", yes):
        write_license_status(output_dir / LICENSE_STATUS_FILENAME, status)
        console.print(f"left as-is: this export is saved locally at {output_dir}, for local/private use "
                       "only - not safe for public release.")
        return

    with get_session() as s:
        for doc_id in status["restricted_doc_ids"]:
            doc = s.get(m.Document, doc_id)
            if doc:
                doc.blacklisted, doc.blacklist_reason = True, "not safely redistributable: license gate failed"

    with get_session() as s:
        result = export_all(s, output_dir, include_images=include_images)
    for stage_name, counts in result.items():
        console.print(f"{stage_name}: {counts}")

    status = classify_export(output_dir, license_cfg["license_allow"], license_cfg["license_flag"])
    write_license_status(output_dir / LICENSE_STATUS_FILENAME, status)
    console.print(f"re-exported without restricted documents -> now safe for public release. "
                  f"Saved locally at {output_dir}."
                  if status["shareable"] else
                  f"[red]still not shareable - see {output_dir / LICENSE_STATUS_FILENAME}[/red]")


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


def _materialize_run_folder(output: Path, config: str, dataset: Path,
                             attribution: dict[str, Any] | None = None) -> Path:
    """Copy this run's exact config and training data into `output`, so the run folder is fully
    self-contained: reproducible and portable even if the shared `configs/*.yaml` or `data/export/`
    tree is later edited, regenerated, or moved. Returns the local dataset copy's path (always
    `output/dataset.jsonl`), which becomes the path recorded in `LICENSE_STATUS.json`'s
    `training_provenance` from this point on."""
    import shutil

    output.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    source_config = settings.configs_dir / f"{config}.yaml"
    if source_config.exists():
        shutil.copy2(source_config, output / "config.yaml")

    local_dataset = output / "dataset.jsonl"
    shutil.copy2(dataset, local_dataset)
    if attribution is not None:
        with open(output / "dataset.attribution.json", "w", encoding="utf-8") as f:
            json.dump(attribution, f, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        source_attribution = dataset.parent / f"{dataset.stem}.attribution.json"
        if source_attribution.exists():
            shutil.copy2(source_attribution, output / "dataset.attribution.json")
    return local_dataset


def _check_license_before_training(dataset: Path, output: Path, stage: str, config: str,
                                    from_adapter: Path | None, yes: bool = False) -> tuple[Path, bool]:
    """Pre-training license gate: warn and offer to drop restricted rows *before* any training
    starts, then record the adapter's shareability (and enough provenance to retrain clean later)
    in `output/LICENSE_STATUS.json` regardless of the outcome. Every branch ends by materializing
    this run's config + final dataset into `output` (see `_materialize_run_folder`) before training
    starts, so the run folder never depends on the shared configs/data/export trees afterward.
    Returns (local dataset copy to actually train on, whether the resulting adapter will be
    shareable)."""
    from batterygemma.export.licensing import (
        LICENSE_STATUS_FILENAME, restricted_doc_ids_for_file, restricted_row_indices_for_file,
        write_filtered_jsonl, write_license_status,
    )

    license_cfg = load_config("sources")
    restricted_doc_ids = restricted_doc_ids_for_file(
        dataset.parent, dataset.name, license_cfg["license_allow"], license_cfg["license_flag"]
    )
    provenance = {"stage": stage, "config": config, "from_adapter": str(from_adapter) if from_adapter else None}

    if not restricted_doc_ids:
        local_dataset = _materialize_run_folder(output, config, dataset)
        write_license_status(output / LICENSE_STATUS_FILENAME, {
            "shareable": True, "restricted_doc_ids": [],
            "training_provenance": {**provenance, "dataset": str(local_dataset)},
        })
        return local_dataset, True

    console.print(
        f"\n[red]WARNING: {dataset.name} includes {len(restricted_doc_ids)} source document(s) that are "
        "NOT safely redistributable[/red] (missing/unclear license, or outside the open-license "
        "allowlist)."
    )
    console.print(
        f"[yellow]Consequence: training on this data as-is will produce an adapter saved locally at "
        f"{output} that must NOT be uploaded to Hugging Face or any public platform - `bg train "
        "push-to-hub` will refuse to publish it unless retrained without this material.[/yellow]"
    )
    if not _confirm("Drop these rows now and train on a clean, filtered copy instead?", yes):
        local_dataset = _materialize_run_folder(output, config, dataset)
        write_license_status(output / LICENSE_STATUS_FILENAME, {
            "shareable": False, "restricted_doc_ids": sorted(restricted_doc_ids),
            "training_provenance": {**provenance, "dataset": str(local_dataset)},
        })
        console.print(f"proceeding with the full dataset. The resulting adapter will be saved locally "
                       f"at {output} for local/private use only - not shareable.")
        return local_dataset, False

    with get_session() as s:
        for doc_id in restricted_doc_ids:
            doc = s.get(m.Document, doc_id)
            if doc:
                doc.blacklisted, doc.blacklist_reason = True, "not safely redistributable: license gate failed"

    drop_indices = restricted_row_indices_for_file(dataset.parent, dataset.name, restricted_doc_ids)
    filtered = dataset.parent / f"{dataset.stem}.open{dataset.suffix}"
    kept = write_filtered_jsonl(dataset, drop_indices, filtered)
    source_attribution = dataset.parent / f"{dataset.stem}.attribution.json"
    clean_attribution = None
    if source_attribution.exists():
        with open(source_attribution, encoding="utf-8") as f:
            clean_attribution = {k: v for k, v in json.load(f).items() if v["doc_id"] not in restricted_doc_ids}
    local_dataset = _materialize_run_folder(output, config, filtered, attribution=clean_attribution)
    console.print(f"wrote {filtered} ({kept} rows, {len(drop_indices)} dropped) and blacklisted the "
                  f"offending document(s) for future exports too. The resulting adapter will be saved "
                  f"locally at {output} (self-contained: config + dataset copied in) and will be shareable.")
    write_license_status(
        output / LICENSE_STATUS_FILENAME,
        {"shareable": True, "restricted_doc_ids": [],
         "training_provenance": {**provenance, "dataset": str(local_dataset)}},
    )
    return local_dataset, True


def _run_training(stage: str, dataset: Path, output: Path, config: str, gguf: bool,
                   from_adapter: Path | None, yes: bool = False) -> None:
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth
    from batterygemma.train.sft import run_cpt, run_sft

    dataset, shareable = _check_license_before_training(dataset, output, stage, config, from_adapter, yes)
    if gguf and not shareable:
        console.print("[yellow]note: the inline GGUF export below will still run for local use, but "
                       "`bg train push-to-hub` will refuse to publish it (see warning above).[/yellow]")
    try:
        ensure_unsloth()
    except UnslothInstallError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg = load_config(config)
    runner = {"cpt": run_cpt, "sft": run_sft}[stage]
    result = runner(dataset, output, cfg, gguf=gguf, from_adapter=from_adapter)

    console.print(f"{stage.upper()} training complete -> {result.output_dir}")
    if from_adapter:
        console.print(f"continued from adapter -> {from_adapter}")
    if result.log_history:
        console.print(f"final logged step: {result.log_history[-1]}")
    if gguf:
        console.print(f"GGUF export -> {result.output_dir}/gguf")


@train_app.command("cpt")
def train_cpt_cmd(
    dataset: Path = typer.Argument(..., help="Path to a CPT JSONL export ({\"text\": ...} per line)"),
    output: Path = typer.Option(Path("outputs/cpt"), help="Directory to save the trained LoRA adapter"),
    config: str = typer.Option("train_cpt", help="Config name under configs/ (without .yaml)"),
    gguf: bool = typer.Option(False, help="Also merge LoRA and export a quantized GGUF file (q4_k_m)"),
    from_adapter: Path | None = typer.Option(
        None, help="Continue training a prior stage's adapter dir instead of starting a fresh LoRA on base"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Non-interactive: auto-confirm dropping any non-open-licensed rows"
    ),
) -> None:
    """Continued pretraining (LoRA) on Apple Silicon via Unsloth's MLX backend."""
    _run_training("cpt", dataset, output, config, gguf, from_adapter, yes)


@train_app.command("sft")
def train_sft_cmd(
    dataset: Path = typer.Argument(..., help="Path to an SFT JSONL export ({\"messages\": [...]} per line)"),
    output: Path = typer.Option(Path("outputs/sft"), help="Directory to save the trained LoRA adapter"),
    config: str = typer.Option("train_sft", help="Config name under configs/ (without .yaml)"),
    gguf: bool = typer.Option(False, help="Also merge LoRA and export a quantized GGUF file (q4_k_m)"),
    from_adapter: Path | None = typer.Option(
        None,
        help="Continue training a prior stage's adapter dir (e.g. outputs/cpt) instead of starting a "
        "fresh LoRA on base - lets CPT -> SFT be chained. Omit to run SFT directly on base.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Non-interactive: auto-confirm dropping any non-open-licensed rows"
    ),
) -> None:
    """Supervised fine-tuning (LoRA) on Apple Silicon via Unsloth's MLX backend.

    DPO is not supported on this backend - see src/batterygemma/train/sft.py module docstring.
    """
    _run_training("sft", dataset, output, config, gguf, from_adapter, yes)


def _retrain_clean(adapter_dir: Path, status: dict, yes: bool = False) -> Path:
    """Blacklist a non-shareable adapter's restricted source documents, filter them out of its
    training dataset, and retrain the same stage from that clean copy. Returns the new adapter
    dir (`<adapter_dir>-open`) to use in place of the original."""
    from batterygemma.export.licensing import (
        restricted_doc_ids_for_file, restricted_row_indices_for_file, write_filtered_jsonl,
    )

    provenance = status.get("training_provenance")
    if not provenance or not provenance.get("dataset"):
        console.print("[red]No training provenance recorded for this adapter - retrain from a clean, "
                       "filtered dataset manually (see `bg export`'s license warning) instead.[/red]")
        raise typer.Exit(1)

    dataset = Path(provenance["dataset"])
    license_cfg = load_config("sources")
    restricted_doc_ids = restricted_doc_ids_for_file(
        dataset.parent, dataset.name, license_cfg["license_allow"], license_cfg["license_flag"]
    )
    with get_session() as s:
        for doc_id in restricted_doc_ids:
            doc = s.get(m.Document, doc_id)
            if doc:
                doc.blacklisted, doc.blacklist_reason = True, "not safely redistributable: license gate failed"
    drop_indices = restricted_row_indices_for_file(dataset.parent, dataset.name, restricted_doc_ids)
    clean_dataset = dataset.parent / f"{dataset.stem}.open{dataset.suffix}"
    write_filtered_jsonl(dataset, drop_indices, clean_dataset)
    clean_output = adapter_dir.parent / f"{adapter_dir.name}-open"
    console.print(f"retraining {provenance['stage']} on {clean_dataset} -> {clean_output} ...")
    _run_training(provenance["stage"], clean_dataset, clean_output, provenance["config"], gguf=False,
                  from_adapter=Path(provenance["from_adapter"]) if provenance.get("from_adapter") else None,
                  yes=yes)
    return clean_output


@train_app.command("export-gguf")
def train_export_gguf_cmd(
    adapter_dir: Path = typer.Argument(..., help="LoRA output dir from a finished `bg train cpt|sft` run"),
    output: Path | None = typer.Option(None, help="Directory to write the GGUF file (default: <adapter_dir>/gguf)"),
    config: str = typer.Option("train_cpt", help="Config name under configs/ (without .yaml) - "
                                "supplies model_name and max_seq_length"),
    quantization: str = typer.Option("q4_k_m", help="llama.cpp quant type, e.g. q4_k_m, q8_0, f16"),
) -> None:
    """Merge a saved LoRA adapter into the base model and export a quantized GGUF file.

    A GGUF file is a local, shareable-or-not-yet-decided artifact - training on non-open-licensed
    material never blocks producing one, only a warning is shown. The hard gate is at
    `bg train push-to-hub`, which refuses to actually publish a GGUF built from restricted data.
    """
    from batterygemma.export.licensing import LICENSE_STATUS_FILENAME, read_license_status, write_license_status
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth
    from batterygemma.train.sft import export_gguf_from_adapter

    status = read_license_status(adapter_dir / LICENSE_STATUS_FILENAME)
    if status is None:
        console.print(f"[yellow]no {LICENSE_STATUS_FILENAME} found in {adapter_dir} - this adapter predates "
                       "the license gate, so its public-release safety is unverified.[/yellow]")
    elif not status["shareable"]:
        console.print(
            f"[yellow]heads up: this adapter was trained on {len(status['restricted_doc_ids'])} "
            "non-open-licensed source document(s). The GGUF file will still be produced for local use, "
            "but `bg train push-to-hub` will refuse to publish it.[/yellow]"
        )

    try:
        ensure_unsloth()
    except UnslothInstallError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    cfg = load_config(config)
    gguf_output = output or adapter_dir
    gguf_dir = export_gguf_from_adapter(
        cfg["model_name"], cfg["max_seq_length"], adapter_dir, gguf_output, quantization_method=quantization,
        text_only=cfg.get("text_only", True),
    )
    if status is not None:
        write_license_status(gguf_dir / LICENSE_STATUS_FILENAME, status)
    console.print(f"GGUF export -> {gguf_dir}")


@train_app.command("push-to-hub")
def train_push_to_hub_cmd(
    gguf_dir: Path = typer.Argument(..., help="GGUF output dir from `bg train export-gguf`"),
    repo_id: str = typer.Option(..., help="Target Hugging Face repo, e.g. username/batterygemma-cpt"),
    private: bool = typer.Option(True, help="Create/push as a private repo"),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Non-interactive: auto-confirm retraining clean if needed, and auto-confirm the publish itself",
    ),
) -> None:
    """Publish a GGUF model to Hugging Face Hub, with a generated model card and attribution manifest.

    This is the hard gate the whole license-tracking pipeline (`bg export`, `bg train cpt|sft`,
    `bg train export-gguf`) exists to enforce: refuses to publish a model trained, even partly, on
    non-open-licensed material. When blocked, offers to retrain on a clean, filtered dataset and
    publish that instead.
    """
    from batterygemma.export.hf_release import attribution_summary, build_model_card, push_gguf_to_hub
    from batterygemma.export.licensing import LICENSE_STATUS_FILENAME, read_license_status, write_license_status
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth
    from batterygemma.train.sft import export_gguf_from_adapter

    status = read_license_status(gguf_dir / LICENSE_STATUS_FILENAME)
    if status is None:
        console.print(
            f"[red]refusing to publish: no {LICENSE_STATUS_FILENAME} found in {gguf_dir}[/red]\n"
            "This GGUF's training provenance is unverified. Re-run `bg train export-gguf` from an "
            "adapter trained under the current license gate, then try again."
        )
        raise typer.Exit(1)

    if not status["shareable"]:
        console.print(
            f"[red]refusing to publish: this model was trained on {len(status['restricted_doc_ids'])} "
            "non-open-licensed source document(s)[/red] and must not be redistributed."
        )
        if not _confirm("Retrain on a clean, filtered dataset now, and publish that instead?", yes):
            console.print(f"nothing published. The existing GGUF remains local-only at {gguf_dir} "
                           "(not shareable) - re-run this command after retraining clean.")
            raise typer.Exit(1)

        adapter_dir = gguf_dir.parent if gguf_dir.name == "gguf" else gguf_dir
        clean_adapter_dir = _retrain_clean(adapter_dir, status, yes=yes)
        clean_status = read_license_status(clean_adapter_dir / LICENSE_STATUS_FILENAME)
        try:
            ensure_unsloth()
        except UnslothInstallError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        cfg = load_config(clean_status["training_provenance"]["config"])
        gguf_dir = export_gguf_from_adapter(cfg["model_name"], cfg["max_seq_length"], clean_adapter_dir,
                                             clean_adapter_dir, quantization_method="q4_k_m",
                                             text_only=cfg.get("text_only", True))
        write_license_status(gguf_dir / LICENSE_STATUS_FILENAME, clean_status)
        status = clean_status
        console.print(f"clean GGUF ready -> {gguf_dir}")

    dataset = Path(status["training_provenance"]["dataset"])
    attribution = attribution_summary(dataset.parent)
    quantization = next((p.stem.rsplit(".", 1)[-1] for p in gguf_dir.glob("*.gguf")), "unknown")
    model_card = build_model_card(
        repo_name=repo_id.split("/")[-1], base_model=load_config(status["training_provenance"]["config"])["model_name"],
        license_id="cc-by-4.0", quantization=quantization,
        training_provenance=status["training_provenance"], attribution=attribution,
    )
    console.print(f"About to publish to [bold]https://huggingface.co/{repo_id}[/bold] "
                  f"({'private' if private else 'public'}) - {len(attribution)} attributed source(s), "
                  f"{sum(1 for _ in gguf_dir.glob('*.gguf'))} GGUF file(s).")
    if not _confirm("Proceed?", yes):
        console.print(f"nothing published. The GGUF remains local-only at {gguf_dir}.")
        raise typer.Exit(1)

    url = push_gguf_to_hub(gguf_dir, repo_id, model_card=model_card, attribution=attribution, private=private)
    console.print(f"published -> {url}\n(local copy remains at {gguf_dir})")


@eval_app.command("build-gold")
def eval_build_gold(
    output: Path = typer.Option(Path("data/eval/gold_v1.jsonl"), help="Where to write the gold JSONL"),
    closed: int = typer.Option(600, help="Max closed Q&A items"),
    open_: int = typer.Option(400, "--open", help="Max open Q&A items"),
    negative: int = typer.Option(300, help="Max negative-detection items"),
    ideation: int = typer.Option(200, help="Max ideation items"),
) -> None:
    """Build the held-out gold benchmark from accepted stage-8 records on eval-split documents."""
    from batterygemma.eval.build_gold import build_gold_set

    targets = {"closed_qa": closed, "open_qa": open_, "negative_detection": negative, "ideation": ideation}
    with get_session() as s:
        counts = build_gold_set(s, output, target_counts=targets)
    console.print(f"gold set written to {output}: {counts} (total {sum(counts.values())})")


@eval_app.command("run")
def eval_run_cmd(
    gold: Path = typer.Argument(..., help="Path to a gold JSONL from `bg eval build-gold`"),
    model: str = typer.Option(..., help="Base model name/path, or a LoRA output dir from `bg train`"),
    output: Path = typer.Option(Path("data/eval/results.json"), help="Where to write per-item results + summary"),
    max_seq_length: int = typer.Option(2048, help="Must match the value used at training time"),
    max_new_tokens: int = typer.Option(512, help="Generation length cap per answer"),
    limit: int | None = typer.Option(None, help="Only evaluate the first N gold items (smoke-testing)"),
) -> None:
    """Generate answers for every gold item with `model` and score them (stage 11)."""
    from batterygemma.eval.metrics import aggregate
    from batterygemma.eval.run_eval import load_gold, run_eval
    from batterygemma.export.unsloth_jsonl import SYSTEM_PROMPT
    from batterygemma.train.environment import UnslothInstallError, ensure_unsloth

    try:
        ensure_unsloth()
    except UnslothInstallError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    import mlx_lm

    mlx_model, tokenizer = FastModel.from_pretrained(model_name=model, max_seq_length=max_seq_length, text_only=True)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def generate_fn(messages: list[dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return mlx_lm.generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)

    gold_items = load_gold(gold)
    if limit:
        gold_items = gold_items[:limit]
    router = build_router()
    with get_session() as s:
        results = run_eval(s, router, gold_items, generate_fn, system_prompt=SYSTEM_PROMPT)

    summary = aggregate(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(json.dumps({"model": model, "summary": summary, "results": results}, indent=2))
    for category, entry in summary.items():
        console.print(f"{category}: {entry}")
    console.print(f"results written to {output}")


def _log_table(entries) -> Table:  # noqa: ANN001 - list[LogEntry], kept loose to avoid importing the type here
    table = Table("id", "ts", "level", "component", "message", "context")
    for e in entries:
        table.add_row(str(e.id), e.ts.strftime("%Y-%m-%d %H:%M:%S"), e.level, e.component,
                      e.message[:120], str(e.context) if e.context else "")
    return table


@logs_app.command("tail")
def logs_tail(
    n: int = typer.Option(50, "-n", help="Number of most recent entries to show"),
    level: str | None = typer.Option(None, help="Filter by level, e.g. WARNING"),
    component: str | None = typer.Option(None, help="Filter by logger name substring, e.g. fetch or router"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Keep polling for new entries (like tail -f)"),
) -> None:
    """Show the most recent log entries, oldest first."""
    import time as _time

    from batterygemma.logs import get_log_session, query_logs

    with get_log_session() as s:
        entries = list(reversed(query_logs(s, level=level, component=component, limit=n)))
    console.print(_log_table(entries))
    if not follow:
        return
    last_id = entries[-1].id if entries else 0
    try:
        while True:
            _time.sleep(1.0)
            with get_log_session() as s:
                new = query_logs(s, level=level, component=component, after_id=last_id, limit=1000)
            if new:
                console.print(_log_table(new))
                last_id = new[-1].id
    except KeyboardInterrupt:
        pass


@logs_app.command("query")
def logs_query(
    level: str | None = typer.Option(None, help="Filter by level, e.g. ERROR"),
    component: str | None = typer.Option(None, help="Filter by logger name substring"),
    contains: str | None = typer.Option(None, help="Filter by a substring in the message"),
    limit: int = typer.Option(100, help="Maximum rows to return"),
) -> None:
    """Search the log, most recent first."""
    from batterygemma.logs import get_log_session, query_logs

    with get_log_session() as s:
        entries = query_logs(s, level=level, component=component, contains=contains, limit=limit)
    console.print(_log_table(entries))
    console.print(f"{len(entries)} entries")


@logs_app.command("stats")
def logs_stats() -> None:
    """Row counts by level and by component."""
    from batterygemma.logs import get_log_session, log_stats

    with get_log_session() as s:
        stats = log_stats(s)
    console.print("by level:", stats["by_level"])
    console.print("by component:", stats["by_component"])


@logs_app.command("clear")
def logs_clear(
    older_than_days: int | None = typer.Option(None, help="Only delete entries older than this many days"),
) -> None:
    """Delete log entries (all of them, unless --older-than-days is given)."""
    from datetime import datetime, timedelta, timezone

    from batterygemma.logs import clear_logs, get_log_session

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days) if older_than_days else None
    with get_log_session() as s:
        n = clear_logs(s, older_than=cutoff)
    console.print(f"deleted {n} log entries")


if __name__ == "__main__":
    app()
