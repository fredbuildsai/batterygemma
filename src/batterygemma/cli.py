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
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
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


if __name__ == "__main__":
    app()
