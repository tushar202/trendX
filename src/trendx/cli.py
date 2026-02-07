from __future__ import annotations

import asyncio

import typer

from trendx.pipeline import run_backfill, run_pipeline
from trendx.utils.logging import get_logger

app = typer.Typer(help="TrendX CLI")
logger = get_logger(__name__)


@app.command()
def ingest(config: str = "configs/default.yaml") -> None:
    """Fetch and normalize sources (placeholder)."""
    typer.echo("ingest is a placeholder and currently runs the full pipeline")
    asyncio.run(run_pipeline(config))


@app.command()
def analyze(config: str = "configs/default.yaml") -> None:
    """Dedup, cluster, and detect trends (placeholder)."""
    typer.echo("analyze is a placeholder and currently runs the full pipeline")
    asyncio.run(run_pipeline(config))


@app.command()
def report(config: str = "configs/default.yaml") -> None:
    """Generate MD + JSON reports (placeholder)."""
    typer.echo("report is a placeholder and currently runs the full pipeline")
    asyncio.run(run_pipeline(config))


@app.command()
def run(config: str = "configs/default.yaml") -> None:
    """Run the full weekly pipeline."""
    asyncio.run(run_pipeline(config))


@app.command()
def backfill(weeks: int = 4, config: str = "configs/default.yaml") -> None:
    """Rebuild reports for the last N weeks."""
    asyncio.run(run_backfill(config, weeks))


@app.command()
def agent(config: str = "configs/default.yaml", week: str | None = None) -> None:
    """(Experimental) Orchestrator-driven execution."""
    from trendx.orchestrator import run_agent

    asyncio.run(run_agent(config, week))
