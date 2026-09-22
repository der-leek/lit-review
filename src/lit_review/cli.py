"""Typer CLI entrypoint."""

from __future__ import annotations

import asyncio

import typer
from rich import console as console_

from lit_review import __version__
from lit_review import pipeline

app = typer.Typer(
    name="lit-review",
    help="Async literature-review tool (Semantic Scholar + LFM2.5 late-interaction + Obsidian).",
    add_completion=False,
)
console = console_.Console()


@app.callback()
def main() -> None:
    """lit-review – build citation graphs for literature reviews."""


@app.command()
def version() -> None:
    """Print version."""
    console.print(__version__)


@app.command()
def run(
    query: str = typer.Argument(..., help="Natural-language literature-review question"),
    run_id: str | None = typer.Option(None, help="Resume or force a specific run id"),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Skip interactive query confirmation"
    ),
    no_vault: bool = typer.Option(False, "--no-vault", help="Skip Obsidian vault export"),
) -> None:
    """Run a full literature-review pipeline (uses LiquidAI/LFM2.5-ColBERT-350M)."""
    asyncio.run(
        pipeline.main(
            query,
            run_id=run_id,
            interactive_optimizer=not no_interactive,
            export_vault=not no_vault,
        )
    )


if __name__ == "__main__":
    app()
