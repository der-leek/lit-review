"""Typer CLI entrypoint."""

from __future__ import annotations

import asyncio

import typer
from rich import console as console_

from lit_review import __version__, pipeline

app = typer.Typer(
    name="lit-review",
    help=(
        "Async literature-review tool (Semantic Scholar + "
        "LFM2.5 late-interaction + Obsidian)."
    ),
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
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="LLM provider: openrouter (default) or apple_fm",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help=(
            "Space-delimited Semantic Scholar paper IDs to use as the three "
            "initial seeds. When set, the LLM only elicits a rich intent "
            "(no automated search); the supplied papers are hydrated, shown "
            "for confirmation, then expanded and ranked."
        ),
    ),
) -> None:
    """Run a full literature-review pipeline (uses LiquidAI/LFM2.5-ColBERT-350M)."""
    seed_ids: list[str] | None = None
    if seeds is not None:
        seed_ids = [s.strip() for s in seeds.split() if s.strip()]
        if not seed_ids:
            console.print("[red]--seeds was provided but no paper IDs were parsed.[/]")
            raise typer.Exit(code=1)

    asyncio.run(
        pipeline.main(
            query,
            run_id=run_id,
            interactive_optimizer=not no_interactive,
            export_vault=not no_vault,
            llm_provider=llm,
            seed_ids=seed_ids,
        )
    )


if __name__ == "__main__":
    app()
