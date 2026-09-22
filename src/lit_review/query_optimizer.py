"""Interactive query optimizer powered by OpenRouter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich import console as csl
from rich.prompt import Confirm, Prompt

from lit_review import config

if TYPE_CHECKING:
    import openrouter as router

console = csl.Console()

SYSTEM_PROMPT = """You are a scientific literature search expert.
Given a natural-language research question, output a single JSON object with these keys:

{
  "optimized_query": "concise, keyword-rich query optimized for
    Semantic Scholar relevance ranking",
  "alternative_queries": ["optional second variant", "optional third variant"],
  "year_range": "2018-2026" or null,
  "fields_of_study": ["Computer Science"] or [],
  "min_citation_count": 5 or null,
  "notes": "one-paragraph explanation of the changes you made"
}

Rules:
- Prefer canonical technical terminology over colloquial phrasing.
- Add discriminative facets that improve precision.
- Keep the optimized_query short enough for Semantic Scholar's ranker.
- Do not invent constraints the user did not imply.
"""


async def optimize_query(
    raw: str,
    openrouter: router.OpenRouter,
    *,
    model: str = config.OPENROUTER_MODEL,
    interactive: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Call an LLM to rewrite the user query.

    Then optionally ask the user to confirm or edit the result.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
    ]

    resp = await openrouter.chat.send_async(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        parsed: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse optimizer JSON:[/] {exc}")
        console.print(content)
        raise

    optimized = parsed.get("optimized_query") or raw
    notes = parsed.get("notes", "")

    if interactive:
        console.print("\n[bold cyan]Proposed optimized query[/]")
        console.print(optimized)
        if notes:
            console.print(f"[dim]{notes}[/]")
        if not Confirm.ask("Accept this query?", default=True):
            optimized = Prompt.ask("Enter your preferred query", default=optimized)
            parsed["optimized_query"] = optimized

    return optimized, parsed
