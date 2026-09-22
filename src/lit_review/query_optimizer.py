"""Interactive query optimizer powered by a pluggable LLMClient.

Tuned for Semantic Scholar relevance search:
- ranking emphasizes title matches, citation count, and recency
- plain-text queries only (no Boolean operators on /paper/search)
- hyphens in terms are harmful; prefer spaces or canonical multi-word phrases
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich import console as csl
from rich import panel, prompt
from rich import table as tbl

if TYPE_CHECKING:
    from lit_review.llm.base import LLMClient

console = csl.Console()

SYSTEM_PROMPT = """You are an expert scientific literature search strategist specializing
 in Semantic Scholar.

Semantic Scholar relevance search facts you MUST respect:
- The query is plain text matched against title, abstract, and author names,
    then re-ranked by a model that strongly prefers title matches, highly cited papers,
    and recent work.
- There is NO Boolean syntax on the relevance endpoint. Do not emit AND/OR/NOT,
    quotes-as-operators, or field prefixes.
- Hyphenated compounds often fail to match; prefer space-separated forms or
    the most common unhyphenated scholarly phrasing (e.g. "self attention" or
    "self-attention" only if that is the dominant form in titles).
- Short, precise, title-like phrases outperform long conversational questions.
- Overly broad queries waste downstream citation-graph expansion; overly narrow queries
    miss seminal work.

Your job: turn a messy natural-language research question into a high-precision
search plan.

Output a single JSON object with exactly these keys:

{
  "intent": "one sentence stating the user's true information need in research terms",
  "core_concepts": ["canonical term 1", "canonical term 2", "..."],
  "disambiguation": "null if unambiguous, otherwise a short note of competing senses",
  "ambiguity_flags": ["list of residual ambiguities that could change results"],
  "clarifying_questions": ["0-3 questions the user should answer if
    ambiguity_flags is non-empty"],
  "optimized_query": "the single best plain-text query for Semantic Scholar
    (title-oriented, 4-12 content words typical)",
  "alternative_queries": ["2-3 diverse variants: broader, narrower, or synonym-shifted"],
  "hyde_pseudo_abstract": "2-4 sentence hypothetical abstract of an ideal matching paper
    (for dense/late-interaction stages; not used as the S2 keyword query)",
  "year_range": "YYYY-YYYY or YYYY- or -YYYY or null",
  "fields_of_study": ["Computer Science"] or [],
  "min_citation_count": null or integer,
  "study_types": ["method", "survey", "empirical", "theory"] or [],
  "confidence": 0.0-1.0,
  "notes": "brief rationale for the rewrite and any constraints you applied"
}

Rules for optimized_query and alternatives:
1. Use canonical scholarly terminology the literature actually uses in titles.
2. Prefer noun phrases that would appear in a paper title over full questions.
3. Include 1-2 discriminative facets (task, method family, domain, scale) when
    the user implies them.
4. Do NOT invent year, field, or citation constraints the user did not imply;
    leave those null/empty.
5. If the query is multi-intent, pick the dominant intent for optimized_query and
    put secondary intents in alternative_queries.
6. confidence < 0.6 when major ambiguity remains; populate clarifying_questions
    in that case.
7. hyde_pseudo_abstract should read like a real paper abstract fragment,
    not a search string.
"""


def _parse_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse optimizer JSON:[/] {exc}")
        console.print(content)
        raise


def _display_plan(raw: str, parsed: dict[str, Any]) -> None:
    console.print()
    console.print(panel.Panel(raw, title="[bold]Original query[/]", border_style="dim"))

    intent = parsed.get("intent") or ""
    if intent:
        console.print(f"[bold]Intent:[/] {intent}")

    concepts = parsed.get("core_concepts") or []
    if concepts:
        console.print(f"[bold]Core concepts:[/] {', '.join(concepts)}")

    disamb = parsed.get("disambiguation")
    if disamb and disamb != "null":
        console.print(f"[yellow]Disambiguation:[/] {disamb}")

    flags = parsed.get("ambiguity_flags") or []
    if flags:
        console.print("[yellow]Ambiguity flags:[/]")
        for f in flags:
            console.print(f"  • {f}")

    questions = parsed.get("clarifying_questions") or []
    if questions:
        console.print("[bold yellow]Suggested clarifying questions:[/]")
        for i, q in enumerate(questions, 1):
            console.print(f"  {i}. {q}")

    table = tbl.Table(title="Search queries", show_header=True, header_style="bold cyan")
    table.add_column("Role", style="cyan", width=14)
    table.add_column("Query")
    table.add_row("primary", str(parsed.get("optimized_query") or ""))
    for i, alt in enumerate(parsed.get("alternative_queries") or [], 1):
        table.add_row(f"alt {i}", str(alt))
    console.print(table)

    constraints = []
    if parsed.get("year_range"):
        constraints.append(f"year={parsed['year_range']}")
    if parsed.get("fields_of_study"):
        constraints.append(f"fields={parsed['fields_of_study']}")
    if parsed.get("min_citation_count") is not None:
        constraints.append(f"min_cites={parsed['min_citation_count']}")
    if parsed.get("study_types"):
        constraints.append(f"types={parsed['study_types']}")
    if constraints:
        console.print(f"[bold]Filters:[/] {', '.join(constraints)}")

    conf = parsed.get("confidence")
    if conf is not None:
        style = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
        console.print(f"[bold]Confidence:[/] [{style}]{conf:.2f}[/{style}]")

    notes = parsed.get("notes") or ""
    if notes:
        console.print(f"[dim]{notes}[/]")

    hyde = parsed.get("hyde_pseudo_abstract") or ""
    if hyde:
        console.print(
            panel.Panel(
                hyde,
                title="[dim]HyDE pseudo-abstract (for embedding stages)[/]",
                border_style="dim",
            )
        )


def _pick_query(parsed: dict[str, Any]) -> str:
    primary = parsed.get("optimized_query") or ""
    alts: list[str] = list(parsed.get("alternative_queries") or [])

    console.print()
    if prompt.Confirm.ask("Accept primary optimized query?", default=True):
        return primary

    if alts:
        console.print("Alternatives:")
        for i, a in enumerate(alts, 1):
            console.print(f"  [{i}] {a}")
        console.print("  [0] type a custom query")
        choice = prompt.Prompt.ask(
            "Choose",
            choices=[str(i) for i in range(len(alts) + 1)],
            default="1" if alts else "0",
        )
        idx = int(choice)
        if idx == 0:
            return prompt.Prompt.ask("Custom query", default=primary)
        return alts[idx - 1]

    return prompt.Prompt.ask("Enter preferred query", default=primary)


async def optimize_query(
    raw: str,
    llm: LLMClient,
    *,
    interactive: bool = True,
    max_clarify_rounds: int = 1,
) -> tuple[str, dict[str, Any]]:
    """Rewrite a natural-language literature question into a Semantic Scholar plan.

    `llm` is any object satisfying `LLMClient`.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
    ]

    content = await llm.complete(messages, response_format="json")
    parsed = _parse_response(content)

    if interactive and max_clarify_rounds > 0:
        confidence_threshold = 0.6
        conf = float(parsed.get("confidence") or 1.0)
        questions = parsed.get("clarifying_questions") or []
        if conf < confidence_threshold and questions:
            console.print(
                "\n[bold yellow]Low confidence — a short clarification "
                "will improve results.[/]"
            )
            _display_plan(raw, parsed)
            answers: list[str] = []
            for q in questions[:3]:
                ans = prompt.Prompt.ask(q, default="")
                if ans.strip():
                    answers.append(f"Q: {q}\nA: {ans.strip()}")
            if answers:
                followup = (
                    "The user answered clarifying questions. "
                    "Produce an updated JSON plan.\n\n" + "\n\n".join(answers)
                )
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append({"role": "user", "content": followup})
                content2 = await llm.complete(messages, response_format="json")
                parsed = _parse_response(content2)

    if interactive:
        _display_plan(raw, parsed)
        optimized = _pick_query(parsed)
        parsed["optimized_query"] = optimized
    else:
        optimized = parsed.get("optimized_query") or raw

    parsed.setdefault("year_range", None)
    parsed.setdefault("fields_of_study", [])
    parsed.setdefault("min_citation_count", None)
    parsed.setdefault("hyde_pseudo_abstract", "")
    parsed.setdefault("alternative_queries", [])

    return optimized, parsed
