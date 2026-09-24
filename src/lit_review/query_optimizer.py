"""Interactive query optimizer powered by a pluggable LLMClient.

Tuned for Semantic Scholar relevance search:
- ranking emphasizes title matches, citation count, and recency
- plain-text queries only (no Boolean operators on /paper/search)
- hyphens in terms are harmful; prefer spaces or canonical multi-word phrases

The optimizer always drives a clarification dialogue. Ambiguity flags shape the
questions the LLM asks so the resulting `intent` is detailed and rich in
semantic information before any seed search runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich import console as csl
from rich import panel, prompt
from rich import table as tbl

if TYPE_CHECKING:
    from lit_review.llm import base

console = csl.Console()

# Draft field constraints shared by the schema and the optimizer: the first
# query is the default seed query; the pipeline ranks the three drafts by
# intent MaxSim and picks a winner.
SEARCH_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "core_concepts",
        "disambiguation",
        "ambiguity_flags",
        "clarifying_questions",
        "drafts",
        "study_types",
        "confidence",
        "notes",
    ],
    "properties": {
        "intent": {
            "type": "string",
            "description": (
                "A detailed, multi-sentence statement of the user's true "
                "information need in research terms. Capture task, method, "
                "domain, constraints, and what would make a paper relevant. "
                "Used as the embedding query for seed MaxSim and for continuous "
                "late-interaction scoring during graph expansion."
            ),
        },
        "core_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Canonical scholarly terms",
        },
        "disambiguation": {
            "type": ["string", "null"],
            "description": "null if unambiguous, else a competing-senses note",
        },
        "ambiguity_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Residual ambiguities that could change results. These flags "
                "drive the clarifying questions posed to the user."
            ),
        },
        "clarifying_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "1-5 concrete questions derived from ambiguity_flags that will "
                "enrich the intent. Always propose questions that extract more "
                "semantic detail even when confidence is high."
            ),
        },
        "drafts": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "Three diverse Semantic Scholar query drafts: broader, "
                "narrower, or synonym-shifted. Queries must differ."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "query",
                    "year_range",
                    "fields_of_study",
                    "min_citation_count",
                ],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Plain-text Semantic Scholar query, title-oriented, "
                            "4-12 content words typical"
                        ),
                    },
                    "year_range": {
                        "type": ["string", "null"],
                        "description": "YYYY-YYYY, YYYY-, -YYYY, or null",
                    },
                    "fields_of_study": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Usually ["Computer Science"] or []',
                    },
                    "min_citation_count": {
                        "type": ["integer", "null"],
                        "description": "null or integer",
                    },
                },
            },
        },
        "study_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": '["method","survey","empirical","theory"] or []',
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "0.0-1.0; below 0.6 only when major ambiguity remains",
        },
        "notes": {
            "type": "string",
            "description": "Brief rationale for the drafts and constraints",
        },
    },
}

_SYSTEM_PROMPT = """You are an expert scientific literature search strategist specializing
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
search plan with a rich semantic intent.

Your output follows the JSON Schema provided with the request. The schema lists
the exact fields and their types; respect it exactly.

Strategy rules:
1. Use canonical scholarly terminology the literature actually uses in titles.
2. Prefer noun phrases that would appear in a paper title over full questions.
3. Include 1-2 discriminative facets (task, method family, domain, scale) when
    the user implies them.
4. Do NOT invent year, field, or citation constraints the user did not imply;
    leave those null/empty.
5. Provide three drafts that genuinely differ from each other: one broader,
    one narrower, and one synonym-shifted relative to the dominant sense of the
    user's question. The three queries must be distinct strings.
6. If the query is multi-intent, pick the dominant intent for `intent` and shape
    the drafts around it.
7. Always populate `ambiguity_flags` and `clarifying_questions`. The questions
    must be driven by the flags and should dig for: specific methods, datasets,
    application domains, evaluation criteria, time period, and what would make a
    paper highly relevant vs only tangentially related. Even when confidence is
    high, ask 1-3 questions that deepen the semantic intent.
8. The `intent` field must be detailed (several sentences). It is the primary
    signal for late-interaction MaxSim scoring of paper abstracts, so capture
    nuance, exclusions, and success criteria — not just a restatement of the
    original query.
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
        console.print(panel.Panel(intent, title="[bold]Intent[/]", border_style="cyan"))

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
        console.print("[bold yellow]Clarifying questions:[/]")
        for i, q in enumerate(questions, 1):
            console.print(f"  {i}. {q}")

    drafts = parsed.get("drafts") or []
    if drafts:
        table = tbl.Table(
            title="Search drafts", show_header=True, header_style="bold cyan"
        )
        table.add_column("#", style="cyan", width=3)
        table.add_column("Query")
        table.add_column("Filters")
        for i, draft in enumerate(drafts, 1):
            filters: list[str] = []
            if draft.get("year_range"):
                filters.append(f"year={draft['year_range']}")
            if draft.get("fields_of_study"):
                filters.append(f"fields={','.join(draft['fields_of_study'])}")
            if draft.get("min_citation_count") is not None:
                filters.append(f"min_cites={draft['min_citation_count']}")
            table.add_row(str(i), str(draft.get("query") or ""), "; ".join(filters))
        console.print(table)

    conf = parsed.get("confidence")
    if conf is not None:
        style = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
        console.print(f"[bold]Confidence:[/] [{style}]{conf:.2f}[/{style}]")

    notes = parsed.get("notes") or ""
    if notes:
        console.print(f"[dim]{notes}[/]")


def display_seed_papers(
    seeds: list[dict[str, Any]],
    *,
    scores: dict[str, float] | None = None,
    winning_query: str | None = None,
) -> None:
    """Render seed papers (title, year, cites, abstract) in readable panels."""
    console.print()
    if winning_query:
        console.print(
            panel.Panel(
                winning_query,
                title="[bold]Winning search query[/]",
                border_style="green",
            )
        )

    scores = scores or {}
    for i, paper in enumerate(seeds, 1):
        pid = paper.get("paperId") or paper.get("paper_id") or "?"
        title = paper.get("title") or "(no title)"
        year = paper.get("year") or "?"
        cites = paper.get("citationCount", paper.get("citation_count", "?"))
        infl = paper.get(
            "influentialCitationCount",
            paper.get("influential_citation_count", "?"),
        )
        abstract = (paper.get("abstract") or "").strip() or (
            "[no abstract — scored by title]"
        )
        maxsim = scores.get(str(pid))
        header = f"#{i}  {title}"
        meta_parts = [
            f"id={pid}",
            f"year={year}",
            f"cites={cites}",
            f"influential={infl}",
        ]
        if maxsim is not None:
            meta_parts.append(f"MaxSim={maxsim:.4f}")
        body = "\n".join([" · ".join(meta_parts), "", abstract])
        console.print(panel.Panel(body, title=header, border_style="blue"))


def _normalize_drafts(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Return up to 3 usable drafts, repeating the best when fewer remain.

    Drops empty query strings; if fewer than three usable drafts remain,
    repeats the best available query and its filters.
    """
    raw = parsed.get("drafts")
    if not isinstance(raw, list):
        raw = []

    def _absent(value: Any) -> bool:
        """Whether a draft value means "unset" (None, empty, or string null)."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip() or value.strip().lower() == "null"
        return False

    usable = [
        {
            "query": str(draft.get("query") or "").strip(),
            "year_range": None
            if _absent(draft.get("year_range"))
            else (draft.get("year_range")),
            "fields_of_study": draft.get("fields_of_study") or [],
            # Apple-guided generation emits `-1`/`0` as its "no minimum"
            # sentinel (it cannot emit null); normalize those to absent so
            # the S2 client never sends `minCitationCount` with a bad value.
            "min_citation_count": (
                draft.get("min_citation_count")
                if draft.get("min_citation_count") not in (None, -1, 0)
                else None
            ),
        }
        for draft in raw
        if isinstance(draft, dict) and str(draft.get("query") or "").strip()
    ]
    if not usable:
        return []
    while len(usable) < 3:
        usable.append(dict(usable[0]))
    return usable[:3]


def _completion_kwargs(llm: base.LLMClient) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"response_format": "json"}
    if getattr(llm, "supports_json_schema", True):
        kwargs["json_schema"] = SEARCH_PLAN_SCHEMA
    return kwargs


def _initial_messages(raw: str, llm: base.LLMClient) -> list[dict[str, str]]:
    if getattr(llm, "supports_json_schema", True):
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ]
    return [
        {
            "role": "system",
            "content": (
                _SYSTEM_PROMPT
                + "\n\nReturn only JSON matching this schema:\n"
                + json.dumps(SEARCH_PLAN_SCHEMA)
            ),
        },
        {"role": "user", "content": raw},
    ]


async def _ask_clarifying_questions(
    parsed: dict[str, Any],
) -> list[str]:
    """Prompt the user for answers to the plan's clarifying questions."""
    questions = list(parsed.get("clarifying_questions") or [])
    flags = list(parsed.get("ambiguity_flags") or [])
    if not questions and flags:
        questions = [f"Can you clarify: {flag}?" for flag in flags[:3]]

    if not questions:
        # Always extract at least some additional signal.
        free = prompt.Prompt.ask(
            "Any extra detail that would make a paper highly relevant "
            "(methods, datasets, exclusions)?",
            default="",
        )
        return [f"Additional detail: {free.strip()}"] if free.strip() else []

    console.print(
        "\n[bold yellow]Clarifying your intent — answer any that help; "
        "press Enter to skip a question.[/]"
    )
    if flags:
        console.print("[dim]Driven by ambiguity flags:[/]")
        for f in flags:
            console.print(f"  [dim]• {f}[/]")

    answers: list[str] = []
    for q in questions[:5]:
        ans = prompt.Prompt.ask(q, default="")
        if ans.strip():
            answers.append(f"Q: {q}\nA: {ans.strip()}")
    return answers


async def _refine_plan(
    llm: base.LLMClient,
    messages: list[dict[str, str]],
    parsed: dict[str, Any],
    user_payload: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Send a follow-up to the LLM and return updated messages + parsed plan."""
    kwargs = _completion_kwargs(llm)
    messages = list(messages)
    messages.append({"role": "assistant", "content": json.dumps(parsed)})
    messages.append({"role": "user", "content": user_payload})
    content = await llm.complete(messages, **kwargs)
    updated = _parse_response(content)
    return messages, updated


async def optimize_query(
    raw: str,
    llm: base.LLMClient,
    *,
    interactive: bool = True,
    intent_only: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Rewrite a natural-language literature question into a search plan.

    Always runs a clarification loop when `interactive` is True. Ambiguity
    flags drive the questions. The loop continues until the user explicitly
    approves the intent (and, when not `intent_only`, is ready for seed search).

    `intent_only` is used with manually supplied seeds: elicit a rich intent
    without requiring draft approval for search.

    Returns `(query, parsed)` where `query` is the first draft's query (or the
    raw query when drafts are empty) and `parsed["drafts"]` / `parsed["intent"]`
    feed the pipeline seed selection.
    """
    messages = _initial_messages(raw, llm)
    kwargs = _completion_kwargs(llm)

    content = await llm.complete(messages, **kwargs)
    parsed = _parse_response(content)

    if not interactive:
        parsed["drafts"] = _normalize_drafts(parsed)
        parsed.setdefault("intent", "")
        drafts = parsed["drafts"]
        optimized = drafts[0]["query"] if drafts else raw
        return optimized, parsed

    while True:
        _display_plan(raw, parsed)

        answers = await _ask_clarifying_questions(parsed)
        if answers:
            followup = (
                "The user answered clarifying questions. Incorporate their "
                "answers into a richer `intent` and updated drafts. Keep "
                "ambiguity_flags / clarifying_questions for any remaining gaps.\n\n"
                + "\n\n".join(answers)
            )
            messages, parsed = await _refine_plan(llm, messages, parsed, followup)
            _display_plan(raw, parsed)

        console.print()
        console.print("[bold]Intent consultation[/]")
        if intent_only:
            console.print("  [cyan]a[/] — approve intent")
        else:
            console.print("  [cyan]a[/] — approve intent and search for seeds")
        console.print("  [cyan]r[/] — refine further (free-form notes to the LLM)")
        console.print("  [cyan]q[/] — ask the suggested clarifying questions again")
        choice = prompt.Prompt.ask(
            "Choice",
            choices=["a", "r", "q"],
            default="a",
        ).lower()

        if choice == "a":
            break
        if choice == "q":
            continue
        if choice == "r":
            notes = prompt.Prompt.ask(
                "Describe what to change or add to the intent",
                default="",
            )
            if not notes.strip():
                continue
            followup = (
                "The user wants the intent refined further. Apply these notes "
                "and produce an updated JSON plan with a richer intent and "
                "fresh drafts.\n\nUser notes:\n" + notes.strip()
            )
            messages, parsed = await _refine_plan(llm, messages, parsed, followup)

    parsed["drafts"] = _normalize_drafts(parsed)
    parsed.setdefault("intent", "")
    drafts = parsed["drafts"]
    optimized = drafts[0]["query"] if drafts else raw
    return optimized, parsed
