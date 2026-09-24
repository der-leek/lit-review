"""Top-level orchestration: optimise → seed → expand → rank → export."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
from rich import console as csl
from rich import prompt as rprompt

from lit_review import (
    client,
    config,
    database,
    expansion,
    graph,
    llm,
    query_optimizer,
    ranking,
)

console = csl.Console()


async def main(
    raw_query: str,
    *,
    run_id: str | None = None,
    interactive_optimizer: bool = True,
    export_vault: bool = True,
    llm_provider: str | None = None,
    seed_ids: list[str] | None = None,
) -> str:
    """Full end-to-end run. Returns the run_id.

    LLM backend via `llm_provider` or `LIT_REVIEW_LLM_PROVIDER`
    (default: openrouter). Use `apple_fm` for on-device Apple Intelligence.
    Late-interaction ranking uses LiquidAI/LFM2.5-ColBERT-350M.

    When `seed_ids` is provided (from ``--seeds``), the pipeline elicits a rich
    intent only, hydrates those papers, presents them for confirmation, then
    runs expansion and ranking — no automated seed search.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    provider = llm_provider or config.LLM_PROVIDER
    console.rule(f"[bold]lit-review run {run_id}[/]  [dim](llm={provider})[/]")

    http = httpx.AsyncClient(timeout=60.0)
    db = await database.get_db()
    model = llm.get_llm_client(provider)

    try:
        # Resume when the run_id already has pending expansion work.
        existing = await database.get_run(db, run_id)
        resuming = existing is not None and existing.get("status") in (
            "running",
            "failed",
        )
        pending_exists = False
        if resuming:
            async with db.execute(
                "SELECT COUNT(*) FROM queue_state WHERE status = 'pending'"
            ) as cur:
                (pending_exists,) = await cur.fetchone()
            pending_exists = bool(pending_exists)

        s2 = client.S2Client(http)
        late = ranking.LateInteractionRanker()

        if resuming and pending_exists:
            console.log(f"Resuming run {run_id} from persisted queue")
            intent = (existing.get("query_optimized") or "").strip() or raw_query
            seeds: list[dict] = []
        else:
            await database.create_run(db, run_id, raw_query)

            if seed_ids:
                seeds, intent = await _seeds_from_manual_ids(
                    s2,
                    model,
                    raw_query,
                    seed_ids,
                    interactive=interactive_optimizer,
                    late=late,
                )
            else:
                seeds, intent = await _seeds_from_search(
                    s2,
                    model,
                    raw_query,
                    interactive=interactive_optimizer,
                    late=late,
                )

            if not seeds:
                console.print("[red]No seeds available – aborting.[/]")
                await database.update_run(db, run_id, status="failed")
                return run_id

            winning_query = intent  # store the rich intent as the optimized query
            await database.update_run(db, run_id, query_optimized=winning_query)

        # Final gate before the expensive expansion/ranking.
        resume_note = " (resuming)" if resuming and pending_exists else ""
        if interactive_optimizer and not (
            resuming and pending_exists
        ):
            if not rprompt.Confirm.ask(
                f"Proceed with expansion & ranking{resume_note}?",
                default=True,
            ):
                console.print("[yellow]Aborted by user.[/]")
                await database.update_run(db, run_id, status="failed")
                return run_id

        # Concurrent MaxSim expansion (no SPECTER, no max-depth).
        candidates = await expansion.expand_graph(
            s2,
            db,
            seeds,
            intent,
            late_ranker=late,
            resume=resuming and pending_exists,
        )
        console.log(f"Expansion produced {len(candidates)} candidates")

        # Final late-interaction re-ranking over scored candidates.
        placeholders = ",".join("?" * len(candidates)) if candidates else "''"
        async with db.execute(
            f"""
            SELECT paper_id, abstract, title FROM papers
            WHERE paper_id IN ({placeholders})
            """,
            candidates or [],
        ) as cur:
            rows = await cur.fetchall()

        if rows and late.available:
            ids = [r[0] for r in rows]
            texts = [
                (r[1] or "").strip() or (r[2] or "").strip() for r in rows
            ]
            ranked = late.score(intent, texts, top_k=config.MAX_CORE_PAPERS)
            final_ids = [ids[i] for i, _ in ranked]
            scores = {ids[i]: s for i, s in ranked}
        else:
            final_ids = candidates[: config.MAX_CORE_PAPERS]
            scores = {}

        await database.mark_core(db, final_ids, scores)
        console.log(f"Marked {len(final_ids)} core papers")

        g = await graph.build_graph(db)
        if export_vault:
            await graph.export_obsidian_vault(db, g)

        await database.update_run(db, run_id, status="finished")
        console.rule("[green]Finished[/]")
        return run_id
    finally:
        await http.aclose()
        await db.close()
        await model.aclose()


# ---------------------------------------------------------------------------
# Seed acquisition
# ---------------------------------------------------------------------------


async def _seeds_from_manual_ids(
    s2: client.S2Client,
    model: Any,
    raw_query: str,
    seed_ids: list[str],
    *,
    interactive: bool,
    late: ranking.LateInteractionRanker,
) -> tuple[list[dict], str]:
    """Elicit intent only, hydrate user-supplied paper IDs, confirm, return."""
    console.log(
        f"Using {len(seed_ids)} manually supplied seed IDs — eliciting intent only"
    )
    _, meta = await query_optimizer.optimize_query(
        raw_query,
        model,
        interactive=interactive,
        intent_only=True,
    )
    intent = (meta.get("intent") or "").strip() or raw_query

    fields = "paperId,title,abstract,year,citationCount,influentialCitationCount"
    hydrated = await s2.batch_papers(seed_ids, fields=fields, graceful=True)
    # Preserve user order as much as possible.
    by_id = {p.get("paperId"): p for p in (hydrated or []) if p.get("paperId")}
    seeds = [by_id[sid] for sid in seed_ids if sid in by_id]

    if not seeds:
        console.print("[red]Could not hydrate any of the supplied seed IDs.[/]")
        return [], intent

    scores: dict[str, float] = {}
    if late.available:
        texts = [ranking.paper_text_for_scoring(p) for p in seeds]
        ranked = late.score(intent, texts, top_k=len(texts))
        for idx, sc in ranked:
            pid = seeds[idx].get("paperId")
            if pid:
                scores[str(pid)] = sc

    query_optimizer.display_seed_papers(seeds, scores=scores)

    if interactive:
        if not rprompt.Confirm.ask("Use these seeds?", default=True):
            console.print("[yellow]Manual seeds rejected.[/]")
            return [], intent

    return seeds[: config.NUM_SEEDS], intent


async def _seeds_from_search(
    s2: client.S2Client,
    model: Any,
    raw_query: str,
    *,
    interactive: bool,
    late: ranking.LateInteractionRanker,
) -> tuple[list[dict], str]:
    """Intent elicitation → three drafts → top hit each → MaxSim → approve loop."""
    while True:
        console.log("Optimizing query / refining intent...")
        opt_query, meta = await query_optimizer.optimize_query(
            raw_query,
            model,
            interactive=interactive,
            intent_only=False,
        )
        intent = (meta.get("intent") or "").strip() or opt_query
        drafts = meta.get("drafts") or []

        if not drafts:
            console.print("[red]Optimizer returned no drafts.[/]")
            return [], intent

        # One top result per draft (SEED_LIMIT defaults to 1).
        seed_lists = await _fetch_draft_seeds(s2, drafts)
        candidates: list[tuple[dict, str, int]] = []  # paper, query, draft_idx
        for i, (draft, results) in enumerate(zip(drafts, seed_lists, strict=False)):
            if not results:
                continue
            top = results[0]
            candidates.append((top, draft.get("query") or "", i))

        if not candidates:
            console.print("[red]No search results for any draft.[/]")
            if not interactive:
                return [], intent
            if not rprompt.Confirm.ask("Retry intent refinement?", default=True):
                return [], intent
            continue

        # MaxSim each top paper against the rich intent.
        papers = [c[0] for c in candidates]
        texts = [ranking.paper_text_for_scoring(p) for p in papers]
        scores: dict[str, float] = {}
        order: list[int] = list(range(len(papers)))
        if late.available and texts:
            ranked = late.score(intent, texts, top_k=len(texts))
            order = [idx for idx, _ in ranked]
            for idx, sc in ranked:
                pid = papers[idx].get("paperId")
                if pid:
                    scores[str(pid)] = sc

        # Present ordered by MaxSim; keep at most NUM_SEEDS.
        ordered_papers = [papers[i] for i in order][: config.NUM_SEEDS]
        # Winning query = draft that produced the highest-MaxSim paper.
        best_idx = order[0] if order else 0
        winning_query = candidates[best_idx][1] if candidates else opt_query

        query_optimizer.display_seed_papers(
            ordered_papers,
            scores=scores,
            winning_query=winning_query,
        )

        if not interactive:
            return ordered_papers, intent

        console.print()
        console.print("[bold]Seed approval[/]")
        console.print("  [cyan]a[/] — approve these seeds and continue")
        console.print("  [cyan]r[/] — refine intent and re-search")
        console.print("  [cyan]m[/] — replace seeds with manual paper IDs")
        console.print("  [cyan]x[/] — abort")
        choice = rprompt.Prompt.ask(
            "Choice",
            choices=["a", "r", "m", "x"],
            default="a",
        ).lower()

        if choice == "a":
            return ordered_papers, intent
        if choice == "x":
            return [], intent
        if choice == "m":
            raw_ids = rprompt.Prompt.ask(
                "Enter paper IDs (space-delimited)",
                default="",
            )
            ids = [x.strip() for x in raw_ids.split() if x.strip()]
            if not ids:
                continue
            return await _seeds_from_manual_ids(
                s2,
                model,
                raw_query,
                ids,
                interactive=interactive,
                late=late,
            )
        # choice == "r" → loop and refine intent again
        continue


async def _fetch_draft_seeds(
    s2: client.S2Client,
    drafts: list[dict],
) -> list[list[dict]]:
    """Run one `s2.search` per draft (in parallel) and return per-draft seeds."""
    if not drafts:
        return []

    async def search_one(draft: dict) -> list[dict]:
        fos = draft.get("fields_of_study") or []
        fos_str = ",".join(fos) if fos else None
        return await s2.search(
            draft.get("query") or "",
            limit=config.SEED_LIMIT,
            year=draft.get("year_range") or None,
            fields_of_study=fos_str,
            min_citation_count=draft.get("min_citation_count"),
            graceful=True,
        )

    return await asyncio.gather(*(search_one(d) for d in drafts))
