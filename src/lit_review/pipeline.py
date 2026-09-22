"""Top-level orchestration: optimise → seed → expand → rank → export."""

from __future__ import annotations

import uuid

import httpx
import openrouter
from rich import console as csl

from lit_review import (
    client,
    config,
    database,
    expansion,
    graph,
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
) -> str:
    """Full end-to-end run. Returns the run_id.

    Uses LiquidAI/LFM2.5-ColBERT-350M for late-interaction ranking.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    console.rule(f"[bold]lit-review run {run_id}")

    http = httpx.AsyncClient(timeout=60.0)
    db = await database.get_db()
    router = openrouter.OpenRouter(api_key=config.OPENROUTER_API_KEY)

    try:
        await database.create_run(db, run_id, raw_query)
        s2 = client.S2Client(http)

        # Query optimiser
        console.log("Optimising query...")
        opt_query, meta = await query_optimizer.optimize_query(
            raw_query,
            router,
            interactive=interactive_optimizer,
        )
        await database.update_run(
            db,
            run_id,
            query_optimized=opt_query,
        )

        # Seed retrieval
        console.log(f"Fetching up to {config.SEED_LIMIT} seeds...")
        year = meta.get("year_range")
        fos = meta.get("fields_of_study") or []
        fos_str = ",".join(fos) if fos else None
        min_cites = meta.get("min_citation_count")

        seeds = await s2.search(
            opt_query,
            limit=config.SEED_LIMIT,
            year=year,
            fields_of_study=fos_str,
            min_citation_count=min_cites,
        )
        if not seeds:
            console.print("[red]No seeds returned – aborting.[/]")
            await database.update_run(db, run_id, status="failed")
            return run_id

        console.log(f"Got {len(seeds)} seeds")

        # Expansion (priority queue + SPECTER prune)
        dense = ranking.DenseRanker()
        candidates = await expansion.expand_graph(
            s2,
            db,
            seeds,
            opt_query,
            dense_ranker=dense,
        )
        console.log(f"Expansion produced {len(candidates)} candidates")

        # Late-interaction re-ranking
        late = ranking.LateInteractionRanker()  # defaults to LiquidAI/LFM2.5-ColBERT-350M
        placeholders = ",".join("?" * len(candidates)) if candidates else "''"
        async with db.execute(
            f"""
            SELECT paper_id, abstract FROM papers
            WHERE paper_id IN ({placeholders})
              AND abstract IS NOT NULL
            """,
            candidates or [],
        ) as cur:
            rows = await cur.fetchall()

        if rows and late.available:
            ids = [r[0] for r in rows]
            abstracts = [r[1] for r in rows]
            ranked = late.score(opt_query, abstracts, top_k=config.MAX_CORE_PAPERS)
            final_ids = [ids[i] for i, _ in ranked]
            scores = {ids[i]: s for i, s in ranked}
        else:
            final_ids = candidates[: config.MAX_CORE_PAPERS]
            scores = {}

        await database.mark_core(db, final_ids, scores)
        console.log(f"Marked {len(final_ids)} core papers")

        # Graph + Obsidian
        g = await graph.build_graph(db)
        if export_vault:
            await graph.export_obsidian_vault(db, g)

        await database.update_run(
            db,
            run_id,
            status="finished",
        )
        console.rule("[green]Finished[/]")
        return run_id
    finally:
        await http.aclose()
        await db.close()
        if hasattr(router, "aclose"):
            await router.aclose()
        elif hasattr(router, "close"):
            router.close()
