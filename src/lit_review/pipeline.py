"""Top-level orchestration: optimise → seed → expand → rank → export."""

from __future__ import annotations

import uuid

import httpx
from rich import console as csl

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
) -> str:
    """Full end-to-end run. Returns the run_id.

    LLM backend via `llm_provider` or `LIT_REVIEW_LLM_PROVIDER`
    (default: openrouter). Use `apple_fm` for on-device Apple Intelligence.
    Late-interaction ranking uses LiquidAI/LFM2.5-ColBERT-350M.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    provider = llm_provider or config.LLM_PROVIDER
    console.rule(f"[bold]lit-review run {run_id}[/]  [dim](llm={provider})[/]")

    http = httpx.AsyncClient(timeout=60.0)
    db = await database.get_db()
    model = llm.get_llm_client(provider)

    try:
        await database.create_run(db, run_id, raw_query)
        s2 = client.S2Client(http)

        # Query optimiser (provider-agnostic)
        console.log("Optimising query...")
        opt_query, meta = await query_optimizer.optimize_query(
            raw_query,
            model,
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

        # Prefer HyDE pseudo-abstract for embedding stages when available
        embed_query = (meta.get("hyde_pseudo_abstract") or "").strip() or opt_query

        # Expansion (priority queue + SPECTER prune)
        dense = ranking.DenseRanker()
        candidates = await expansion.expand_graph(
            s2,
            db,
            seeds,
            embed_query,
            dense_ranker=dense,
        )
        console.log(f"Expansion produced {len(candidates)} candidates")

        # Late-interaction re-ranking
        late = ranking.LateInteractionRanker()
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
            ranked = late.score(embed_query, abstracts, top_k=config.MAX_CORE_PAPERS)
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
        await model.aclose()
