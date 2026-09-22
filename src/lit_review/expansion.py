"""Priority-queue driven citation/reference expansion."""

from __future__ import annotations

import dataclasses
import heapq
import math
from typing import TYPE_CHECKING

from rich import console as csl

from lit_review import client, config, database, ranking

if TYPE_CHECKING:
    import aiosqlite

console = csl.Console()


@dataclasses.dataclass(order=True)
class QueueItem:
    priority: float
    paper_id: str = dataclasses.field(compare=False)
    depth: int = dataclasses.field(compare=False)


async def expand_graph(
    s2: client.S2Client,
    db: aiosqlite.Connection,
    seed_papers: list[dict],
    query_text: str,
    *,
    dense_ranker: ranking.DenseRanker | None = None,
    max_total: int = config.MAX_TOTAL_RETRIEVED,
    max_depth: int = config.MAX_DEPTH,
) -> list[str]:
    """Expansion guided by a priority queue ordered by (dense relevance x influence).

    Returns candidate paper IDs that survived dense pruning
    (ready for late-interaction scoring with LFM2.5).
    """
    dense_ranker = dense_ranker or ranking.DenseRanker()
    query_emb = dense_ranker.encode_query(query_text)

    priority_queue: list[QueueItem] = []
    visited: set[str] = set()
    core_candidates: list[str] = []

    for rank, paper in enumerate(seed_papers):
        paper_id = paper.get("paperId")
        if not paper_id:
            continue

        influence = paper.get("influentialCitationCount", 0)
        priority = -((len(seed_papers) - rank) * math.log1p(influence + 1))
        heapq.heappush(priority_queue, QueueItem(priority, paper_id, depth=0))
        visited.add(paper_id)

        await database.save_queue_item(db, paper_id, priority, 0, "pending")

    await database.upsert_papers(db, seed_papers, depth=0)
    total_retrieved = len(seed_papers)

    while priority_queue and total_retrieved < max_total:
        item = heapq.heappop(priority_queue)
        if item.depth >= max_depth:
            continue

        await database.save_queue_item(
            db,
            item.paper_id,
            item.priority,
            item.depth,
            "expanded",
        )

        cites = await s2.citations(item.paper_id, limit=config.CITATIONS_PER_PAPER)
        refs = await s2.references(item.paper_id, limit=config.REFS_PER_PAPER)

        new_ids: list[str] = []
        for c in cites:
            cp = c.get("citingPaper") or {}
            paper_id = cp.get("paperId")
            if paper_id and paper_id not in visited:
                visited.add(paper_id)
                new_ids.append(paper_id)
                await database.add_edge(
                    db,
                    src_id=paper_id,
                    dst_id=item.paper_id,
                    edge_type="cites",
                    is_influential=bool(c.get("isInfluential")),
                )

        for r in refs:
            cp = r.get("citedPaper") or {}
            paper_id = cp.get("paperId")
            if paper_id and paper_id not in visited:
                visited.add(paper_id)
                new_ids.append(paper_id)
                await database.add_edge(
                    db,
                    src_id=item.paper_id,
                    dst_id=paper_id,
                    edge_type="referenced_by",
                    is_influential=bool(r.get("isInfluential")),
                )
        await db.commit()

        if not new_ids:
            continue

        fields = (
            "paperId,title,abstract,year,citationCount,"
            "influentialCitationCount,embedding.specter_v2"
        )
        hydrated = await s2.batch_papers(new_ids, fields=fields)
        total_retrieved += len(hydrated)
        await database.upsert_papers(db, hydrated, depth=item.depth + 1)

        abstracts = []
        infl_counts = []
        valid_papers = []
        for paper in hydrated:
            if not paper.get("abstract") or not paper.get("paperId"):
                continue
            abstracts.append(paper["abstract"])
            infl_counts.append(paper.get("influentialCitationCount", 0))
            valid_papers.append(paper)

        if not abstracts:
            continue

        paper_embeddings = dense_ranker.encode_papers(abstracts)
        dense_scores = dense_ranker.score(query_emb, paper_embeddings)
        combined = ranking.combine_dense_and_influence(dense_scores, infl_counts)

        order = combined.argsort()[::-1][: config.SPECTER_TOP_K]
        for idx in order:
            paper = valid_papers[idx]
            paper_id = paper["paperId"]

            score = float(combined[idx])
            heapq.heappush(priority_queue, QueueItem(-score, paper_id, item.depth + 1))
            core_candidates.append(paper_id)

            await database.save_queue_item(
                db, paper_id, -score, item.depth + 1, "pending"
            )

        console.log(
            f"Retrieved {total_retrieved:>6} | queue {len(priority_queue):>4} | "
            f"candidates {len(core_candidates):>5} | depth {item.depth}"
        )

    return core_candidates
