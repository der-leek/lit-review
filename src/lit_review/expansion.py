"""Concurrent MaxSim-driven citation/reference expansion.

Three cooperating async roles share a priority queue and a score buffer:

1. **Expander** – pops the highest-priority paper, fetches its citations and
   references, hydrates abstracts via the batch endpoint, and pushes candidates
   into the score buffer.
2. **Scorer** – continuously drains the buffer, scores each paper's abstract
   (or title when the abstract is missing) against the user's detailed intent
   with the late-interaction MaxSim model, and publishes scored items.
3. **Prioritizer** – turns MaxSim scores into queue priorities
   (`MaxSim * max(1, influentialCitationCount)`) and pushes new nodes onto the
   priority queue for further expansion.

Late-interaction encoding runs concurrently with network I/O so batch-query
latency hides most of the encoder cost. SPECTER is not used; relevance is
query↔content MaxSim. Expansion stops when the priority queue and score buffer
are empty or when ``MAX_TOTAL_RETRIEVED`` papers have been scored.
"""

from __future__ import annotations

import asyncio
import dataclasses
import heapq
from typing import TYPE_CHECKING

from rich import console as csl

from lit_review import client, config, database, ranking

if TYPE_CHECKING:
    import aiosqlite

console = csl.Console()

# Sentinel objects for shutting down worker tasks.
_STOP = object()


@dataclasses.dataclass(order=True)
class QueueItem:
    priority: float
    paper_id: str = dataclasses.field(compare=False)


@dataclasses.dataclass
class ScoredPaper:
    paper: dict
    maxsim: float
    priority: float


async def expand_graph(
    s2: client.S2Client,
    db: aiosqlite.Connection,
    seed_papers: list[dict],
    query_text: str,
    *,
    late_ranker: ranking.LateInteractionRanker | None = None,
    max_total: int = config.MAX_TOTAL_RETRIEVED,
    resume: bool = False,
) -> list[str]:
    """Expand the citation graph guided by continuous MaxSim relevance.

    Returns candidate paper IDs that were scored (ready for final core-set
    ranking). When ``resume=True``, pending queue items are rehydrated from
    SQLite so an interrupted run continues where it left off.
    """
    late = late_ranker or ranking.LateInteractionRanker()

    priority_queue: list[QueueItem] = []
    queue_lock = asyncio.Lock()
    score_buffer: asyncio.Queue[dict | object] = asyncio.Queue(
        maxsize=config.SCORE_BUFFER_MAX
    )
    scored_out: asyncio.Queue[ScoredPaper | object] = asyncio.Queue()

    visited: set[str] = set()
    core_candidates: list[str] = []
    total_scored = 0
    stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Resume or seed the priority queue
    # ------------------------------------------------------------------
    if resume:
        pending = await database.load_queue(db)
        for row in pending:
            heapq.heappush(
                priority_queue,
                QueueItem(float(row[0]), str(row[1])),
            )
            core_candidates.append(str(row[1]))

        async with db.execute(
            "SELECT DISTINCT paper_id FROM queue_state WHERE status = 'expanded'"
        ) as cur:
            rows = await cur.fetchall()
            core_candidates.extend(str(r[0]) for r in rows)

        async with db.execute("SELECT DISTINCT paper_id FROM papers") as cur:
            rows = await cur.fetchall()
            visited.update(str(r[0]) for r in rows)

        async with db.execute("SELECT COUNT(*) FROM papers") as cur:
            (existing_count,) = await cur.fetchone()
        total_scored = int(existing_count)
    else:
        for paper in seed_papers:
            paper_id = paper.get("paperId")
            if not paper_id or paper_id in visited:
                continue
            text = ranking.paper_text_for_scoring(paper)
            maxsim = late.score_one(query_text, text) if late.available else 1.0
            infl = paper.get("influentialCitationCount", 0)
            pri = ranking.maxsim_priority(maxsim, infl)
            heapq.heappush(priority_queue, QueueItem(-pri, paper_id))
            visited.add(paper_id)
            core_candidates.append(paper_id)
            await database.save_queue_item(db, paper_id, -pri, 0, "pending")

        await database.upsert_papers(db, seed_papers, depth=0)
        total_scored = len(seed_papers)

    # ------------------------------------------------------------------
    # Worker: score buffer → scored_out
    # ------------------------------------------------------------------
    async def scorer_worker() -> None:
        batch: list[dict] = []
        batch_deadline = 0.0

        async def flush() -> None:
            nonlocal batch
            if not batch:
                return
            texts = [ranking.paper_text_for_scoring(p) for p in batch]
            if late.available:
                ranked = late.score(query_text, texts, top_k=len(texts))
                score_map = {idx: sc for idx, sc in ranked}
            else:
                score_map = {i: 0.0 for i in range(len(batch))}

            for i, paper in enumerate(batch):
                maxsim = float(score_map.get(i, 0.0))
                infl = paper.get("influentialCitationCount", 0)
                pri = ranking.maxsim_priority(maxsim, infl)
                await scored_out.put(ScoredPaper(paper=paper, maxsim=maxsim, priority=pri))
            batch = []

        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(score_buffer.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if batch:
                    await flush()
                continue

            if item is _STOP:
                await flush()
                await scored_out.put(_STOP)
                return

            assert isinstance(item, dict)
            batch.append(item)
            # Small batches keep the priority queue fed without waiting for a
            # full encode window; network latency still dominates.
            if len(batch) >= 8:
                await flush()

    # ------------------------------------------------------------------
    # Worker: scored_out → priority queue
    # ------------------------------------------------------------------
    async def prioritizer_worker() -> None:
        nonlocal total_scored
        while not stop_event.is_set():
            try:
                item = await asyncio.wait_for(scored_out.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if item is _STOP:
                return

            assert isinstance(item, ScoredPaper)
            paper = item.paper
            paper_id = paper.get("paperId")
            if not paper_id:
                continue

            total_scored += 1
            core_candidates.append(paper_id)
            await database.upsert_papers(db, [paper], depth=0)
            await database.save_queue_item(
                db, paper_id, -item.priority, 0, "pending"
            )

            async with queue_lock:
                heapq.heappush(
                    priority_queue, QueueItem(-item.priority, paper_id)
                )

            if total_scored >= max_total:
                stop_event.set()
                return

    # ------------------------------------------------------------------
    # Worker: priority queue → expand → score buffer
    # ------------------------------------------------------------------
    async def expander_worker() -> None:
        idle_rounds = 0
        while not stop_event.is_set() and total_scored < max_total:
            item: QueueItem | None = None
            async with queue_lock:
                if priority_queue:
                    item = heapq.heappop(priority_queue)

            if item is None:
                idle_rounds += 1
                # Exit when nothing is left to expand and the score pipeline
                # has drained (or has been idle long enough after seeds).
                if (
                    score_buffer.empty()
                    and scored_out.empty()
                    and idle_rounds > 50
                ):
                    stop_event.set()
                    break
                await asyncio.sleep(0.1)
                continue

            idle_rounds = 0
            await database.save_queue_item(
                db, item.paper_id, item.priority, 0, "expanded"
            )

            cites = await s2.citations(
                item.paper_id,
                limit=config.CITATIONS_PER_PAPER,
                graceful=True,
            )
            refs = await s2.references(
                item.paper_id,
                limit=config.REFS_PER_PAPER,
                graceful=True,
            )

            new_ids: list[str] = []
            for c in cites or []:
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

            for r in refs or []:
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
                "paperId,title,abstract,year,citationCount,influentialCitationCount"
            )
            hydrated = await s2.batch_papers(new_ids, fields=fields, graceful=True)
            for paper in hydrated or []:
                if not paper.get("paperId"):
                    continue
                # Back-pressure: wait if the score buffer is full so the
                # late-interaction encoder can catch up.
                while score_buffer.full() and not stop_event.is_set():
                    await asyncio.sleep(0.05)
                if stop_event.is_set():
                    break
                await score_buffer.put(paper)

            console.log(
                f"Scored {total_scored:>6} | queue {len(priority_queue):>4} | "
                f"buffer {score_buffer.qsize():>3} | candidates {len(core_candidates):>5}"
            )

    scorer_task = asyncio.create_task(scorer_worker(), name="maxsim-scorer")
    prioritizer_task = asyncio.create_task(
        prioritizer_worker(), name="maxsim-prioritizer"
    )
    expander_task = asyncio.create_task(expander_worker(), name="maxsim-expander")

    try:
        await expander_task
    finally:
        stop_event.set()
        # Drain path: signal scorer → prioritizer shutdown.
        try:
            await score_buffer.put(_STOP)
        except Exception:
            pass
        await asyncio.gather(scorer_task, prioritizer_task, return_exceptions=True)

    return list(dict.fromkeys(core_candidates))
