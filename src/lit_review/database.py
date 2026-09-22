"""SQLite schema, connection helpers, and repository methods."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import aiosqlite

from lit_review import config

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    query_raw       TEXT NOT NULL,
    query_optimized TEXT,
    status          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id                    TEXT PRIMARY KEY,
    title                       TEXT,
    abstract                    TEXT,
    year                        INTEGER,
    citation_count              INTEGER,
    influential_citation_count  INTEGER,
    specter                     BLOB,
    late_score                  REAL,
    is_core                     INTEGER DEFAULT 0,
    depth                       INTEGER DEFAULT 0,
    raw_json                    TEXT,
    created_at                  REAL
);

CREATE TABLE IF NOT EXISTS edges (
    src_id          TEXT NOT NULL,
    dst_id          TEXT NOT NULL,
    edge_type       TEXT NOT NULL,
    is_influential  INTEGER DEFAULT 0,
    PRIMARY KEY (src_id, dst_id, edge_type)
);

CREATE TABLE IF NOT EXISTS queue_state (
    paper_id        TEXT PRIMARY KEY,
    priority        REAL NOT NULL,
    depth           INTEGER NOT NULL,
    status          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_core ON papers(is_core);
CREATE INDEX IF NOT EXISTS idx_papers_depth ON papers(depth);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_state(status);
"""


async def get_db(
    path: str | pathlib.Path | None = None,
) -> aiosqlite.Connection:
    """Open (or create) the database and ensure schema exists."""
    db_path = path or config.DB_PATH
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA foreign_keys=ON;")
    await db.executescript(SCHEMA)
    await db.commit()
    return db


async def create_run(
    db: aiosqlite.Connection,
    run_id: str,
    query_raw: str,
) -> None:
    now = time.time()
    await db.execute(
        """
        INSERT OR REPLACE INTO runs
            (run_id, query_raw, status, created_at, updated_at)
        VALUES (?, ?, 'running', ?, ?)
        """,
        (run_id, query_raw, now, now),
    )
    await db.commit()


async def update_run(
    db: aiosqlite.Connection,
    run_id: str,
    *,
    query_optimized: str | None = None,
    status: str | None = None,
) -> None:
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [time.time()]
    if query_optimized is not None:
        fields.append("query_optimized = ?")
        values.append(query_optimized)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    values.append(run_id)
    await db.execute(
        f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?",
        values,
    )
    await db.commit()


async def upsert_papers(
    db: aiosqlite.Connection,
    papers: Sequence[dict],
    depth: int = 0,
) -> None:
    """Insert or replace paper records. Expects S2-style dicts."""
    now = time.time()
    rows = []
    for p in papers:
        pid = p.get("paperId")
        if not pid:
            continue
        emb = None
        embedding = p.get("embedding") or {}
        if isinstance(embedding, dict):
            vec = embedding.get("vector")
            if vec is not None:
                emb = json.dumps(vec).encode("utf-8")
        rows.append(
            (
                pid,
                p.get("title"),
                p.get("abstract"),
                p.get("year"),
                p.get("citationCount"),
                p.get("influentialCitationCount"),
                emb,
                depth,
                json.dumps(p),
                now,
            )
        )
    if not rows:
        return
    await db.executemany(
        """
        INSERT OR REPLACE INTO papers
            (paper_id, title, abstract, year, citation_count,
             influential_citation_count, specter, depth, raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    await db.commit()


async def add_edge(
    db: aiosqlite.Connection,
    src_id: str,
    dst_id: str,
    edge_type: str,
    is_influential: bool = False,
) -> None:
    await db.execute(
        """
        INSERT OR IGNORE INTO edges (src_id, dst_id, edge_type, is_influential)
        VALUES (?, ?, ?, ?)
        """,
        (src_id, dst_id, edge_type, int(is_influential)),
    )


async def mark_core(
    db: aiosqlite.Connection,
    paper_ids: Iterable[str],
    scores: dict[str, float] | None = None,
) -> None:
    scores = scores or {}
    for pid in paper_ids:
        await db.execute(
            "UPDATE papers SET is_core = 1, late_score = ? WHERE paper_id = ?",
            (scores.get(pid, 1.0), pid),
        )
    await db.commit()


async def get_core_papers(db: aiosqlite.Connection) -> list[dict]:
    async with db.execute(
        """
        SELECT paper_id, title, abstract, year, citation_count,
               influential_citation_count, late_score, depth
        FROM papers WHERE is_core = 1
        ORDER BY late_score DESC NULLS LAST
        """
    ) as cur:
        rows = await cur.fetchall()
    cols = [
        "paper_id",
        "title",
        "abstract",
        "year",
        "citation_count",
        "influential_citation_count",
        "late_score",
        "depth",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


async def load_queue(db: aiosqlite.Connection) -> Iterable[aiosqlite.Row]:
    """Return pending queue items as (priority, paper_id, depth)."""
    async with db.execute(
        "SELECT priority, paper_id, depth FROM queue_state WHERE status = 'pending'"
    ) as cur:
        return await cur.fetchall()


async def save_queue_item(
    db: aiosqlite.Connection,
    paper_id: str,
    priority: float,
    depth: int,
    status: str = "pending",
) -> None:
    await db.execute(
        """
        INSERT OR REPLACE INTO queue_state (paper_id, priority, depth, status)
        VALUES (?, ?, ?, ?)
        """,
        (paper_id, priority, depth, status),
    )
