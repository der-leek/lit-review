"""NetworkX graph construction and Obsidian vault export."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import networkx as nx
from rich import console as csl

from lit_review import config, database

if TYPE_CHECKING:
    import aiosqlite

console = csl.Console()


async def build_graph(db: aiosqlite.Connection) -> nx.DiGraph:
    """Load papers + edges into a directed NetworkX graph."""
    g = nx.DiGraph()

    async with db.execute(
        """
        SELECT paper_id, title, year, late_score, is_core, citation_count,
               influential_citation_count, depth
        FROM papers
        """
    ) as cur:
        async for row in cur:
            (
                paper_id,
                title,
                year,
                score,
                is_core,
                cites,
                influence,
                depth,
            ) = row

            g.add_node(
                paper_id,
                title=title or "",
                year=year,
                score=score,
                core=bool(is_core),
                citation_count=cites,
                influential_citation_count=influence,
                depth=depth,
            )

    async with db.execute(
        "SELECT src_id, dst_id, edge_type, is_influential FROM edges"
    ) as cur:
        async for row in cur:
            src, dst, etype, influence = row
            if src in g and dst in g:
                g.add_edge(src, dst, type=etype, influential=bool(influence))

    return g


def _safe_filename(title: str, paper_id: str) -> str:
    """Create a filesystem-safe note name."""
    clean = "".join(c if c.isalnum() or c in " -_" else "" for c in (title or ""))
    clean = clean.strip()[:80] or paper_id[:12]
    return f"{clean} – {paper_id[:8]}.md"


async def export_obsidian_vault(
    db: aiosqlite.Connection,
    g: nx.DiGraph,
    vault_dir: pathlib.Path | None = None,
    *,
    only_core: bool = True,
) -> pathlib.Path:
    """Write one Markdown note per paper (core by default).

    Includes YAML frontmatter and wikilinks for outgoing citation edges.
    """
    vault = pathlib.Path(vault_dir or config.VAULT_DIR)
    vault.mkdir(parents=True, exist_ok=True)

    if only_core:
        papers = await database.get_core_papers(db)
    else:
        async with db.execute(
            "SELECT paper_id, title, abstract, year, late_score, depth FROM papers"
        ) as cur:
            rows = await cur.fetchall()

        papers = [
            {
                "paper_id": r[0],
                "title": r[1],
                "abstract": r[2],
                "year": r[3],
                "late_score": r[4],
                "depth": r[5],
            }
            for r in rows
        ]

    id_to_title = {n: (g.nodes[n].get("title") or n) for n in g.nodes}

    for p in papers:
        paper_id = p["paper_id"]
        title = p.get("title") or paper_id
        abstract = p.get("abstract") or ""
        year = p.get("year")
        score = p.get("late_score")
        depth = p.get("depth")

        out_links = []
        if paper_id in g:
            for _, dst, data in g.out_edges(paper_id, data=True):
                dst_title = id_to_title.get(dst, dst)
                label = (
                    "influential"
                    if data.get("influential")
                    else data.get("type", "cites")
                )
                out_links.append(f"- [[{dst_title}]] ({label})")

        frontmatter = (
            f"---\n"
            f"paper_id: {paper_id}\n"
            f'title: "{title.replace(chr(34), "")}"\n'
            f"year: {year or ''}\n"
            f"score: {score if score is not None else ''}\n"
            f"depth: {depth if depth is not None else ''}\n"
            f"tags: [lit-review]\n"
            f"---\n\n"
        )
        body = f"# {title}\n\n"
        if abstract:
            body += f"## Abstract\n\n{abstract}\n\n"
        if out_links:
            body += "## Related papers\n\n" + "\n".join(out_links) + "\n"

        note_path = vault / _safe_filename(title, paper_id)
        note_path.write_text(frontmatter + body, encoding="utf-8")

    console.log(f"[green]Obsidian vault written → {vault} ({len(papers)} notes)[/]")
    return vault
