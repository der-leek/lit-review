# lit-review
Async literature-review tool that builds a citation graph from Semantic Scholar, ranks papers with dense (SPECTER) + late-interaction models, and exports an Obsidian vault with backlinks.

## Features
- Interactive query optimizer (OpenRouter) that turns natural-language questions into high-precision Semantic Scholar queries
- Rate-limited async Semantic Scholar client (batch + citations/references)
- Priority-queue expansion guided by relevance × influence
- SPECTER dense pruning followed by late-interaction re-ranking
- Late-interaction model: LiquidAI/LFM2.5-ColBERT-350M
- Resumable SQLite state
- NetworkX graph + Obsidian vault export with wikilinks

## Quick start
```bash
# Install (uv recommended)
uv sync

# Set keys
export SEMANTIC_SCHOLAR_API_KEY=...
export OPENROUTER_API_KEY=...

# Run
uv run lit-review run "How do late-interaction retrievers scale on Apple Silicon?"
```

## Project layout
```
src/lit_review/
├── cli.py               # Typer entrypoint
├── client.py            # Semantic Scholar async client
├── config.py            # settings (LFM2.5 is the default late-interaction model)
├── database.py          # SQLite schema & repository
├── expansion.py         # priority-queue expansion
├── graph.py             # NetworkX + Obsidian export
└── pipeline.py          # orchestration & resume
├── query_optimizer.py   # OpenRouter interactive optimizer
├── ranking.py           # SPECTER + LFM2.5-ColBERT late-interaction
```

## Late-interaction model
Default: `LiquidAI/LFM2.5-ColBERT-350M`

- Late-interaction (token-level MaxSim)
- Loaded through `sentence-transformers` MultiVectorEncoder
- Override with env var `LIT_REVIEW_LATE_MODEL`

## License
MIT
