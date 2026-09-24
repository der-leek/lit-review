# lit-review
Async literature-review tool that builds a citation graph from Semantic Scholar, ranks papers with late-interaction MaxSim (LFM2.5-ColBERT), and exports an Obsidian vault with backlinks.

## Features
- Interactive query optimizer (OpenAI-compatible, OpenRouter by default) that always elicits a rich semantic intent via clarifying questions driven by ambiguity flags
- Seed approval loop: three diverse draft queries → top hit each → MaxSim vs intent → present abstracts for user confirmation (re-search until approved)
- Optional `--seeds` flag for manual paper IDs (intent elicitation only, then expand)
- Rate-limited async Semantic Scholar client (batch + citations/references)
- Concurrent priority-queue expansion guided by continuous MaxSim × influence
- Late-interaction model: LiquidAI/LFM2.5-ColBERT-350M (title fallback when abstract missing)
- Resumable SQLite state
- NetworkX graph + Obsidian vault export with wikilinks

## Quick start
```bash
# Install (uv recommended)
uv sync

# Set keys
export SEMANTIC_SCHOLAR_API_KEY=...
export OPENROUTER_API_KEY=...

# Run (interactive intent + seed approval)
uv run lit-review run "How do late-interaction retrievers scale on Apple Silicon?"

# Or supply three seed paper IDs (intent elicitation only, then expand)
uv run lit-review run "How do late-interaction retrievers scale on Apple Silicon?" \
  --seeds "CorpusId:123 CorpusId:456 CorpusId:789"
```

The default LLM provider uses the OpenAI SDK against OpenRouter's
OpenAI-compatible endpoint (`https://openrouter.ai/api/v1` by default;
override with `OPENROUTER_BASE_URL`). Set `OPENROUTER_API_KEY` (or
`OPENAI_API_KEY`). Use `--llm-provider apple_fm` for on-device Apple
Intelligence on macOS 26+.

## Project layout
```
src/lit_review/
├── cli.py               # Typer entrypoint (--seeds, --llm, …)
├── client.py            # Semantic Scholar async client
├── config.py            # settings (LFM2.5 is the default late-interaction model)
├── database.py          # SQLite schema & repository
├── expansion.py         # concurrent MaxSim priority-queue expansion
├── graph.py             # NetworkX + Obsidian export
├── pipeline.py          # orchestration, seed approval, resume
├── query_optimizer.py   # interactive intent elicitation (schema-driven)
├── ranking.py           # LFM2.5-ColBERT late-interaction MaxSim
└── llm/                 # provider backends (openrouter, apple_fm)
```

## Late-interaction model
Default: `LiquidAI/LFM2.5-ColBERT-350M`

- Late-interaction (token-level MaxSim)
- Loaded through `sentence-transformers` MultiVectorEncoder
- Override with env var `LIT_REVIEW_LATE_MODEL`
- Expansion priority: `MaxSim × max(1, influentialCitationCount)`
- Papers without abstracts are scored by title

## License
MIT
