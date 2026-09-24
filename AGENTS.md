# AGENTS.md

## Project overview

`lit-review` is an async literature-review tool. It builds a citation graph from the Semantic Scholar API, ranks papers with dense (SPECTER) + late-interaction (LFM2.5-ColBERT-350M) models, and exports an Obsidian vault with backlinks.

Key components (in `src/lit_review/`):

- `cli.py` — Typer entrypoint
- `client.py` — rate-limited async Semantic Scholar client
- `query_optimizer.py` — OpenRouter-driven query expansion
- `expansion.py` — priority-queue expansion guided by relevance × influence
- `ranking.py` — SPECTER dense pruning + late-interaction re-ranking
- `database.py` — resumable SQLite state
- `graph.py` — NetworkX graph + Obsidian vault export
- `pipeline.py` — orchestration & resume

The project uses `uv` for dependency and environment management. Do not add or remove dependencies without updating `pyproject.toml` and running `uv sync`.

## Code quality gates

Always run both checks before considering work complete:

```bash
uv run ruff check .
uv run ty check .
```

- `uv run ruff check .` must pass with zero lint errors. Run `uv run ruff format .` if formatting issues arise.
- `uv run ty check .` must pass with zero type errors. Do not weaken type checking with `# type: ignore` comments unless truly necessary, and never silence errors that reveal real type issues.

## Coding conventions (Google Python Style Guide)

The project follows the Google Python Style Guide (see references in `pyproject.toml`).

### Line length

- **80-character line limit.** Do not exceed 80 characters per line. Break long lines, imports, and strings accordingly.

### Imports

- **Imports at the top of files only.** Never import inside functions, methods, or conditionals.
- **Import modules only — not members or classes.** Prefer `import os` / `import networkx as nx` over `from os import path`. Importing individual members from a module is discouraged; only import from a module's namespace when necessary (e.g. for typing constructs).
- Sort imports with isort (`I`) rules: standard library, third-party, then first-party, alphabetically ordered with proper grouping.

### Function limits

- **Maximum 50 statements per function** (Pylint `PLR0915` / `too-many-statements`). If a function exceeds this, refactor it into smaller helpers.
- **Maximum 12 branches per function** (Pylint `PLR0912` / `too-many-branches`). Count `if`, `elif`, `for`, `while`, and `except` as branches. If a function exceeds this, extract logic into sub-functions.
- **Positional/kwarg limits** (configured in `pyproject.toml` under `[tool.ruff.lint.pylint]`):
  - `max-positional-args = 5` — a maximum of 5 positional arguments per function; use keyword-only arguments beyond that.
  - `max-args = 15` — a maximum of 15 total arguments per function.

### Other conventions enforced by ruff

- Google-style docstrings (pydocstyle `D`).
- Full type annotations (pep8-naming `N`, `ANN`, `TC`).
- Pylint, bugbear, comprehension, and perflint rules are enabled — write idiomatic, allocation-avoiding code (no `list()` comprehensions when a generator suffices, no unused variables, etc.).
- Double quotes for strings (`[tool.ruff.format] quote-style = "double"`).