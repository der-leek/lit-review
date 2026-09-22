"""Central configuration and constants."""

from __future__ import annotations

import os
import pathlib

import dotenv

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
S2_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH: pathlib.Path = pathlib.Path(os.getenv("LIT_REVIEW_DB", "litreview.db"))
VAULT_DIR: pathlib.Path = pathlib.Path(os.getenv("LIT_REVIEW_VAULT", "obsidian_vault"))

# ---------------------------------------------------------------------------
# Scale & expansion knobs
# ---------------------------------------------------------------------------
MAX_CORE_PAPERS: int = int(os.getenv("LIT_REVIEW_MAX_CORE", "500"))
MAX_TOTAL_RETRIEVED: int = int(os.getenv("LIT_REVIEW_MAX_TOTAL", "30000"))
SEED_LIMIT: int = int(os.getenv("LIT_REVIEW_SEED_LIMIT", "40"))
CITATIONS_PER_PAPER: int = int(os.getenv("LIT_REVIEW_CITATIONS", "25"))
REFS_PER_PAPER: int = int(os.getenv("LIT_REVIEW_REFS", "20"))
MAX_DEPTH: int = int(os.getenv("LIT_REVIEW_MAX_DEPTH", "3"))

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 400  # S2 hard limit is 500
SPECTER_TOP_K: int = 800
LATE_INTERACTION_TOP_K: int = 500
SPECTER_MODEL: str = "allenai/specter2"

# Late-interaction model (Metal / MPS friendly)
# LiquidAI LFM2.5-ColBERT-350M – token-level late interaction
LATE_INTERACTION_MODEL: str = os.getenv(
    "LIT_REVIEW_LATE_MODEL", "LiquidAI/LFM2.5-ColBERT-350M"
)
DEVICE: str = os.getenv("LIT_REVIEW_DEVICE", "mps")  # "mps" | "cuda" | "cpu"

# ---------------------------------------------------------------------------
# Rate limiting (with API key aim for ~1 RPS on heavy endpoints)
# ---------------------------------------------------------------------------
RATE_LIMIT_RPS: float = float(os.getenv("LIT_REVIEW_RPS", "0.9"))
S2_BASE: str = "https://api.semanticscholar.org/graph/v1"
HEADERS: dict[str, str] = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

# ---------------------------------------------------------------------------
# LLM provider (openrouter | apple_fm)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LIT_REVIEW_LLM_PROVIDER", "openrouter")
OPENROUTER_MODEL: str = os.getenv(
    "LIT_REVIEW_OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash"
)
