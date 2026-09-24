"""Central configuration and constants."""

from __future__ import annotations

import os
import pathlib

import dotenv

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
S2_API_KEY: str | None = os.getenv("S2_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
# OpenRouter exposes an OpenAI-compatible API; point the OpenAI client here.
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
# Legacy key name still honored so existing env files keep working.
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH: pathlib.Path = pathlib.Path(os.getenv("LIT_REVIEW_DB", "lit-review.db"))
VAULT_DIR: pathlib.Path = pathlib.Path(os.getenv("LIT_REVIEW_VAULT", "vault"))

# ---------------------------------------------------------------------------
# Scale & expansion knobs
# ---------------------------------------------------------------------------
MAX_CORE_PAPERS: int = int(os.getenv("LIT_REVIEW_MAX_CORE", "50"))
MAX_TOTAL_RETRIEVED: int = int(os.getenv("LIT_REVIEW_MAX_TOTAL", "5000"))
# Per-draft search limit. The seed selector keeps the top hit from each of
# three drafts, so 1 is enough; raise if you want a larger pool to MaxSim.
SEED_LIMIT: int = int(os.getenv("LIT_REVIEW_SEED_LIMIT", "1"))
NUM_SEEDS: int = int(os.getenv("LIT_REVIEW_NUM_SEEDS", "3"))
CITATIONS_PER_PAPER: int = int(os.getenv("LIT_REVIEW_CITATIONS", "25"))
REFS_PER_PAPER: int = int(os.getenv("LIT_REVIEW_REFS", "20"))
# Soft ceiling on papers waiting to be MaxSim-scored. Back-pressure for the
# expander when the late-interaction encoder falls behind network I/O.
SCORE_BUFFER_MAX: int = int(os.getenv("LIT_REVIEW_SCORE_BUFFER_MAX", "128"))

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 400  # S2 hard limit is 500
LATE_INTERACTION_TOP_K: int = 500
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

# /paper/search is throttled frequently; poll every 1s for up to this long
SEARCH_POLL_TIMEOUT: float = float(os.getenv("LIT_REVIEW_SEARCH_POLL_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# LLM provider (openrouter | apple_fm)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LIT_REVIEW_LLM_PROVIDER", "openrouter")
OPENROUTER_MODEL: str = os.getenv(
    "LIT_REVIEW_OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731"
)
# Default model for OpenAI-compatible providers (used when a provider-specific
# model like OPENROUTER_MODEL is not set).
OPENAI_MODEL: str = os.getenv("LIT_REVIEW_OPENAI_MODEL", OPENROUTER_MODEL)
