"""Late-interaction (LFM2.5-ColBERT) ranking helpers.

Relevance is driven by token-level MaxSim between the user's detailed intent
and each paper's abstract (falling back to title when no abstract is present).
SPECTER dense embeddings are no longer used for expansion or pruning — the
Semantic Scholar API already surfaces citation-based influence signals, and
MaxSim is a stronger query↔content match for the expansion priority queue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sentence_transformers as st
import torch
from rich import console as csl

from lit_review import config

if TYPE_CHECKING:
    from collections.abc import Sequence

console = csl.Console()


def _get_device() -> str:
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return "mps"
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def paper_text_for_scoring(paper: dict) -> str:
    """Return the best available text for MaxSim scoring.

    Prefer a non-empty abstract; fall back to the title when the abstract is
    missing so papers without abstracts still receive a relevance signal.
    """
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        return abstract
    title = (paper.get("title") or "").strip()
    return title


def maxsim_priority(maxsim: float, influential_citation_count: int | None) -> float:
    """Combine MaxSim relevance with citation influence for the priority queue.

    priority = MaxSim * max(1, influentialCitationCount)
    """
    influence = max(1, int(influential_citation_count or 0))
    return float(maxsim) * float(influence)


class LateInteractionRanker:
    """Late-interaction ranker using LiquidAI/LFM2.5-ColBERT-350M.

    Loaded via sentence-transformers MultiVectorEncoder.
    Runs on Apple Silicon (MPS) when available.
    """

    def __init__(
        self,
        model_name: str = config.LATE_INTERACTION_MODEL,
        device: str | None = None,
    ):
        self.device = device or _get_device()
        self.model = None
        self.model_name = model_name
        console.log(f"Loading late-interaction model {model_name} on {self.device}")

        try:
            self.model = st.MultiVectorEncoder(model_name, trust_remote_code=True)
            self.model.to(self.device)
        except Exception as exc:
            console.log(
                f"[yellow]Could not load MultiVectorEncoder ({exc}). "
                "Late-interaction scoring will be skipped.[/]"
            )

    @property
    def available(self) -> bool:
        """Return whether the model is available."""
        return self.model is not None

    def score(
        self,
        query: str,
        texts: Sequence[str],
        top_k: int = 500,
    ) -> list[tuple[int, float]]:
        """Return list of (index_into_texts, score) sorted descending.

        Uses MaxSim-style scoring when the MultiVectorEncoder API supports it;
        otherwise falls back to a mean-pool approximation.
        """
        if not self.model or not texts:
            return [(i, 0.0) for i in range(min(len(texts), top_k))]

        try:
            # Encode query and documents (token-level multi-vectors)
            q_emb = self.model.encode([query], batch_size=1)
            d_embs = self.model.encode(
                list(texts), batch_size=16, show_progress_bar=False
            )

            scores: list[tuple[int, float]] = []
            for i, d in enumerate(d_embs):
                # Prefer official MaxSim if exposed; otherwise mean-pool fallback
                if hasattr(self.model, "similarity"):
                    score = float(self.model.similarity(q_emb[0], d))
                else:
                    # Approximate: mean over token interactions
                    score = (
                        float(np.mean(d @ q_emb[0].T))
                        if hasattr(d, "__matmul__")
                        else 0.0
                    )
                scores.append((i, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
        except Exception as exc:
            console.log(f"[yellow]Late-interaction scoring failed: {exc}[/]")
            return [(i, 0.0) for i in range(min(len(texts), top_k))]

    def score_one(self, query: str, text: str) -> float:
        """Return the MaxSim score of a single text against the query."""
        if not text.strip():
            return 0.0
        ranked = self.score(query, [text], top_k=1)
        return ranked[0][1] if ranked else 0.0
