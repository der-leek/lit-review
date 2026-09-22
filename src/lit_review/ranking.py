"""Dense (SPECTER) and late-interaction (LFM2.5-ColBERT) ranking helpers."""

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


class DenseRanker:
    """SPECTER2 dense embeddings for cheap pruning."""

    def __init__(self, model_name: str = config.SPECTER_MODEL, device: str | None = None):
        self.device = device or _get_device()
        console.log(f"Loading dense model {model_name} on {self.device}")
        self.model = st.SentenceTransformer(model_name, device=self.device)

    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(emb[0], dtype=np.float32)

    def encode_papers(self, abstracts: Sequence[str]) -> np.ndarray:
        if not abstracts:
            if not (embedding_dim := self.model.get_embedding_dimension()):
                raise ValueError()

            return np.zeros((0, embedding_dim), dtype=np.float32)

        emb = self.model.encode(
            list(abstracts),
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(abstracts) > 64,
        )
        return np.asarray(emb, dtype=np.float32)

    def score(
        self,
        query_emb: np.ndarray,
        paper_embs: np.ndarray,
    ) -> np.ndarray:
        """Cosine similarity (embeddings are already L2-normalised)."""
        if paper_embs.size == 0:
            return np.array([], dtype=np.float32)
        return paper_embs @ query_emb


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
        abstracts: Sequence[str],
        top_k: int = 500,
    ) -> list[tuple[int, float]]:
        """Return list of (index_into_abstracts, score) sorted descending.

        Uses MaxSim-style scoring when the MultiVectorEncoder API supports it;
        otherwise falls back to a mean-pool approximation.
        """
        if not self.model or not abstracts:
            return [(i, 0.0) for i in range(min(len(abstracts), top_k))]

        try:
            # Encode query and documents (token-level multi-vectors)
            q_emb = self.model.encode([query], batch_size=1)
            d_embs = self.model.encode(
                list(abstracts), batch_size=16, show_progress_bar=True
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
            return [(i, 0.0) for i in range(min(len(abstracts), top_k))]


def combine_dense_and_influence(
    dense_relevance_scores: np.ndarray,
    influential_counts: Sequence[int],
) -> np.ndarray:
    """Relevance × log(1 + influential citations)."""
    influence = np.asarray(influential_counts, dtype=np.float32)
    return dense_relevance_scores * np.log1p(influence + 1.0)
