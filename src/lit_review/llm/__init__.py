"""LLM provider abstraction.

Usage
-----
::

    from lit_review.llm import get_llm_client, LLMClient

    client = get_llm_client("openrouter")   # or "apple_fm"
    text = await client.complete(messages, response_format="json")
"""

from __future__ import annotations

import os

from lit_review.llm import apple_fm, base, openrouter


def get_llm_client(
    provider: str | None = None,
    **kwargs: str,
) -> base.LLMClient:
    """Factory for LLM backends.

    Args:
        provider: `"openrouter"` (default) or `"apple_fm"`.
            Falls back to env `LIT_REVIEW_LLM_PROVIDER`.
        **kwargs: Forwarded to the concrete client constructor.
    """
    name = (provider or os.getenv("LIT_REVIEW_LLM_PROVIDER") or "openrouter").lower()

    if name in ("openrouter", "or"):
        return openrouter.OpenRouterClient(**kwargs)

    if name in ("apple_fm", "apple", "fm", "foundation"):
        return apple_fm.AppleFMClient(**kwargs)

    raise ValueError(f"Unknown LLM provider {name!r}. Supported: openrouter, apple_fm")


# Lazy re-exports so `from lit_review.llm import OpenRouterClient` works
def __getattr__(
    name: str,
) -> type[openrouter.OpenRouterClient] | type[apple_fm.AppleFMClient]:
    if name == "OpenRouterClient":
        return openrouter.OpenRouterClient
    if name == "AppleFMClient":
        return apple_fm.AppleFMClient

    raise AttributeError(name)
