"""LLM provider abstraction.

Usage
-----
::

    from lit_review.llm import get_llm_client, LLMClient

    client = get_llm_client("openrouter")   # or "apple_fm"
    text = await client.complete(messages, response_format="json")
"""

from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import sys
from typing import Any

from lit_review.llm import base

_IMPORTED_MODULES: set[str] = set()


def _provider_modules() -> list[str]:
    """Return dotted module names for every `.py` in this package."""
    pkg_dir = pathlib.Path(__file__).parent
    modules: list[str] = []
    for path in sorted(pkg_dir.glob("*.py")):
        if path.name in ("__init__.py",):
            continue
        modules.append(f"{__name__}.{path.stem}")
    return modules


def _ensure_providers_loaded() -> None:
    """Import every provider module once; skip modules that already ran.

    Optional backends (e.g. `apple_fm` needing `apple-fm-sdk`) are skipped on
    `ModuleNotFoundError`, so a missing optional dependency never breaks the
    default path. Each module's `__init_subclass__` self-registers in
    `BaseLLMClient._registry`.
    """
    for module_name in _provider_modules():
        if module_name in _IMPORTED_MODULES:
            continue
        _IMPORTED_MODULES.add(module_name)
        if module_name in sys.modules:
            continue

        with contextlib.suppress(ModuleNotFoundError):
            importlib.import_module(module_name)


def get_llm_client(
    provider: str | None = None,
    **kwargs: Any,
) -> base.BaseLLMClient:
    """Factory for LLM backends.

    Args:
        provider: `"openrouter"` (default) or `"apple_fm"`.
            Falls back to env `LIT_REVIEW_LLM_PROVIDER`.
        **kwargs: Forwarded to the concrete client constructor.
    """
    name = (provider or os.getenv("LIT_REVIEW_LLM_PROVIDER") or "openrouter").lower()

    cls = base.BaseLLMClient._registry.get(name)
    if cls is None:
        # Cold start (or the name's module wasn't imported yet): scan once.
        _ensure_providers_loaded()
        cls = base.BaseLLMClient._registry.get(name)

    if cls is None:
        supported = ", ".join(sorted(base.BaseLLMClient._registry))
        raise ValueError(f"Unknown LLM provider {name!r}. Supported: {supported}")

    return cls(**kwargs)


# Lazy re-exports so `from lit_review.llm import OpenAIClient` works.
def __getattr__(name: str) -> type[Any]:
    for cls in base.BaseLLMClient._registry.values():
        if cls.__name__ == name:
            return cls
    # Fall back to scanning provider modules (e.g. for classes declared in a
    # file that hasn't been imported yet).
    _ensure_providers_loaded()
    for cls in base.BaseLLMClient._registry.values():
        if cls.__name__ == name:
            return cls
    raise AttributeError(name)
