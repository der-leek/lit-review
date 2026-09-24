"""Provider-agnostic LLM interface.

Implement `LLMClient` for any backend (OpenRouter, Apple FM, local vLLM, …).
The rest of lit-review only depends on this protocol.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable


class BaseLLMClient:
    """Shared base for LLM backends; subclasses auto-register as providers.

    A provider is just a subclass of `BaseLLMClient` that declares a name::

        class MyClient(BaseLLMClient):
            provider_name = "my_provider"
            aliases = ("my",)

    `__init_subclass__` registers the class under `provider_name` and each
    alias in `BaseLLMClient._registry`, so `get_llm_client(name)` finds it
    without any separate registry edits. Concrete backends that cannot enforce
    a JSON Schema for `complete` (e.g. a local vLLM without guided
    generation) override `supports_json_schema = False`. Behavior is driven
    by this capability, never by a provider-name check in the optimizer or
    pipeline.
    """

    # Registry of provider name/alias -> client class. Populated by
    # `__init_subclass__`; nothing else writes to it.
    _registry: dict[str, type[BaseLLMClient]] = {}

    # Class-level declarations a provider should set:
    provider_name: str = ""
    aliases: tuple[str, ...] = ()

    supports_json_schema: bool = True

    def __init_subclass__(
        cls,
        **kwargs: Any,
    ) -> None:
        """Register the subclass under its provider name and aliases."""
        super().__init_subclass__(**kwargs)
        name = cls.provider_name or _snake_case(cls.__name__)
        BaseLLMClient._registry[name] = cls
        for alias in cls.aliases:
            BaseLLMClient._registry[alias] = cls
        cls.provider_name = name

    async def aclose(self) -> None:
        """Release resources (HTTP sessions, model handles, …). Optional no-op."""
        return None


def _snake_case(text: str) -> str:
    """Convert a CamelCase class name to snake_case (e.g. MyClient -> my_client)."""
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
    return text.replace("__", "_")


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat interface used by the query optimizer."""

    supports_json_schema: bool

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None = None,
        **kwargs: dict[str, Any],
    ) -> str:
        """Run a chat-style completion.

        Args:
            messages: OpenAI-style list of `{"role": "system"|"user"|"assistant",
                "content": "..."}`.
            response_format: `"json"` to request a JSON object string, or `None` for
                free text. Providers that cannot enforce JSON should still return a
                string that looks like JSON when asked (the caller will parse it).
            kwargs: Dictionary of arguments to pass to the downstream llm provider.
                The query optimizer passes `json_schema=<dict>` (the JSON Schema
                dict for guided generation, following the OpenAI strict subset),
                plus any provider-specific extras.

        Returns:
        str
            The assistant message content.
        """
        ...

    async def aclose(self) -> None:
        """Release resources (HTTP sessions, model handles, …). Optional no-op."""
        return None
