"""Provider-agnostic LLM interface.

Implement ``LLMClient`` for any backend (OpenRouter, Apple FM, local vLLM, …).
The rest of lit-review only depends on this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat interface used by the query optimizer."""

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
            kwargs: Dictionary of arguments to pass to the downstream llm provider

        Returns:
        str
            The assistant message content.
        """
        ...

    async def aclose(self) -> None:
        """Release resources (HTTP sessions, model handles, …). Optional no-op."""
        return None
