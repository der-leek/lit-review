"""OpenRouter backend for ``LLMClient``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openrouter as router

from lit_review import config

if TYPE_CHECKING:
    from lit_review.llm import base


class OpenRouterClient:
    """Thin adapter around the official `openrouter` SDK.

    Satisfies `LLMClient`. Requires `OPENROUTER_API_KEY`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.OPENROUTER_MODEL,
    ):
        key = api_key or config.OPENROUTER_API_KEY
        if not key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key=."
            )
        self._client = router.OpenRouter(api_key=key)
        self.model = model

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None = None,
        **kwargs: dict[str, Any],
    ) -> str:
        params: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
        }
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.send_async(**params)
        return resp.choices[0].message.content

    async def aclose(self) -> None:
        if hasattr(self._client, "aclose"):
            await self._client.aclose()
        elif hasattr(self._client, "close"):
            self._client.close()


# Explicit structural match for type checkers
def _assert_protocol() -> None:
    _: base.LLMClient = OpenRouterClient.__new__(OpenRouterClient)
