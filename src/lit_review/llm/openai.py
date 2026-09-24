"""OpenAI-compatible backend for ``LLMClient``.

Uses the standard OpenAI client, defaulting to OpenRouter's
OpenAI-compatible endpoint (`config.OPENROUTER_BASE_URL`), so the request
shape is the ordinary OpenAI Chat Completions API. OpenRouter-specific
routing knobs (such as ``provider.require_parameters``) ride along in
``extra_body``.
"""

from __future__ import annotations

from typing import Any

import openai
from rich import console as csl

from lit_review import config
from lit_review.llm import base

console = csl.Console()

# Downstream providers behind OpenRouter reject the schema only after routing;
# retry without the endpoint filter once when this happens. A 400 is a schema
# complaint, 422 an unprocessable routing miss.
_RETRYABLE_ROUTING_STATUS = {400, 422}


class OpenAIClient(base.BaseLLMClient):
    """OpenAI-compatible API (OpenRouter by default).

    Satisfies `LLMClient`. Requires an API key (`OPENROUTER_API_KEY`, or
    `OPENAI_API_KEY` since the endpoint is OpenAI-shaped).
    """

    provider_name = "openai"
    aliases = ("openrouter", "or")

    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.OPENROUTER_MODEL,
        base_url: str = config.OPENROUTER_BASE_URL,
    ):
        """Create the client.

        Args:
            api_key: OpenRouter (or OpenAI) API key. Falls back to the env
                `OPENROUTER_API_KEY` then `OPENAI_API_KEY`.
            model: OpenRouter model slug to use by default.
            base_url: OpenAI-compatible endpoint. Defaults to OpenRouter.
        """
        key = api_key or config.OPENROUTER_API_KEY or config.OPENAI_API_KEY
        if not key:
            raise ValueError(
                "OpenRouter needs an API key. Set OPENROUTER_API_KEY "
                "(or OPENAI_API_KEY) or pass api_key=."
            )
        self._client: openai.AsyncOpenAI = openai.AsyncOpenAI(
            api_key=key,
            base_url=base_url,
        )
        self.base_url = base_url
        self.model = model

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None = None,
        **kwargs: dict[str, Any],
    ) -> str:
        """Complete a chat transcript via the OpenAI-compatible endpoint.

        Args:
            messages: OpenAI-style message list.
            response_format: `"json"` requests structured output via the
                OpenAI `json_schema` response format (strict). `None` is free
                text.
            kwargs: `json_schema=<dict>` is required for `"json"`; `model`
                overrides the default model; other keys are ignored.

        Returns:
            The assistant message content (a JSON string when `"json"`).
        """
        params: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
        }

        json_schema = kwargs.get("json_schema")
        require_parameters = False
        if response_format == "json":
            if json_schema is None:
                raise ValueError(
                    "response_format='json' requires json_schema=<dict>; "
                    "OpenRouter has no usable JSON fallback."
                )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "search_plan",
                    "schema": json_schema,
                    "strict": True,
                },
            }
            require_parameters = True

        if require_parameters:
            params["extra_body"] = {"provider": {"require_parameters": True}}

        try:
            resp = await self._client.chat.completions.create(**params)
        except Exception as exc:
            # R2: a downstream provider may reject the strict schema while
            # advertising structured outputs; retry once without the endpoint
            # filter (schema stays, only the route loosens).
            if require_parameters and _is_routing_rejection(exc):
                console.log(
                    "[yellow]OpenRouter schema/routing 4xx — retrying without "
                    "require_parameters.[/]"
                )
                params.pop("extra_body", None)
                resp = await self._client.chat.completions.create(**params)
            else:
                raise
        return resp.choices[0].message.content

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _is_routing_rejection(exc: Exception) -> bool:
    """Return whether `exc` is a schema/routing 4xx from OpenRouter."""
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_ROUTING_STATUS


# Explicit structural match for type checkers
def _assert_protocol() -> None:
    _: base.LLMClient = OpenAIClient.__new__(OpenAIClient)
