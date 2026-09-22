"""Apple Foundation Models (on-device) backend for `LLMClient`.

Requires macOS 26+, Apple Intelligence enabled, and:

    pip install apple-fm-sdk

This adapter maps OpenAI-style messages to a single prompt. When
`response_format="json"` it always uses the SDK's built-in guided
generation (structured output) backed by a generation schema — there is
no prompt-based fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import apple_fm_sdk as fm

if TYPE_CHECKING:
    from lit_review.llm import base


@fm.generable("Structured search-plan output for a literature review")
class _SearchPlan:
    """Default schema used whenever the caller only asks for JSON.

    Mirrors the keys the query optimizer expects from `query_optimizer.py`
    (see `SYSTEM_PROMPT` there). Nothing here is written to disk as Python
    objects; the generated `GeneratedContent` is returned as its JSON string.
    """

    intent: str = fm.guide("One-sentence statement of the user's information need")
    core_concepts: list[str] = fm.guide("Canonical scholarly terms")
    disambiguation: str | None = fm.guide("null or a short competing-senses note")
    ambiguity_flags: list[str] = fm.guide("Residual ambiguities")
    clarifying_questions: list[str] = fm.guide("0-3 questions if ambiguous, else empty")
    optimized_query: str = fm.guide("Best plain-text Semantic Scholar query")
    alternative_queries: list[str] = fm.guide("Broader/narrower/synonym variants")
    hyde_pseudo_abstract: str = fm.guide(
        "Hypothetical abstract of an ideal matching paper"
    )
    year_range: str | None = fm.guide("YYYY-YYYY, YYYY-, -YYYY, or null")
    fields_of_study: list[str] = fm.guide('Usually ["Computer Science"] or []')
    min_citation_count: int | None = fm.guide("null or integer")
    study_types: list[str] = fm.guide('["method","survey","empirical","theory"] or []')
    confidence: float = fm.guide("0.0-1.0", range=(0.0, 1.0))
    notes: str = fm.guide("Brief rationale")


def _generation_response_json(response: Any) -> str:
    """Return the JSON string from a `GeneratedContent` response."""
    if hasattr(response, "to_json"):
        return response.to_json()
    if isinstance(response, str):
        return response
    text = getattr(response, "content", None) or getattr(response, "text", None)
    if text is None:
        text = str(response)
    return text


class AppleFMClient:
    """On-device Apple Intelligence via `apple-fm-sdk`.

    Satisfies `LLMClient`. No API key required.
    """

    def __init__(self, instructions: str | None = None):

        self._fm = fm
        model = fm.SystemLanguageModel()
        available, reason = model.is_available()
        if not available:
            raise RuntimeError(f"Apple Foundation Model unavailable: {reason}")

        self._instructions = instructions or (
            "You are a careful scientific research assistant. "
            "Follow the user's instructions exactly. "
            "When asked for JSON, reply with only valid JSON and no markdown fences."
        )
        # One long-lived session; transcript accumulates across complete() calls
        # within a single optimizer run. Callers that need isolation can create
        # a fresh client.
        self._session = fm.LanguageModelSession(
            model=model,
            instructions=self._instructions,
        )

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt string."""
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                # Already covered by session instructions; include as context
                parts.append(f"[System guidance]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(f"[User]\n{content}")
        return "\n\n".join(parts)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None = None,
        **kwargs: dict[str, Any],
    ) -> str:
        """Complete a chat transcript.

        Args:
            messages: OpenAI-style message list.
            response_format: If "json" or "json_object", request JSON output
                via the SDK's guided generation. When None, free text.
            **kwargs: Extra arguments. `schema` may be provided; it is either
                a `fm.GenerationSchema`, a `@fm.generable` class, or a plain
                JSON Schema dict passed through as `json_schema`. Without it,
                the default `_SearchPlan` schema is used.

        Returns:
            The model's text response. When JSON was requested, this is a
            JSON string without markdown fences.
        """
        prompt = self._messages_to_prompt(messages)

        schema = kwargs.pop("schema", None)
        json_schema = kwargs.pop("json_schema", None)
        json_requested = (
            response_format in ("json", "json_object")
            or schema is not None
            or json_schema is not None
        )

        if json_requested:
            # Always drive the SDK's native structured output with a schema.
            if json_schema is not None:
                response = await self._session.respond(prompt, json_schema=json_schema)
            elif schema is not None:
                # Accept a generable class, a GenerationSchema, or a plain JSON dict.
                generation_schema = _resolve_generation_schema(schema)
                response = await self._session.respond(prompt, schema=generation_schema)
            else:
                response = await self._session.respond(prompt, generating=_SearchPlan)

            return _generation_response_json(response)

        response = await self._session.respond(prompt)
        return self._extract_text(response)

    def _extract_text(self, response: Any) -> str:
        """Extract plain text from a free-text SDK response."""
        if isinstance(response, str):
            text = response
        else:
            text = (
                getattr(response, "content", None)
                or getattr(response, "text", None)
                or str(response)
            )
        return text.strip()

    async def aclose(self) -> None:
        # LanguageModelSession has no public close in current SDK; no-op.
        return None


def _resolve_generation_schema(schema: Any) -> fm.GenerationSchema:
    """Coerce a caller-supplied schema into an `fm.GenerationSchema`.

    Accepts a `@fm.generable` class (it already carries a
    `generation_schema()` classmethod), an `fm.GenerationSchema` itself,
    or a raw `GenerationSchema` instance — anything else is raised.
    """
    if isinstance(schema, fm.GenerationSchema):
        return schema
    cls = schema if isinstance(schema, type) else type(schema)
    generation_schema = getattr(cls, "generation_schema", None)
    if callable(generation_schema):
        return generation_schema()
    raise TypeError(
        "`schema` must be a @fm.generable class or an fm.GenerationSchema, "
        f"got {schema!r}"
    )


def _assert_protocol() -> None:
    _: base.LLMClient = AppleFMClient.__new__(AppleFMClient)
