"""Provider auto-registration tests.

A provider must be discoverable by ``get_llm_client`` purely by subclassing
``BaseLLMClient`` and declaring a name — no registry edits anywhere.
"""

from __future__ import annotations

import pytest

from lit_review.llm import get_llm_client
from lit_review.llm.base import BaseLLMClient


class FakeProvider(BaseLLMClient):
    """Dummy provider for registration tests (no real backend)."""

    provider_name = "fake"
    aliases = ("f", "fake_alias")

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy


def test_subclass_auto_registers() -> None:
    """Subclassing BaseLLMClient registers under name and aliases."""
    assert BaseLLMClient._registry["fake"] is FakeProvider
    assert BaseLLMClient._registry["f"] is FakeProvider
    assert BaseLLMClient._registry["fake_alias"] is FakeProvider


def test_get_llm_client_by_name() -> None:
    """The factory resolves a provider by its declared name."""
    client = get_llm_client("fake", proxy="http://localhost:8080")
    assert isinstance(client, FakeProvider)
    assert client.proxy == "http://localhost:8080"


def test_get_llm_client_by_alias() -> None:
    """The factory resolves aliases too."""
    client = get_llm_client("f")
    assert isinstance(client, FakeProvider)


def test_unknown_provider_raises() -> None:
    """An unknown provider name raises a readable ValueError."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("definitely_not_a_provider")


def test_default_resolves_to_openai() -> None:
    """The default provider resolves via the openrouter alias to OpenAIClient."""
    from lit_review.llm.openai import OpenAIClient

    assert BaseLLMClient._registry["openrouter"] is OpenAIClient
    assert BaseLLMClient._registry["openai"] is OpenAIClient
