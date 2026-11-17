"""Regression tests for runtime WebSocket and AI provider wiring."""

import pytest

from app.ai.factory import get_ai_companion_provider, reset_providers
from app.ai.gemini_provider import GeminiAICompanionProvider
from app.ai.groq_provider import GroqAICompanionProvider
from app.ai.mock_provider import MockAICompanionProvider


def test_auto_provider_prefers_gemini_and_wires_groq_fallback(monkeypatch):
    from app.ai import factory

    monkeypatch.setattr(factory.settings, "AI_PROVIDER", "auto")
    monkeypatch.setattr(factory.settings, "GEMINI_API_KEY", "google-test-key")
    monkeypatch.setattr(factory.settings, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(factory.settings, "GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(factory.settings, "GROQ_MODEL", "openai/gpt-oss-20b")
    reset_providers()
    try:
        provider = get_ai_companion_provider()
        assert isinstance(provider, GeminiAICompanionProvider)
        assert isinstance(provider.fallback, GroqAICompanionProvider)
        assert provider.fallback.model == "openai/gpt-oss-20b"
    finally:
        reset_providers()


def test_auto_provider_uses_groq_when_google_key_is_missing(monkeypatch):
    from app.ai import factory

    monkeypatch.setattr(factory.settings, "AI_PROVIDER", "auto")
    monkeypatch.setattr(factory.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(factory.settings, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(factory.settings, "GROQ_API_KEY", "groq-test-key")
    monkeypatch.setattr(factory.settings, "GROQ_MODEL", "openai/gpt-oss-20b")
    reset_providers()
    try:
        provider = get_ai_companion_provider()
        assert isinstance(provider, GroqAICompanionProvider)
        assert provider.model == "openai/gpt-oss-20b"
    finally:
        reset_providers()


def test_auto_provider_uses_mock_without_credentials(monkeypatch):
    from app.ai import factory

    monkeypatch.setattr(factory.settings, "AI_PROVIDER", "auto")
    monkeypatch.setattr(factory.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(factory.settings, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(factory.settings, "GROQ_API_KEY", None)
    reset_providers()
    try:
        assert isinstance(get_ai_companion_provider(), MockAICompanionProvider)
    finally:
        reset_providers()


@pytest.mark.asyncio
async def test_groq_provider_uses_current_model_and_completion_token_parameter():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello from Groq"}}]}

    class FakeClient:
        def __init__(self):
            self.payload = None

        async def post(self, path, json):
            assert path == "/chat/completions"
            self.payload = json
            return FakeResponse()

    client = FakeClient()
    provider = GroqAICompanionProvider(
        api_key="groq-test-key",
        model="openai/gpt-oss-20b",
        client=client,
    )
    try:
        reply = await provider.generate_reply("session-1", [], "hello")
        assert reply == "hello from Groq"
        assert client.payload["model"] == "openai/gpt-oss-20b"
        assert client.payload["max_completion_tokens"] == 256
        assert "max_tokens" not in client.payload
    finally:
        await provider.close()


def test_only_one_websocket_route_is_registered():
    from app.main import app

    websocket_routes = [
        route for route in app.routes
        if route.__class__.__name__ == "APIWebSocketRoute" and route.path == "/ws"
    ]
    assert len(websocket_routes) == 1, f"Expected exactly one /ws route, found {len(websocket_routes)}"

@pytest.mark.asyncio
async def test_gemini_runtime_failure_really_calls_groq_fallback():
    from app.ai.gemini_provider import GeminiAICompanionProvider

    class FakeGeminiModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("simulated Gemini outage")

    class FakeGeminiClient:
        class Aio:
            models = FakeGeminiModels()

        aio = Aio()

    class FakeGroqResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "reply from Groq fallback"}}]}

    class FakeGroqClient:
        def __init__(self):
            self.called = False

        async def post(self, path, json):
            self.called = True
            return FakeGroqResponse()

    groq_client = FakeGroqClient()
    groq = GroqAICompanionProvider(
        api_key="groq-test-key",
        model="openai/gpt-oss-20b",
        client=groq_client,
    )
    gemini = GeminiAICompanionProvider(
        api_key="google-test-key",
        client=FakeGeminiClient(),
        fallback_provider=groq,
        timeout=0.1,
    )

    try:
        reply = await gemini.generate_reply("session-fallback", [], "hello")
        assert reply == "reply from Groq fallback"
        assert groq_client.called is True
    finally:
        await gemini.close()
        await groq.close()

def test_groq_provider_replaces_retired_model_even_if_set_in_environment(monkeypatch):
    from app.ai import groq_provider

    monkeypatch.setattr(groq_provider.settings, "GROQ_MODEL", "llama-3.1-8b-instant")
    provider = GroqAICompanionProvider(api_key="groq-test-key", client=object())
    try:
        assert provider.model == "openai/gpt-oss-20b"
    finally:
        # object() is only a construction stub and owns no resources.
        provider._client = None

