"""Provider Factory with singleton resolution, environment config, and test overrides."""

import logging
import threading
from typing import Optional

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.ai.gemini_provider import GeminiAICompanionProvider, GeminiModerationProvider
from app.ai.groq_provider import GroqAICompanionProvider
from app.ai.mock_provider import MockAICompanionProvider, MockModerationProvider
from app.config import settings

logger = logging.getLogger(__name__)

_companion_provider: Optional[BaseAICompanionProvider] = None
_moderation_provider: Optional[BaseModerationProvider] = None
_lock = threading.Lock()


def _configured_groq_provider(api_key: str) -> GroqAICompanionProvider:
    """Build Groq with the configured model and an offline safety net."""
    return GroqAICompanionProvider(
        api_key=api_key.strip(),
        model=(getattr(settings, "GROQ_MODEL", None) or "openai/gpt-oss-20b").strip(),
        fallback_provider=MockAICompanionProvider(),
    )


def get_ai_companion_provider() -> BaseAICompanionProvider:
    """Resolve the provider once; in auto mode prefer Gemini, then Groq, then Mock.

    With both API keys configured, Groq is wired as Gemini's runtime fallback.
    Explicit AI_PROVIDER=mock always disables external providers.
    """
    global _companion_provider
    if _companion_provider is not None:
        return _companion_provider

    with _lock:
        if _companion_provider is not None:
            return _companion_provider

        provider_name = (getattr(settings, "AI_PROVIDER", "auto") or "auto").strip().lower()
        gemini_key = getattr(settings, "GEMINI_API_KEY", None) or getattr(settings, "GOOGLE_API_KEY", None)
        groq_key = getattr(settings, "GROQ_API_KEY", None)
        gemini_key = gemini_key.strip() if isinstance(gemini_key, str) and gemini_key.strip() else None
        groq_key = groq_key.strip() if isinstance(groq_key, str) and groq_key.strip() else None

        if provider_name == "auto":
            provider_name = "gemini" if gemini_key else "groq" if groq_key else "mock"

        if provider_name == "gemini":
            if not gemini_key:
                if groq_key:
                    logger.warning("Gemini selected without a Google API key; using Groq as primary provider.")
                    _companion_provider = _configured_groq_provider(groq_key)
                    provider_name = "groq"
                else:
                    logger.warning("Gemini selected but no Google or Groq API key is configured; using Mock.")
                    _companion_provider = MockAICompanionProvider()
                    provider_name = "mock"
            else:
                fallback = _configured_groq_provider(groq_key) if groq_key else MockAICompanionProvider()
                _companion_provider = GeminiAICompanionProvider(
                    api_key=gemini_key,
                    model=getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash"),
                    fallback_provider=fallback,
                )
        elif provider_name == "groq":
            if groq_key:
                _companion_provider = _configured_groq_provider(groq_key)
            else:
                logger.warning("Groq selected but GROQ_API_KEY is not configured; using Mock.")
                _companion_provider = MockAICompanionProvider()
                provider_name = "mock"
        elif provider_name == "mock":
            _companion_provider = MockAICompanionProvider()
        else:
            logger.error("Unsupported AI_PROVIDER=%r; using Mock.", provider_name)
            _companion_provider = MockAICompanionProvider()
            provider_name = "mock"

        logger.info("AI companion provider resolved to %s", provider_name)
        return _companion_provider


def get_moderation_provider() -> BaseModerationProvider:
    """Retrieve singleton AI moderation provider, instantiating based on settings if needed."""
    global _moderation_provider
    if _moderation_provider is not None:
        return _moderation_provider

    with _lock:
        if _moderation_provider is not None:
            return _moderation_provider

        provider_name = (getattr(settings, "MODERATION_PROVIDER", "mock") or "mock").strip().lower()
        gemini_key = getattr(settings, "GEMINI_API_KEY", None) or getattr(settings, "GOOGLE_API_KEY", None)

        if provider_name == "gemini":
            if gemini_key and gemini_key.strip():
                _moderation_provider = GeminiModerationProvider(
                    api_key=gemini_key.strip(),
                    model=getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash"),
                )
            else:
                logger.error("MODERATION_PROVIDER is 'gemini' but no Google API key is configured; moderation will fail closed.")
                _moderation_provider = GeminiModerationProvider(
                    api_key=None,
                    model=getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash"),
                )
        else:
            _moderation_provider = MockModerationProvider()

        return _moderation_provider


def set_ai_companion_provider(provider: Optional[BaseAICompanionProvider]) -> None:
    """Inject a custom AI companion provider (primarily for tests)."""
    global _companion_provider
    with _lock:
        _companion_provider = provider


def set_moderation_provider(provider: Optional[BaseModerationProvider]) -> None:
    """Inject a custom moderation provider (primarily for tests)."""
    global _moderation_provider
    with _lock:
        _moderation_provider = provider


def reset_providers() -> None:
    """Reset cached singleton providers to None (essential for test isolation)."""
    global _companion_provider, _moderation_provider
    with _lock:
        _companion_provider = None
        _moderation_provider = None
