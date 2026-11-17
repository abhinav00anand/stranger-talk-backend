"""Provider Factory with Singleton Resolution, Environment Config, and Test Overrides."""

import logging
import threading
from typing import Optional

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.ai.gemini_provider import GeminiAICompanionProvider, GeminiModerationProvider
from app.ai.mock_provider import MockAICompanionProvider, MockModerationProvider
from app.config import settings

logger = logging.getLogger(__name__)

_companion_provider: Optional[BaseAICompanionProvider] = None
_moderation_provider: Optional[BaseModerationProvider] = None
_lock = threading.Lock()


def get_ai_companion_provider() -> BaseAICompanionProvider:
    """Retrieve singleton AI companion provider, instantiating based on settings if needed."""
    global _companion_provider
    if _companion_provider is not None:
        return _companion_provider

    with _lock:
        if _companion_provider is not None:
            return _companion_provider

        provider_name = (getattr(settings, "AI_PROVIDER", "mock") or "mock").strip().lower()
        gemini_key = getattr(settings, "GEMINI_API_KEY", None) or getattr(settings, "GOOGLE_API_KEY", None)

        if provider_name == "gemini":
            if gemini_key and gemini_key.strip():
                _companion_provider = GeminiAICompanionProvider(
                    api_key=gemini_key.strip(),
                    model=getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash"),
                )
            else:
                logger.warning("AI_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not configured. Falling back to Mock.")
                _companion_provider = MockAICompanionProvider()
        else:
            # Handles 'mock' and any unknown/unsupported providers gracefully
            _companion_provider = MockAICompanionProvider()

        return _companion_provider


def get_moderation_provider() -> BaseModerationProvider:
    """Retrieve singleton moderation provider, instantiating based on settings if needed."""
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
                logger.error("MODERATION_PROVIDER is 'gemini' but GEMINI_API_KEY is unset; moderation will fail closed.")
                _moderation_provider = GeminiModerationProvider(api_key=None, model=getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash"))
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
