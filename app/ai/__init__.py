"""AI Fallback Companion and Content Moderation Provider subsystem."""

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.ai.companion_manager import AISessionState, CompanionManager, QueueWatchdog
from app.ai.factory import (
    get_ai_companion_provider,
    get_moderation_provider,
    reset_providers,
    set_ai_companion_provider,
    set_moderation_provider,
)
from app.ai.gemini_provider import GeminiAICompanionProvider, GeminiModerationProvider
from app.ai.mock_provider import MockAICompanionProvider, MockModerationProvider

__all__ = [
    "AISessionState",
    "BaseAICompanionProvider",
    "BaseModerationProvider",
    "CompanionManager",
    "GeminiAICompanionProvider",
    "GeminiModerationProvider",
    "MockAICompanionProvider",
    "MockModerationProvider",
    "QueueWatchdog",
    "get_ai_companion_provider",
    "get_moderation_provider",
    "reset_providers",
    "set_ai_companion_provider",
    "set_moderation_provider",
]
