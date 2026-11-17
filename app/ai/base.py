"""Abstract Base Classes for AI Companion and Moderation Providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union
from app.models.envelopes import ChatMessagePayload, ServerChatMessagePayload


class BaseAICompanionProvider(ABC):
    """Abstract interface for AI conversational companion providers."""

    @abstractmethod
    async def generate_reply(
        self,
        session_id: str,
        history: Sequence[Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]],
        latest_message: str,
    ) -> str:
        """Generate a natural, conversational response given session context and latest message.

        Args:
            session_id: Active session identifier. Must be non-empty string.
            history: Sliding window sequence of prior messages.
            latest_message: The latest text message sent by the user. Must be non-empty string.

        Returns:
            The AI companion reply string.

        Raises:
            ValueError: If session_id or latest_message is empty or invalid.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not latest_message or not isinstance(latest_message, str) or not latest_message.strip():
            raise ValueError("latest_message must be a non-empty string")
        return ""

    async def close(self) -> None:
        """Release any underlying client sessions or resources."""
        pass

    async def __aenter__(self) -> "BaseAICompanionProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class BaseModerationProvider(ABC):
    """Abstract interface for AI toxicity and content moderation providers."""

    @abstractmethod
    async def score_toxicity(self, text: str) -> float:
        """Score the toxicity of input text.

        Args:
            text: Text content to evaluate.

        Returns:
            Float toxicity score between 0.0 (completely benign) and 1.0 (severe violation).
        """
        pass

    async def moderate_text(self, text: str) -> float:
        """Convenience delegation method for ModerationPipeline compatibility."""
        return await self.score_toxicity(text)

    async def close(self) -> None:
        """Release any underlying client sessions or resources."""
        pass

    async def __aenter__(self) -> "BaseModerationProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
