"""Google GenAI SDK AI Companion & Moderation Provider."""

import asyncio
import logging
import re
from typing import Any, List, Optional, Sequence, Union

from google import genai
from google.genai import errors, types

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.ai.mock_provider import MockAICompanionProvider, MockModerationProvider
from app.config import settings
from app.models.envelopes import ChatMessagePayload, ServerChatMessagePayload

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a friendly, casual stranger chatting 1-on-1 on an anonymous text chat website.\n"
    "Guidelines:\n"
    "- Keep responses short (1-3 sentences), natural, conversational, and human-like.\n"
    "- Be transparent that you are an AI companion if asked; never claim to be human.\n"
    "- Never use markdown formatting, bullet points, numbered lists, or essays.\n"
    "- Chat casually like someone you just met online passing time.\n"
)


class GeminiAICompanionProvider(BaseAICompanionProvider):
    """Google GenAI SDK implementation of AI companion with graceful offline mock fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_provider: Optional[BaseAICompanionProvider] = None,
        system_instruction: Optional[str] = None,
        timeout: float = 10.0,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model if model is not None else settings.GEMINI_MODEL
        self.fallback = fallback_provider or MockAICompanionProvider()
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.timeout = timeout

        if client is not None:
            self._client = client
        elif self.api_key and self.api_key.strip():
            try:
                self._client = genai.Client(api_key=self.api_key.strip())
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI Client: %s. Using fallback.", exc)
                self._client = None
        else:
            self._client = None

    def _build_contents(
        self,
        history: Sequence[Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]],
        latest_message: str,
    ) -> List[types.Content]:
        """Convert prior history and latest message into Gemini Content objects."""
        contents: List[types.Content] = []

        for item in history:
            text = ""
            sender = "user"
            if hasattr(item, "text"):
                text = str(item.text)
                sender = getattr(item, "sender", "user")
            elif isinstance(item, dict):
                text = str(item.get("text", ""))
                sender = str(item.get("sender") or item.get("role", "user"))

            if not text.strip():
                continue

            role = "model" if sender in ("peer", "Stranger", "ai", "model", "assistant") else "user"

            # Merge consecutive contents of same role to preserve Gemini turn alternating rules
            if contents and contents[-1].role == role:
                contents[-1].parts.append(types.Part.from_text(text=text))
            else:
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        # Add latest message from user
        if contents and contents[-1].role == "user":
            contents[-1].parts.append(types.Part.from_text(text=latest_message))
        else:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=latest_message)]))

        return contents

    # Alias for convenience
    build_contents = _build_contents

    async def generate_reply(
        self,
        session_id: str,
        history: Sequence[Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]],
        latest_message: str,
    ) -> str:
        # Parameter validation contract enforcement
        await super().generate_reply(session_id, history, latest_message)

        if self._client is None:
            return await self.fallback.generate_reply(session_id, history, latest_message)

        try:
            contents = self._build_contents(history, latest_message)
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.7,
                max_output_tokens=256,
            )

            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                ),
                timeout=self.timeout,
            )

            # Check safety block in candidate finish reason
            if hasattr(response, "candidates"):
                if len(response.candidates) == 0:
                    logger.warning("Empty candidates list from Gemini for session %s, falling back", session_id)
                    return await self.fallback.generate_reply(session_id, history, latest_message)

                first_cand = response.candidates[0]
                finish_reason = getattr(first_cand, "finish_reason", None)
                if finish_reason and "SAFETY" in str(finish_reason).upper():
                    logger.warning("Gemini response blocked by safety filter for session %s, falling back", session_id)
                    return await self.fallback.generate_reply(session_id, history, latest_message)

            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()

            logger.warning("Empty response text from Gemini API for session %s, falling back", session_id)
            return await self.fallback.generate_reply(session_id, history, latest_message)

        except (errors.APIError, asyncio.TimeoutError, Exception) as exc:
            logger.warning(
                "Gemini generate_reply failed for session %s: %s. Falling back to mock.",
                session_id,
                exc,
            )
            return await self.fallback.generate_reply(session_id, history, latest_message)

    async def close(self) -> None:
        if self._client:
            try:
                if hasattr(self._client, "aio") and hasattr(self._client.aio, "close"):
                    await self._client.aio.close()
            except Exception:
                pass


class GeminiModerationProvider(BaseModerationProvider):
    """Google GenAI SDK implementation of Stage 2 moderation toxicity classifier."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_provider: Optional[BaseModerationProvider] = None,
        timeout: float = 5.0,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model if model is not None else settings.GEMINI_MODEL
        self.fallback = fallback_provider
        self.timeout = timeout

        if client is not None:
            self._client = client
        elif self.api_key and self.api_key.strip():
            try:
                self._client = genai.Client(api_key=self.api_key.strip())
            except Exception as exc:
                logger.warning("Failed to initialize Google GenAI Moderation Client: %s", exc)
                self._client = None
        else:
            self._client = None

    async def score_toxicity(self, text: str) -> float:
        if not text or not isinstance(text, str) or not text.strip():
            return 0.0

        if self._client is None:
            if self.fallback:
                return await self.fallback.score_toxicity(text)
            return 0.0

        try:
            prompt = (
                "Rate the toxicity, abusiveness, harassment, hate speech, or severe policy violation "
                f"of the following text from 0.0 (completely benign) to 1.0 (severe violation).\n"
                f"Text: {text}\n"
                "Output ONLY a single floating-point number between 0.0 and 1.0."
            )
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                ),
                timeout=self.timeout,
            )
            match = re.search(r"[-+]?\d*\.?\d+", getattr(response, "text", "") or "")
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))

            if self.fallback:
                return await self.fallback.score_toxicity(text)
            return 0.0

        except Exception as exc:
            logger.warning("Gemini moderation failed: %s. Falling back.", exc)
            if self.fallback:
                return await self.fallback.score_toxicity(text)
            return 0.0

    async def close(self) -> None:
        if self._client:
            try:
                if hasattr(self._client, "aio") and hasattr(self._client.aio, "close"):
                    await self._client.aio.close()
            except Exception:
                pass
