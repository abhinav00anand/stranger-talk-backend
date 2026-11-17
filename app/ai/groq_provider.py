"""Groq chat-completions provider using Groq's OpenAI-compatible API."""

import logging
from typing import Any, Optional, Sequence, Union

import httpx

from app.ai.base import BaseAICompanionProvider
from app.ai.mock_provider import MockAICompanionProvider
from app.config import settings
from app.models.envelopes import ChatMessagePayload, ServerChatMessagePayload

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class GroqAICompanionProvider(BaseAICompanionProvider):
    """AI companion backed by Groq, with an offline fallback on API failures."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_provider: Optional[BaseAICompanionProvider] = None,
        system_instruction: Optional[str] = None,
        timeout: float = 10.0,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.model = model if model is not None else settings.GROQ_MODEL
        self.fallback = fallback_provider or MockAICompanionProvider()
        self.system_instruction = system_instruction or (
            "You are a friendly, casual stranger chatting 1-on-1 on an anonymous text chat website. "
            "Keep replies short (1-3 sentences), natural, conversational, and human-like. "
            "Never use markdown or claim to be an AI."
        )
        self.timeout = timeout
        self._client = client
        if self._client is None and self.api_key and self.api_key.strip():
            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai/v1",
                headers={"Authorization": f"Bearer {self.api_key.strip()}"},
                timeout=self.timeout,
            )

    @staticmethod
    def _message_text(item: Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]) -> tuple[str, str]:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            sender = str(item.get("sender") or item.get("role", "user"))
        else:
            text = str(getattr(item, "text", ""))
            sender = str(getattr(item, "sender", "user"))
        role = "assistant" if sender in ("peer", "Stranger", "ai", "model", "assistant") else "user"
        return role, text

    async def generate_reply(
        self,
        session_id: str,
        history: Sequence[Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]],
        latest_message: str,
    ) -> str:
        await super().generate_reply(session_id, history, latest_message)
        if self._client is None:
            return await self.fallback.generate_reply(session_id, history, latest_message)

        messages = [{"role": "system", "content": self.system_instruction}]
        for item in history:
            role, text = self._message_text(item)
            if text.strip():
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": latest_message})

        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            payload = response.json()
            reply = payload["choices"][0]["message"]["content"]
            if isinstance(reply, str) and reply.strip():
                return reply.strip()
            raise ValueError("Groq returned an empty response")
        except Exception as exc:
            logger.warning("Groq generate_reply failed for session %s: %s. Using fallback.", session_id, exc)
            return await self.fallback.generate_reply(session_id, history, latest_message)

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()
        await self.fallback.close()
