"""Offline Mock Providers for AI Companion and Moderation.

Zero external dependencies, casual internet stranger persona, and deterministic scoring.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.models.envelopes import ChatMessagePayload, ServerChatMessagePayload

logger = logging.getLogger(__name__)

# Fallback pool of casual, authentic internet stranger responses
CASUAL_FALLBACK_REPLIES = [
    "haha yeah totally agree with that",
    "that's really interesting, tell me more about it!",
    "oh wow, really? never thought about it like that before",
    "yeah for sure! what else have you been up to today?",
    "that makes a lot of sense actually",
    "nice! always cool running into people who get that",
    "honestly same here haha",
    "sounds pretty fun! how long have you been doing that?",
]

KEYWORD_RESPONSES = [
    # Color
    (["favorite color", "favorite colour", "favourite color", "color", "what is your favorite color"], "definitely dark teal or forest green, feels really peaceful. what's yours?"),
    # Why like color
    (["why do you like", "why that color", "why"], "honestly it just reminds me of nature and hiking. always felt calming to look at."),
    # How are you
    (["how are you", "how are u", "how's it going", "hows it going", "how are you doing", "what's up", "whats up"], "not bad at all, just chilling and listening to some music. what about you?"),
    # Emotional / Support
    (["rough day", "bad day", "sad", "depressed", "stressed", "feeling down", "feeling sad", "terrible day", "having a rough"], "oh man, sorry to hear that. anything you want to vent about, or would you rather distract yourself?"),
    # Greetings
    ([r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\byo\b", r"\bsup\b", r"\bhowdy\b"], "hey there! how's your day going?"),
    # Books
    (["recommend a book", "good book", "favorite book", "book", "reading", "read"], "you should definitely check out 'Project Hail Mary' by Andy Weir, such an amazing read! do you like sci-fi?"),
    # Genre
    (["what genre", "genre"], "it's sci-fi / survival, lots of clever problem solving. are you into fiction or non-fiction mostly?"),
    # Movies
    (["favorite movie", "movie", "film", "cinema", "watch"], "interstellar is one of my all-time favorites, the soundtrack gives me chills. seen any good movies lately?"),
    # Music
    (["music", "song", "band", "listen"], "i listen to a lot of indie rock and lo-fi beats honestly. what kind of tunes are you into?"),
    # Weather
    (["weather", "rain", "sunny", "cold", "snow", "hot outside"], "pretty chilly outside here today, perfect excuse to stay in with tea. how's the weather by you?"),
    # Leaving
    (["bye", "cya", "leaving", "gotta go", "goodnight", "see you"], "take care! was nice chatting with you"),
    # Identity
    (["who are you", "what are you", "are you ai", "are you an ai", "are you a bot", "bot"], "I'm the AI companion here to keep you company. what brings you here?"),
]


class MockAICompanionProvider(BaseAICompanionProvider):
    """Deterministic, zero-dependency offline mock for AI stranger companion."""

    def __init__(
        self,
        custom_responses: Optional[Dict[str, str]] = None,
        fixed_reply: Optional[str] = None,
    ) -> None:
        self.custom_responses = custom_responses or {}
        self.fixed_reply = fixed_reply

    async def generate_reply(
        self,
        session_id: str,
        history: Sequence[Union[ChatMessagePayload, ServerChatMessagePayload, dict, Any]],
        latest_message: str,
    ) -> str:
        # Parameter validation contract enforcement
        await super().generate_reply(session_id, history, latest_message)

        if self.fixed_reply is not None:
            return self.fixed_reply

        clean_text = latest_message.strip().lower()

        # 1. Custom exact match overrides
        if clean_text in self.custom_responses:
            return self.custom_responses[clean_text]

        # 2. Keyword/Intent matching
        for keywords, response in KEYWORD_RESPONSES:
            for kw in keywords:
                if "\\" in kw:
                    if re.search(kw, clean_text):
                        return response
                elif kw in clean_text:
                    return response

        # 3. Contextual fallback rotation using session_id, message, and history length
        history_len = len(history) if history else 0
        h = int(hashlib.md5(f"{session_id}:{clean_text}:{history_len}".encode()).hexdigest(), 16)
        return CASUAL_FALLBACK_REPLIES[h % len(CASUAL_FALLBACK_REPLIES)]


class MockModerationProvider(BaseModerationProvider):
    """Deterministic offline mock for moderation toxicity scoring."""

    DEFAULT_TOXIC_PATTERNS = [
        "kill yourself",
        "kys",
        "kill you",
        "go die",
        "murder",
        "suicide",
        "hate you",
        "trash",
        "worthless",
        "stupid trash",
        "piece of trash",
        "bitch",
        "nazi",
        "die in a fire",
    ]

    DEFAULT_BORDERLINE_PATTERNS = [
        "shut up",
        "idiot",
        "damn",
        "crap",
        "stupid",
        "annoying",
        "dumb",
        "fool",
    ]

    def __init__(
        self,
        default_score: float = 0.0,
        toxic_score: float = 0.95,
        borderline_score: float = 0.5,
        toxic_patterns: Optional[List[str]] = None,
        borderline_patterns: Optional[List[str]] = None,
        custom_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        self.default_score = default_score
        self.toxic_score = toxic_score
        self.borderline_score = borderline_score
        self.toxic_patterns = [p.lower() for p in (toxic_patterns or self.DEFAULT_TOXIC_PATTERNS)]
        self.borderline_patterns = [p.lower() for p in (borderline_patterns or self.DEFAULT_BORDERLINE_PATTERNS)]
        self.custom_scores = {k.lower(): v for k, v in (custom_scores or {}).items()}

    def set_score(self, text_or_pattern: str, score: float) -> None:
        """Inject a custom score for testing."""
        self.custom_scores[text_or_pattern.lower()] = score

    async def score_toxicity(self, text: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0

        clean = text.strip().lower()
        if not clean:
            return 0.0

        # Exact match in custom_scores
        if clean in self.custom_scores:
            return self.custom_scores[clean]

        # Toxic pattern match (score >= 0.85)
        for pattern in self.toxic_patterns:
            if pattern in clean:
                return self.toxic_score

        # Borderline pattern match (intermediate score, e.g. 0.5)
        for pattern in self.borderline_patterns:
            if pattern in clean:
                return self.borderline_score

        # Substring match in custom_scores
        for pattern, score in self.custom_scores.items():
            if pattern in clean:
                return score

        return self.default_score
