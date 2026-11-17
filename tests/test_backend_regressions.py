"""Regression tests for AI history, moderation outages, and matchmaking rollback."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.ai.companion_manager import CompanionManager, QueueWatchdog
from app.ai.gemini_provider import GeminiAICompanionProvider, GeminiModerationProvider
from app.matchmaker.queue import MatchmakerQueue
from app.matchmaker.session_manager import SessionManager
from app.models.envelopes import JoinQueuePayload


def test_gemini_history_preserves_assistant_role():
    provider = GeminiAICompanionProvider(api_key="", client=None)
    contents = provider._build_contents(
        [
            {"sender": "user", "text": "Hey"},
            {"sender": "assistant", "text": "Hello!"},
        ],
        "How are you?",
    )

    assert [item.role for item in contents] == ["user", "model", "user"]
    assert contents[0].parts[0].text == "Hey"
    assert contents[1].parts[0].text == "Hello!"
    assert contents[2].parts[0].text == "How are you?"


@pytest.mark.asyncio
async def test_gemini_moderation_unavailable_raises_instead_of_allowing():
    provider = GeminiModerationProvider(api_key="", client=None)

    with pytest.raises(RuntimeError, match="unavailable"):
        await provider.score_toxicity("a message that needs moderation")


def test_join_queue_topics_are_bounded_and_validated():
    assert JoinQueuePayload.model_validate({"topics": ["music", "technology"]}).topics == [
        "music",
        "technology",
    ]

    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": "technology"})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": [123]})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": ["x" * 41]})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": [str(i) for i in range(11)]})


@pytest.mark.asyncio
async def test_watchdog_rolls_back_session_when_match_delivery_fails():
    queue = MatchmakerQueue()
    sessions = SessionManager()
    companions = CompanionManager(session_manager=sessions, queue=queue)
    await queue.enqueue("disconnected-client")
    ticket = await queue.get_ticket("disconnected-client")
    ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=10)

    async def fail_delivery(_client_id, _envelope):
        raise RuntimeError("socket is gone")

    watchdog = QueueWatchdog(
        queue=queue,
        session_manager=sessions,
        companion_manager=companions,
        timeout=0,
        send_matched_callback=fail_delivery,
    )

    paired = await watchdog.check_timeouts()

    assert paired == []
    assert sessions.active_session_count() == 0
    assert companions.active_sessions_count() == 0
