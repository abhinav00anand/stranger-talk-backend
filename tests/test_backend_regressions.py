"""Regression tests for deterministic human-only matchmaking."""

import pytest
from pydantic import ValidationError

from app.matchmaker.queue import MatchmakerQueue
from app.matchmaker.session_manager import SessionManager
from app.models.envelopes import JoinQueuePayload


def test_join_queue_topics_are_bounded_and_validated():
    assert JoinQueuePayload.model_validate({"topics": ["music", "technology"]}).topics == ["music", "technology"]
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": "technology"})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": [123]})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": ["x" * 41]})
    with pytest.raises(ValidationError):
        JoinQueuePayload.model_validate({"topics": [str(i) for i in range(11)]})


@pytest.mark.asyncio
async def test_queue_pairs_two_distinct_humans_fifo():
    queue = MatchmakerQueue()
    await queue.enqueue("human-a")
    await queue.enqueue("human-b")
    assert await queue.try_pop_pair() == ("human-a", "human-b")
    assert await queue.try_pop_pair() is None
    assert queue.queue_size() == 0


@pytest.mark.asyncio
async def test_queue_prefers_shared_topics_then_falls_back_to_fifo():
    queue = MatchmakerQueue()
    await queue.enqueue("human-a", topics=["music"])
    await queue.enqueue("human-b", topics=["coding"])
    await queue.enqueue("human-c", topics=["MUSIC"])
    assert await queue.try_pop_pair() == ("human-a", "human-c")
    assert await queue.try_pop_pair() is None
    assert queue.queue_size() == 1


@pytest.mark.asyncio
async def test_session_manager_creates_two_distinct_human_participants():
    sessions = SessionManager()
    session = await sessions.create_session("human-a", "human-b")
    assert session.client_a == "human-a"
    assert session.client_b == "human-b"
    assert session.get_partner_id("human-a") == "human-b"
    assert session.get_partner_id("human-b") == "human-a"
    assert not hasattr(session, "is_ai")


@pytest.mark.asyncio
async def test_websocket_matches_two_real_clients_without_ai_metadata(make_client):
    client_a = await make_client()
    client_b = await make_client()
    await client_a.join_queue()
    queued = await client_a.recv_type("queued", timeout=3.0)
    assert queued["payload"]["position"] == 1
    await client_b.join_queue()
    match_a = await client_a.recv_type("matched", timeout=3.0)
    match_b = await client_b.recv_type("matched", timeout=3.0)
    assert match_a["session_id"] == match_b["session_id"]
    assert isinstance(match_a["payload"]["partner_id"], str)
    assert isinstance(match_b["payload"]["partner_id"], str)
    assert match_a["payload"]["partner_id"] != match_b["payload"]["partner_id"]
    assert "peer_is_ai" not in match_a["payload"]
    assert "peer_is_ai" not in match_b["payload"]
    assert match_a["payload"]["is_initiator"] is True
    assert match_b["payload"]["is_initiator"] is False
