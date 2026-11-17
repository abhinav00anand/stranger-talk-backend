"""Regression tests for local moderation and WebSocket hardening."""

import pytest

from app.moderation.pipeline import ModerationPipeline


def test_local_text_moderation_allows_benign_text():
    result = ModerationPipeline(enabled=True).moderate_text_fastpath("A harmless sentence for another person.")
    assert result.is_clean
    assert result.stage == "stage1_clean"


def test_sliding_window_rate_limiter_enforces_budget_and_expiry():
    from app.main import SlidingWindowRateLimiter
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=1.0)
    assert limiter.allow(now=10.0)
    assert limiter.allow(now=10.1)
    assert not limiter.allow(now=10.2)
    assert limiter.allow(now=11.1)


@pytest.mark.asyncio
async def test_websocket_chat_relays_between_two_humans(make_client):
    client_a = await make_client()
    client_b = await make_client()
    await client_a.join_queue()
    await client_b.join_queue()
    match_a = await client_a.recv_type("matched", timeout=5.0)
    match_b = await client_b.recv_type("matched", timeout=5.0)
    assert match_a["session_id"] == match_b["session_id"]
    await client_a.send_chat("Hello, real stranger!", session_id=match_a["session_id"])
    delivered = await client_b.recv_type("chat", timeout=3.0)
    assert delivered["payload"]["text"] == "Hello, real stranger!"
    assert delivered["payload"]["sender"] == "peer"


@pytest.mark.asyncio
async def test_websocket_rejects_stale_session_chat(make_client):
    client_a = await make_client()
    client_b = await make_client()
    await client_a.join_queue()
    await client_b.join_queue()
    match_a = await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)
    await client_a.send_chat("This must not be delivered.", session_id="stale-session")
    error = await client_a.recv_type("error", timeout=3.0)
    assert error["payload"]["code"] == "INVALID_SESSION"
    partner_messages = await client_b.drain(timeout=0.3)
    assert not any(message.get("type") == "chat" for message in partner_messages)
    await client_a.send_chat("This valid message should still work.", session_id=match_a["session_id"])
    delivered = await client_b.recv_type("chat", timeout=3.0)
    assert delivered["payload"]["text"] == "This valid message should still work."


@pytest.mark.asyncio
async def test_websocket_closes_oversized_frame(stranger_client):
    assert stranger_client.ws is not None
    await stranger_client.ws.send("x" * (262_144 + 1))
    try:
        await stranger_client.ws.recv()
    except Exception:
        pass
    assert stranger_client.close_code == 1009
