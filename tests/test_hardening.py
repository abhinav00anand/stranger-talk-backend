"""Regression tests for chat routing, moderation, and WebSocket abuse controls."""

import asyncio

import pytest

from app.ai.companion_manager import CompanionManager
from app.moderation.pipeline import ModerationPipeline


class BrokenModerationProvider:
    async def moderate_text(self, text: str):
        raise RuntimeError("moderation provider unavailable")


@pytest.mark.asyncio
async def test_stage_two_moderation_failure_blocks_without_banning():
    pipeline = ModerationPipeline(enabled=True)
    result = await pipeline.moderate_text(
        "A harmless sentence for the provider to inspect.",
        provider=BrokenModerationProvider(),
    )
    assert result.is_blocked
    assert not result.is_bannable
    assert result.stage == "stage2_error_block"


@pytest.mark.asyncio
async def test_stage_two_moderation_provider_is_consulted():
    class FlaggingProvider:
        async def moderate_text(self, text: str):
            assert text == "A benign sentence that Stage 1 does not flag."
            return 0.99

    pipeline = ModerationPipeline(enabled=True)
    result = await pipeline.moderate_text(
        "A benign sentence that Stage 1 does not flag.",
        provider=FlaggingProvider(),
    )
    assert result.is_bannable
    assert result.stage == "stage2_deep"


def test_sliding_window_rate_limiter_enforces_budget_and_expiry():
    from app.main import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=1.0)
    assert limiter.allow(now=10.0)
    assert limiter.allow(now=10.1)
    assert not limiter.allow(now=10.2)
    assert limiter.allow(now=11.1)


@pytest.mark.asyncio
async def test_companion_manager_serializes_concurrent_turns():
    observed = []

    class RecordingProvider:
        async def generate_reply(self, session_id, history, latest_message):
            observed.append((latest_message, [item.text for item in history]))
            await asyncio.sleep(0.01)
            return f"reply to {latest_message}"

    manager = CompanionManager(provider=RecordingProvider(), typing_delay=0.0)
    manager.register_session("serial-session", "client-1")

    await asyncio.gather(
        manager.process_user_message("serial-session", "first"),
        manager.process_user_message("serial-session", "second"),
    )

    assert [entry[0] for entry in observed] == ["first", "second"]
    assert observed[1][1] == ["first", "reply to first"]
    assert [item.text for item in manager.get_history("serial-session")] == [
        "first", "reply to first", "second", "reply to second"
    ]


@pytest.mark.asyncio
async def test_websocket_chat_uses_configured_stage_two_provider(monkeypatch, make_client):
    import app.main as main

    class FlaggingProvider:
        async def moderate_text(self, text: str):
            return 0.99

    monkeypatch.setattr(main, "get_moderation_provider", lambda: FlaggingProvider())

    client_a = await make_client()
    client_b = await make_client()
    await client_a.join_queue()
    await client_b.join_queue()
    match_a = await client_a.recv_type("matched", timeout=5.0)
    match_b = await client_b.recv_type("matched", timeout=5.0)
    assert match_a["session_id"] == match_b["session_id"]

    await client_a.send_chat("A harmless message for the configured moderation provider.")
    banned = await client_a.recv_type("banned", timeout=5.0)
    assert banned["type"] == "banned"
    disconnected = await client_b.recv_type("peer_disconnected", timeout=5.0)
    assert disconnected["payload"]["reason"] == "banned"


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
