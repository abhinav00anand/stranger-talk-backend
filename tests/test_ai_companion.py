"""AI Fallback Companion Tests (Tier 1 & Tier 2).

Covers:
- F7: AI Fallback Companion Watchdog, Persona, Typing Cadence & Session Parity
"""
import asyncio
from datetime import datetime, timezone
import pytest


# ============================================================================
# Helpers & Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def fast_ai_timeout(monkeypatch):
    """Speed up AI fallback watchdog timer for snappy test execution when settings are loaded."""
    try:
        from app.config import settings
        # Set short timeout so tests do not wait 5+ seconds
        monkeypatch.setattr(settings, "AI_FALLBACK_TIMEOUT_SECONDS", 0.5)
    except Exception:
        pass


async def _pair_with_ai(make_client, timeout: float = 8.0):
    """Helper to connect a single client, wait for queue timeout, and pair with AI companion."""
    client = await make_client()
    await client.join_queue()
    await client.recv_type("queued", timeout=3.0)

    msg = await client.recv_type("matched", timeout=timeout)
    assert msg["type"] == "matched"
    assert msg.get("session_id") is not None
    assert msg.get("payload", {}).get("peer_is_ai") is True
    return client, msg


# ============================================================================
# Tier 1: Feature Coverage (Happy Path)
# ============================================================================

@pytest.mark.asyncio
async def test_t1_f7_fallback_triggered_on_timeout(make_client):
    """Lone client waiting in queue is automatically matched with AI companion after timeout."""
    client, match_msg = await _pair_with_ai(make_client)

    payload = match_msg.get("payload", {})
    assert payload.get("peer_is_ai") is True
    assert match_msg.get("session_id") is not None


@pytest.mark.asyncio
async def test_t1_f7_ai_companion_peer_alias(make_client):
    """AI companion match event provides a friendly peer alias (e.g. 'Stranger')."""
    client, match_msg = await _pair_with_ai(make_client)

    payload = match_msg.get("payload", {})
    alias = payload.get("peer_alias", "")
    assert isinstance(alias, str)
    assert len(alias) > 0
    assert alias in ("Stranger", "AI Companion", "ChatFreely Companion") or bool(alias)


@pytest.mark.asyncio
async def test_t1_f7_ai_responds_to_chat(make_client):
    """Sending a text message to AI companion yields a valid, non-empty conversational reply."""
    client, _ = await _pair_with_ai(make_client)

    await client.send_chat("Hello! How is your day going?")

    msg = await client.recv_type("chat", timeout=8.0)
    assert msg["type"] == "chat"
    assert msg["session_id"] == client.session_id

    payload = msg.get("payload", {})
    text = payload.get("text", "")
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert payload.get("sender") in ("peer", "Stranger", "ai")


@pytest.mark.asyncio
async def test_t1_f7_ai_typing_before_response(make_client):
    """AI companion emits a typing indicator event prior to or alongside delivering its reply."""
    client, _ = await _pair_with_ai(make_client)

    await client.send_chat("Tell me an interesting fun fact!")

    received_typing = False
    start_time = asyncio.get_event_loop().time()
    # Read incoming messages until chat reply is delivered
    while asyncio.get_event_loop().time() - start_time < 8.0:
        envelope = await client.recv_json(timeout=4.0)
        msg_type = envelope.get("type")
        if msg_type == "typing" and envelope.get("payload", {}).get("is_typing") is True:
            received_typing = True
        elif msg_type == "chat":
            break

    assert received_typing is True


@pytest.mark.asyncio
async def test_t1_f7_ai_skip_requeues_user(make_client):
    """User invoking skip during an AI session re-enters the matchmaking queue seamlessly."""
    client, _ = await _pair_with_ai(make_client)

    await client.skip()

    # Client must receive queued confirmation
    msg = await client.recv_type("queued", timeout=5.0)
    assert msg["type"] == "queued"


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

@pytest.mark.asyncio
async def test_t2_f7_human_join_cancels_ai_timer(make_client):
    """Arrival of a second human peer cancels the AI fallback timer and pairs humans."""
    client_a = await make_client()
    client_b = await make_client()

    # Client A joins queue
    await client_a.join_queue()
    await client_a.recv_type("queued", timeout=3.0)

    # Client B joins immediately before AI fallback fires
    await client_b.join_queue()

    # Both clients should be matched with each other
    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    assert msg_a["session_id"] == msg_b["session_id"]
    # Neither should be marked as AI
    assert msg_a.get("payload", {}).get("peer_is_ai") is not True
    assert msg_b.get("payload", {}).get("peer_is_ai") is not True


@pytest.mark.asyncio
async def test_t2_f7_ai_sliding_context_window(make_client):
    """Multi-turn conversation with AI companion functions reliably across successive turns."""
    client, _ = await _pair_with_ai(make_client)

    prompts = [
        "Hi, what is your favorite color?",
        "Why do you like that color?",
        "Can you recommend a good book?",
        "What genre is that?",
    ]

    for prompt in prompts:
        await client.send_chat(prompt)
        reply = await client.recv_type("chat", timeout=8.0)
        text = reply.get("payload", {}).get("text", "")
        assert isinstance(text, str)
        assert len(text.strip()) > 0


@pytest.mark.asyncio
async def test_t2_f7_ai_leave_returns_to_idle(make_client):
    """User invoking leave during an AI session cleanly exits to idle and can re-queue."""
    client, _ = await _pair_with_ai(make_client)

    await client.leave()

    # Drain any leave acknowledgement / quiet buffer
    await client.drain(timeout=0.5)

    # User re-queues successfully from idle
    await client.join_queue()
    msg = await client.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"


@pytest.mark.asyncio
async def test_t2_f7_ai_disconnect_cleanup(make_client):
    """Abrupt client socket drop during AI companion session terminates session cleanly."""
    client_a, _ = await _pair_with_ai(make_client)

    # Client abruptly closes socket
    await client_a.close(code=1000)

    # Subsequent new user can join queue without interference
    client_b = await make_client()
    await client_b.join_queue()
    msg = await client_b.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"


@pytest.mark.asyncio
async def test_t2_f7_ai_toxic_message_moderated(make_client):
    """Toxic message sent to AI companion triggers moderation filter and ban identically to human sessions."""
    client, _ = await _pair_with_ai(make_client)

    # Send toxic content
    toxic_text = "I hate you, kill yourself you stupid trash"
    await client.send_chat(toxic_text)

    # Violator receives banned event or socket is closed with 1008
    try:
        msg = await client.recv_type("banned", timeout=3.0)
        assert msg["type"] == "banned"
        assert "reason" in msg.get("payload", {})
    except Exception:
        # Check if socket closed with policy violation code
        await client.drain(timeout=0.5)
        assert client.close_code in (1008, 4403)
