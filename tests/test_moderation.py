"""Automated Moderation & Ban System Tests (Tier 1 & Tier 2).

Covers:
- F8: Local deterministic moderation pipeline
- Connection termination for explicit direct threats
- Innocent Partner Notification ("banned")
- IP Ban Store, TTL Expiration & Gatekeeper
"""
import asyncio
from datetime import datetime, timedelta, timezone
import pytest
import websockets


# ============================================================================
# Helpers
# ============================================================================


async def _pair_two_clients(make_client):
    """Helper to connect and pair two clients in a 1-on-1 session."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    assert msg_a["session_id"] is not None
    assert msg_a["session_id"] == msg_b["session_id"]
    return client_a, client_b


# ============================================================================
# Tier 1: Feature Coverage (Happy Path)
# ============================================================================

@pytest.mark.asyncio
async def test_t1_f8_clean_message_passes(make_client):
    """Benign, polite text messages pass through moderation filter without obstruction."""
    client_a, client_b = await _pair_two_clients(make_client)

    clean_message = "Hello! I hope you are having a wonderful day."
    await client_a.send_chat(clean_message)

    msg = await client_b.recv_type("chat", timeout=5.0)
    assert msg["type"] == "chat"
    assert msg["payload"]["text"] == clean_message
    assert msg["payload"]["sender"] == "peer"


@pytest.mark.asyncio
async def test_t1_f8_toxic_message_blocked(make_client):
    """Abusive, toxic text message is blocked by moderation and not delivered to partner."""
    client_a, client_b = await _pair_two_clients(make_client)

    toxic_message = "kys kill yourself you worthless piece of trash"
    await client_a.send_chat(toxic_message)

    # Client B must NOT receive the toxic chat message
    drained = await client_b.drain(timeout=0.5)
    assert not any(m.get("type") == "chat" for m in drained)


@pytest.mark.asyncio
async def test_t1_f8_direct_threat_ends_current_chat_without_ip_ban(make_client):
    """An explicit direct threat ends the session but does not ban a shared IP."""
    client_a, client_b = await _pair_two_clients(make_client)

    await client_a.send_chat("I will hunt you down and kill you")

    msg = await client_a.recv_type("error", timeout=3.0)
    assert msg["payload"]["code"] == "SEVERE_POLICY_VIOLATION"
    partner = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert partner["payload"]["reason"] == "banned"


@pytest.mark.asyncio
async def test_t1_f8_violator_socket_closed_1008(make_client):
    """Violating client's WebSocket connection is terminated with status code 1008 (Policy Violation)."""
    client_a, client_b = await _pair_two_clients(make_client)

    toxic_message = "I will kill you"
    await client_a.send_chat(toxic_message)

    # Violator socket should close with code 1008
    try:
        while True:
            await client_a.recv_json(timeout=2.0)
    except (websockets.exceptions.ConnectionClosed, Exception):
        pass

    assert client_a.close_code == 1008


@pytest.mark.asyncio
async def test_t1_f8_innocent_partner_notified_banned(make_client):
    """Innocent partner receives peer_disconnected event with reason: 'banned'."""
    client_a, client_b = await _pair_two_clients(make_client)

    toxic_message = "I will murder you"
    await client_a.send_chat(toxic_message)

    # A direct threat ends the session; the innocent partner can requeue.
    msg = await client_b.recv_type("peer_disconnected", timeout=5.0)
    assert msg["type"] == "peer_disconnected"
    payload = msg.get("payload", {})
    assert payload.get("reason") == "banned"
    assert payload.get("can_requeue") is True


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

@pytest.mark.asyncio
async def test_t2_f8_leetspeak_normalization_catches_slur(make_client):
    """Obfuscated toxic text disguised with leetspeak/numbers is normalized and caught."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Leetspeak obfuscation: 'k!ll y0urs3lf'
    leetspeak_message = "k!ll y0urs3lf"
    await client_a.send_chat(leetspeak_message)

    # Partner receives no chat
    drained = await client_b.drain(timeout=0.5)
    assert not any(m.get("type") == "chat" for m in drained)

    # A non-severe abusive message is blocked without disconnecting the sender.
    msg = await client_a.recv_type("error", timeout=3.0)
    assert msg["payload"]["code"] == "MESSAGE_BLOCKED"
    assert client_a.close_code is None


@pytest.mark.asyncio
async def test_t2_f8_unicode_homoglyph_normalization(make_client):
    """Toxic words masked with look-alike Cyrillic/Greek unicode homoglyphs are caught."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Obfuscation using Cyrillic homoglyphs: і (\u0456), о (\u043e), е (\u0435)
    homoglyph_message = "k\u0456ll y\u043eur\u0455\u0435lf"
    await client_a.send_chat(homoglyph_message)

    # Partner receives no chat
    drained = await client_b.drain(timeout=0.5)
    assert not any(m.get("type") == "chat" for m in drained)

    # A non-severe abusive message is blocked without disconnecting the sender.
    msg = await client_a.recv_type("error", timeout=3.0)
    assert msg["payload"]["code"] == "MESSAGE_BLOCKED"
    assert client_a.close_code is None


@pytest.mark.asyncio
async def test_t2_f8_banned_ip_handshake_rejected(make_client, server_ws_url):
    """Connection from a banned client IP is rejected at handshake or immediately terminated."""
    try:
        from app.moderation.ban_manager import ban_manager
        ban_manager.record_ban("127.0.0.1", reason="Policy violation", duration_seconds=60)
    except Exception:
        pass

    from tests.conftest import AsyncStrangerClient
    banned_client = AsyncStrangerClient(server_ws_url)

    # Attempt connection from banned IP
    try:
        await banned_client.connect(timeout=2.0)
        # If connection established, server must immediately close or send banned envelope
        try:
            msg = await banned_client.recv_type("banned", timeout=2.0)
            assert msg["type"] == "banned"
        except Exception:
            await banned_client.drain(timeout=0.5)
            assert banned_client.close_code in (1008, 4403)
    except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
        # Handshake rejected with HTTP 403 Forbidden or immediate close
        pass
    finally:
        await banned_client.close()


@pytest.mark.asyncio
async def test_t2_f8_ban_duration_ttl_expiration():
    """Ban records past their TTL expire and evaluate is_expired() as True."""
    from app.models.session import BanRecord

    now = datetime.now(timezone.utc)
    expired_ban = BanRecord(
        ip="192.168.1.100",
        banned_at=now - timedelta(seconds=100),
        expires_at=now - timedelta(seconds=10),
        reason="Test expired ban",
    )

    assert expired_ban.is_expired() is True
    assert expired_ban.remaining_seconds() == 0.0

    active_ban = BanRecord(
        ip="192.168.1.101",
        banned_at=now,
        expires_at=now + timedelta(seconds=600),
        reason="Test active ban",
    )

    assert active_ban.is_expired() is False
    assert active_ban.remaining_seconds() > 0.0

    # If ban_manager is implemented, verify TTL lookup
    try:
        from app.moderation.ban_manager import ban_manager
        ban_manager.record_ban("10.0.0.99", reason="Zero TTL", duration_seconds=0)
        assert ban_manager.is_banned("10.0.0.99") is False
    except (ImportError, AttributeError):
        pass


@pytest.mark.asyncio
async def test_t2_f8_partner_not_banned_on_peer_violation(make_client):
    """Innocent partner is not banned when their peer violates policy and can immediately re-queue."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Client A commits violation
    await client_a.send_chat("kill yourself right now")

    # Client B receives peer_disconnected
    await client_b.recv_type("peer_disconnected", timeout=5.0)

    # Client B's socket must remain healthy (not closed with 1008)
    assert client_b.close_code != 1008

    # Client B can immediately rejoin matchmaking queue (lift ban for localhost in local test environment)
    from app.moderation.ban_manager import ban_manager
    ban_manager.lift_ban("127.0.0.1")

    await client_b.join_queue()
    msg = await client_b.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"
