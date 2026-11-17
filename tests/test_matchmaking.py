"""Matchmaking and Session Lifecycle Tests (Tier 1 & Tier 2).

Covers:
- F1: Anonymous Matchmaking & FIFO Pairing
- F2: Session Skip Transition
- F3: Session Leave Transition & Disconnect Detection
"""
import asyncio
from datetime import datetime
import pytest


# ============================================================================
# Tier 1: Feature Coverage (Happy Path)
# ============================================================================

@pytest.mark.asyncio
async def test_t1_f1_join_queue_returns_queued(make_client):
    """Client joins queue and receives queued status confirmation."""
    client = await make_client()
    await client.join_queue()

    msg = await client.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"
    assert "payload" in msg
    assert msg.get("session_id") is None
    payload = msg["payload"]
    assert "position" in payload or "estimated_wait_seconds" in payload


@pytest.mark.asyncio
async def test_t1_f1_two_clients_matched(make_client):
    """Two clients joining queue are paired into the same 1-on-1 session."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    assert msg_a["type"] == "matched"
    assert msg_b["type"] == "matched"
    assert msg_a["session_id"] is not None
    assert msg_a["session_id"] == msg_b["session_id"]


@pytest.mark.asyncio
async def test_t1_f1_stun_config_delivered(make_client):
    """Matched event payload contains public STUN server configuration."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg = await client_a.recv_type("matched", timeout=5.0)
    payload = msg.get("payload", {})

    assert "ice_servers" in payload
    ice_servers = payload["ice_servers"]
    assert isinstance(ice_servers, list)
    assert len(ice_servers) > 0

    all_urls = []
    for entry in ice_servers:
        urls = entry.get("urls", [])
        if isinstance(urls, list):
            all_urls.extend(urls)
        elif isinstance(urls, str):
            all_urls.append(urls)

    assert any("stun:stun.l.google.com:19302" in u for u in all_urls)


@pytest.mark.asyncio
async def test_t1_f1_initiator_assigned(make_client):
    """Exactly one matched peer is designated as initiator to prevent glare."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    init_a = bool(msg_a.get("payload", {}).get("is_initiator"))
    init_b = bool(msg_b.get("payload", {}).get("is_initiator"))

    # Exactly one peer must be initiator
    assert init_a != init_b


@pytest.mark.asyncio
async def test_t1_f1_leave_queue_cancels(make_client):
    """Client can cancel queue wait with leave_queue and avoid pairing."""
    client_a = await make_client()
    await client_a.join_queue()
    await client_a.recv_type("queued", timeout=3.0)

    # Cancel queue entry
    await client_a.leave_queue()

    # Now a second client enters; client_a should NOT match with client_b
    client_b = await make_client()
    await client_b.join_queue()

    msg_b = await client_b.recv_type("queued", timeout=3.0)
    assert msg_b["type"] == "queued"

    # Client A should not receive a matched event
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await client_a.recv_type("matched", timeout=1.0)


@pytest.mark.asyncio
async def test_t1_f2_skip_requeues_initiator(make_client):
    """Skipping a session automatically re-queues the skipping client."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Client A skips
    await client_a.skip()

    msg_a = await client_a.recv_type("queued", timeout=3.0)
    assert msg_a["type"] == "queued"


@pytest.mark.asyncio
async def test_t1_f2_skip_notifies_partner(make_client):
    """Partner receives peer_disconnected notification with reason 'skipped'."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.skip()

    msg_b = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert msg_b["type"] == "peer_disconnected"
    assert msg_b.get("payload", {}).get("reason") == "skipped"


@pytest.mark.asyncio
async def test_t1_f2_partner_can_requeue(make_client):
    """Partner's peer_disconnected envelope specifies can_requeue is true."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.skip()

    msg_b = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert msg_b.get("payload", {}).get("can_requeue") is True


@pytest.mark.asyncio
async def test_t1_f2_skip_cleans_session_state(make_client):
    """Sending chat messages to a skipped/terminated session fails."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)
    old_session_id = msg_a["session_id"]

    await client_a.skip()
    await client_b.recv_type("peer_disconnected", timeout=3.0)

    # Client B attempts to send chat to terminated session
    await client_b.send_chat("Hello?", session_id=old_session_id)
    err = await client_b.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") in ("NOT_IN_SESSION", "SESSION_TERMINATED")


@pytest.mark.asyncio
async def test_t1_f2_subsequent_match_after_skip(make_client):
    """A skipping user cleanly matches with a third available client."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg1 = await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)
    first_session_id = msg1["session_id"]

    # Client A skips and is requeued
    await client_a.skip()
    await client_a.recv_type("queued", timeout=3.0)

    # Client C joins queue
    client_c = await make_client()
    await client_c.join_queue()

    # Client A and C should match
    msg_a2 = await client_a.recv_type("matched", timeout=5.0)
    msg_c = await client_c.recv_type("matched", timeout=5.0)

    assert msg_a2["session_id"] == msg_c["session_id"]
    assert msg_a2["session_id"] != first_session_id


@pytest.mark.asyncio
async def test_t1_f3_leave_transitions_to_idle(make_client):
    """Matched user sending leave transitions to idle without requeueing."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.leave()

    # Drain any message to client_a; it should NOT receive 'queued'
    drained = await client_a.drain(timeout=0.5)
    types = [m.get("type") for m in drained]
    assert "queued" not in types


@pytest.mark.asyncio
async def test_t1_f3_leave_notifies_partner(make_client):
    """Partner receives peer_disconnected notification with reason 'left'."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.leave()

    msg_b = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert msg_b["type"] == "peer_disconnected"
    assert msg_b.get("payload", {}).get("reason") == "left"


@pytest.mark.asyncio
async def test_t1_f3_ungraceful_disconnect_detected(make_client):
    """Abrupt WebSocket closure by peer triggers peer_disconnected with reason 'disconnected'."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Client A disconnects abruptly
    await client_a.close(code=1001)

    msg_b = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert msg_b["type"] == "peer_disconnected"
    assert msg_b.get("payload", {}).get("reason") == "disconnected"


@pytest.mark.asyncio
async def test_t1_f3_leave_cleans_pool(make_client):
    """After leave, neither participant can send messages on the old session."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg = await client_a.recv_type("matched", timeout=5.0)
    session_id = msg["session_id"]

    await client_a.leave()

    # Attempting to send chat on destroyed session returns NOT_IN_SESSION
    await client_a.send_chat("Testing", session_id=session_id)
    err = await client_a.recv_type("error", timeout=3.0)
    assert err.get("payload", {}).get("code") in ("NOT_IN_SESSION", "SESSION_TERMINATED")


@pytest.mark.asyncio
async def test_t1_f3_reconnect_after_leave(make_client):
    """Client that leaves a session can join the queue again successfully."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.leave()

    # Client A re-joins queue
    await client_a.join_queue()
    msg = await client_a.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

@pytest.mark.asyncio
async def test_t2_f1_duplicate_join_queue_rejected(make_client):
    """Sending join_queue twice returns ALREADY_QUEUED error."""
    client = await make_client()
    await client.join_queue()
    await client.recv_type("queued", timeout=3.0)

    # Second join_queue
    await client.join_queue()
    err = await client.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") == "ALREADY_QUEUED"


@pytest.mark.asyncio
async def test_t2_f1_empty_queue_no_match(make_client):
    """A lone client in the matchmaking queue remains queued without ghost pairing."""
    client = await make_client()
    await client.join_queue()

    msg = await client.recv_type("queued", timeout=3.0)
    assert msg["type"] == "queued"

    # Ensure no human match message arrives immediately
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        msg = await client.recv_type("matched", timeout=1.0)
        # If matched arrives, it must NOT be an empty human peer
        assert msg.get("payload", {}).get("peer_is_ai") is True


@pytest.mark.asyncio
async def test_t2_f1_rapid_join_and_cancel(make_client):
    """Client rapidly sending join_queue then leave_queue transitions cleanly."""
    client = await make_client()
    # Rapid send back to back
    await client.join_queue()
    await client.leave_queue()

    drained = await client.drain(timeout=0.5)
    # No error code should occur
    error_msgs = [m for m in drained if m.get("type") == "error"]
    assert len(error_msgs) == 0


@pytest.mark.asyncio
async def test_t2_f1_disconnect_while_queued(make_client):
    """Client disconnecting while queued is purged and does not pair with newcomers."""
    client_a = await make_client()
    await client_a.join_queue()
    await client_a.recv_type("queued", timeout=3.0)

    # Disconnect Client A while queued
    await client_a.close(code=1000)

    # Client B joins
    client_b = await make_client()
    await client_b.join_queue()

    msg_b = await client_b.recv_type("queued", timeout=3.0)
    assert msg_b["type"] == "queued"

    # Client B should not match with dead Client A
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await client_b.recv_type("matched", timeout=1.0)


@pytest.mark.asyncio
async def test_t2_f1_odd_number_of_clients(make_client):
    """3 clients join queue: first 2 match, 3rd remains queued."""
    client_a = await make_client()
    client_b = await make_client()
    client_c = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()
    await client_c.join_queue()

    # First two match
    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)
    assert msg_a["session_id"] == msg_b["session_id"]

    # Client C remains queued
    msg_c = await client_c.recv_type("queued", timeout=3.0)
    assert msg_c["type"] == "queued"


@pytest.mark.asyncio
async def test_t2_f2_skip_when_not_in_session(make_client):
    """Sending skip while idle returns NOT_IN_SESSION error."""
    client = await make_client()
    await client.skip()

    err = await client.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") == "NOT_IN_SESSION"


@pytest.mark.asyncio
async def test_t2_f2_simultaneous_skip_race(make_client):
    """Both clients sending skip simultaneously is handled without crash or unhandled error."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Fire both skips concurrently
    await asyncio.gather(
        client_a.skip(),
        client_b.skip(),
    )

    # Both clients must either receive queued or peer_disconnected cleanly
    drained_a = await client_a.drain(timeout=1.0)
    drained_b = await client_b.drain(timeout=1.0)

    types_a = [m.get("type") for m in drained_a]
    types_b = [m.get("type") for m in drained_b]

    assert any(t in ("queued", "peer_disconnected") for t in types_a), "Client A did not receive queued or peer_disconnected"
    assert any(t in ("queued", "peer_disconnected") for t in types_b), "Client B did not receive queued or peer_disconnected"

    # No 500 or fatal unhandled error in payloads
    for m in drained_a + drained_b:
        if m.get("type") == "error":
            assert m.get("payload", {}).get("code") in ("NOT_IN_SESSION", "ALREADY_QUEUED")


@pytest.mark.asyncio
async def test_t2_f2_rapid_consecutive_skips(make_client):
    """Client sending skip multiple times in rapid succession is handled safely."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Send 3 skips rapidly
    await client_a.skip()
    await client_a.skip()
    await client_a.skip()

    drained = await client_a.drain(timeout=1.0)
    types = [m.get("type") for m in drained]
    # First skip yields queued
    assert "queued" in types or "matched" in types


@pytest.mark.asyncio
async def test_t2_f2_skip_during_partner_disconnect(make_client):
    """Partner disconnects right as user sends skip: clean requeue without hanging."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Concurrently close B and skip A
    await asyncio.gather(
        client_b.close(code=1000),
        client_a.skip(),
    )

    drained_a = await client_a.drain(timeout=1.0)
    types_a = [m.get("type") for m in drained_a]
    # Client A should cleanly reach queued or handle disconnect
    assert any(t in ("queued", "peer_disconnected") for t in types_a)


@pytest.mark.asyncio
async def test_t2_f2_skip_with_malformed_session_id(make_client):
    """Sending skip with invalid or mismatched session_id returns error."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Send skip with bogus session_id
    await client_a.skip(session_id="00000000-0000-0000-0000-000000000000")

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") in ("NOT_IN_SESSION", "INVALID_PAYLOAD", "INVALID_SESSION")


@pytest.mark.asyncio
async def test_t2_f3_leave_when_idle(make_client):
    """Sending leave while idle in lobby returns NOT_IN_SESSION error."""
    client = await make_client()
    await client.leave()

    err = await client.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") == "NOT_IN_SESSION"


@pytest.mark.asyncio
async def test_t2_f3_both_leave_simultaneously(make_client):
    """Both clients calling leave simultaneously cleans up room with no deadlock."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Simultaneous leave
    await asyncio.gather(
        client_a.leave(),
        client_b.leave(),
    )

    drained_a = await client_a.drain(timeout=0.5)
    drained_b = await client_b.drain(timeout=0.5)

    # Both clients cleanly transition without unhandled crashes
    assert all(m.get("type") in ("left_session", "peer_disconnected", "error") for m in drained_a + drained_b)


@pytest.mark.asyncio
async def test_t2_f3_leave_during_matching(make_client):
    """Client leaving immediately while match is being negotiated cleans up safely."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    # Client A immediately leaves right as B joins
    await asyncio.gather(
        client_a.leave_queue(),
        client_b.join_queue(),
    )

    # Ensure no orphaned or corrupted state
    drained_a = await client_a.drain(timeout=0.5)
    assert not any(m.get("type") == "error" and m.get("payload", {}).get("code") == "INTERNAL_ERROR" for m in drained_a)


@pytest.mark.asyncio
async def test_t2_f3_abrupt_network_drop_during_signaling(make_client):
    """Abrupt network drop mid-call notifies remaining partner and terminates session."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # Start signaling
    if msg_a.get("payload", {}).get("is_initiator"):
        await client_a.send_sdp_offer("v=0\r\ntest-sdp")
        await client_b.recv_type("offer", timeout=3.0)
        # Drop initiator abruptly
        await client_a.close(code=1006)
        initiator = client_a
    else:
        await client_b.send_sdp_offer("v=0\r\ntest-sdp")
        await client_a.recv_type("offer", timeout=3.0)
        # Drop initiator abruptly
        await client_b.close(code=1006)
        initiator = client_b

    # Remaining peer must be notified
    surviving_client = client_b if initiator is client_a else client_a
    msg = await surviving_client.recv_type("peer_disconnected", timeout=3.0)
    assert msg["type"] == "peer_disconnected"
    assert msg.get("payload", {}).get("reason") == "disconnected"


@pytest.mark.asyncio
async def test_t2_f3_ping_pong_heartbeat_idle(make_client):
    """Client sending ping envelope receives valid pong envelope with timestamp."""
    client = await make_client()
    await client.send_ping()

    pong = await client.recv_type("pong", timeout=3.0)
    assert pong["type"] == "pong"
    assert "timestamp" in pong
