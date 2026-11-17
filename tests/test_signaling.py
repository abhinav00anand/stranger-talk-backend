"""WebRTC Signaling Relay Tests (Tier 1 & Tier 2).

Covers:
- F4: WebRTC Audio Signaling Relay (SDP Offer/Answer & Trickle ICE Candidates)
"""
import asyncio
from datetime import datetime, timezone
import pytest


# ============================================================================
# Tier 1: Feature Coverage (Happy Path)
# ============================================================================

@pytest.mark.asyncio
async def test_t1_f4_sdp_offer_relay(make_client):
    """Initiator client sends SDP offer; partner receives relayed offer with matching SDP."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    test_sdp = (
        "v=0\r\n"
        "o=- 42 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )

    await initiator.send_sdp_offer(test_sdp)

    relayed = await receiver.recv_type("offer", timeout=5.0)
    assert relayed["type"] == "offer"
    assert relayed["session_id"] == initiator.session_id
    payload = relayed.get("payload", {})
    assert payload.get("sdp") == test_sdp
    assert payload.get("type") == "offer"


@pytest.mark.asyncio
async def test_t1_f4_sdp_answer_relay(make_client):
    """Receiver sends SDP answer; initiator receives relayed answer with matching SDP."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    # Send initial offer
    offer_sdp = "v=0\r\no=- 42 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 54321 RTP/SAVPF 111\r\n"
    await initiator.send_sdp_offer(offer_sdp)
    await receiver.recv_type("offer", timeout=5.0)

    # Receiver answers
    answer_sdp = "v=0\r\no=- 99 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 54321 RTP/SAVPF 111\r\n"
    await receiver.send_sdp_answer(answer_sdp)

    relayed = await initiator.recv_type("answer", timeout=5.0)
    assert relayed["type"] == "answer"
    assert relayed["session_id"] == receiver.session_id
    payload = relayed.get("payload", {})
    assert payload.get("sdp") == answer_sdp
    assert payload.get("type") == "answer"


@pytest.mark.asyncio
async def test_t1_f4_ice_candidate_relay(make_client):
    """Client sends ICE candidate; partner receives exact candidate metadata."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    candidate_str = (
        "candidate:842163049 1 udp 1677729535 192.168.1.100 52312 typ srflx "
        "raddr 192.168.1.100 rport 52312"
    )

    await client_a.send_ice_candidate(candidate=candidate_str, sdpMid="0", sdpMLineIndex=0)

    relayed = await client_b.recv_type("ice_candidate", timeout=5.0)
    assert relayed["type"] == "ice_candidate"
    payload = relayed.get("payload", {})
    assert payload.get("candidate") == candidate_str
    assert str(payload.get("sdpMid")) == "0"
    assert int(payload.get("sdpMLineIndex")) == 0


@pytest.mark.asyncio
async def test_t1_f4_multiple_ice_candidates(make_client):
    """Multiple trickle ICE candidates stream without loss or corruption."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    candidates = [
        "candidate:1 1 udp 2122260223 192.168.1.100 50000 typ host",
        "candidate:2 1 udp 1686052607 203.0.113.1 50001 typ srflx raddr 192.168.1.100 rport 50000",
        "candidate:3 1 tcp 1518280447 192.168.1.100 9 typ host tcptype active",
    ]

    for c in candidates:
        await client_a.send_ice_candidate(candidate=c, sdpMid="0", sdpMLineIndex=0)

    received_candidates = []
    for _ in range(len(candidates)):
        msg = await client_b.recv_type("ice_candidate", timeout=5.0)
        received_candidates.append(msg.get("payload", {}).get("candidate"))

    assert received_candidates == candidates


@pytest.mark.asyncio
async def test_t1_f4_signaling_timestamp_preserved(make_client):
    """Relayed signaling envelopes contain valid ISO-8601 UTC timestamps."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    await initiator.send_sdp_offer("v=0\r\ntest-sdp")
    relayed = await receiver.recv_type("offer", timeout=5.0)

    assert "timestamp" in relayed
    ts_str = relayed["timestamp"]
    assert isinstance(ts_str, str)
    # Validate ISO-8601 parsing
    parsed_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    assert parsed_ts is not None


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

@pytest.mark.asyncio
async def test_t2_f4_signaling_when_not_in_session(make_client):
    """Sending WebRTC offer while idle returns NOT_IN_SESSION error."""
    client = await make_client()
    await client.send_sdp_offer("v=0\r\ntest", session_id="00000000-0000-0000-0000-000000000000")

    err = await client.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") == "NOT_IN_SESSION"


@pytest.mark.asyncio
async def test_t2_f4_empty_ice_candidate_relay(make_client):
    """Empty candidate string (End-of-Candidates) relays without error or crash."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    # WebRTC End-of-Candidates indication
    await client_a.send_ice_candidate(candidate="", sdpMid="0", sdpMLineIndex=0)

    relayed = await client_b.recv_type("ice_candidate", timeout=5.0)
    assert relayed["type"] == "ice_candidate"
    assert relayed.get("payload", {}).get("candidate") == ""


@pytest.mark.asyncio
async def test_t2_f4_oversized_sdp_payload(make_client):
    """Large SDP payload (64KB) is handled safely without crashing server."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    # 64KB SDP payload
    large_sdp = "v=0\r\no=- 42 2 IN IP4 127.0.0.1\r\ns=-\r\n" + ("a=custom-attr:x\r\n" * 3000)

    await initiator.send_sdp_offer(large_sdp)

    # Server either successfully relays or returns structured validation error (no unhandled 500 crash)
    try:
        msg = await receiver.recv_type("offer", timeout=5.0)
        assert msg.get("payload", {}).get("sdp") == large_sdp
    except (TimeoutError, asyncio.TimeoutError):
        # In case server rejected as oversized
        err = await initiator.recv_type("error", timeout=3.0)
        assert err.get("payload", {}).get("code") in ("MESSAGE_TOO_LONG", "INVALID_PAYLOAD")


@pytest.mark.asyncio
async def test_t2_f4_malformed_sdp_json(make_client):
    """Sending offer message with missing sdp field returns INVALID_PAYLOAD error."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    initiator = client_a if client_a.is_initiator else client_b

    # Send malformed offer payload (missing sdp)
    await initiator.send_envelope(
        "offer",
        session_id=initiator.session_id,
        payload={"type": "offer", "invalid_key": 123},
    )

    err = await initiator.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    assert err.get("payload", {}).get("code") == "INVALID_PAYLOAD"


@pytest.mark.asyncio
async def test_t2_f4_offer_to_ai_companion_ignored(make_client):
    """SDP offer sent during an AI companion session is handled gracefully without error."""
    client = await make_client()
    await client.join_queue()

    # Wait for match (may be AI companion after timeout or fallback)
    # If client sends an offer when peer_is_ai is true, server acknowledges or ignores safely
    await client.send_sdp_offer("v=0\r\ntest-sdp", session_id="ai-virtual-session")

    drained = await client.drain(timeout=0.5)
    # Server should not disconnect with internal error
    assert not any(m.get("payload", {}).get("code") == "INTERNAL_ERROR" for m in drained)
