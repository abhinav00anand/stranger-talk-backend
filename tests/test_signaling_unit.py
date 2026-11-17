"""Unit tests for WebRTC SignalingRelay (STUN delivery, SDP relay, ICE candidates, glare elimination)."""

import pytest
from pydantic import ValidationError

from app.config import settings
from app.models.envelopes import IceCandidatePayload, SdpPayload, ServerMessageType
from app.models.session import Session, SessionStatus
from app.signaling.rtc_signaling import SignalingRelay


# ============================================================================
# STUN Server Delivery and Configuration Tests
# ============================================================================


def test_signaling_relay_default_stun_servers():
    """SignalingRelay initializes with default STUN servers from application settings."""
    relay = SignalingRelay()
    ice_servers = relay.get_ice_servers()

    assert isinstance(ice_servers, list)
    assert len(ice_servers) == 1
    urls = ice_servers[0].get("urls", [])
    assert isinstance(urls, list)
    assert "stun:stun.l.google.com:19302" in urls
    assert "stun:stun1.l.google.com:19302" in urls


def test_signaling_relay_custom_stun_servers():
    """SignalingRelay accepts custom STUN server lists."""
    custom_servers = ["stun:custom.relay.org:3478", "stun:turn.relay.org:3478"]
    relay = SignalingRelay(stun_servers=custom_servers)
    ice_servers = relay.get_ice_servers()

    assert ice_servers == [{"urls": custom_servers}]


def test_signaling_relay_empty_stun_servers():
    """SignalingRelay handles empty STUN server configuration gracefully."""
    relay = SignalingRelay(stun_servers=[])
    ice_servers = relay.get_ice_servers()

    assert ice_servers == []


# ============================================================================
# SDP Offer Relay and Glare Elimination Tests
# ============================================================================


def test_relay_offer_from_initiator_success():
    """Offer sent by designated initiator succeeds and returns valid WebSocketEnvelope."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-100",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    sdp_str = "v=0\r\no=- 123 2 IN IP4 127.0.0.1\r\ns=-\r\nm=audio 50000 RTP/SAVPF 111\r\n"
    envelope = relay.relay_offer(session, "initiator_peer", {"sdp": sdp_str, "type": "offer"})

    assert envelope.type == ServerMessageType.OFFER.value
    assert envelope.session_id == "sess-100"
    assert envelope.payload["sdp"] == sdp_str
    assert envelope.payload["type"] == "offer"
    assert envelope.timestamp is not None


def test_relay_offer_with_sdp_payload_object():
    """Offer sent using SdpPayload model instance succeeds."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-101",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    payload = SdpPayload(sdp="v=0\r\ntest", type="offer")
    envelope = relay.relay_offer(session, "initiator_peer", payload)

    assert envelope.type == ServerMessageType.OFFER.value
    assert envelope.payload["sdp"] == "v=0\r\ntest"


def test_relay_offer_from_non_initiator_raises_glare_error():
    """Offer sent by non-initiator peer is rejected to prevent WebRTC glare."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-102",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    with pytest.raises(ValueError, match="SIGNALING_STATE_INVALID: Non-initiator cannot send offer"):
        relay.relay_offer(session, "receiver_peer", {"sdp": "v=0\r\ntest", "type": "offer"})


def test_relay_offer_invalid_sdp_type_raises_error():
    """Sending an SDP payload with type='answer' to relay_offer raises error."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-103",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    with pytest.raises(ValueError, match="SDP payload type must be 'offer'"):
        relay.relay_offer(session, "initiator_peer", {"sdp": "v=0\r\ntest", "type": "answer"})


# ============================================================================
# SDP Answer Relay and Glare Elimination Tests
# ============================================================================


def test_relay_answer_from_non_initiator_success():
    """Answer sent by non-initiator peer succeeds and transitions session to ACTIVE."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-200",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    sdp_str = "v=0\r\no=- 456 2 IN IP4 127.0.0.1\r\ns=-\r\nm=audio 50000 RTP/SAVPF 111\r\n"
    envelope = relay.relay_answer(session, "receiver_peer", {"sdp": sdp_str, "type": "answer"})

    assert envelope.type == ServerMessageType.ANSWER.value
    assert envelope.session_id == "sess-200"
    assert envelope.payload["sdp"] == sdp_str
    assert envelope.payload["type"] == "answer"
    # Status progression check
    assert session.status == SessionStatus.ACTIVE


def test_relay_answer_from_initiator_raises_glare_error():
    """Answer sent by designated initiator is rejected to prevent signaling protocol violation."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-201",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    with pytest.raises(ValueError, match="SIGNALING_STATE_INVALID: Initiator cannot send answer"):
        relay.relay_answer(session, "initiator_peer", {"sdp": "v=0\r\ntest", "type": "answer"})


def test_relay_answer_invalid_sdp_type_raises_error():
    """Sending an SDP payload with type='offer' to relay_answer raises error."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-202",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.PAIRED,
    )

    with pytest.raises(ValueError, match="SDP payload type must be 'answer'"):
        relay.relay_answer(session, "receiver_peer", {"sdp": "v=0\r\ntest", "type": "offer"})


# ============================================================================
# ICE Candidate Relay Tests
# ============================================================================


def test_relay_ice_candidate_from_initiator_and_receiver():
    """ICE candidates can be relayed bidirectionally by initiator and receiver."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-300",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.ACTIVE,
    )

    # Initiator sends candidate
    cand_init = {
        "candidate": "candidate:1 1 udp 2122260223 192.168.1.1 50000 typ host",
        "sdpMid": "0",
        "sdpMLineIndex": 0,
    }
    env_init = relay.relay_ice_candidate(session, "initiator_peer", cand_init)
    assert env_init.type == ServerMessageType.ICE_CANDIDATE.value
    assert env_init.session_id == "sess-300"
    assert env_init.payload["candidate"] == cand_init["candidate"]
    assert env_init.payload["sdpMid"] == "0"
    assert env_init.payload["sdpMLineIndex"] == 0

    # Receiver sends candidate
    cand_recv = {
        "candidate": "candidate:2 1 udp 1686052607 203.0.113.1 50001 typ srflx",
        "sdpMid": "audio",
        "sdpMLineIndex": 1,
    }
    env_recv = relay.relay_ice_candidate(session, "receiver_peer", cand_recv)
    assert env_recv.type == ServerMessageType.ICE_CANDIDATE.value
    assert env_recv.session_id == "sess-300"
    assert env_recv.payload["candidate"] == cand_recv["candidate"]


def test_relay_ice_candidate_end_of_candidates():
    """Empty candidate string representing WebRTC End-of-Candidates relays safely."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-301",
        client_a="initiator_peer",
        client_b="receiver_peer",
        initiator_id="initiator_peer",
        status=SessionStatus.ACTIVE,
    )

    env = relay.relay_ice_candidate(
        session, "initiator_peer", {"candidate": "", "sdpMid": "0", "sdpMLineIndex": 0}
    )
    assert env.payload["candidate"] == ""


# ============================================================================
# Inactive Session and Non-Participant Validation Tests
# ============================================================================


@pytest.mark.parametrize(
    "inactive_status",
    [
        SessionStatus.TERMINATED,
        SessionStatus.LEAVING,
        SessionStatus.SKIPPING,
        SessionStatus.BANNED,
        SessionStatus.DISCONNECTED,
    ],
)
def test_signaling_inactive_session_rejected(inactive_status):
    """Any signaling attempt on an inactive or terminated session raises error."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-400",
        client_a="peer_a",
        client_b="peer_b",
        initiator_id="peer_a",
        status=inactive_status,
    )

    with pytest.raises(ValueError, match="SESSION_INACTIVE"):
        relay.relay_offer(session, "peer_a", {"sdp": "v=0", "type": "offer"})

    with pytest.raises(ValueError, match="SESSION_INACTIVE"):
        relay.relay_answer(session, "peer_b", {"sdp": "v=0", "type": "answer"})

    with pytest.raises(ValueError, match="SESSION_INACTIVE"):
        relay.relay_ice_candidate(session, "peer_a", {"candidate": "c1"})


def test_signaling_non_participant_sender_rejected():
    """Signaling attempt by a client not belonging to the session is rejected."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-500",
        client_a="peer_a",
        client_b="peer_b",
        initiator_id="peer_a",
        status=SessionStatus.ACTIVE,
    )

    with pytest.raises(ValueError, match="INVALID_PARTICIPANT"):
        relay.relay_offer(session, "intruder_user", {"sdp": "v=0", "type": "offer"})

    with pytest.raises(ValueError, match="INVALID_PARTICIPANT"):
        relay.relay_answer(session, "intruder_user", {"sdp": "v=0", "type": "answer"})

    with pytest.raises(ValueError, match="INVALID_PARTICIPANT"):
        relay.relay_ice_candidate(session, "intruder_user", {"candidate": "c1"})


def test_signaling_malformed_sdp_missing_field_rejected():
    """Sending an offer without the required 'sdp' attribute raises a validation error."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-600",
        client_a="peer_a",
        client_b="peer_b",
        initiator_id="peer_a",
        status=SessionStatus.PAIRED,
    )

    with pytest.raises((ValueError, ValidationError)):
        relay.relay_offer(session, "peer_a", {"type": "offer"})


def test_signaling_malformed_candidate_rejected():
    """Sending a candidate with invalid non-dict/non-model payload raises ValueError."""
    relay = SignalingRelay()
    session = Session(
        session_id="sess-601",
        client_a="peer_a",
        client_b="peer_b",
        initiator_id="peer_a",
        status=SessionStatus.ACTIVE,
    )

    with pytest.raises(ValueError):
        relay.relay_ice_candidate(session, "peer_a", "not_a_candidate_dict")  # type: ignore
