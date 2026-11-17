"""WebRTC Signaling Relay for SDP Offer/Answer and Trickle ICE Candidates."""

from typing import Any, Dict, List, Optional

from app.config import settings
from app.models.envelopes import (
    IceCandidatePayload,
    SdpPayload,
    ServerMessageType,
    WebSocketEnvelope,
)
from app.models.session import Session, SessionStatus


class SignalingRelay:
    """Manages WebRTC SDP Offer, SDP Answer, and Trickle ICE Candidate forwarding."""

    def __init__(self, stun_servers: Optional[List[str]] = None) -> None:
        """Initialize SignalingRelay with STUN servers or fallback to application settings."""
        if stun_servers is not None:
            self.stun_servers: List[str] = list(stun_servers)
        else:
            self.stun_servers = list(settings.STUN_SERVERS)

    def get_ice_servers(self) -> List[Dict[str, Any]]:
        """Return RTCIceServer configuration structures for WebRTC client connection."""
        if not self.stun_servers:
            return []
        return [{"urls": list(self.stun_servers)}]

    def relay_offer(
        self, session: Session, sender_id: str, sdp_dict: Dict[str, Any]
    ) -> WebSocketEnvelope:
        """Relay SDP Offer from designated initiator to receiver peer.

        Validations:
        - Session must be active (PAIRED or ACTIVE).
        - Sender must be a participant in the session.
        - Glare Elimination: Sender must be the designated session initiator.
        - Payload must be a valid SDP offer.
        """
        if not session.is_active():
            raise ValueError(
                f"SESSION_INACTIVE: Session '{session.session_id}' is not active (status: {session.status.value})"
            )

        if not session.contains_client(sender_id):
            raise ValueError(
                f"INVALID_PARTICIPANT: Client '{sender_id}' is not a participant in session '{session.session_id}'"
            )

        # Glare Elimination: Only designated initiator may send SDP offer
        if session.initiator_id != sender_id:
            raise ValueError(
                "SIGNALING_STATE_INVALID: Non-initiator cannot send offer"
            )

        # Validate SDP payload
        if isinstance(sdp_dict, SdpPayload):
            payload_obj = sdp_dict
            if payload_obj.type != "offer":
                raise ValueError("SIGNALING_PAYLOAD_INVALID: SDP payload type must be 'offer'")
        elif isinstance(sdp_dict, dict):
            payload_data = dict(sdp_dict)
            if "type" in payload_data and payload_data["type"] != "offer":
                raise ValueError("SIGNALING_PAYLOAD_INVALID: SDP payload type must be 'offer'")
            payload_data.setdefault("type", "offer")
            payload_obj = SdpPayload.model_validate(payload_data)
        else:
            raise ValueError("SIGNALING_PAYLOAD_INVALID: Invalid SDP payload format")

        return WebSocketEnvelope.create(
            msg_type=ServerMessageType.OFFER,
            payload=payload_obj,
            session_id=session.session_id,
        )

    def relay_answer(
        self, session: Session, sender_id: str, sdp_dict: Dict[str, Any]
    ) -> WebSocketEnvelope:
        """Relay SDP Answer from receiver back to the initiator.

        Validations:
        - Session must be active (PAIRED or ACTIVE).
        - Sender must be a participant in the session.
        - Glare Elimination: Sender must NOT be the session initiator.
        - Payload must be a valid SDP answer.
        - Transitions session status to ACTIVE upon valid answer.
        """
        if not session.is_active():
            raise ValueError(
                f"SESSION_INACTIVE: Session '{session.session_id}' is not active (status: {session.status.value})"
            )

        if not session.contains_client(sender_id):
            raise ValueError(
                f"INVALID_PARTICIPANT: Client '{sender_id}' is not a participant in session '{session.session_id}'"
            )

        # Glare Elimination: Initiator cannot send SDP answer
        if session.initiator_id == sender_id:
            raise ValueError(
                "SIGNALING_STATE_INVALID: Initiator cannot send answer"
            )

        # Validate SDP payload
        if isinstance(sdp_dict, SdpPayload):
            payload_obj = sdp_dict
            if payload_obj.type != "answer":
                raise ValueError("SIGNALING_PAYLOAD_INVALID: SDP payload type must be 'answer'")
        elif isinstance(sdp_dict, dict):
            payload_data = dict(sdp_dict)
            if "type" in payload_data and payload_data["type"] != "answer":
                raise ValueError("SIGNALING_PAYLOAD_INVALID: SDP payload type must be 'answer'")
            payload_data.setdefault("type", "answer")
            payload_obj = SdpPayload.model_validate(payload_data)
        else:
            raise ValueError("SIGNALING_PAYLOAD_INVALID: Invalid SDP payload format")

        # Transition status from PAIRED to ACTIVE
        if session.status == SessionStatus.PAIRED:
            session.status = SessionStatus.ACTIVE

        return WebSocketEnvelope.create(
            msg_type=ServerMessageType.ANSWER,
            payload=payload_obj,
            session_id=session.session_id,
        )

    def relay_ice_candidate(
        self, session: Session, sender_id: str, candidate_dict: Dict[str, Any]
    ) -> WebSocketEnvelope:
        """Relay Trickle ICE Candidate between paired peers in either direction.

        Validations:
        - Session must be active (PAIRED or ACTIVE).
        - Sender must be a participant in the session.
        - Payload must be a valid ICE candidate structure (allows candidate="" for End-of-Candidates).
        """
        if not session.is_active():
            raise ValueError(
                f"SESSION_INACTIVE: Session '{session.session_id}' is not active (status: {session.status.value})"
            )

        if not session.contains_client(sender_id):
            raise ValueError(
                f"INVALID_PARTICIPANT: Client '{sender_id}' is not a participant in session '{session.session_id}'"
            )

        # Validate candidate payload
        if isinstance(candidate_dict, IceCandidatePayload):
            payload_obj = candidate_dict
        elif isinstance(candidate_dict, dict):
            payload_obj = IceCandidatePayload.model_validate(candidate_dict)
        else:
            raise ValueError("SIGNALING_PAYLOAD_INVALID: Invalid ICE candidate payload format")

        return WebSocketEnvelope.create(
            msg_type=ServerMessageType.ICE_CANDIDATE,
            payload=payload_obj,
            session_id=session.session_id,
        )
