"""WebSocket message envelopes, payload schemas, and message type enums."""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ClientMessageType(str, Enum):
    """Client-to-server WebSocket message type identifiers."""

    JOIN_QUEUE = "join_queue"
    LEAVE_QUEUE = "leave_queue"
    OFFER = "offer"
    ANSWER = "answer"
    ICE_CANDIDATE = "ice_candidate"
    CHAT = "chat"
    GIF = "gif"
    TYPING = "typing"
    SKIP = "skip"
    LEAVE = "leave"
    PING = "ping"


class ServerMessageType(str, Enum):
    """Server-to-client WebSocket message type identifiers."""

    QUEUED = "queued"
    MATCHED = "matched"
    OFFER = "offer"
    ANSWER = "answer"
    ICE_CANDIDATE = "ice_candidate"
    CHAT = "chat"
    GIF = "gif"
    TYPING = "typing"
    PEER_DISCONNECTED = "peer_disconnected"
    BANNED = "banned"
    ERROR = "error"
    PONG = "pong"
    IDLE = "idle"
    LEFT_SESSION = "left_session"


# ---------------------------------------------------------------------------
# Message Payload Models
# ---------------------------------------------------------------------------


class JoinQueuePayload(BaseModel):
    """Payload for client join_queue request."""

    topics: Optional[List[str]] = Field(default=None, max_length=10)

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = []
        for topic in value:
            if not isinstance(topic, str):
                raise ValueError("Each topic must be a string")
            topic = topic.strip()
            if not topic or len(topic) > 40:
                raise ValueError("Topics must contain 1 to 40 non-whitespace characters")
            cleaned.append(topic)
        return cleaned


class SdpPayload(BaseModel):
    """Payload for WebRTC SDP offer and answer exchange."""

    sdp: str
    type: Literal["offer", "answer"]


class IceCandidatePayload(BaseModel):
    """Payload for WebRTC trickle ICE candidate exchange."""

    candidate: str
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None
    usernameFragment: Optional[str] = None


class ChatMessagePayload(BaseModel):
    """Payload for client-sent text chat messages."""

    text: str = Field(..., min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Chat message text cannot be empty or only whitespace")
        return stripped


class GifMessagePayload(BaseModel):
    """Payload for client-sent GIF messages."""

    url: str = Field(..., min_length=1, max_length=2048)
    id: Optional[str] = None
    preview_url: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class TypingPayload(BaseModel):
    """Payload for typing indicator state updates."""

    is_typing: bool


class PeerDisconnectedPayload(BaseModel):
    """Payload emitted when a matched peer disconnects or leaves."""

    reason: Literal["skipped", "left", "disconnected", "banned"]
    can_requeue: bool = True
    message: Optional[str] = None


class MatchedPayload(BaseModel):
    """Payload emitted when two human clients are paired."""

    is_initiator: bool
    peer_alias: str = "Stranger"
    ice_servers: List[Dict[str, Any]] = Field(default_factory=list)


class BannedPayload(BaseModel):
    """Payload sent directly to a client when banned for policy violations."""

    reason: str
    ban_duration_seconds: int
    banned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorPayload(BaseModel):
    """Payload for structured errors returned to clients."""

    code: str
    message: str


class QueuedPayload(BaseModel):
    """Payload emitted when a client is enqueued."""

    position: int = 1
    estimated_wait_seconds: Optional[int] = None


class ServerChatMessagePayload(BaseModel):
    """Payload emitted to clients when a chat message is relayed or delivered."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = "peer"
    text: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ServerGifMessagePayload(BaseModel):
    """Payload emitted to clients when a GIF message is relayed or delivered."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = "peer"
    url: str
    preview_url: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PongPayload(BaseModel):
    """Payload emitted in response to client ping."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Canonical WebSocket Envelope
# ---------------------------------------------------------------------------


class WebSocketEnvelope(BaseModel):
    """Unified JSON envelope for all client and server WebSocket communication."""

    type: str
    session_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("timestamp", mode="before")
    @classmethod
    def format_timestamp(cls, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def create(
        cls,
        msg_type: Union[ClientMessageType, ServerMessageType, str],
        payload: Union[BaseModel, Dict[str, Any], None] = None,
        session_id: Optional[str] = None,
    ) -> "WebSocketEnvelope":
        """Factory method to construct a WebSocketEnvelope with automatic serialization."""
        type_str = msg_type.value if hasattr(msg_type, "value") else str(msg_type)
        if payload is None:
            payload_dict: Dict[str, Any] = {}
        elif isinstance(payload, BaseModel):
            payload_dict = payload.model_dump(mode="json")
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = {"data": payload}

        return cls(
            type=type_str,
            session_id=session_id,
            payload=payload_dict,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self) -> str:
        """Serialize envelope to JSON string."""
        return self.model_dump_json()

    def to_dict(self) -> Dict[str, Any]:
        """Convert envelope to dictionary."""
        return self.model_dump()
