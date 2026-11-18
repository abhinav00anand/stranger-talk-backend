"""Data models for matchmaking queues, sessions, and ban records."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


def _to_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ClientState(str, Enum):
    """Lifecycle states of an active WebSocket client connection."""

    IDLE = "idle"
    QUEUED = "queued"
    MATCHING = "matching"
    PAIRED = "paired"
    SKIPPING = "skipping"
    LEAVING = "leaving"
    BANNED = "banned"
    DISCONNECTED = "disconnected"


class SessionStatus(str, Enum):
    """Lifecycle states of a 1-on-1 session between peers."""

    IDLE = "idle"
    QUEUED = "queued"
    MATCHING = "matching"
    PAIRED = "paired"
    ACTIVE = "active"
    SKIPPING = "skipping"
    LEAVING = "leaving"
    BANNED = "banned"
    TERMINATED = "terminated"
    DISCONNECTED = "disconnected"


class Session(BaseModel):
    """Represents an active or recently terminated 1-on-1 session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_a: str
    client_b: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: SessionStatus = SessionStatus.PAIRED
    initiator_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_distinct_clients(self) -> "Session":
        if self.client_a == self.client_b:
            raise ValueError("Session participants client_a and client_b must be distinct")
        return self

    def get_partner_id(self, client_id: str) -> Optional[str]:
        """Return the partner's client_id given one participant's client_id."""
        if client_id == self.client_a:
            return self.client_b
        if client_id == self.client_b:
            return self.client_a
        return None

    def contains_client(self, client_id: str) -> bool:
        """Check if a client is one of the session participants."""
        return client_id in (self.client_a, self.client_b)

    def is_active(self) -> bool:
        """Check if session is currently active and can route messages."""
        return self.status in (SessionStatus.PAIRED, SessionStatus.ACTIVE)


class QueueTicket(BaseModel):
    """Represents a client's entry in the matchmaking queue."""

    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str
    queued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ClientState = ClientState.QUEUED
    topics: Optional[List[str]] = None
    ip: Optional[str] = None
    is_bot: bool = False

    def elapsed_seconds(self) -> float:
        """Calculate how many seconds this ticket has been waiting in queue."""
        now = datetime.now(timezone.utc)
        queued = _to_utc(self.queued_at)
        return max(0.0, (now - queued).total_seconds())


class BanRecord(BaseModel):
    """Represents a moderation ban placed on an IP address or identifier."""

    ip: str
    banned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    reason: str

    def is_expired(self) -> bool:
        """Check if the ban record has expired past its validity window."""
        now = datetime.now(timezone.utc)
        return now >= _to_utc(self.expires_at)

    def remaining_seconds(self) -> float:
        """Calculate the remaining duration of the ban in seconds."""
        now = datetime.now(timezone.utc)
        return max(0.0, (_to_utc(self.expires_at) - now).total_seconds())
