"""1-on-1 session manager and state machine with atomic concurrency protection."""

import asyncio
from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple

from app.models.session import ClientState, Session, SessionStatus

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages active 1-on-1 sessions, client routing, and lifecycle state transitions."""

    def __init__(self, max_history: int = 1000) -> None:
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, Session] = {}
        self._client_to_session: Dict[str, str] = {}
        self._client_states: Dict[str, ClientState] = {}
        self._terminated_sessions: OrderedDict[str, Session] = OrderedDict()
        self._max_history = max_history

    def _record_terminated(self, session: Session) -> None:
        """Cache terminated session in bounded history."""
        self._terminated_sessions[session.session_id] = session
        if len(self._terminated_sessions) > self._max_history:
            self._terminated_sessions.popitem(last=False)

    def _cleanup_stale_mapping(self, client_id: str) -> None:
        """Internal helper to clean up old session mapping for a client before re-pairing."""
        old_session_id = self._client_to_session.pop(client_id, None)
        if old_session_id and old_session_id in self._sessions:
            old_sess = self._sessions.pop(old_session_id)
            old_sess.status = SessionStatus.TERMINATED
            self._record_terminated(old_sess)
            partner = old_sess.get_partner_id(client_id)
            if partner:
                self._client_to_session.pop(partner, None)
                self._client_states[partner] = ClientState.IDLE

    async def create_session(
        self,
        client_a: str,
        client_b: str,
        is_ai: bool = False,
    ) -> Session:
        """Atomically create a new 1-on-1 session with deterministic initiator assignment.

        client_a is designated as initiator (initiator_id = client_a, is_initiator = True).
        client_b is designated as receiver (is_initiator = False).
        """
        if not client_a or not client_b:
            raise ValueError("client_a and client_b must be non-empty strings")
        if client_a == client_b:
            raise ValueError("Session participants client_a and client_b must be distinct")

        async with self._lock:
            # Clean up prior stale mappings for these clients if any exist
            self._cleanup_stale_mapping(client_a)
            self._cleanup_stale_mapping(client_b)

            session = Session(
                client_a=client_a,
                client_b=client_b,
                is_ai=is_ai,
                initiator_id=client_a,
                status=SessionStatus.PAIRED,
            )

            self._sessions[session.session_id] = session
            self._client_to_session[client_a] = session.session_id
            self._client_to_session[client_b] = session.session_id

            self._client_states[client_a] = ClientState.PAIRED
            self._client_states[client_b] = ClientState.PAIRED

            logger.info(
                "Created session: session_id=%s, client_a=%s, client_b=%s, is_ai=%s",
                session.session_id,
                client_a,
                client_b,
                is_ai,
            )
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by session_id (active or recently terminated)."""
        return self._sessions.get(session_id) or self._terminated_sessions.get(session_id)

    def get_session_by_client(self, client_id: str) -> Optional[Session]:
        """Retrieve currently active session for a given client ID."""
        session_id = self._client_to_session.get(client_id)
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def get_partner_id(self, client_id: str) -> Optional[str]:
        """Retrieve the partner client ID for an active session."""
        session = self.get_session_by_client(client_id)
        if not session:
            return None
        return session.get_partner_id(client_id)

    def get_client_state(self, client_id: str) -> ClientState:
        """Return the current lifecycle state of a client (defaults to IDLE)."""
        return self._client_states.get(client_id, ClientState.IDLE)

    def set_client_state(self, client_id: str, state: ClientState) -> None:
        """Update client state directly (e.g. QUEUED, BANNED, IDLE)."""
        self._client_states[client_id] = state

    def active_session_count(self) -> int:
        """Return the number of currently active sessions."""
        return len(self._sessions)

    def get_active_sessions(self) -> List[Session]:
        """Return list of all currently active sessions."""
        return list(self._sessions.values())

    async def activate_session(self, session_id: str) -> Optional[Session]:
        """Transition session status from PAIRED to ACTIVE upon signaling completion."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == SessionStatus.PAIRED:
                session.status = SessionStatus.ACTIVE
            return session

    async def handle_skip(
        self,
        client_id: str,
        expected_session_id: Optional[str] = None,
    ) -> Tuple[Optional[Session], Optional[str]]:
        """Atomically transition active session to TERMINATED when a user skips.

        Sets skipping client to QUEUED.
        Sets partner to IDLE.
        Handles simultaneous skip races gracefully: first call terminates session and returns
        (session, partner_id), subsequent call returns (None, None).
        """
        async with self._lock:
            session_id = self._client_to_session.get(client_id)
            if not session_id:
                return None, None

            if expected_session_id is not None and session_id != expected_session_id:
                return None, None

            session = self._sessions.get(session_id)
            if not session or not session.is_active():
                self._client_to_session.pop(client_id, None)
                return None, None

            partner_id = session.get_partner_id(client_id)
            session.status = SessionStatus.TERMINATED

            self._client_to_session.pop(session.client_a, None)
            self._client_to_session.pop(session.client_b, None)

            self._client_states[client_id] = ClientState.QUEUED
            if partner_id:
                self._client_states[partner_id] = ClientState.IDLE

            self._sessions.pop(session_id, None)
            self._record_terminated(session)

            logger.info(
                "Handled skip: client_id=%s, partner_id=%s, session_id=%s",
                client_id,
                partner_id,
                session_id,
            )
            return session, partner_id

    async def handle_leave(
        self,
        client_id: str,
        expected_session_id: Optional[str] = None,
    ) -> Tuple[Optional[Session], Optional[str]]:
        """Atomically terminate active session when a user leaves to idle lobby.

        Sets leaving client to IDLE.
        Sets partner to IDLE.
        Handles simultaneous leave races gracefully: first call terminates session and returns
        (session, partner_id), subsequent call returns (None, None).
        """
        async with self._lock:
            session_id = self._client_to_session.get(client_id)
            if not session_id:
                return None, None

            if expected_session_id is not None and session_id != expected_session_id:
                return None, None

            session = self._sessions.get(session_id)
            if not session or not session.is_active():
                self._client_to_session.pop(client_id, None)
                return None, None

            partner_id = session.get_partner_id(client_id)
            session.status = SessionStatus.TERMINATED

            self._client_to_session.pop(session.client_a, None)
            self._client_to_session.pop(session.client_b, None)

            self._client_states[client_id] = ClientState.IDLE
            if partner_id:
                self._client_states[partner_id] = ClientState.IDLE

            self._sessions.pop(session_id, None)
            self._record_terminated(session)

            logger.info(
                "Handled leave: client_id=%s, partner_id=%s, session_id=%s",
                client_id,
                partner_id,
                session_id,
            )
            return session, partner_id

    async def handle_disconnect(
        self,
        client_id: str,
    ) -> Tuple[Optional[Session], Optional[str]]:
        """Clean up disconnected client, terminate active session, and set state to DISCONNECTED."""
        async with self._lock:
            self._client_states[client_id] = ClientState.DISCONNECTED
            session_id = self._client_to_session.pop(client_id, None)

            if not session_id:
                return None, None

            session = self._sessions.get(session_id)
            if not session or not session.is_active():
                return None, None

            partner_id = session.get_partner_id(client_id)
            session.status = SessionStatus.TERMINATED

            if partner_id:
                self._client_to_session.pop(partner_id, None)
                self._client_states[partner_id] = ClientState.IDLE

            self._sessions.pop(session_id, None)
            self._record_terminated(session)

            logger.info(
                "Handled disconnect: client_id=%s, partner_id=%s, session_id=%s",
                client_id,
                partner_id,
                session_id,
            )
            return session, partner_id

    async def terminate_session(
        self,
        session_id: str,
        reason: str = "terminated",
    ) -> Optional[Session]:
        """Directly terminate a session by session ID with status mapping."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active():
                return None

            if reason == "banned":
                session.status = SessionStatus.BANNED
            elif reason == "disconnected":
                session.status = SessionStatus.DISCONNECTED
            elif reason == "leaving":
                session.status = SessionStatus.LEAVING
            elif reason == "skipping":
                session.status = SessionStatus.SKIPPING
            else:
                session.status = SessionStatus.TERMINATED

            self._client_to_session.pop(session.client_a, None)
            self._client_to_session.pop(session.client_b, None)

            self._sessions.pop(session_id, None)
            self._record_terminated(session)

            logger.info(
                "Terminated session: session_id=%s, reason=%s",
                session_id,
                reason,
            )
            return session

    def reset(self) -> None:
        """Reset all internal state maps (used in tests/fixtures)."""
        self._sessions.clear()
        self._client_to_session.clear()
        self._client_states.clear()
        self._terminated_sessions.clear()
