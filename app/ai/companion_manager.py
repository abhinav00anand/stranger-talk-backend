"""AI Companion Manager, Conversation State Tracker, and Queue Fallback Watchdog."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from app.config import settings
from app.matchmaker.queue import MatchmakerQueue
from app.matchmaker.session_manager import SessionManager
from app.models.envelopes import (
    ChatMessagePayload,
    MatchedPayload,
    ServerChatMessagePayload,
    ServerMessageType,
    TypingPayload,
    WebSocketEnvelope,
)
from app.models.session import ClientState, QueueTicket, Session, SessionStatus

logger = logging.getLogger(__name__)

DEFAULT_AI_TIMEOUT = 5.0
DEFAULT_TYPING_MIN = 0.5
DEFAULT_TYPING_MAX = 1.2
DEFAULT_PEER_ALIAS = "Stranger"


@dataclass
class AISessionState:
    """Encapsulates in-memory context and sliding window history for an AI companion session."""

    session_id: str
    client_id: str
    companion_id: str = "ai_companion"
    history: deque[Any] = field(default_factory=lambda: deque(maxlen=10))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    active_task: Optional[asyncio.Task] = None
    active_tasks: set[asyncio.Task] = field(default_factory=set, repr=False)
    message_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class CompanionManager:
    """Manages active AI companion sessions, 10-turn sliding memory, and typing pipeline."""

    def __init__(
        self,
        provider: Any = None,
        session_manager: Optional[SessionManager] = None,
        queue: Optional[MatchmakerQueue] = None,
        typing_delay: Optional[float] = None,
        min_typing_delay: float = DEFAULT_TYPING_MIN,
        max_typing_delay: float = DEFAULT_TYPING_MAX,
        peer_alias: str = DEFAULT_PEER_ALIAS,
        send_callback: Optional[Callable[[str, WebSocketEnvelope], Awaitable[None]]] = None,
    ) -> None:
        self._provider = provider
        self.session_manager = session_manager
        self.queue = queue
        self.typing_delay = typing_delay
        self.min_typing_delay = min_typing_delay
        self.max_typing_delay = max_typing_delay
        self.peer_alias = peer_alias
        self.send_callback = send_callback

        self._sessions: Dict[str, AISessionState] = {}
        self._lock = asyncio.Lock()

    def _get_provider(self) -> Any:
        """Resolve AI companion provider from injection or factory."""
        if self._provider is not None:
            return self._provider
        try:
            from app.ai.factory import get_ai_companion_provider
            return get_ai_companion_provider()
        except ImportError:
            from app.ai.mock_provider import MockAICompanionProvider
            return MockAICompanionProvider()

    def _get_typing_delay(self) -> float:
        """Calculate typing delay duration."""
        if self.typing_delay is not None:
            return max(0.0, float(self.typing_delay))
        return random.uniform(self.min_typing_delay, self.max_typing_delay)

    def register_session(
        self,
        session_id: str,
        client_id: str,
        companion_id: str = "ai_companion",
    ) -> AISessionState:
        """Register a new active AI session in the state store."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        state = AISessionState(
            session_id=session_id,
            client_id=client_id,
            companion_id=companion_id,
            history=deque(maxlen=10),
            is_active=True,
        )
        self._sessions[session_id] = state
        logger.info("Registered AI session %s for client %s", session_id, client_id)
        return state

    def get_session(self, session_id: str) -> Optional[AISessionState]:
        """Retrieve active AI session state if present."""
        return self._sessions.get(session_id)

    def is_ai_session(self, session_id: str) -> bool:
        """Check if session is currently tracked as an active AI session."""
        sess = self._sessions.get(session_id)
        return sess is not None and sess.is_active

    def unregister_session(self, session_id: str) -> Optional[AISessionState]:
        """Unregister and cancel ongoing tasks for an AI session."""
        sess = self._sessions.pop(session_id, None)
        if sess:
            sess.is_active = False
            tasks = set(sess.active_tasks)
            if sess.active_task is not None:
                tasks.add(sess.active_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
            logger.info("Unregistered AI session %s", session_id)
        return sess

    def get_history(self, session_id: str) -> List[ChatMessagePayload]:
        """Return a snapshot list of the sliding context history for a session."""
        sess = self._sessions.get(session_id)
        if not sess:
            return []
        return list(sess.history)

    def clear_history(self, session_id: str) -> None:
        """Clear message history for a given session."""
        sess = self._sessions.get(session_id)
        if sess:
            sess.history.clear()

    def active_sessions_count(self) -> int:
        """Return number of currently registered active AI sessions."""
        return len(self._sessions)

    async def _send_typing_indicator(
        self,
        send_fn: Optional[Callable[[str, WebSocketEnvelope], Awaitable[None]]],
        client_id: str,
        session_id: str,
        is_typing: bool,
    ) -> None:
        """Helper to send typing envelope to client."""
        if not send_fn:
            return
        envelope = WebSocketEnvelope.create(
            msg_type=ServerMessageType.TYPING,
            payload=TypingPayload(is_typing=is_typing),
            session_id=session_id,
        )
        try:
            await send_fn(client_id, envelope)
        except Exception as exc:
            logger.warning("Failed to deliver typing envelope to %s: %s", client_id, exc)

    async def process_user_message(
        self,
        session_id: str,
        text: str,
        send_envelope_fn: Optional[Callable[[str, WebSocketEnvelope], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """Process one user turn serially and deliver the companion response."""
        sess = self._sessions.get(session_id)
        if not sess or not sess.is_active:
            logger.warning("Received message for unknown or inactive AI session: %s", session_id)
            return None
        if not text or not isinstance(text, str) or not text.strip():
            return None

        current_task = asyncio.current_task()
        if current_task is not None:
            sess.active_tasks.add(current_task)

        try:
            # Serializing each session preserves conversation order when users send
            # multiple messages before the previous model response has completed.
            async with sess.message_lock:
                if not sess.is_active or self._sessions.get(session_id) is not sess:
                    return None
                return await self._process_user_message_locked(
                    sess=sess,
                    session_id=session_id,
                    clean_text=text.strip(),
                    send_fn=send_envelope_fn or self.send_callback,
                )
        finally:
            if current_task is not None:
                sess.active_tasks.discard(current_task)

    async def _process_user_message_locked(
        self,
        sess: AISessionState,
        session_id: str,
        clean_text: str,
        send_fn: Optional[Callable[[str, WebSocketEnvelope], Awaitable[None]]],
    ) -> Optional[str]:
        """Run typing, provider, history, and delivery steps under the session lock."""
        sess.history.append({"sender": "user", "text": clean_text})
        history_snapshot = list(sess.history)[:-1]
        typing_emitted = False

        try:
            await self._send_typing_indicator(send_fn, sess.client_id, session_id, is_typing=True)
            typing_emitted = True

            delay = self._get_typing_delay()
            if delay > 0:
                await asyncio.sleep(delay)

            if not sess.is_active or self._sessions.get(session_id) is not sess:
                return None

            await self._send_typing_indicator(send_fn, sess.client_id, session_id, is_typing=False)
            typing_emitted = False

            provider = self._get_provider()
            try:
                reply = await provider.generate_reply(
                    session_id=session_id,
                    history=history_snapshot,
                    latest_message=clean_text,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("AI provider error in session %s: %s (falling back)", session_id, exc)
                reply = "Hey, sorry about that! My connection hitched. What were you saying?"

            if not isinstance(reply, str) or not reply.strip():
                reply = "Haha yeah, totally!"
            clean_reply = reply.strip()

            if not sess.is_active or self._sessions.get(session_id) is not sess:
                return None

            sess.history.append({"sender": "assistant", "text": clean_reply})
            chat_payload = ServerChatMessagePayload(sender=self.peer_alias, text=clean_reply)
            envelope = WebSocketEnvelope.create(
                msg_type=ServerMessageType.CHAT,
                payload=chat_payload,
                session_id=session_id,
            )
            if send_fn:
                await send_fn(sess.client_id, envelope)
            return clean_reply
        finally:
            if typing_emitted and sess.is_active and self._sessions.get(session_id) is sess:
                await self._send_typing_indicator(send_fn, sess.client_id, session_id, is_typing=False)

    async def handle_skip(
        self,
        client_id: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[Session], bool]:
        """Handle skip transition with full human session parity.

        Cleans up AI session state, delegates to session_manager.handle_skip,
        and re-enqueues client into MatchmakerQueue.
        """
        target_session_id = session_id
        if not target_session_id and self.session_manager:
            sess = self.session_manager.get_session_by_client(client_id)
            if sess:
                target_session_id = sess.session_id

        if target_session_id:
            self.unregister_session(target_session_id)

        session = None
        if self.session_manager:
            session, _ = await self.session_manager.handle_skip(
                client_id=client_id,
                expected_session_id=target_session_id,
            )

        requeued = False
        if self.queue:
            try:
                await self.queue.enqueue(client_id)
                requeued = True
            except Exception as exc:
                logger.warning("Could not re-enqueue client %s on skip: %s", client_id, exc)

        return session, requeued

    async def handle_leave(
        self,
        client_id: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[Session], bool]:
        """Handle leave transition with full human session parity.

        Cleans up AI session state, delegates to session_manager.handle_leave,
        and transitions user to IDLE.
        """
        target_session_id = session_id
        if not target_session_id and self.session_manager:
            sess = self.session_manager.get_session_by_client(client_id)
            if sess:
                target_session_id = sess.session_id

        if target_session_id:
            self.unregister_session(target_session_id)

        session = None
        if self.session_manager:
            session, _ = await self.session_manager.handle_leave(
                client_id=client_id,
                expected_session_id=target_session_id,
            )

        return session, True

    def reset(self) -> None:
        """Reset all active session states."""
        for state in self._sessions.values():
            state.is_active = False
            tasks = set(state.active_tasks)
            if state.active_task is not None:
                tasks.add(state.active_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
        self._sessions.clear()


class QueueWatchdog:
    """Background watchdog monitoring matchmaking queue for timeout expiration."""

    def __init__(
        self,
        queue: MatchmakerQueue,
        session_manager: SessionManager,
        companion_manager: CompanionManager,
        timeout: Optional[float] = None,
        poll_interval: float = 0.2,
        send_matched_callback: Optional[Callable[[str, WebSocketEnvelope], Awaitable[None]]] = None,
        peer_alias: str = DEFAULT_PEER_ALIAS,
    ) -> None:
        self.queue = queue
        self.session_manager = session_manager
        self.companion_manager = companion_manager
        self._custom_timeout = timeout
        self.poll_interval = poll_interval
        self.send_matched_callback = send_matched_callback
        self.peer_alias = peer_alias

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def timeout(self) -> float:
        """Effective timeout: custom override or dynamic settings.AI_FALLBACK_TIMEOUT_SECONDS."""
        if self._custom_timeout is not None:
            return self._custom_timeout
        return getattr(settings, "AI_FALLBACK_TIMEOUT_SECONDS", DEFAULT_AI_TIMEOUT)

    @timeout.setter
    def timeout(self, value: Optional[float]) -> None:
        self._custom_timeout = value

    @property
    def is_running(self) -> bool:
        """Return True if background watchdog polling task is active."""
        return self._running and self._task is not None and not self._task.done()

    async def check_timeouts(self) -> List[Tuple[str, str]]:
        """Inspect waiting tickets and pop expired clients to match with AI companion.

        Returns list of (client_id, session_id) pairs created in this tick.
        """
        paired: List[Tuple[str, str]] = []
        tickets = await self.queue.get_waiting_tickets()
        current_timeout = self.timeout

        for ticket in tickets:
            if ticket.elapsed_seconds() >= current_timeout:
                # 1. Atomically pop client from queue
                popped_ticket = await self.queue.pop_client(ticket.client_id)
                if popped_ticket is None:
                    # Client was already popped by concurrent human pair or canceled
                    continue

                client_id = ticket.client_id
                # Unique companion ID per client prevents _cleanup_stale_mapping collisions
                companion_id = f"ai_companion_{client_id}"

                # 2. Atomically create 1-on-1 session in session_manager
                session = await self.session_manager.create_session(
                    client_a=client_id,
                    client_b=companion_id,
                    is_ai=True,
                )

                # 3. Register session in companion_manager
                self.companion_manager.register_session(
                    session_id=session.session_id,
                    client_id=client_id,
                    companion_id=companion_id,
                )

                # 4. Construct STUN ice_servers list
                stun_servers = getattr(settings, "STUN_SERVERS", [])
                ice_servers = [{"urls": s} for s in stun_servers]

                # 5. Build matched envelope
                payload = {
                    "is_initiator": True,
                    "peer_is_ai": True,
                    "peer_alias": self.peer_alias,
                    "partner_id": "ai_companion",
                    "ice_servers": ice_servers,
                }
                envelope = WebSocketEnvelope.create(
                    msg_type=ServerMessageType.MATCHED,
                    payload=payload,
                    session_id=session.session_id,
                )

                # 6. Deliver envelope via callback if registered
                if self.send_matched_callback:
                    try:
                        await self.send_matched_callback(client_id, envelope)
                    except Exception as exc:
                        # Delivery can fail when the socket disconnects after the
                        # queue ticket was popped. Roll back both registries so the
                        # client cannot leave behind an unreachable ghost session.
                        logger.warning("Failed to send matched envelope to client %s; rolling back session %s: %s", client_id, session.session_id, exc)
                        await self.session_manager.terminate_session(
                            session.session_id,
                            reason="disconnected",
                        )
                        self.companion_manager.unregister_session(session.session_id)
                        continue

                paired.append((client_id, session.session_id))

        return paired

    async def start(self) -> None:
        """Start the background watchdog polling loop."""
        async with self._lock:
            if self._running:
                return
            self._running = True
            self._task = asyncio.create_task(self._watchdog_loop())
            logger.info("QueueWatchdog started (timeout=%.2fs, interval=%.2fs)", self.timeout, self.poll_interval)

    async def stop(self) -> None:
        """Stop and cleanly await shutdown of the background watchdog loop."""
        async with self._lock:
            if not self._running:
                return
            self._running = False
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
            logger.info("QueueWatchdog stopped")

    async def _watchdog_loop(self) -> None:
        """Periodic loop executing check_timeouts."""
        while self._running:
            try:
                await self.check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("QueueWatchdog iteration error: %s", exc, exc_info=True)

            try:
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> "QueueWatchdog":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
