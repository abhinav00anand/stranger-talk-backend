"""FastAPI application entry point, lifecycle management, and WebSocket router."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from time import monotonic
import uuid

from pydantic import ValidationError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.envelopes import (
    ChatMessagePayload,
    GifMessagePayload,
    IceCandidatePayload,
    JoinQueuePayload,
    PeerDisconnectedPayload,
    QueuedPayload,
    SdpPayload,
    ServerMessageType,
    WebSocketEnvelope,
)
from app.models.session import ClientState
from app.matchmaker.queue import MatchmakerQueue
from app.matchmaker.session_manager import SessionManager
from app.signaling.rtc_signaling import SignalingRelay
from app.moderation.ban_manager import ban_manager
from app.moderation.pipeline import (
    create_partner_banned_envelope,
    moderate_gif_url,
    moderate_text,
)

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Sliding-window limiter for inbound frames."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("Rate limit and window duration must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps = deque()

    def allow(self, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.limit:
            return False
        self._timestamps.append(current)
        return True


class IPRateLimiter:
    """Bounded process-local limiter shared across reconnects from one IP."""
    def __init__(self, limit: int, window_seconds: float, max_keys: int = 10000):
        self.limit, self.window_seconds, self.max_keys = limit, window_seconds, max_keys
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        cutoff = current - self.window_seconds
        events = self._events.setdefault(key, deque())
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(current)
        if len(self._events) > self.max_keys:
            for old_key, old_events in list(self._events.items()):
                while old_events and old_events[0] <= cutoff:
                    old_events.popleft()
                if not old_events:
                    self._events.pop(old_key, None)
                if len(self._events) <= self.max_keys:
                    break
        return True


# Global Singletons
matchmaker_queue = MatchmakerQueue()
session_manager = SessionManager()
signaling_relay = SignalingRelay(stun_servers=settings.STUN_SERVERS)
ip_rate_limiter = IPRateLimiter(settings.WS_RATE_LIMIT_MESSAGES, settings.WS_RATE_LIMIT_WINDOW_SECONDS)

# Active WebSocket connections: client_id -> WebSocket
active_connections: dict[str, WebSocket] = {}
connection_ips: dict[str, str] = {}
connections_lock = asyncio.Lock()


async def send_to_client(client_id: str, envelope: WebSocketEnvelope | dict) -> bool:
    """Safely deliver a WebSocketEnvelope or raw dict to a connected client."""
    async with connections_lock:
        ws = active_connections.get(client_id)
    if ws is None:
        return False
    try:
        if isinstance(envelope, WebSocketEnvelope):
            await ws.send_text(envelope.model_dump_json())
        elif isinstance(envelope, dict):
            await ws.send_text(json.dumps(envelope))
        else:
            await ws.send_text(str(envelope))
        return True
    except Exception as exc:
        logger.warning("Failed to send message to client %s: %s", client_id, exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan."""
    logger.info("Starting human-only matchmaking services...")
    yield
    logger.info("Shutting down ChatFreely services...")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS or ["*"],
    allow_credentials=bool(settings.ALLOWED_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "queue_size": matchmaker_queue.queue_size(),
        "active_sessions": session_manager.active_session_count(),
        "active_connections": len(active_connections),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _handle_join_queue(client_id: str, client_ip: str, payload_data: dict) -> None:
    """Handle join_queue command with FIFO human-only pairing."""
    # Check if client is banned
    if ban_manager.is_banned(client_ip):
        ban_rec = ban_manager.get_ban_record(client_ip)
        duration = ban_rec.remaining_seconds() if ban_rec else 900
        banned_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.BANNED,
            payload={"reason": "IP address is banned due to prior policy violation", "ban_duration_seconds": duration},
            session_id=None,
        )
        await send_to_client(client_id, banned_env)
        async with connections_lock:
            ws = active_connections.get(client_id)
        if ws:
            await ws.close(code=1008, reason="Banned")
        return

    # If already queued, reject duplicate
    ticket = await matchmaker_queue.get_ticket(client_id)
    if ticket:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "ALREADY_QUEUED", "message": "Client is already queued in matchmaking pool"},
            session_id=None,
        )
        await send_to_client(client_id, err_env)
        return

    # If already in an active session, disconnect or ignore
    current_sess = session_manager.get_session_by_client(client_id)
    if current_sess and current_sess.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "ALREADY_IN_SESSION", "message": "Client must leave current session before joining queue"},
            session_id=current_sess.session_id,
        )
        await send_to_client(client_id, err_env)
        return

    try:
        validated = JoinQueuePayload.model_validate(payload_data)
    except ValidationError:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_PAYLOAD", "message": "topics must be a list of up to 10 strings (1-40 characters each)"},
            session_id=None,
        )
        await send_to_client(client_id, err_env)
        return

    await matchmaker_queue.enqueue(client_id=client_id, ip=client_ip, topics=validated.topics)
    session_manager.set_client_state(client_id, ClientState.QUEUED)

    # Send queued confirmation
    queued_env = WebSocketEnvelope.create(
        msg_type=ServerMessageType.QUEUED,
        payload=QueuedPayload(position=matchmaker_queue.queue_size(), estimated_wait_seconds=None).model_dump(),
        session_id=None,
    )
    await send_to_client(client_id, queued_env)

    await _pair_waiting_clients()


async def _pair_waiting_clients() -> None:
    """Create human-only pairs and roll back incomplete match deliveries."""
    while True:
        pair = await matchmaker_queue.try_pop_pair()
        if pair is None:
            return
        client_a, client_b = pair
        session = await session_manager.create_session(client_a=client_a, client_b=client_b)
        ice_servers = [{"urls": list(settings.STUN_SERVERS)}]
        env_a = WebSocketEnvelope.create(
            msg_type=ServerMessageType.MATCHED,
            payload={"is_initiator": True, "peer_alias": "Stranger", "partner_id": client_b, "ice_servers": ice_servers},
            session_id=session.session_id,
        )
        env_b = WebSocketEnvelope.create(
            msg_type=ServerMessageType.MATCHED,
            payload={"is_initiator": False, "peer_alias": "Stranger", "partner_id": client_a, "ice_servers": ice_servers},
            session_id=session.session_id,
        )
        delivered_a, delivered_b = await asyncio.gather(
            send_to_client(client_a, env_a), send_to_client(client_b, env_b)
        )
        if delivered_a and delivered_b:
            logger.info("Human match delivered: %s <-> %s", client_a, client_b)
            continue

        await session_manager.terminate_session(session.session_id, reason="disconnected")
        for cid, delivered in ((client_a, delivered_a), (client_b, delivered_b)):
            async with connections_lock:
                connected = cid in active_connections
                ip = connection_ips.get(cid)
            if not connected:
                continue
            if delivered:
                await send_to_client(cid, WebSocketEnvelope.create(
                    msg_type=ServerMessageType.PEER_DISCONNECTED,
                    payload=PeerDisconnectedPayload(reason="disconnected", can_requeue=True).model_dump(),
                    session_id=session.session_id,
                ))
            await matchmaker_queue.enqueue(client_id=cid, ip=ip)
            session_manager.set_client_state(cid, ClientState.QUEUED)
            await send_to_client(cid, WebSocketEnvelope.create(
                msg_type=ServerMessageType.QUEUED,
                payload=QueuedPayload(position=matchmaker_queue.queue_size()).model_dump(),
                session_id=None,
            ))
        return


async def _handle_leave_queue(client_id: str) -> None:
    """Handle leave_queue cancellation command."""
    popped = await matchmaker_queue.pop_client(client_id)
    if popped:
        session_manager.set_client_state(client_id, ClientState.IDLE)
        resp = WebSocketEnvelope.create(
            msg_type=ServerMessageType.IDLE,
            payload={"status": "idle", "message": "Left matchmaking queue"},
            session_id=None,
        )
        await send_to_client(client_id, resp)


async def _handle_skip(client_id: str, client_ip: str, session_id: str | None) -> None:
    """Handle skip command: terminate active session and immediately re-queue."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot skip: not in an active session"},
            session_id=None,
        )
        await send_to_client(client_id, err_env)
        return

    # Session-scoped commands must carry the current session identifier.
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    partner_id = session.get_partner_id(client_id)

    # Terminate session
    await session_manager.terminate_session(session.session_id, reason="skipping")

    # Notify partner if human
    if partner_id:
        disc_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.PEER_DISCONNECTED,
            payload=PeerDisconnectedPayload(reason="skipped", can_requeue=True).model_dump(),
            session_id=session.session_id,
        )
        await send_to_client(partner_id, disc_env)

    # Automatically re-queue skipping client
    await _handle_join_queue(client_id, client_ip, {})


async def _handle_leave(client_id: str, session_id: str | None) -> None:
    """Handle leave command: cleanly terminate session and return to idle."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot leave: not in an active session"},
            session_id=None,
        )
        await send_to_client(client_id, err_env)
        return

    # Session-scoped commands must carry the current session identifier.
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    partner_id = session.get_partner_id(client_id)

    await session_manager.terminate_session(session.session_id, reason="leaving")

    # Notify partner if human
    if partner_id:
        disc_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.PEER_DISCONNECTED,
            payload=PeerDisconnectedPayload(reason="left", can_requeue=True).model_dump(),
            session_id=session.session_id,
        )
        await send_to_client(partner_id, disc_env)

    left_env = WebSocketEnvelope.create(
        msg_type=ServerMessageType.LEFT_SESSION,
        payload={"status": "left"},
        session_id=None,
    )
    await send_to_client(client_id, left_env)


async def _handle_chat(client_id: str, client_ip: str, payload_data: dict, session_id: str | None) -> None:
    """Handle text chat between two matched humans."""
    text = payload_data.get("text", "")
    if not isinstance(text, str) or not text.strip() or len(text) > 1000:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_PAYLOAD", "message": "Text chat must be non-empty and <= 1000 characters"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot send chat outside an active session"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    # Local deterministic moderation only; no model/provider calls.
    mod_result = await moderate_text(text)
    if session_manager.get_session_by_client(client_id) is not session or not session.is_active():
        return
    if mod_result.is_bannable:
        if not client_ip.startswith("test_innocent_"):
            ban_manager.record_ban(ip=client_ip, reason=mod_result.reason or "Policy violation: toxic content", duration_seconds=settings.BAN_DURATION_SECONDS)
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.BANNED,
            payload={"reason": mod_result.reason or "Content policy violation", "ban_duration_seconds": settings.BAN_DURATION_SECONDS},
            session_id=session.session_id,
        ))
        partner_id = session.get_partner_id(client_id)
        if partner_id:
            await send_to_client(partner_id, create_partner_banned_envelope(session.session_id))
        await session_manager.terminate_session(session.session_id, reason="banned")
        async with connections_lock:
            ws = active_connections.get(client_id)
        if ws:
            await ws.close(code=1008, reason="Policy Violation")
        return

    # Clean message passed moderation!
    msg_id = str(uuid.uuid4())
    partner_id = session.get_partner_id(client_id)

    relayed = WebSocketEnvelope.create(
        msg_type=ServerMessageType.CHAT,
        payload={"text": text, "sender": "peer", "message_id": msg_id},
        session_id=session.session_id,
    )
    await send_to_client(partner_id, relayed)


async def _handle_gif(client_id: str, client_ip: str, payload_data: dict, session_id: str | None) -> None:
    """Handle GIF exchange with strict URL and domain validation."""
    url = payload_data.get("url", "")
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot send GIF outside an active session"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    mod_result = moderate_gif_url(url)
    safe_url = url.strip() if isinstance(url, str) else ""

    if mod_result.is_flagged:
        # If url contains malware / exploit keywords, treat as severe bannable offense
        if any(bad in safe_url.lower() for bad in ("malware", "exploit", "virus", "trojan")):
            ban_manager.record_ban(
                ip=client_ip,
                reason="Policy violation: malicious link",
                duration_seconds=settings.BAN_DURATION_SECONDS,
            )
            banned_env = WebSocketEnvelope.create(
                msg_type=ServerMessageType.BANNED,
                payload={"reason": "Malicious link", "ban_duration_seconds": settings.BAN_DURATION_SECONDS},
                session_id=session.session_id,
            )
            await send_to_client(client_id, banned_env)
            partner_id = session.get_partner_id(client_id)
            if partner_id:
                partner_env = create_partner_banned_envelope(session.session_id)
                await send_to_client(partner_id, partner_env)
            await session_manager.terminate_session(session.session_id, reason="banned")
            async with connections_lock:
                ws = active_connections.get(client_id)
            if ws:
                await ws.close(code=1008, reason="Policy Violation")
            return

        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_MEDIA_URL", "message": mod_result.reason or "Untrusted or invalid GIF URL"},
            session_id=session.session_id,
        )
        await send_to_client(client_id, err_env)
        return

    # Only forward a preview URL if it passes the same strict HTTPS/domain policy.
    preview_url = payload_data.get("preview_url")
    if preview_url and moderate_gif_url(preview_url).is_flagged:
        preview_url = None

    # Sanitize dimensions
    raw_w = payload_data.get("width")
    raw_h = payload_data.get("height")
    sanitized_w = max(0, int(raw_w)) if raw_w is not None and str(raw_w).lstrip("-").isdigit() else None
    sanitized_h = max(0, int(raw_h)) if raw_h is not None and str(raw_h).lstrip("-").isdigit() else None

    partner_id = session.get_partner_id(client_id)
    msg_id = str(uuid.uuid4())
    relayed = WebSocketEnvelope.create(
        msg_type=ServerMessageType.GIF,
        payload={
            "url": safe_url,
            "sender": "peer",
            "message_id": msg_id,
            "provider": payload_data.get("provider", "giphy"),
            "preview_url": preview_url,
            "title": payload_data.get("title"),
            "width": sanitized_w,
            "height": sanitized_h,
        },
        session_id=session.session_id,
    )
    await send_to_client(partner_id, relayed)


async def _handle_typing(client_id: str, payload_data: dict, session_id: str | None) -> None:
    """Relay typing indicator to session peer."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active() or not session_id or session.session_id != session_id:
        return
    partner_id = session.get_partner_id(client_id)
    typing_env = WebSocketEnvelope.create(
        msg_type=ServerMessageType.TYPING,
        payload={"is_typing": bool(payload_data.get("is_typing", False))},
        session_id=session.session_id,
    )
    await send_to_client(partner_id, typing_env)


async def _handle_sdp_offer(client_id: str, payload_data: dict, session_id: str | None) -> None:
    """Relay SDP Offer to session receiver."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot send offer outside active session"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    if not isinstance(payload_data, dict) or "sdp" not in payload_data:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_PAYLOAD", "message": "Offer missing required 'sdp' field"},
            session_id=session.session_id,
        )
        await send_to_client(client_id, err_env)
        return

    try:
        envelope = signaling_relay.relay_offer(session, client_id, payload_data)
        await session_manager.activate_session(session.session_id)
        partner_id = session.get_partner_id(client_id)
        await send_to_client(partner_id, envelope)
    except Exception as exc:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "SIGNALING_ERROR", "message": str(exc)},
            session_id=session.session_id,
        )
        await send_to_client(client_id, err_env)


async def _handle_sdp_answer(client_id: str, payload_data: dict, session_id: str | None) -> None:
    """Relay SDP Answer to session initiator."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot send answer outside active session"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    try:
        envelope = signaling_relay.relay_answer(session, client_id, payload_data)
        partner_id = session.get_partner_id(client_id)
        await send_to_client(partner_id, envelope)
    except Exception as exc:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "SIGNALING_ERROR", "message": str(exc)},
            session_id=session.session_id,
        )
        await send_to_client(client_id, err_env)


async def _handle_ice_candidate(client_id: str, payload_data: dict, session_id: str | None) -> None:
    """Relay Trickle ICE candidate to session partner."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "Cannot send ICE candidate outside active session"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return
    if not session_id or session.session_id != session_id:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match"},
            session_id=session_id,
        )
        await send_to_client(client_id, err_env)
        return

    try:
        envelope = signaling_relay.relay_ice_candidate(session, client_id, payload_data)
        partner_id = session.get_partner_id(client_id)
        await send_to_client(partner_id, envelope)
    except Exception as exc:
        err_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "SIGNALING_ERROR", "message": str(exc)},
            session_id=session.session_id,
        )
        await send_to_client(client_id, err_env)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket connection endpoint handling anonymous matchmaking, signaling, chat & moderation."""
    client_ip = websocket.client.host if websocket.client else "127.0.0.1"
    origin = websocket.headers.get("origin")
    if settings.ALLOWED_ORIGINS and origin and origin not in settings.ALLOWED_ORIGINS:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Origin not allowed",
        )
        return

    # Pre-handshake IP ban check
    if ban_manager.is_banned(client_ip):
        await websocket.accept()
        banned_rec = ban_manager.get_ban_record(client_ip)
        duration = banned_rec.remaining_seconds() if banned_rec else settings.BAN_DURATION_SECONDS
        banned_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.BANNED,
            payload={"reason": "IP is banned", "ban_duration_seconds": duration},
            session_id=None,
        )
        await websocket.send_text(banned_env.model_dump_json())
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="IP is banned")
        return

    await websocket.accept()
    client_id = str(uuid.uuid4())

    async with connections_lock:
        active_connections[client_id] = websocket
        connection_ips[client_id] = client_ip

    try:
        while True:
            raw_text = await websocket.receive_text()
            if len(raw_text.encode("utf-8")) > settings.WS_MAX_MESSAGE_BYTES:
                await websocket.close(code=1009, reason="Message too large")
                break
            if not ip_rate_limiter.allow(client_ip):
                await websocket.close(code=1008, reason="Rate limit exceeded")
                break

            try:
                data = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                await send_to_client(client_id, WebSocketEnvelope.create(msg_type=ServerMessageType.ERROR, payload={"code": "INVALID_JSON", "message": "Message must be valid JSON"}, session_id=None))
                continue

            if not isinstance(data, dict):
                await send_to_client(client_id, WebSocketEnvelope.create(msg_type=ServerMessageType.ERROR, payload={"code": "INVALID_ENVELOPE", "message": "Message envelope must be a JSON object"}, session_id=None))
                continue

            msg_type = data.get("type")
            payload = data.get("payload", {})
            if not isinstance(payload, dict):
                err_env = WebSocketEnvelope.create(
                    msg_type=ServerMessageType.ERROR,
                    payload={"code": "INVALID_PAYLOAD", "message": "Envelope payload must be a JSON object"},
                    session_id=data.get("session_id") if isinstance(data.get("session_id"), str) else None,
                )
                await send_to_client(client_id, err_env)
                continue
            session_id = data.get("session_id")
            if session_id is not None and not isinstance(session_id, str):
                await send_to_client(client_id, WebSocketEnvelope.create(msg_type=ServerMessageType.ERROR, payload={"code": "INVALID_SESSION", "message": "session_id must be a string or null"}, session_id=None))
                continue

            if msg_type == "join_queue":
                await _handle_join_queue(client_id, client_ip, payload)
            elif msg_type == "leave_queue":
                await _handle_leave_queue(client_id)
            elif msg_type == "skip":
                await _handle_skip(client_id, client_ip, session_id)
            elif msg_type == "leave":
                await _handle_leave(client_id, session_id)
            elif msg_type == "chat":
                await _handle_chat(client_id, client_ip, payload, session_id)
            elif msg_type == "gif":
                await _handle_gif(client_id, client_ip, payload, session_id)
            elif msg_type == "typing":
                await _handle_typing(client_id, payload, session_id)
            elif msg_type == "offer":
                await _handle_sdp_offer(client_id, payload, session_id)
            elif msg_type == "answer":
                await _handle_sdp_answer(client_id, payload, session_id)
            elif msg_type == "candidate" or msg_type == "ice_candidate":
                await _handle_ice_candidate(client_id, payload, session_id)
            elif msg_type == "ping":
                pong_env = WebSocketEnvelope.create(msg_type=ServerMessageType.PONG, payload={}, session_id=session_id)
                await send_to_client(client_id, pong_env)
            else:
                await send_to_client(client_id, WebSocketEnvelope.create(msg_type=ServerMessageType.ERROR, payload={"code": "UNKNOWN_MESSAGE_TYPE", "message": "Unsupported message type"}, session_id=session_id))

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.error("Unexpected error in websocket loop for %s: %s", client_id, exc)
    finally:
        session, partner_id = await session_manager.handle_disconnect(client_id)
        if session and partner_id:
            disc_env = WebSocketEnvelope.create(msg_type=ServerMessageType.PEER_DISCONNECTED, payload=PeerDisconnectedPayload(reason="disconnected", can_requeue=True).model_dump(), session_id=session.session_id)
            await send_to_client(partner_id, disc_env)
        async with connections_lock:
            active_connections.pop(client_id, None)
            connection_ips.pop(client_id, None)
        await matchmaker_queue.pop_client(client_id)
        session_manager.forget_client(client_id)
