"""FastAPI application entry point, lifecycle management, and WebSocket router."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import sys
import hmac
from time import monotonic
import uuid

from pydantic import ValidationError
from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.envelopes import (
    ChatMessagePayload,
    GifMessagePayload,
    IceCandidatePayload,
    JoinQueuePayload,
    PeerDisconnectedPayload,
    QueuedPayload,
    ReportPayload,
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
from app.safety.report_store import ReportStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _stdout_handler = logging.StreamHandler(sys.stdout)
    _stdout_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_stdout_handler)
logger.propagate = False


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
report_rate_limiter = IPRateLimiter(3, 3600.0)
report_ip_rate_limiter = IPRateLimiter(20, 3600.0)
report_store = ReportStore(settings.REPORTS_DB_PATH, settings.REPORT_RETENTION_DAYS)
reports_ready = False
blocked_pairs: set[tuple[str, str]] = set()
reported_sessions: set[tuple[str, str]] = set()

# Active WebSocket connections: client_id -> WebSocket
active_connections: dict[str, WebSocket] = {}
# Populated only after a WebSocket proves possession of BOT_AUTH_TOKEN.
bot_connections: set[str] = set()
connection_ips: dict[str, str] = {}
moderation_strikes: dict[str, int] = {}
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


async def _cleanup_empty_sessions_loop():
    """Periodically scan and purge empty or disconnected sessions every 10 seconds."""
    while True:
        try:
            await asyncio.sleep(10.0)
            async with connections_lock:
                active_ids = set(active_connections.keys())
            pruned = await session_manager.prune_empty_sessions(active_ids)
            if pruned:
                logger.info("Automatically cleaned up %d empty session(s): %s", len(pruned), pruned)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in empty session cleanup task")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan."""
    global reports_ready
    logger.info("Starting human-only matchmaking services...")
    try:
        await asyncio.to_thread(report_store.initialize)
        reports_ready = True
        logger.info("Safety report store initialized")
    except Exception:
        reports_ready = False
        logger.exception("Safety report storage is unavailable; report submissions will fail closed")
    
    cleanup_task = asyncio.create_task(_cleanup_empty_sessions_loop())
    yield
    cleanup_task.cancel()
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


@app.get("/admin/reports")
async def list_safety_reports(
    limit: int = Query(default=50, ge=1, le=200),
    x_admin_token: str | None = Header(default=None),
):
    """Read recent moderation reports. This endpoint requires a private admin token."""
    if not settings.REPORTS_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Report administration is not configured.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.REPORTS_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    if not reports_ready:
        raise HTTPException(status_code=503, detail="Report storage is unavailable.")
    reports = await asyncio.to_thread(report_store.list_reports, limit)
    return {"reports": reports}


@app.get("/health")
async def health_check():
    """Check that the core in-process matchmaking components can be queried."""
    try:
        queue_size = matchmaker_queue.queue_size()
        active_sessions = session_manager.active_session_count()
        active_connection_count = len(active_connections)
    except Exception:
        logger.exception("Health check failed while querying core services")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": settings.APP_NAME,
                "environment": settings.APP_ENV,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return {
        "status": "healthy",
        "checks": {"matchmaker": "ok", "session_manager": "ok"},
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "queue_size": queue_size,
        "active_sessions": active_sessions,
        "active_connections": active_connection_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _return_bot_to_standby(bot_id: str) -> bool:
    """Return a still-connected authenticated bot to standby after its human leaves."""
    if bot_id not in bot_connections:
        return False
    async with connections_lock:
        is_connected = bot_id in active_connections
    if not is_connected:
        return False
    await matchmaker_queue.register_standby_bot(bot_id)
    session_manager.set_client_state(bot_id, ClientState.IDLE)
    await send_to_client(
        bot_id,
        WebSocketEnvelope.create(
            msg_type=ServerMessageType.QUEUED,
            payload=QueuedPayload(position=0, estimated_wait_seconds=None).model_dump(),
            session_id=None,
        ),
    )
    logger.info("Authenticated bot returned to standby client_id=%s", bot_id)
    return True


async def _handle_join_queue(client_id: str, client_ip: str, payload_data: dict) -> None:
    """Handle join_queue command with FIFO human-only pairing."""
    # Check if client is banned
    if ban_manager.is_banned(client_ip):
        ban_rec = ban_manager.get_ban_record(client_ip)
        duration = ban_rec.remaining_seconds() if ban_rec else 900
        banned_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.BANNED,
            payload={
                "reason": ban_rec.reason if ban_rec else "Access from this network was temporarily restricted.",
                "ban_duration_seconds": duration,
                "ban_scope": "ip",
                "appeal_email": "indrohelpdesk@gmail.com",
            },
            session_id=None,
        )
        session_manager.set_client_state(client_id, ClientState.BANNED)
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

    # A client-supplied is_bot flag is not proof of identity. Only a WebSocket
    # authenticated with the server-side bot token may enter the standby pool.
    is_authenticated_bot = client_id in bot_connections
    if validated.is_bot and not is_authenticated_bot:
        await send_to_client(
            client_id,
            WebSocketEnvelope.create(
                msg_type=ServerMessageType.ERROR,
                payload={
                    "code": "BOT_AUTH_REQUIRED",
                    "message": "Bot connections must authenticate with the server-issued bot token.",
                },
                session_id=None,
            ),
        )
        return

    if is_authenticated_bot:
        await matchmaker_queue.register_standby_bot(client_id)
        session_manager.set_client_state(client_id, ClientState.IDLE)
        queued_env = WebSocketEnvelope.create(
            msg_type=ServerMessageType.QUEUED,
            payload=QueuedPayload(position=0, estimated_wait_seconds=None).model_dump(),
            session_id=None,
        )
        await send_to_client(client_id, queued_env)
        logger.info("Authenticated bot registered to standby pool client_id=%s total_standby=%d", client_id, matchmaker_queue.standby_bot_count())
        return

    await matchmaker_queue.enqueue(
        client_id=client_id,
        ip=client_ip,
        topics=validated.topics,
        is_bot=False,
    )
    session_manager.set_client_state(client_id, ClientState.QUEUED)

    # Send queued confirmation
    queued_env = WebSocketEnvelope.create(
        msg_type=ServerMessageType.QUEUED,
        payload=QueuedPayload(position=matchmaker_queue.queue_size(), estimated_wait_seconds=None).model_dump(),
        session_id=None,
    )
    await send_to_client(client_id, queued_env)

    await _pair_waiting_clients()

    # On-Demand Bot Match: If this human has no human partner, immediately check for a standby bot!
    current_state = session_manager.get_client_state(client_id)
    if current_state == ClientState.QUEUED:
        async with connections_lock:
            active_ids = set(active_connections.keys())
        bot_id = await matchmaker_queue.pop_standby_bot(
            human_client_id=client_id,
            blocked_pairs=blocked_pairs,
            active_connection_ids=active_ids,
        )
        if bot_id:
            # Atomically pop human from waiting tickets. If another concurrent
            # transition already removed it, release the reserved bot immediately.
            human_ticket = await matchmaker_queue.pop_client(client_id)
            if not human_ticket:
                await _return_bot_to_standby(bot_id)
                return

            session = await session_manager.create_session(client_a=client_id, client_b=bot_id)
            ice_servers = [{"urls": list(settings.STUN_SERVERS)}]
            env_human = WebSocketEnvelope.create(
                msg_type=ServerMessageType.MATCHED,
                payload={"is_initiator": False, "peer_alias": "TalkFreely bot", "peer_is_bot": True, "partner_id": bot_id, "ice_servers": ice_servers},
                session_id=session.session_id,
            )
            env_bot = WebSocketEnvelope.create(
                msg_type=ServerMessageType.MATCHED,
                payload={"is_initiator": True, "peer_alias": "Stranger", "peer_is_bot": False, "partner_id": client_id, "ice_servers": ice_servers},
                session_id=session.session_id,
            )
            del_h, del_b = await asyncio.gather(
                send_to_client(client_id, env_human), send_to_client(bot_id, env_bot)
            )
            if del_h and del_b:
                logger.info("On-demand bot match delivered: Human %s <-> Bot %s", client_id, bot_id)
            else:
                await session_manager.terminate_session(session.session_id, reason="disconnected")
                async with connections_lock:
                    h_alive = client_id in active_connections
                    b_alive = bot_id in active_connections
                if h_alive:
                    if del_h:
                        await send_to_client(client_id, WebSocketEnvelope.create(
                            msg_type=ServerMessageType.PEER_DISCONNECTED,
                            payload=PeerDisconnectedPayload(reason="disconnected", can_requeue=True).model_dump(),
                            session_id=session.session_id,
                        ))
                    await matchmaker_queue.enqueue(client_id=client_id, ip=client_ip, is_bot=False)
                    session_manager.set_client_state(client_id, ClientState.QUEUED)
                    await send_to_client(client_id, WebSocketEnvelope.create(
                        msg_type=ServerMessageType.QUEUED,
                        payload=QueuedPayload(position=matchmaker_queue.queue_size()).model_dump(),
                        session_id=None,
                    ))
                if b_alive:
                    if del_b:
                        await send_to_client(bot_id, WebSocketEnvelope.create(
                            msg_type=ServerMessageType.PEER_DISCONNECTED,
                            payload=PeerDisconnectedPayload(reason="disconnected", can_requeue=True).model_dump(),
                            session_id=session.session_id,
                        ))
                    await _return_bot_to_standby(bot_id)


async def _pair_waiting_clients() -> None:
    """Create human-only pairs and roll back incomplete match deliveries."""
    while True:
        pair = await matchmaker_queue.try_pop_pair(blocked_pairs=blocked_pairs)
        if pair is None:
            return
        client_a, client_b = pair

        # Ghost validation: If either client is no longer an active connection, discard it immediately!
        async with connections_lock:
            a_alive = client_a in active_connections
            b_alive = client_b in active_connections
        if not a_alive or not b_alive:
            for dead_cid, is_dead in ((client_a, not a_alive), (client_b, not b_alive)):
                if is_dead:
                    await matchmaker_queue.pop_client(dead_cid)
                    session_manager.forget_client(dead_cid)
            # If one was alive, put it back at the front of the queue
            for alive_cid, is_alive in ((client_a, a_alive), (client_b, b_alive)):
                if is_alive:
                    async with connections_lock:
                        ip = connection_ips.get(alive_cid)
                    await matchmaker_queue.enqueue(client_id=alive_cid, ip=ip)
            continue

        session = await session_manager.create_session(client_a=client_a, client_b=client_b)
        ice_servers = [{"urls": list(settings.STUN_SERVERS)}]
        env_a = WebSocketEnvelope.create(
            msg_type=ServerMessageType.MATCHED,
            payload={"is_initiator": True, "peer_alias": "Stranger", "peer_is_bot": False, "partner_id": client_b, "ice_servers": ice_servers},
            session_id=session.session_id,
        )
        env_b = WebSocketEnvelope.create(
            msg_type=ServerMessageType.MATCHED,
            payload={"is_initiator": False, "peer_alias": "Stranger", "peer_is_bot": False, "partner_id": client_a, "ice_servers": ice_servers},
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
    await matchmaker_queue.unregister_standby_bot(client_id)
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

    # "Skip / Next" means this exact pair should not be selected again during
    # either live connection, including when the partner is a standby bot.
    if partner_id:
        blocked_pairs.add(_blocked_pair_key(client_id, partner_id))

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
        await _return_bot_to_standby(partner_id)

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
        await _return_bot_to_standby(partner_id)

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

    if mod_result.is_blocked:
        # Escalate repeated blocked messages on this connection only. Never punish
        # an IP shared by family members, schools, offices, or mobile carriers.
        strikes = moderation_strikes.get(client_id, 0) + 1
        moderation_strikes[client_id] = strikes
        if strikes >= 3:
            session_manager.set_client_state(client_id, ClientState.BANNED)
            await send_to_client(client_id, WebSocketEnvelope.create(
                msg_type=ServerMessageType.BANNED,
                payload={
                    "reason": "Repeated messages continued to violate the TalkFreely Community Rules and Terms & Conditions.",
                    "policy": "Repeated abusive messages are not permitted.",
                    "ban_scope": "connection",
                    "appeal_email": "indrohelpdesk@gmail.com",
                },
                session_id=session.session_id,
            ))
            partner_id = session.get_partner_id(client_id)
            if partner_id:
                await send_to_client(partner_id, create_partner_banned_envelope(
                    session.session_id,
                    reason="banned",
                    message="Your partner was disconnected after repeated safety violations.",
                ))
            await session_manager.terminate_session(session.session_id, reason="banned")
            session_manager.set_client_state(client_id, ClientState.BANNED)
            async with connections_lock:
                ws = active_connections.get(client_id)
            if ws:
                await ws.close(code=1008, reason="Repeated policy violations")
            return

        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={
                "code": "MESSAGE_BLOCKED",
                "message": "That message was blocked by the community safety rules. Please rephrase it respectfully.",
            },
            session_id=session.session_id,
        ))
        return

    if mod_result.is_bannable:
        # Explicit, directed violent threats end this connection and session.
        # They do not create an IP ban: the current ban store is process-local,
        # and an IP may be shared by unrelated users.
        session_manager.set_client_state(client_id, ClientState.BANNED)
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.BANNED,
            payload={
                "reason": "A direct violent threat was detected. Threats and intimidation violate the TalkFreely Community Rules and Terms & Conditions.",
                "policy": "Threats of violence are prohibited.",
                "ban_scope": "connection",
                "appeal_email": "indrohelpdesk@gmail.com",
            },
            session_id=session.session_id,
        ))
        partner_id = session.get_partner_id(client_id)
        if partner_id:
            await send_to_client(partner_id, create_partner_banned_envelope(
                session.session_id,
                reason="banned",
                message="Your partner was disconnected for a serious safety violation.",
            ))
        await session_manager.terminate_session(session.session_id, reason="banned")
        session_manager.set_client_state(client_id, ClientState.BANNED)
        async with connections_lock:
            ws = active_connections.get(client_id)
        if ws:
            await ws.close(code=1008, reason="Severe policy violation")
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


def _sanitize_gif_dimension(value: object) -> int | None:
    """Accept small positive integer dimensions only; cap untrusted client values."""
    if value is None or isinstance(value, bool):
        return None
    raw = str(value).strip()
    if not raw or len(raw) > 5 or not raw.isascii() or not raw.isdecimal():
        return None
    dimension = int(raw)
    if dimension < 1:
        return None
    return min(dimension, 10_000)


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
        # Reject untrusted media, but a keyword in a URL path is not proof of intent
        # and must never trigger an IP ban.
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

    # Sanitize untrusted dimensions and title before relaying provider metadata.
    sanitized_w = _sanitize_gif_dimension(payload_data.get("width"))
    sanitized_h = _sanitize_gif_dimension(payload_data.get("height"))
    raw_title = payload_data.get("title")
    safe_title = raw_title.strip()[:160] if isinstance(raw_title, str) and raw_title.strip() else "GIF"

    partner_id = session.get_partner_id(client_id)
    msg_id = str(uuid.uuid4())
    relayed = WebSocketEnvelope.create(
        msg_type=ServerMessageType.GIF,
        payload={
            "url": safe_url,
            "sender": "peer",
            "message_id": msg_id,
            "provider": payload_data.get("provider") if payload_data.get("provider") in {"gifsnap", "tenor", "giphy"} else "gifsnap",
            "id": payload_data.get("id") if isinstance(payload_data.get("id"), str) else None,
            "preview_url": preview_url,
            "title": safe_title,
            "width": sanitized_w,
            "height": sanitized_h,
        },
        session_id=session.session_id,
    )
    await send_to_client(partner_id, relayed)


def _blocked_pair_key(left: str, right: str) -> tuple[str, str]:
    """Return a stable key for a pair of live anonymous connections."""
    return tuple(sorted((left, right)))


async def _end_session_after_safety_action(
    client_id: str,
    client_ip: str,
    session,
    partner_id: str | None,
    success_type: ServerMessageType,
    success_payload: dict,
) -> None:
    """End a reported/blocked chat, prevent this pair being rematched, and requeue reporter."""
    if partner_id:
        blocked_pairs.add(_blocked_pair_key(client_id, partner_id))

    await session_manager.terminate_session(session.session_id, reason="blocked")
    if partner_id:
        await send_to_client(partner_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.PEER_DISCONNECTED,
            payload=PeerDisconnectedPayload(
                reason="left",
                can_requeue=True,
                message="The other participant ended this conversation.",
            ).model_dump(),
            session_id=session.session_id,
        ))
        await _return_bot_to_standby(partner_id)

    await send_to_client(client_id, WebSocketEnvelope.create(
        msg_type=success_type,
        payload=success_payload,
        session_id=session.session_id,
    ))
    # Keep the same WebSocket client ID so the temporary block survives re-queueing.
    await _handle_join_queue(client_id, client_ip, {})


async def _handle_block(client_id: str, client_ip: str, session_id: str | None) -> None:
    """Block the current stranger for the lifetime of these anonymous connections."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "There is no active stranger to block."},
            session_id=session_id,
        ))
        return
    if not session_id or session.session_id != session_id:
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match."},
            session_id=session_id,
        ))
        return

    partner_id = session.get_partner_id(client_id)
    await _end_session_after_safety_action(
        client_id,
        connection_ips.get(client_id, "127.0.0.1"),
        session,
        partner_id,
        ServerMessageType.BLOCKED,
        {"message": "This stranger has been blocked for this connection. Finding another person."},
    )


async def _handle_report(
    client_id: str,
    client_ip: str,
    payload_data: dict,
    session_id: str | None,
) -> None:
    """Persist a report for the current peer, then block the pair and requeue reporter."""
    session = session_manager.get_session_by_client(client_id)
    if not session or not session.is_active():
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "NOT_IN_SESSION", "message": "There is no active stranger to report."},
            session_id=session_id,
        ))
        return
    if not session_id or session.session_id != session_id:
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_SESSION", "message": "A current session ID is required and must match."},
            session_id=session_id,
        ))
        return
    if not reports_ready:
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "REPORT_STORAGE_UNAVAILABLE", "message": "Reports are temporarily unavailable. Please email indrohelpdesk@gmail.com instead."},
            session_id=session.session_id,
        ))
        return
    try:
        report = ReportPayload.model_validate(payload_data)
    except ValidationError:
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "INVALID_REPORT", "message": "Choose a valid report reason and keep details within 1,000 characters."},
            session_id=session.session_id,
        ))
        return

    report_key = (client_id, session.session_id)
    if report_key in reported_sessions:
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "REPORT_ALREADY_SUBMITTED", "message": "A report has already been submitted for this conversation."},
            session_id=session.session_id,
        ))
        return
    if not report_rate_limiter.allow(client_id) or not report_ip_rate_limiter.allow(client_ip):
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "REPORT_RATE_LIMITED", "message": "You have reached the report limit. Please contact indrohelpdesk@gmail.com for urgent concerns."},
            session_id=session.session_id,
        ))
        return

    partner_id = session.get_partner_id(client_id)
    report_id = str(uuid.uuid4())
    try:
        await asyncio.to_thread(
            report_store.create_report,
            report_id=report_id,
            session_id=session.session_id,
            reporter_client_id=client_id,
            reported_client_id=partner_id or "unknown",
            category=report.category,
            details=report.details,
            message_excerpt=report.message_excerpt,
        )
    except Exception:
        logger.exception("Unable to store a safety report")
        await send_to_client(client_id, WebSocketEnvelope.create(
            msg_type=ServerMessageType.ERROR,
            payload={"code": "REPORT_STORAGE_ERROR", "message": "Your report could not be saved. Please email indrohelpdesk@gmail.com."},
            session_id=session.session_id,
        ))
        return

    reported_sessions.add(report_key)
    await _end_session_after_safety_action(
        client_id,
        client_ip,
        session,
        partner_id,
        ServerMessageType.REPORT_SUBMITTED,
        {
            "report_id": report_id,
            "message": "Report received. This stranger has been blocked and you are being matched with someone else.",
        },
    )


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
        logger.warning("WebSocket rejected reason=origin_not_allowed")
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
            payload={
                "reason": banned_rec.reason if banned_rec else "Access from this network was temporarily restricted.",
                "ban_duration_seconds": duration,
                "ban_scope": "ip",
                "appeal_email": "indrohelpdesk@gmail.com",
            },
            session_id=None,
        )
        await websocket.send_text(banned_env.model_dump_json())
        logger.warning("WebSocket close requested reason=ip_banned code=1008")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="IP is banned")
        return

    supplied_bot_token = websocket.headers.get("x-talkfreely-bot-token")
    is_authenticated_bot = bool(
        settings.BOT_AUTH_TOKEN
        and supplied_bot_token
        and hmac.compare_digest(supplied_bot_token, settings.BOT_AUTH_TOKEN)
    )
    if supplied_bot_token and not is_authenticated_bot:
        logger.warning("WebSocket rejected reason=invalid_bot_token")
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid bot credentials",
        )
        return

    await websocket.accept()
    client_id = str(uuid.uuid4())
    if is_authenticated_bot:
        bot_connections.add(client_id)
    connected_at = monotonic()
    logger.info("WebSocket connected client_id=%s", client_id)

    async with connections_lock:
        active_connections[client_id] = websocket
        connection_ips[client_id] = client_ip

    try:
        while True:
            raw_text = await websocket.receive_text()
            if len(raw_text.encode("utf-8")) > settings.WS_MAX_MESSAGE_BYTES:
                logger.warning("WebSocket close requested client_id=%s reason=message_too_large code=1009", client_id)
                await websocket.close(code=1009, reason="Message too large")
                break
            if not ip_rate_limiter.allow(client_ip):
                logger.warning("WebSocket close requested client_id=%s reason=ip_rate_limit code=1008", client_id)
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
                # _handle_chat may close this socket after sending a "banned" notice.
                # Break here rather than reading another frame from a closed ASGI socket.
                if session_manager.get_client_state(client_id) == ClientState.BANNED:
                    break
            elif msg_type == "gif":
                await _handle_gif(client_id, client_ip, payload, session_id)
            elif msg_type == "report":
                await _handle_report(client_id, client_ip, payload, session_id)
            elif msg_type == "block":
                await _handle_block(client_id, client_ip, session_id)
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

    except WebSocketDisconnect as exc:
        logger.info(
            "WebSocket disconnected client_id=%s close_code=%s duration_seconds=%.1f",
            client_id,
            exc.code,
            monotonic() - connected_at,
        )
    except asyncio.CancelledError:
        logger.info("WebSocket task cancelled client_id=%s", client_id)
        raise
    except Exception:
        logger.exception("Unexpected error in websocket loop client_id=%s", client_id)
    finally:
        logger.info("WebSocket cleanup started client_id=%s duration_seconds=%.1f", client_id, monotonic() - connected_at)
        session, partner_id = await session_manager.handle_disconnect(client_id)
        if session and partner_id:
            disc_env = WebSocketEnvelope.create(msg_type=ServerMessageType.PEER_DISCONNECTED, payload=PeerDisconnectedPayload(reason="disconnected", can_requeue=True).model_dump(), session_id=session.session_id)
            await send_to_client(partner_id, disc_env)
            await _return_bot_to_standby(partner_id)
        async with connections_lock:
            active_connections.pop(client_id, None)
            connection_ips.pop(client_id, None)
            moderation_strikes.pop(client_id, None)
        bot_connections.discard(client_id)
        blocked_pairs.difference_update(pair for pair in list(blocked_pairs) if client_id in pair)
        reported_sessions.difference_update(key for key in list(reported_sessions) if key[0] == client_id)
        report_rate_limiter._events.pop(client_id, None)
        await matchmaker_queue.unregister_standby_bot(client_id)
        await matchmaker_queue.pop_client(client_id)
        session_manager.forget_client(client_id)
        logger.info(
            "WebSocket cleanup complete client_id=%s duration_seconds=%.1f",
            client_id,
            monotonic() - connected_at,
        )
