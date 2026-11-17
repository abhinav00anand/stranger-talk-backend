"""Pytest configuration and shared test harness fixtures."""
import sys

# Python 3.14 Windows ujson patch
sys.modules.setdefault("ujson", None)

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Dict, List, Optional

import pytest
import uvicorn
import websockets

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def get_app():
    """Load the real application; import failures must fail the test suite."""
    from app.main import app
    return app


class AsyncStrangerClient:
    """Async WebSocket test client encapsulating ChatFreely protocol envelopes."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.is_initiator: Optional[bool] = None
        self.matched_payload: Optional[Dict[str, Any]] = None
        self._close_code: Optional[int] = None
        self._close_reason: Optional[str] = None

    @property
    def close_code(self) -> Optional[int]:
        if self._close_code is not None:
            return self._close_code
        if self.ws and hasattr(self.ws, "close_code"):
            return self.ws.close_code
        return None

    @property
    def close_reason(self) -> Optional[str]:
        if self._close_reason is not None:
            return self._close_reason
        if self.ws and hasattr(self.ws, "close_reason"):
            return self.ws.close_reason
        return None

    async def connect(self, timeout: float = 5.0) -> "AsyncStrangerClient":
        """Connect to WebSocket endpoint."""
        self.ws = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=timeout)
        return self

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send raw JSON object over WebSocket."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        await self.ws.send(json.dumps(data))

    async def send_envelope(
        self,
        msg_type: str,
        payload: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Send standardized ChatFreely WebSocket message envelope."""
        envelope = {
            "type": msg_type,
            "session_id": session_id if session_id is not None else self.session_id,
            "payload": payload if payload is not None else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.send_json(envelope)

    async def recv_json(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Receive and parse JSON message envelope."""
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            data = json.loads(raw)
            if isinstance(data, dict):
                if data.get("session_id"):
                    self.session_id = data["session_id"]
                if data.get("type") == "matched":
                    payload = data.get("payload", {})
                    self.is_initiator = payload.get("is_initiator")
                    self.matched_payload = payload
            return data
        except websockets.exceptions.ConnectionClosed as exc:
            self._close_code = exc.code
            self._close_reason = exc.reason
            raise

    async def recv_type(self, expected_type: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for a specific message type, collecting and returning the matching envelope."""
        start = time.time()
        while time.time() - start < timeout:
            remaining = max(0.1, timeout - (time.time() - start))
            msg = await self.recv_json(timeout=remaining)
            if msg.get("type") == expected_type:
                return msg
        raise TimeoutError(f"Did not receive message type '{expected_type}' within {timeout}s")

    async def drain(self, timeout: float = 0.2) -> List[Dict[str, Any]]:
        """Drain all pending messages from socket buffer until quiet."""
        messages: List[Dict[str, Any]] = []
        while True:
            try:
                msg = await self.recv_json(timeout=timeout)
                messages.append(msg)
            except (asyncio.TimeoutError, TimeoutError, websockets.exceptions.ConnectionClosed):
                break
        return messages

    async def join_queue(self, topics: Optional[List[str]] = None) -> None:
        """Send join_queue command."""
        payload = {}
        if topics is not None:
            payload["topics"] = topics
        await self.send_envelope("join_queue", payload=payload, session_id=None)

    async def leave_queue(self) -> None:
        """Send leave_queue command."""
        await self.send_envelope("leave_queue", payload={}, session_id=None)

    async def skip(self, session_id: Optional[str] = None) -> None:
        """Send skip command for current or specific session."""
        await self.send_envelope("skip", session_id=session_id or self.session_id, payload={})

    async def leave(self, session_id: Optional[str] = None) -> None:
        """Send leave command for current or specific session."""
        await self.send_envelope("leave", session_id=session_id or self.session_id, payload={})

    async def send_chat(self, text: str, session_id: Optional[str] = None) -> None:
        """Send chat text message."""
        await self.send_envelope(
            "chat",
            session_id=session_id or self.session_id,
            payload={"text": text},
        )

    async def send_gif(self, url: str, session_id: Optional[str] = None, **kwargs: Any) -> None:
        """Send GIF URL message with optional metadata."""
        payload = {"url": url, **kwargs}
        await self.send_envelope("gif", session_id=session_id or self.session_id, payload=payload)

    async def send_typing(self, is_typing: bool = True, session_id: Optional[str] = None) -> None:
        """Send typing indicator state."""
        await self.send_envelope(
            "typing",
            session_id=session_id or self.session_id,
            payload={"is_typing": is_typing},
        )

    async def send_sdp_offer(self, sdp: str, session_id: Optional[str] = None) -> None:
        """Send WebRTC SDP offer."""
        await self.send_envelope(
            "offer",
            session_id=session_id or self.session_id,
            payload={"sdp": sdp, "type": "offer"},
        )

    async def send_sdp_answer(self, sdp: str, session_id: Optional[str] = None) -> None:
        """Send WebRTC SDP answer."""
        await self.send_envelope(
            "answer",
            session_id=session_id or self.session_id,
            payload={"sdp": sdp, "type": "answer"},
        )

    async def send_ice_candidate(
        self,
        candidate: str,
        sdpMid: Any = "0",
        sdpMLineIndex: int = 0,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Send WebRTC ICE candidate."""
        payload = {
            "candidate": candidate,
            "sdpMid": str(sdpMid) if sdpMid is not None else None,
            "sdpMLineIndex": sdpMLineIndex,
            **kwargs,
        }
        await self.send_envelope("ice_candidate", session_id=session_id or self.session_id, payload=payload)

    async def send_ping(self) -> None:
        """Send heartbeat ping envelope."""
        await self.send_envelope("ping", session_id=None, payload={})

    async def close(self, code: int = 1000) -> None:
        """Close WebSocket connection gracefully or abruptly simulate network drop."""
        self._close_code = code
        if self.ws:
            try:
                if code == 1006:
                    # Simulate abnormal network closure / abrupt TCP drop
                    if hasattr(self.ws, "transport") and self.ws.transport:
                        self.ws.transport.close()
                    else:
                        await self.ws.close(code=1000)
                else:
                    await self.ws.close(code=code)
            except Exception:
                pass


@pytest.fixture(scope="session")
def live_server():
    """Start ephemeral FastAPI/uvicorn server on 127.0.0.1:0 (free OS port)."""
    app = get_app()

    # Find free OS port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    timeout = 10.0
    start_time = time.time()
    while not server.started:
        if time.time() - start_time > timeout:
            raise TimeoutError("Uvicorn test server failed to start within 10 seconds")
        time.sleep(0.01)

    ws_url = f"ws://127.0.0.1:{port}/ws"
    http_url = f"http://127.0.0.1:{port}"

    yield {"ws_url": ws_url, "http_url": http_url, "port": port, "server": server}

    server.should_exit = True
    thread.join(timeout=2.0)


@pytest.fixture(scope="session")
def server_ws_url(live_server) -> str:
    """Return WebSocket connection URL."""
    return live_server["ws_url"]


@pytest.fixture(scope="session")
def server_http_url(live_server) -> str:
    """Return HTTP base URL."""
    return live_server["http_url"]


@pytest.fixture
async def make_client(server_ws_url):
    """Factory fixture returning an auto-cleaned AsyncStrangerClient connected to the server."""
    clients: List[AsyncStrangerClient] = []

    async def _factory() -> AsyncStrangerClient:
        client = AsyncStrangerClient(server_ws_url)
        await client.connect()
        clients.append(client)
        return client

    yield _factory

    # Teardown all opened clients
    for client in clients:
        try:
            await client.close()
        except Exception:
            pass


@pytest.fixture
async def stranger_client(make_client) -> AsyncStrangerClient:
    """Single pre-connected AsyncStrangerClient fixture."""
    return await make_client()


def _clear_ip_bans():
    """Helper to clear in-memory IP bans from ban_manager if initialized."""
    try:
        from app.moderation.ban_manager import ban_manager
        if hasattr(ban_manager, "clear"):
            ban_manager.clear()
        elif hasattr(ban_manager, "bans") and isinstance(ban_manager.bans, dict):
            ban_manager.bans.clear()
        elif hasattr(ban_manager, "banned_ips") and isinstance(ban_manager.banned_ips, dict):
            ban_manager.banned_ips.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_ip_bans():
    """Clear process-local abuse controls before and after each test."""
    _clear_ip_bans()
    try:
        from app.main import ip_rate_limiter
        ip_rate_limiter._events.clear()
    except ImportError:
        raise
    yield
    _clear_ip_bans()
    from app.main import ip_rate_limiter
    ip_rate_limiter._events.clear()

