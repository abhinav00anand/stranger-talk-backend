# Stranger Talk Backend (ChatFreely)

Real-time anonymous stranger chat and live WebRTC audio calling service inspired by OpenTalk. Built with FastAPI, WebSockets, STUN audio signaling, deterministic local moderation, and human-only matchmaking.

---

## 🌟 Core Features

- **Live 1-on-1 WebRTC Voice Signaling**: Peer-to-peer audio streaming using Google public STUN (`stun:stun.l.google.com:19302`) with deterministic initiator assignment for glare-free SDP negotiation.
- **Dynamic Matchmaking Engine**: FIFO pairing queue with topic matching, instant skip re-queueing, and graceful leave disconnects.
- **In-Session Messaging & GIF Validation**: Real-time text messaging with typing indicators (`is_typing`), and strict HTTPS validation for trusted GIF providers (Giphy, Tenor).
- **Local Content Moderation & Auto-Ban**:
  - *Stage 1*: Fast-path regex and advanced normalization (leetspeak, zero-width spaces, Cyrillic/Greek homoglyphs).
  - *Gatekeeper*: Instant WebSocket termination (`CloseCode 1008`) and IP ban persistence. Innocent partners are shielded and given one-click re-queue.

---

## 📡 Live Production Endpoints (Railway)

- **HTTP Base URL**: `https://stranger-talk-api-production.up.railway.app`
- **Health Check**: `https://stranger-talk-api-production.up.railway.app/health`
- **WebSocket Signaling**: `wss://stranger-talk-api-production.up.railway.app/ws`

---

## 🛠️ WebSocket Protocol Specification

All WebSocket frames are JSON strings conforming to the standardized envelope:

```json
{
  "type": "<message_type>",
  "session_id": "<uuid or null>",
  "payload": { ... },
  "timestamp": "ISO-8601 UTC string"
}
```

### 1. Client-to-Server Envelopes

Every session-scoped message (chat, GIF, typing, skip, leave, and WebRTC signaling) must include the exact session_id from the latest matched envelope. Messages with a missing or stale session ID are rejected or ignored; clients must not reuse an old ID after skipping or reconnecting.

| Type | Payload | Description |
| :--- | :--- | :--- |
| `join_queue` | `{"topics": ["tech", "music"]}` | Enter the matchmaking queue. |
| `leave_queue` | `{}` | Exit queue before being paired. |
| `skip` | `{}` | Skip current partner; instantly re-queues you and notifies partner. |
| `leave` | `{}` | Terminate session and return to idle state. |
| `chat` | `{"text": "Hello stranger!"}` | Send a text message to current partner. |
| `gif` | `{"url": "https://media.giphy.com/...", "width": 400, "height": 300}` | Send a validated GIF link. |
| `typing` | `{"is_typing": true}` | Dispatch typing status indicator. |
| `offer` | `{"sdp": "...", "type": "offer"}` | Relay WebRTC SDP Offer to peer. |
| `answer` | `{"sdp": "...", "type": "answer"}` | Relay WebRTC SDP Answer to initiator. |
| `candidate` | `{"candidate": "...", "sdpMid": "0", "sdpMLineIndex": 0}` | Relay Trickle ICE Candidate. |
| `ping` | `{}` | Application-level heartbeat ping. |

### 2. Server-to-Client Envelopes

| Type | Payload Example | Description |
| :--- | :--- | :--- |
| `queued` | `{"position": 1, "estimated_wait_seconds": 2}` | Queue status confirmation. |
| `matched` | `{"is_initiator": true, "peer_alias": "Stranger", "partner_id": "...", "ice_servers": [...]}` | Paired with another connected human. |
| `chat` | `{"text": "Hey!", "sender": "peer", "message_id": "..."}` | Inbound text message. |
| `gif` | `{"url": "...", "width": 400, "height": 300, "sender": "peer"}` | Inbound validated GIF. |
| `typing` | `{"is_typing": true}` | Partner typing status. |
| `offer` / `answer` | `{"sdp": "..."}` | WebRTC signaling data. |
| `ice_candidate` | `{"candidate": "...", "sdpMid": "0", "sdpMLineIndex": 0}` | Trickle ICE Candidate. |
| `peer_disconnected`| `{"reason": "disconnected", "can_requeue": true}` | Partner left or dropped. |
| `banned` | `{"reason": "Content policy violation", "ban_duration_seconds": 900}` | Notifies violator before socket termination. |
| `pong` | `{}` | Heartbeat response with timestamp. |
| `error` | `{"code": "INVALID_PAYLOAD", "message": "..."}` | Protocol or operational error. |

---

## 🧪 Running Automated Tests

The backend includes automated unit and WebSocket integration tests:

```bash
# Run complete test suite
py -3.14 -m pytest -v

# Run with concise summary
py -3.14 -m pytest -q
```

---

## Runtime Security Configuration

- ALLOWED_ORIGINS: optional JSON array of exact browser origins, for example ["https://your-frontend.example"]. When configured, the API applies this allowlist to credentialed CORS and browser WebSocket handshakes.
- WS_MAX_MESSAGE_BYTES: maximum inbound WebSocket frame size (default 262,144 bytes).
- WS_RATE_LIMIT_MESSAGES and WS_RATE_LIMIT_WINDOW_SECONDS: per-connection inbound frame budget (defaults to 60 messages per 10 seconds).

Deployment note: matchmaking queues, sessions, active connections, and IP bans are process-local in-memory state. Run exactly one application process/replica until shared Redis-backed state is implemented. Multiple workers or replicas maintain separate queues and cannot match users connected to different instances.

## 🚀 Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
