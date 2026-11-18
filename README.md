# Stranger Talk Backend (ChatFreely)

Real-time anonymous stranger chat and live WebRTC audio calling service inspired by OpenTalk. Built with FastAPI, WebSockets, STUN audio signaling, deterministic local moderation, and human-only matchmaking.

---

## 🌟 Core Features

- **Live 1-on-1 WebRTC Voice Signaling**: Peer-to-peer audio streaming using Google public STUN (`stun:stun.l.google.com:19302`) with deterministic initiator assignment for glare-free SDP negotiation.
- **Dynamic Matchmaking Engine**: FIFO pairing queue with topic matching, instant skip re-queueing, and graceful leave disconnects.
- **In-Session Messaging, Reports, Blocking & GIFs**: Real-time text messaging, safety reports stored in SQLite, connection-scoped block controls, and server-side Tenor GIF search. GIF URLs are still validated before relay.
- **Local Content Moderation & Enforcement**:
  - *Stage 1*: Fast-path regex and advanced normalization (leetspeak, zero-width spaces, Cyrillic/Greek homoglyphs).
  - Explicit direct threats and repeated blocked messages end the offending connection with a clear `banned` event. This connection-scoped action does not issue a new IP ban, avoiding collateral bans on shared networks.
  - Users can report or block the current stranger; reports are rate-limited, persisted, and do not automatically ban the reported person.

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
| `report` | `{"category":"harassment","details":"...","message_excerpt":"..." }` | Submit a report for the current peer. The server derives the reported participant from the active session; reporting also blocks that pair for the current live connections. |
| `block` | `{}` | End the current chat, block the peer for the lifetime of both live connections, and requeue you. |
| `ping` | `{}` | Application-level heartbeat ping. |

### 2. Server-to-Client Envelopes

| Type | Payload Example | Description |
| :--- | :--- | :--- |
| `queued` | `{"position": 1, "estimated_wait_seconds": null}` | Queue status confirmation; wait estimate is currently unavailable. |
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

## Safety Reports and Tenor GIF Search

- `REPORTS_DB_PATH`: SQLite report database path. For Railway, attach a persistent volume mounted at `/data` and set this to `/data/talkfreely_reports.sqlite3`. Without a persistent volume, reports can be lost when the container is replaced.
- `REPORT_RETENTION_DAYS`: report retention window (default 90 days; expired reports are pruned during startup and report access/submission).
- `REPORTS_ADMIN_TOKEN`: set a long random secret to enable `GET /admin/reports`; pass it in the `X-Admin-Token` header. Never put this token in the frontend.
- `TENOR_API_KEY`: an existing Tenor API key. The key is used only by the backend proxy at `GET /api/gifs/search?q=...`; the frontend never receives it. Tenor's developer documentation says new API clients have not been accepted since January 2026, so an existing key is required.
- `TENOR_CLIENT_KEY`: optional Tenor integration identifier (default `talkfreely_web`). GIF results use Tenor's `contentfilter=high` and are URL-validated again before being relayed.

Reports contain the selected reason, optional details, the optional message excerpt the reporter chooses to submit, the session ID, and temporary random connection IDs. Reports do not automatically ban the reported person. Block lists apply only while both anonymous WebSocket connections remain alive; a reconnect creates a new anonymous connection identity.

## Runtime Security Configuration

- ALLOWED_ORIGINS: optional JSON array of exact browser origins, for example ["https://your-frontend.example"]. When configured, the API applies this allowlist to credentialed CORS and browser WebSocket handshakes.
- WS_MAX_MESSAGE_BYTES: maximum inbound WebSocket frame size (default 262,144 bytes).
- WS_RATE_LIMIT_MESSAGES and WS_RATE_LIMIT_WINDOW_SECONDS: per-IP inbound frame budget shared across connections and reconnects in this process (defaults to 60 messages per 10 seconds).
- FORWARDED_ALLOW_IPS: trusted reverse-proxy IPs/CIDRs used by Uvicorn to resolve client IPs. Restrict this to trusted proxy ranges when directly exposing the app; the container default is `*` for managed proxy deployments.

Deployment note: matchmaking queues, sessions, active connections, and IP bans are process-local in-memory state. Run exactly one application process/replica until shared Redis-backed state is implemented. Multiple workers or replicas maintain separate queues and cannot match users connected to different instances.

## 🚀 Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
