# Stranger Talk Backend (ChatFreely)

Real-time anonymous stranger chat and live WebRTC audio calling service inspired by OpenTalk. Built with FastAPI, WebSockets, STUN audio signaling, two-stage AI moderation, auto-ban gatekeeping, and human-like AI companion fallbacks.

---

## 🌟 Core Features

- **Live 1-on-1 WebRTC Voice Signaling**: Peer-to-peer audio streaming using Google public STUN (`stun:stun.l.google.com:19302`) with deterministic initiator assignment for glare-free SDP negotiation.
- **Dynamic Matchmaking Engine**: FIFO pairing queue with topic matching, instant skip re-queueing, and graceful leave disconnects.
- **In-Session Messaging & GIF Validation**: Real-time text messaging with typing indicators (`is_typing`), and strict HTTPS validation for trusted GIF providers (Giphy, Tenor).
- **Two-Stage Content Moderation & Auto-Ban**:
  - *Stage 1*: Fast-path regex and advanced normalization (leetspeak, zero-width spaces, Cyrillic/Greek homoglyphs).
  - *Stage 2*: AI toxicity classifier.
  - *Gatekeeper*: Instant WebSocket termination (`CloseCode 1008`) and IP ban persistence. Innocent partners are shielded and given one-click re-queue.
- **Human-like AI Companion Fallback**: When matchmaking queues are empty, queued users automatically pair with an AI companion after a configurable timeout (default 5s) featuring natural typing delays, realistic typing cadence, and conversational memory.

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
| `matched` | `{"is_initiator": true, "peer_is_ai": false, "peer_alias": "Stranger", "ice_servers": [...]}` | Paired with a peer or AI companion. |
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

The backend includes a 292-test test harness (199 unit tests + 93 E2E integration scenarios):

```bash
# Run complete test suite
py -3.14 -m pytest -v

# Run with concise summary
py -3.14 -m pytest -q
```

---

## 🚀 Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
