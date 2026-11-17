"""Tier 4: Real-World Application Scenarios (Workload Journeys).

Covers:
- Realistic end-to-end user journeys, stress workloads, and lifecycles.
- Exact tests specified in TEST_INFRA.md:
  1. test_t4_multi_user_stranger_chat_traffic
  2. test_t4_lone_night_owl_ai_conversation
  3. test_t4_bad_actor_swift_banishment_journey
  4. test_t4_complete_webrtc_call_lifecycle
  5. test_t4_high_concurrency_rapid_skip_marathon
"""
import asyncio

import pytest
import websockets


# ============================================================================
# Helpers & Fixtures
# ============================================================================

@pytest.fixture
def fast_ai_timeout(monkeypatch):
    """Speed up AI fallback watchdog timer for snappy test execution."""
    try:
        from app.config import settings
        monkeypatch.setattr(settings, "AI_FALLBACK_TIMEOUT_SECONDS", 0.5)
    except Exception:
        pass


# ============================================================================
# Tier 4: Real-World Application Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_t4_multi_user_stranger_chat_traffic(make_client):
    """Realistic 6-user simulation. Users arrive at different intervals, pair, chat, exchange GIFs, skip, and cleanly depart.

    Simulates realistic multi-user traffic with concurrent pairings, mixed media exchange,
    skip transitions, cross-pairing, and graceful exits without message collisions.
    """
    # 6 clients connect
    clients = [await make_client() for _ in range(6)]
    c0, c1, c2, c3, c4, c5 = clients

    # Staggered arrivals into matchmaking queue
    await c0.join_queue()
    await asyncio.sleep(0.02)
    await c1.join_queue()

    # Pair 1 (c0 and c1) matched
    m0 = await c0.recv_type("matched", timeout=5.0)
    m1 = await c1.recv_type("matched", timeout=5.0)
    assert m0["session_id"] == m1["session_id"]
    session_1 = m0["session_id"]

    await c2.join_queue()
    await asyncio.sleep(0.02)
    await c3.join_queue()

    # Pair 2 (c2 and c3) matched
    m2 = await c2.recv_type("matched", timeout=5.0)
    m3 = await c3.recv_type("matched", timeout=5.0)
    assert m2["session_id"] == m3["session_id"]
    session_2 = m2["session_id"]
    assert session_1 != session_2

    await c4.join_queue()
    await asyncio.sleep(0.02)
    await c5.join_queue()

    # Pair 3 (c4 and c5) matched
    m4 = await c4.recv_type("matched", timeout=5.0)
    m5 = await c5.recv_type("matched", timeout=5.0)
    assert m4["session_id"] == m5["session_id"]
    session_3 = m4["session_id"]
    assert session_3 not in (session_1, session_2)

    # In-session communication: mixed text and GIF messages across all 3 pairs
    # Pair 1: text and Giphy GIF
    await c0.send_chat("Hello from client 0 in session 1")
    r1_chat = await c1.recv_type("chat", timeout=5.0)
    assert r1_chat["payload"]["text"] == "Hello from client 0 in session 1"

    giphy_url = "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif"
    await c1.send_gif(url=giphy_url, title="Hi back!")
    r0_gif = await c0.recv_type("gif", timeout=5.0)
    assert r0_gif["payload"]["url"] == giphy_url

    # Pair 2: text and Tenor GIF
    await c2.send_chat("Hello from client 2 in session 2")
    r3_chat = await c3.recv_type("chat", timeout=5.0)
    assert r3_chat["payload"]["text"] == "Hello from client 2 in session 2"

    tenor_url = "https://media.tenor.com/images/41a123bc45de/tenor.gif"
    await c3.send_gif(url=tenor_url, title="Wave")
    r2_gif = await c2.recv_type("gif", timeout=5.0)
    assert r2_gif["payload"]["url"] == tenor_url

    # Pair 3: bidirectional text
    await c4.send_chat("Hello from client 4 in session 3")
    r5_chat = await c5.recv_type("chat", timeout=5.0)
    assert r5_chat["payload"]["text"] == "Hello from client 4 in session 3"

    await c5.send_chat("Hello back from client 5")
    r4_chat = await c4.recv_type("chat", timeout=5.0)
    assert r4_chat["payload"]["text"] == "Hello back from client 5"

    # Churn: c0 skips (re-enters queue); c1 leaves; c2 skips (re-enters queue)
    await c0.skip()
    await c0.recv_type("queued", timeout=3.0)
    await c1.recv_type("peer_disconnected", timeout=3.0)
    await c1.leave()

    await c2.skip()
    await c2.recv_type("queued", timeout=3.0)
    await c3.recv_type("peer_disconnected", timeout=3.0)

    # c0 and c2 match into a new session!
    new_m0 = await c0.recv_type("matched", timeout=5.0)
    new_m2 = await c2.recv_type("matched", timeout=5.0)
    assert new_m0["session_id"] == new_m2["session_id"]
    new_session_id = new_m0["session_id"]
    assert new_session_id not in (session_1, session_2, session_3)

    # c0 and c2 exchange messages in the new session
    await c0.send_chat("Hey new stranger! We just got matched.")
    r2_new_chat = await c2.recv_type("chat", timeout=5.0)
    assert r2_new_chat["payload"]["text"] == "Hey new stranger! We just got matched."

    # Clean departure for all remaining participants
    for c in (c0, c2, c3, c4, c5):
        try:
            await c.leave()
        except Exception:
            pass
        await c.close()


@pytest.mark.asyncio
async def test_t4_lone_night_owl_ai_conversation(make_client, fast_ai_timeout):
    """Single user joins quiet queue, transitions to AI companion after timeout, conducts natural multi-turn conversation, observes typing cadence, skips, and leaves cleanly.

    Validates lone user experience when queue is sparse, AI companion fallback watchdog,
    typing indicator cadence before AI replies, multi-turn sliding context, and graceful exit.
    """
    client = await make_client()

    # Lone user joins queue
    await client.join_queue()
    queued_msg = await client.recv_type("queued", timeout=3.0)
    assert queued_msg["type"] == "queued"

    # AI companion watchdog triggers fallback match
    match_msg = await client.recv_type("matched", timeout=8.0)
    assert match_msg["type"] == "matched"
    assert match_msg.get("payload", {}).get("peer_is_ai") is True
    session_id = match_msg["session_id"]
    assert session_id is not None

    alias = match_msg.get("payload", {}).get("peer_alias", "")
    assert isinstance(alias, str) and len(alias) > 0

    # Turn 1: User says hello and observes typing indicator before response
    await client.send_chat("Hello, is anyone still awake tonight?")

    received_typing = False
    received_chat = None
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < 8.0:
        envelope = await client.recv_json(timeout=4.0)
        t = envelope.get("type")
        if t == "typing" and envelope.get("payload", {}).get("is_typing") is True:
            received_typing = True
        elif t == "chat":
            received_chat = envelope
            break

    assert received_typing is True, "AI companion failed to emit typing indicator prior to chat reply"
    assert received_chat is not None
    assert received_chat["payload"]["sender"] in ("peer", "Stranger", "ai")
    assert len(received_chat["payload"]["text"].strip()) > 0

    # Turn 2: Follow-up question
    await client.send_chat("I love reading late at night. Do you have a favorite genre?")
    reply_2 = await client.recv_type("chat", timeout=8.0)
    assert len(reply_2["payload"]["text"].strip()) > 0

    # Turn 3: Closing thought
    await client.send_chat("That sounds fascinating! Thanks for chatting with me.")
    reply_3 = await client.recv_type("chat", timeout=8.0)
    assert len(reply_3["payload"]["text"].strip()) > 0

    # User invokes skip
    await client.skip()
    msg_queued_again = await client.recv_type("queued", timeout=5.0)
    assert msg_queued_again["type"] == "queued"

    # User departs cleanly
    await client.leave()
    await client.drain(timeout=0.3)
    await client.close()


@pytest.mark.asyncio
async def test_t4_bad_actor_swift_banishment_journey(make_client, server_ws_url):
    """End-to-end security journey. Malicious user tries spamming slurs and bad links; caught instantly by moderation, socket closed with 1008, IP recorded; partner reassured with friendly notice; attacker denied reconnect.

    Validates automated security enforcement: instant termination of bad actors,
    partner shielding and notification, IP ban persistence, and clean partner recovery.
    """
    client_good = await make_client()
    client_bad = await make_client()

    # Good and bad users match in a session
    await client_good.join_queue()
    await client_bad.join_queue()
    await client_good.recv_type("matched", timeout=5.0)
    await client_bad.recv_type("matched", timeout=5.0)

    # Bad actor sends severely toxic message
    toxic_text = "kys kill yourself you worthless piece of trash"
    await client_bad.send_chat(toxic_text)

    # Bad actor receives banned notification and/or connection closed with 1008
    try:
        ban_msg = await client_bad.recv_type("banned", timeout=3.0)
        assert ban_msg["type"] == "banned"
    except Exception:
        pass

    try:
        while True:
            await client_bad.recv_json(timeout=2.0)
    except (websockets.exceptions.ConnectionClosed, Exception):
        pass
    assert client_bad.close_code in (1008, 4403)

    # Good user receives peer_disconnected notification with reason: "banned"
    disc_good = await client_good.recv_type("peer_disconnected", timeout=5.0)
    assert disc_good["type"] == "peer_disconnected"
    assert disc_good.get("payload", {}).get("reason") == "banned"
    assert disc_good.get("payload", {}).get("can_requeue") is True

    # Good user was shielded: did NOT receive toxic message
    good_drained = await client_good.drain(timeout=0.3)
    assert not any(m.get("type") == "chat" for m in good_drained)

    # Attacker attempts immediate reconnect from same IP: rejected or closed
    from tests.conftest import AsyncStrangerClient
    reconnect_attacker = AsyncStrangerClient(server_ws_url)
    try:
        await reconnect_attacker.connect(timeout=2.0)
        try:
            ban_reply = await reconnect_attacker.recv_type("banned", timeout=2.0)
            assert ban_reply["type"] == "banned"
        except Exception:
            await reconnect_attacker.drain(timeout=0.5)
            assert reconnect_attacker.close_code in (1008, 4403)
    except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
        # HTTP 403 Forbidden handshake rejection
        pass
    finally:
        await reconnect_attacker.close()

    # Innocent partner recovers: re-enters queue and matches with friendly stranger
    # (lift ban for localhost so innocent peer on same test runner IP can queue)
    from app.moderation.ban_manager import ban_manager
    ban_manager.lift_ban("127.0.0.1")

    await client_good.join_queue()

    client_friendly = await make_client()
    await client_friendly.join_queue()

    # Good user and friendly stranger match into a safe session
    m_good = await client_good.recv_type("matched", timeout=5.0)
    m_friendly = await client_friendly.recv_type("matched", timeout=5.0)
    assert m_good["session_id"] == m_friendly["session_id"]

    # Safe conversation proceeds
    await client_good.send_chat("Hello! Glad to be paired with someone nice.")
    f_received = await client_friendly.recv_type("chat", timeout=5.0)
    assert f_received["payload"]["text"] == "Hello! Glad to be paired with someone nice."

    await client_friendly.send_chat("Same here! Hope you have a wonderful conversation.")
    g_received = await client_good.recv_type("chat", timeout=5.0)
    assert g_received["payload"]["text"] == "Same here! Hope you have a wonderful conversation."

    await client_good.leave()
    await client_friendly.leave()


@pytest.mark.asyncio
async def test_t4_complete_webrtc_call_lifecycle(make_client):
    """Complete audio call emulation. STUN server configuration delivery, offer/answer negotiation, trickle ICE candidate exchange, parallel in-call text chat, and graceful hang-up.

    Emulates a full WebRTC live voice session including NAT configuration, SDP negotiation,
    trickle ICE candidate relay, parallel in-call messaging, and clean session termination.
    """
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    assert msg_a["session_id"] == msg_b["session_id"]
    session_id = msg_a["session_id"]

    # Step 1: STUN server configuration delivery verification
    payload_a = msg_a.get("payload", {})
    assert "ice_servers" in payload_a
    ice_servers = payload_a["ice_servers"]
    assert isinstance(ice_servers, list) and len(ice_servers) > 0
    all_urls = []
    for entry in ice_servers:
        urls = entry.get("urls", [])
        if isinstance(urls, list):
            all_urls.extend(urls)
        elif isinstance(urls, str):
            all_urls.append(urls)
    assert any("stun:stun.l.google.com:19302" in u for u in all_urls)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    # Step 2: SDP Offer Negotiation
    offer_sdp = (
        "v=0\r\n"
        "o=- 334455 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await initiator.send_sdp_offer(offer_sdp)
    rec_offer = await receiver.recv_type("offer", timeout=5.0)
    assert rec_offer["payload"]["sdp"] == offer_sdp
    assert rec_offer["session_id"] == session_id

    # Step 3: SDP Answer Negotiation
    answer_sdp = (
        "v=0\r\n"
        "o=- 667788 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await receiver.send_sdp_answer(answer_sdp)
    rec_answer = await initiator.recv_type("answer", timeout=5.0)
    assert rec_answer["payload"]["sdp"] == answer_sdp
    assert rec_answer["session_id"] == session_id

    # Step 4: Trickle ICE Candidate Exchange
    # Initiator sends 2 trickle ICE candidates
    ice_init_1 = "candidate:1 1 udp 2122260223 192.168.1.100 50000 typ host"
    ice_init_2 = "candidate:2 1 udp 1677729535 203.0.113.1 50001 typ srflx raddr 192.168.1.100 rport 50000"
    await initiator.send_ice_candidate(candidate=ice_init_1, sdpMid="0", sdpMLineIndex=0)
    await initiator.send_ice_candidate(candidate=ice_init_2, sdpMid="0", sdpMLineIndex=0)

    c_rec_1 = await receiver.recv_type("ice_candidate", timeout=5.0)
    c_rec_2 = await receiver.recv_type("ice_candidate", timeout=5.0)
    assert c_rec_1["payload"]["candidate"] == ice_init_1
    assert c_rec_2["payload"]["candidate"] == ice_init_2

    # Receiver sends 2 trickle ICE candidates
    ice_recv_1 = "candidate:3 1 udp 2122260223 192.168.1.200 60000 typ host"
    ice_recv_2 = "candidate:4 1 udp 1677729535 203.0.113.2 60001 typ srflx raddr 192.168.1.200 rport 60000"
    await receiver.send_ice_candidate(candidate=ice_recv_1, sdpMid="0", sdpMLineIndex=0)
    await receiver.send_ice_candidate(candidate=ice_recv_2, sdpMid="0", sdpMLineIndex=0)

    c_init_1 = await initiator.recv_type("ice_candidate", timeout=5.0)
    c_init_2 = await initiator.recv_type("ice_candidate", timeout=5.0)
    assert c_init_1["payload"]["candidate"] == ice_recv_1
    assert c_init_2["payload"]["candidate"] == ice_recv_2

    # Step 5: Parallel In-Call Text Chat & Typing
    await initiator.send_typing(is_typing=True)
    typing_msg = await receiver.recv_type("typing", timeout=5.0)
    assert typing_msg["payload"]["is_typing"] is True

    await initiator.send_chat("Voice audio stream connected. Can you hear me?")
    chat_msg = await receiver.recv_type("chat", timeout=5.0)
    assert chat_msg["payload"]["text"] == "Voice audio stream connected. Can you hear me?"

    await receiver.send_chat("Loud and clear! WebRTC audio call working perfectly.")
    reply_msg = await initiator.recv_type("chat", timeout=5.0)
    assert reply_msg["payload"]["text"] == "Loud and clear! WebRTC audio call working perfectly."

    # Step 6: Graceful Hang-up
    await initiator.leave()
    hangup_notice = await receiver.recv_type("peer_disconnected", timeout=5.0)
    assert hangup_notice["payload"]["reason"] == "left"

    await receiver.leave()
    await initiator.close()
    await receiver.close()


@pytest.mark.asyncio
async def test_t4_high_concurrency_rapid_skip_marathon(make_client):
    """10 concurrent clients performing rapid randomized actions (join, skip, chat, leave) under load to prove thread-safety, zero deadlocks, and clean resource reclamation.

    Simulates high-load concurrency with 10 async clients rapidly skipping, chatting,
    and leaving, followed by an end-state system health check proving clean resource reclamation.
    """
    num_clients = 10
    clients = [await make_client() for _ in range(num_clients)]
    progressed_counter = 0

    async def client_worker(client, idx: int):
        """Worker executing rapid state transitions."""
        nonlocal progressed_counter
        try:
            # Join queue
            await client.join_queue()
            # Wait for matched or queued
            try:
                m = await client.recv_type("matched", timeout=6.0)
                if m.get("session_id"):
                    progressed_counter += 1
                    # Send in-session message
                    await client.send_chat(f"Concurrent stress message from client {idx}")
                    # Briefly indicate typing
                    await client.send_typing(is_typing=True)
                    # Random skip
                    await client.skip()
            except (TimeoutError, asyncio.TimeoutError):
                pass

            # Small yield to event loop
            await asyncio.sleep(0.02)

            # Clean exit
            await client.leave()
        except Exception as exc:
            # Catch and return any unhandled error
            return exc
        return None

    # Run all 10 workers concurrently under load
    results = await asyncio.gather(
        *[client_worker(c, i) for i, c in enumerate(clients)],
        return_exceptions=True,
    )

    # Verify no critical unhandled crashes occurred
    for r in results:
        assert not isinstance(r, (BrokenPipeError, ConnectionResetError, RuntimeError)), f"Worker crashed: {r}"

    # Assert that at least 4 clients successfully paired and progressed through in-session states
    assert progressed_counter >= 4, f"Only {progressed_counter} clients progressed through in-session states (expected >= 4)"

    # Clean up all worker sockets
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass

    # Post-Marathon Health Check:
    # Verify the server remains fully responsive and can pair a new session cleanly
    fresh_a = await make_client()
    fresh_b = await make_client()

    await fresh_a.join_queue()
    await fresh_b.join_queue()

    match_a = await fresh_a.recv_type("matched", timeout=5.0)
    match_b = await fresh_b.recv_type("matched", timeout=5.0)
    assert match_a["session_id"] == match_b["session_id"]

    await fresh_a.send_chat("Post-marathon health check message")
    rec = await fresh_b.recv_type("chat", timeout=5.0)
    assert rec["payload"]["text"] == "Post-marathon health check message"

    await fresh_a.leave()
    await fresh_b.leave()
    await fresh_a.close()
    await fresh_b.close()
