"""Tier 3: Cross-Feature Combinations (Pairwise Interaction Flows).

Covers:
- Pairwise feature interactions across matchmaking, skip, leave, signaling,
  messaging, GIF exchange, AI companion fallback, and moderation bans.
- Exact tests specified in TEST_INFRA.md:
  1. test_t3_matchmaking_skip_and_chat
  2. test_t3_matchmaking_signaling_and_gif
  3. test_t3_ai_fallback_chat_and_skip_to_human
  4. test_t3_chat_moderation_ban_and_partner_requeue
  5. test_t3_gif_moderation_ban_and_ip_blacklist
  6. test_t3_signaling_sudden_disconnect_and_requeue
  7. test_t3_simultaneous_skip_cross_session
  8. test_t3_rapid_churn_skip_and_leave_cycle
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


async def _pair_two_clients(make_client):
    """Helper to connect and pair two clients in a 1-on-1 session."""
    client_a = await make_client()
    client_b = await make_client()

    await client_a.join_queue()
    await client_b.join_queue()

    msg_a = await client_a.recv_type("matched", timeout=5.0)
    msg_b = await client_b.recv_type("matched", timeout=5.0)

    assert msg_a["session_id"] is not None
    assert msg_a["session_id"] == msg_b["session_id"]
    return client_a, client_b


# ============================================================================
# Tier 3: Cross-Feature Tests
# ============================================================================

@pytest.mark.asyncio
async def test_t3_matchmaking_skip_and_chat(make_client):
    """Users match, exchange chat, user A skips, matches with user C, and exchanges chat.

    Verifies seamless chat relay before skip, clean skip transition with partner notification,
    re-queueing of skipping user, subsequent pairing with a third user, and isolated chat in the new session.
    """
    client_a, client_b = await _pair_two_clients(make_client)
    first_session_id = client_a.session_id

    # Session 1: Client A and Client B exchange chat messages
    msg_to_b = "Hello stranger B!"
    await client_a.send_chat(msg_to_b)
    received_by_b = await client_b.recv_type("chat", timeout=5.0)
    assert received_by_b["payload"]["text"] == msg_to_b
    assert received_by_b["payload"]["sender"] == "peer"

    msg_to_a = "Hello stranger A!"
    await client_b.send_chat(msg_to_a)
    received_by_a = await client_a.recv_type("chat", timeout=5.0)
    assert received_by_a["payload"]["text"] == msg_to_a
    assert received_by_a["payload"]["sender"] == "peer"

    # Client A skips the session
    await client_a.skip()

    # Client A must receive queued confirmation
    queued_a = await client_a.recv_type("queued", timeout=3.0)
    assert queued_a["type"] == "queued"

    # Client B must receive peer_disconnected notification with reason 'skipped'
    disc_b = await client_b.recv_type("peer_disconnected", timeout=3.0)
    assert disc_b["type"] == "peer_disconnected"
    assert disc_b.get("payload", {}).get("reason") == "skipped"
    assert disc_b.get("payload", {}).get("can_requeue") is True

    # Client C connects and joins matchmaking queue
    client_c = await make_client()
    await client_c.join_queue()

    # Client A and Client C must match in a fresh session
    msg_a2 = await client_a.recv_type("matched", timeout=5.0)
    msg_c = await client_c.recv_type("matched", timeout=5.0)

    assert msg_a2["session_id"] == msg_c["session_id"]
    second_session_id = msg_a2["session_id"]
    assert second_session_id != first_session_id

    # Session 2: Client A and Client C exchange chat messages
    msg_to_c = "Hello new partner C!"
    await client_a.send_chat(msg_to_c)
    received_by_c = await client_c.recv_type("chat", timeout=5.0)
    assert received_by_c["payload"]["text"] == msg_to_c
    assert received_by_c["payload"]["sender"] == "peer"

    msg_to_a2 = "Nice to meet you A!"
    await client_c.send_chat(msg_to_a2)
    received_by_a2 = await client_a.recv_type("chat", timeout=5.0)
    assert received_by_a2["payload"]["text"] == msg_to_a2
    assert received_by_a2["payload"]["sender"] == "peer"

    # Client B must NOT receive any messages sent in Session 2
    b_drained = await client_b.drain(timeout=0.3)
    assert not any(m.get("type") == "chat" for m in b_drained)


@pytest.mark.asyncio
async def test_t3_matchmaking_signaling_and_gif(make_client):
    """Users match, perform WebRTC offer/answer, and exchange GIF messages concurrently.

    Verifies full compatibility and non-interference between WebRTC signaling frames
    (SDP offer, answer, ICE candidates) and media envelopes (GIF URLs, metadata).
    """
    client_a, client_b = await _pair_two_clients(make_client)

    initiator = client_a if client_a.is_initiator else client_b
    receiver = client_b if initiator is client_a else client_a

    # 1. Initiator sends WebRTC SDP Offer
    offer_sdp = (
        "v=0\r\n"
        "o=- 551122 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await initiator.send_sdp_offer(offer_sdp)
    received_offer = await receiver.recv_type("offer", timeout=5.0)
    assert received_offer["payload"]["sdp"] == offer_sdp
    assert received_offer["payload"]["type"] == "offer"

    # 2. Receiver responds with WebRTC SDP Answer
    answer_sdp = (
        "v=0\r\n"
        "o=- 993344 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await receiver.send_sdp_answer(answer_sdp)
    received_answer = await initiator.recv_type("answer", timeout=5.0)
    assert received_answer["payload"]["sdp"] == answer_sdp
    assert received_answer["payload"]["type"] == "answer"

    # 3. Initiator sends ICE candidate and a GIF concurrently
    ice_init = "candidate:1001 1 udp 2122260223 192.168.1.10 50000 typ host"
    gif_giphy = "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif"
    await initiator.send_ice_candidate(candidate=ice_init, sdpMid="0", sdpMLineIndex=0)
    await initiator.send_gif(
        url=gif_giphy,
        title="Welcome Dance",
        width=480,
        height=360,
        preview_url="https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/preview.gif",
    )

    # Receiver collects both ICE candidate and GIF envelopes
    received_ice = await receiver.recv_type("ice_candidate", timeout=5.0)
    assert received_ice["payload"]["candidate"] == ice_init

    received_gif = await receiver.recv_type("gif", timeout=5.0)
    assert received_gif["payload"]["url"] == gif_giphy
    assert received_gif["payload"]["title"] == "Welcome Dance"
    assert received_gif["payload"]["sender"] == "peer"
    assert received_gif["payload"]["width"] == 480
    assert received_gif["payload"]["height"] == 360

    # 4. Receiver sends return ICE candidate and return Tenor GIF
    ice_recv = "candidate:2002 1 udp 1677729535 10.0.0.25 51234 typ srflx raddr 10.0.0.25 rport 51234"
    gif_tenor = "https://media.tenor.com/images/41a123bc45de/tenor.gif"
    await receiver.send_ice_candidate(candidate=ice_recv, sdpMid="0", sdpMLineIndex=0)
    await receiver.send_gif(url=gif_tenor, title="Thumbs Up", width=320, height=240)

    # Initiator collects return ICE candidate and GIF envelopes
    received_ice2 = await initiator.recv_type("ice_candidate", timeout=5.0)
    assert received_ice2["payload"]["candidate"] == ice_recv

    received_gif2 = await initiator.recv_type("gif", timeout=5.0)
    assert received_gif2["payload"]["url"] == gif_tenor
    assert received_gif2["payload"]["title"] == "Thumbs Up"
    assert received_gif2["payload"]["sender"] == "peer"


@pytest.mark.asyncio
async def test_t3_ai_fallback_chat_and_skip_to_human(make_client, fast_ai_timeout):
    """User waits for AI match, chats with AI, skips, and immediately matches with a newly connected human peer.

    Verifies queue timeout watchdog trigger, companion conversation responses, skip parity
    when matched with AI, and seamless re-entry leading to a human-to-human match.
    """
    client_a = await make_client()
    await client_a.join_queue()
    await client_a.recv_type("queued", timeout=3.0)

    # Wait for AI companion watchdog fallback trigger
    ai_match = await client_a.recv_type("matched", timeout=8.0)
    assert ai_match["type"] == "matched"
    assert ai_match.get("payload", {}).get("peer_is_ai") is True
    ai_session_id = ai_match["session_id"]
    assert ai_session_id is not None

    # Chat with AI companion
    await client_a.send_chat("Hello AI companion! How are you doing tonight?")
    ai_reply = await client_a.recv_type("chat", timeout=8.0)
    assert ai_reply["type"] == "chat"
    assert ai_reply["session_id"] == ai_session_id
    text = ai_reply.get("payload", {}).get("text", "")
    assert isinstance(text, str) and len(text.strip()) > 0
    assert ai_reply.get("payload", {}).get("sender") in ("peer", "Stranger", "ai")

    # User A skips the AI companion session
    await client_a.skip()
    queued_msg = await client_a.recv_type("queued", timeout=5.0)
    assert queued_msg["type"] == "queued"

    # Human user B connects and joins the queue
    client_b = await make_client()
    await client_b.join_queue()

    # User A and User B match into a human-to-human session
    human_match_a = await client_a.recv_type("matched", timeout=5.0)
    human_match_b = await client_b.recv_type("matched", timeout=5.0)

    assert human_match_a["session_id"] == human_match_b["session_id"]
    assert human_match_a["session_id"] != ai_session_id
    assert human_match_a.get("payload", {}).get("peer_is_ai") is not True
    assert human_match_b.get("payload", {}).get("peer_is_ai") is not True

    # Exchange messages in the human session
    await client_a.send_chat("Hello human friend!")
    b_received = await client_b.recv_type("chat", timeout=5.0)
    assert b_received["payload"]["text"] == "Hello human friend!"
    assert b_received["payload"]["sender"] == "peer"

    await client_b.send_chat("Great to meet a fellow human!")
    a_received = await client_a.recv_type("chat", timeout=5.0)
    assert a_received["payload"]["text"] == "Great to meet a fellow human!"
    assert a_received["payload"]["sender"] == "peer"


@pytest.mark.asyncio
async def test_t3_chat_moderation_ban_and_partner_requeue(make_client):
    """Users chat; violator sends toxic message and is banned; innocent partner automatically re-enters queue and matches with another user.

    Verifies policy violation detection, socket closure for offender, partner notification
    with reason 'banned', and smooth partner re-queueing to pair with another innocent peer.
    """
    client_violator, client_partner = await _pair_two_clients(make_client)

    # Initial friendly message
    await client_partner.send_chat("Hope you are having a nice day!")
    init_chat = await client_violator.recv_type("chat", timeout=5.0)
    assert init_chat["payload"]["text"] == "Hope you are having a nice day!"

    # Violator sends toxic message violating policies
    toxic_text = "kys kill yourself you worthless trash"
    await client_violator.send_chat(toxic_text)

    # Violator receives banned envelope and/or socket termination (code 1008)
    try:
        ban_msg = await client_violator.recv_type("banned", timeout=3.0)
        assert ban_msg["type"] == "banned"
    except Exception:
        pass

    # Ensure violator connection closes
    try:
        while True:
            await client_violator.recv_json(timeout=2.0)
    except (websockets.exceptions.ConnectionClosed, Exception):
        pass
    assert client_violator.close_code in (1008, 4403)

    # Innocent partner receives peer_disconnected notification with reason: "banned"
    disc_partner = await client_partner.recv_type("peer_disconnected", timeout=5.0)
    assert disc_partner["type"] == "peer_disconnected"
    assert disc_partner.get("payload", {}).get("reason") == "banned"
    assert disc_partner.get("payload", {}).get("can_requeue") is True

    # Innocent partner did NOT receive the toxic message
    partner_drained = await client_partner.drain(timeout=0.3)
    assert not any(m.get("type") == "chat" for m in partner_drained)

    # Innocent partner re-enters matchmaking queue
    try:
        from app.moderation.ban_manager import ban_manager
        ban_manager.lift_ban("127.0.0.1")
    except Exception:
        pass
    await client_partner.join_queue()

    # New client C enters queue
    client_c = await make_client()
    await client_c.join_queue()

    # Partner and Client C match cleanly
    match_partner = await client_partner.recv_type("matched", timeout=5.0)
    match_c = await client_c.recv_type("matched", timeout=5.0)
    assert match_partner["session_id"] == match_c["session_id"]

    # They can exchange messages in peace
    await client_partner.send_chat("Hello new friend!")
    c_chat = await client_c.recv_type("chat", timeout=5.0)
    assert c_chat["payload"]["text"] == "Hello new friend!"


@pytest.mark.asyncio
async def test_t3_gif_moderation_ban_and_ip_blacklist(make_client, server_ws_url):
    """Violator sends forbidden URL, gets banned, and subsequent connection attempt from same IP is blocked.

    Verifies media domain whitelist enforcement, IP ban caching, and handshake/pre-admission gatekeeper.
    """
    client_a, client_b = await _pair_two_clients(make_client)

    # Client A sends forbidden, non-allowlisted insecure HTTP GIF URL
    forbidden_url = "http://malicious-exploit-server.com/malware.gif"
    await client_a.send_gif(url=forbidden_url)

    # Client A receives banned event or socket terminated
    try:
        ban_msg = await client_a.recv_type("banned", timeout=3.0)
        assert ban_msg["type"] == "banned"
    except Exception:
        pass

    try:
        while True:
            await client_a.recv_json(timeout=2.0)
    except (websockets.exceptions.ConnectionClosed, Exception):
        pass
    assert client_a.close_code in (1008, 4403)

    # Partner B does NOT receive the malicious GIF
    partner_drained = await client_b.drain(timeout=0.5)
    assert not any(m.get("type") == "gif" for m in partner_drained)

    # Subsequent connection attempt from banned IP is rejected at handshake or immediately closed
    from tests.conftest import AsyncStrangerClient
    banned_client = AsyncStrangerClient(server_ws_url)
    try:
        await banned_client.connect(timeout=2.0)
        try:
            msg = await banned_client.recv_type("banned", timeout=2.0)
            assert msg["type"] == "banned"
        except Exception:
            await banned_client.drain(timeout=0.5)
            assert banned_client.close_code in (1008, 4403)
    except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
        # HTTP 403 Forbidden handshake rejection or immediate close
        pass
    finally:
        await banned_client.close()


@pytest.mark.asyncio
async def test_t3_signaling_sudden_disconnect_and_requeue(make_client):
    """Mid-signaling partner drops; remaining user receives disconnect notice, re-queues, and establishes new session.

    Verifies handling of abrupt socket termination during WebRTC negotiation,
    clean session reclamation, and re-enqueue capability for the survivor.
    """
    client_a, client_b = await _pair_two_clients(make_client)

    # Client A initiates signaling by sending SDP offer
    offer_sdp = (
        "v=0\r\n"
        "o=- 778899 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await client_a.send_sdp_offer(offer_sdp)
    await client_b.recv_type("offer", timeout=5.0)

    # Before answering, Client B abruptly drops connection
    await client_b.close(code=1000)

    # Client A receives peer_disconnected notification
    disc_a = await client_a.recv_type("peer_disconnected", timeout=5.0)
    assert disc_a["type"] == "peer_disconnected"
    assert disc_a.get("payload", {}).get("reason") in ("disconnected", "left", "abandoned")

    # Client A re-enters matchmaking queue
    await client_a.join_queue()
    msg_queued = await client_a.recv_type("queued", timeout=3.0)
    assert msg_queued["type"] == "queued"

    # Client C connects and joins queue
    client_c = await make_client()
    await client_c.join_queue()

    # Client A and Client C match in a new session
    match_a2 = await client_a.recv_type("matched", timeout=5.0)
    match_c = await client_c.recv_type("matched", timeout=5.0)
    assert match_a2["session_id"] == match_c["session_id"]

    # Signaling succeeds in the new session
    initiator = client_a if client_a.is_initiator else client_c
    receiver = client_c if initiator is client_a else client_a

    await initiator.send_sdp_offer(offer_sdp)
    new_offer = await receiver.recv_type("offer", timeout=5.0)
    assert new_offer["payload"]["sdp"] == offer_sdp

    answer_sdp = (
        "v=0\r\n"
        "o=- 889900 2 IN IP4 127.0.0.1\r\n"
        "s=-\r\n"
        "t=0 0\r\n"
        "m=audio 54321 RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    await receiver.send_sdp_answer(answer_sdp)
    new_answer = await initiator.recv_type("answer", timeout=5.0)
    assert new_answer["payload"]["sdp"] == answer_sdp


@pytest.mark.asyncio
async def test_t3_simultaneous_skip_cross_session(make_client):
    """Two pairs of users skip simultaneously; all 4 users re-queue and pair cleanly without cross-talk or race conditions.

    Verifies matchmaking mutex lock safety, simultaneous re-queueing of multiple pairs,
    and mutual session isolation when re-paired.
    """
    # Create two separate concurrent sessions (A+B and C+D)
    client_a = await make_client()
    client_b = await make_client()
    await client_a.join_queue()
    await client_b.join_queue()
    m_a = await client_a.recv_type("matched", timeout=5.0)
    m_b = await client_b.recv_type("matched", timeout=5.0)
    session_1 = m_a["session_id"]

    client_c = await make_client()
    client_d = await make_client()
    await client_c.join_queue()
    await client_d.join_queue()
    m_c = await client_c.recv_type("matched", timeout=5.0)
    m_d = await client_d.recv_type("matched", timeout=5.0)
    session_2 = m_c["session_id"]
    assert session_1 != session_2

    # Clients A and C skip concurrently
    await asyncio.gather(client_a.skip(), client_c.skip())

    # A and C receive queued
    await asyncio.gather(
        client_a.recv_type("queued", timeout=3.0),
        client_c.recv_type("queued", timeout=3.0),
    )

    # B and D receive peer_disconnected
    disc_b, disc_d = await asyncio.gather(
        client_b.recv_type("peer_disconnected", timeout=3.0),
        client_d.recv_type("peer_disconnected", timeout=3.0),
    )
    assert disc_b.get("payload", {}).get("reason") == "skipped"
    assert disc_d.get("payload", {}).get("reason") == "skipped"

    # B and D re-enter queue concurrently
    await asyncio.gather(client_b.join_queue(), client_d.join_queue())

    # All 4 clients receive matched events into two new sessions
    new_m_a, new_m_b, new_m_c, new_m_d = await asyncio.gather(
        client_a.recv_type("matched", timeout=5.0),
        client_b.recv_type("matched", timeout=5.0),
        client_c.recv_type("matched", timeout=5.0),
        client_d.recv_type("matched", timeout=5.0),
    )

    all_new_sessions = {
        new_m_a["session_id"],
        new_m_b["session_id"],
        new_m_c["session_id"],
        new_m_d["session_id"],
    }
    # Must have 2 distinct sessions, neither of which is the original session_1 or session_2
    assert len(all_new_sessions) == 2
    assert session_1 not in all_new_sessions
    assert session_2 not in all_new_sessions

    # Verify session isolation: send chat in client_a's new session
    test_msg = "Isolation check message from client A"
    await client_a.send_chat(test_msg)

    # Determine who A's partner is
    partner_of_a = None
    for cand, cand_match in [(client_b, new_m_b), (client_c, new_m_c), (client_d, new_m_d)]:
        if cand_match["session_id"] == new_m_a["session_id"]:
            partner_of_a = cand
            break

    assert partner_of_a is not None
    rec_partner = await partner_of_a.recv_type("chat", timeout=5.0)
    assert rec_partner["payload"]["text"] == test_msg

    # Non-partner clients must receive nothing
    for non_partner in [client_b, client_c, client_d]:
        if non_partner is not partner_of_a:
            drained = await non_partner.drain(timeout=0.3)
            assert not any(m.get("type") == "chat" for m in drained)


@pytest.mark.asyncio
async def test_t3_rapid_churn_skip_and_leave_cycle(make_client):
    """Users rapidly cycle through skip, leave, join_queue, and chat in alternating sequences.

    Verifies robustness under rapid lifecycle churn, absence of memory leaks,
    correct transitions between idle, queued, and matched states, and clean connection shutdowns.
    """
    client_a = await make_client()
    client_b = await make_client()

    # Cycle 1: A and B match, chat, A skips
    await client_a.join_queue()
    await client_b.join_queue()
    await client_a.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_a.send_chat("Cycle 1 message")
    msg_b = await client_b.recv_type("chat", timeout=5.0)
    assert msg_b["payload"]["text"] == "Cycle 1 message"

    await client_a.skip()
    await client_a.recv_type("queued", timeout=3.0)
    await client_b.recv_type("peer_disconnected", timeout=3.0)

    # B leaves cleanly to idle
    await client_b.leave()
    await client_b.drain(timeout=0.3)

    # Cycle 2: Client C connects, joins queue, matches with A
    client_c = await make_client()
    await client_c.join_queue()
    await client_a.recv_type("matched", timeout=5.0)
    await client_c.recv_type("matched", timeout=5.0)

    await client_c.send_chat("Cycle 2 message from C")
    msg_a2 = await client_a.recv_type("chat", timeout=5.0)
    assert msg_a2["payload"]["text"] == "Cycle 2 message from C"

    # C skips, A notified
    await client_c.skip()
    await client_c.recv_type("queued", timeout=3.0)
    await client_a.recv_type("peer_disconnected", timeout=3.0)

    # Cycle 3: B re-enters queue, matches with C
    await client_b.join_queue()
    await client_c.recv_type("matched", timeout=5.0)
    await client_b.recv_type("matched", timeout=5.0)

    await client_b.send_chat("Cycle 3 message from B")
    msg_c2 = await client_c.recv_type("chat", timeout=5.0)
    assert msg_c2["payload"]["text"] == "Cycle 3 message from B"

    # Final cleanup: all clients leave cleanly
    await client_a.leave()
    await client_b.leave()
    await client_c.leave()

    # Verify all sockets close without error
    await client_a.close()
    await client_b.close()
    await client_c.close()
