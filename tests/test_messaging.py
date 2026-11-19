"""In-Session Messaging and GIF Exchange Tests (Tier 1 & Tier 2).

Covers:
- F5: In-Session Text Chat Relay & Typing Indicators
- F6: GIF Exchange, Allowlist & Metadata Validation
"""
import asyncio
from datetime import datetime, timezone
import pytest


# ============================================================================
# Helpers
# ============================================================================

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
# Tier 1: Feature Coverage (Happy Path)
# ============================================================================

# --- F5: In-Session Text Chat ---

@pytest.mark.asyncio
async def test_t1_f5_text_chat_delivered(make_client):
    """Client A sends text chat; Client B receives chat envelope with sender: 'peer'."""
    client_a, client_b = await _pair_two_clients(make_client)

    chat_text = "Hello stranger, nice to meet you!"
    await client_a.send_chat(chat_text)

    msg = await client_b.recv_type("chat", timeout=5.0)
    assert msg["type"] == "chat"
    assert msg["session_id"] == client_a.session_id

    payload = msg.get("payload", {})
    assert payload.get("text") == chat_text
    assert payload.get("sender") == "peer"


@pytest.mark.asyncio
async def test_t1_f5_message_id_generated(make_client):
    """Relayed chat message contains a unique server-generated message_id."""
    client_a, client_b = await _pair_two_clients(make_client)

    await client_a.send_chat("Testing unique message id assignment")

    msg = await client_b.recv_type("chat", timeout=5.0)
    payload = msg.get("payload", {})
    assert "message_id" in payload
    assert isinstance(payload["message_id"], str)
    assert len(payload["message_id"]) > 0


@pytest.mark.asyncio
async def test_t1_f5_bidirectional_chat(make_client):
    """Both clients in a session can exchange messages back and forth."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Client A to Client B
    await client_a.send_chat("Message from client A")
    msg_b = await client_b.recv_type("chat", timeout=5.0)
    assert msg_b["payload"]["text"] == "Message from client A"
    assert msg_b["payload"]["sender"] == "peer"

    # Client B to Client A
    await client_b.send_chat("Reply from client B")
    msg_a = await client_a.recv_type("chat", timeout=5.0)
    assert msg_a["payload"]["text"] == "Reply from client B"
    assert msg_a["payload"]["sender"] == "peer"


@pytest.mark.asyncio
async def test_t1_f5_typing_indicator_on(make_client):
    """Client sends typing indicator true; partner receives typing: true event."""
    client_a, client_b = await _pair_two_clients(make_client)

    await client_a.send_typing(is_typing=True)

    msg = await client_b.recv_type("typing", timeout=5.0)
    assert msg["type"] == "typing"
    assert msg["session_id"] == client_a.session_id
    payload = msg.get("payload", {})
    assert payload.get("is_typing") is True


@pytest.mark.asyncio
async def test_t1_f5_typing_indicator_off(make_client):
    """Client sends typing indicator false; partner receives typing: false event."""
    client_a, client_b = await _pair_two_clients(make_client)

    await client_a.send_typing(is_typing=False)

    msg = await client_b.recv_type("typing", timeout=5.0)
    assert msg["type"] == "typing"
    assert msg["session_id"] == client_a.session_id
    payload = msg.get("payload", {})
    assert payload.get("is_typing") is False


# --- F6: GIF Exchange ---

@pytest.mark.asyncio
async def test_t1_f6_valid_giphy_url_relayed(make_client):
    """Client sends valid Giphy HTTPS URL; partner receives gif message with matching URL."""
    client_a, client_b = await _pair_two_clients(make_client)

    giphy_url = "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif"
    await client_a.send_gif(url=giphy_url)

    msg = await client_b.recv_type("gif", timeout=5.0)
    assert msg["type"] == "gif"
    assert msg["session_id"] == client_a.session_id

    payload = msg.get("payload", {})
    assert payload.get("url") == giphy_url
    assert payload.get("sender") == "peer"


@pytest.mark.asyncio
async def test_t1_f6_gifsnap_provider_and_id_are_preserved(make_client):
    """GifSnap provider metadata is retained in the relayed GIF envelope."""
    client_a, client_b = await _pair_two_clients(make_client)

    media_url = "https://media.giphy.com/media/test/giphy.gif"
    await client_a.send_gif(url=media_url, provider="gifsnap", id="gif_123")

    msg = await client_b.recv_type("gif", timeout=5.0)
    payload = msg.get("payload", {})
    assert payload.get("url") == media_url
    assert payload.get("provider") == "gifsnap"
    assert payload.get("id") == "gif_123"


@pytest.mark.asyncio
async def test_t1_f6_valid_tenor_url_relayed(make_client):
    """Client sends valid Tenor HTTPS URL; partner receives gif message with matching URL."""
    client_a, client_b = await _pair_two_clients(make_client)

    tenor_url = "https://media.tenor.com/images/41a123bc45de/tenor.gif"
    await client_a.send_gif(url=tenor_url)

    msg = await client_b.recv_type("gif", timeout=5.0)
    assert msg["type"] == "gif"
    assert msg["session_id"] == client_a.session_id

    payload = msg.get("payload", {})
    assert payload.get("url") == tenor_url
    assert payload.get("sender") == "peer"


@pytest.mark.asyncio
async def test_t1_f6_gif_metadata_preserved(make_client):
    """Width, height, preview_url, and title metadata are preserved when relayed."""
    client_a, client_b = await _pair_two_clients(make_client)

    gif_url = "https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif"
    preview_url = "https://media.giphy.com/media/3o7aD2saalBwwftBIY/preview.gif"
    title = "Dancing Panda"
    width = 480
    height = 360

    await client_a.send_gif(
        url=gif_url,
        preview_url=preview_url,
        title=title,
        width=width,
        height=height,
    )

    msg = await client_b.recv_type("gif", timeout=5.0)
    payload = msg.get("payload", {})
    assert payload.get("url") == gif_url
    assert payload.get("preview_url") == preview_url
    assert payload.get("title") == title
    assert payload.get("width") == width
    assert payload.get("height") == height


@pytest.mark.asyncio
async def test_t1_f6_gif_message_id_assigned(make_client):
    """Relayed GIF message contains a unique server-generated message_id."""
    client_a, client_b = await _pair_two_clients(make_client)

    await client_a.send_gif("https://media.giphy.com/media/test/giphy.gif")

    msg = await client_b.recv_type("gif", timeout=5.0)
    payload = msg.get("payload", {})
    assert "message_id" in payload
    assert isinstance(payload["message_id"], str)
    assert len(payload["message_id"]) > 0


@pytest.mark.asyncio
async def test_t1_f6_alternate_chat_and_gif(make_client):
    """Peers can smoothly alternate between sending text messages and GIFs."""
    client_a, client_b = await _pair_two_clients(make_client)

    # 1. A sends text
    await client_a.send_chat("Here is a funny reaction:")
    msg1 = await client_b.recv_type("chat", timeout=5.0)
    assert msg1["payload"]["text"] == "Here is a funny reaction:"

    # 2. A sends GIF
    gif1 = "https://media.giphy.com/media/reaction1/giphy.gif"
    await client_a.send_gif(gif1)
    msg2 = await client_b.recv_type("gif", timeout=5.0)
    assert msg2["payload"]["url"] == gif1

    # 3. B replies with text
    await client_b.send_chat("Haha that is great!")
    msg3 = await client_a.recv_type("chat", timeout=5.0)
    assert msg3["payload"]["text"] == "Haha that is great!"

    # 4. B replies with GIF
    gif2 = "https://media.tenor.com/laughing/tenor.gif"
    await client_b.send_gif(gif2)
    msg4 = await client_a.recv_type("gif", timeout=5.0)
    assert msg4["payload"]["url"] == gif2


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

# --- F5 Boundaries ---

@pytest.mark.asyncio
async def test_t2_f5_empty_chat_text_rejected(make_client):
    """Empty string chat message is rejected with validation error and not relayed."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Send empty text payload
    await client_a.send_envelope("chat", payload={"text": ""})

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_PAYLOAD", "VALIDATION_ERROR", "EMPTY_MESSAGE")

    # Client B should receive nothing
    drained = await client_b.drain(timeout=0.3)
    assert not any(m.get("type") == "chat" for m in drained)


@pytest.mark.asyncio
async def test_t2_f5_whitespace_only_chat_rejected(make_client):
    """Whitespace-only chat message is rejected with validation error and not relayed."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Send whitespace-only string
    await client_a.send_envelope("chat", payload={"text": "   \t \r\n   "})

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_PAYLOAD", "VALIDATION_ERROR", "EMPTY_MESSAGE")

    # Client B should receive nothing
    drained = await client_b.drain(timeout=0.3)
    assert not any(m.get("type") == "chat" for m in drained)


@pytest.mark.asyncio
async def test_t2_f5_max_length_chat_1000_chars(make_client):
    """1000-character message passes; 1001-character message is rejected."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Exactly 1000 characters should succeed
    msg_1000 = "x" * 1000
    await client_a.send_chat(msg_1000)
    received = await client_b.recv_type("chat", timeout=5.0)
    assert received["payload"]["text"] == msg_1000

    # 1001 characters should fail validation
    msg_1001 = "y" * 1001
    await client_a.send_chat(msg_1001)
    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_PAYLOAD", "VALIDATION_ERROR", "MESSAGE_TOO_LONG")


@pytest.mark.asyncio
async def test_t2_f5_unicode_emoji_and_special_chars(make_client):
    """Complex unicode, emojis, ZWJ sequences, RTL text and HTML tags relay intact."""
    client_a, client_b = await _pair_two_clients(make_client)

    complex_message = (
        "Complex text: 👨‍👩‍👧‍👦 🚀 🔥\n"
        "RTL test: مرحبا بالعالم שלום עולם\n"
        "Escaping: <script>alert('xss')</script> &amp; \"quotes\" 'single'"
    )

    await client_a.send_chat(complex_message)
    msg = await client_b.recv_type("chat", timeout=5.0)
    assert msg["payload"]["text"] == complex_message


@pytest.mark.asyncio
async def test_t2_f5_chat_sent_after_partner_left(make_client):
    """Sending chat after partner has left returns NOT_IN_SESSION error."""
    client_a, client_b = await _pair_two_clients(make_client)

    # Client B leaves session
    await client_b.leave()
    await client_a.recv_type("peer_disconnected", timeout=5.0)

    # Client A attempts to chat
    await client_a.send_chat("Are you still there?")

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("NOT_IN_SESSION", "SESSION_TERMINATED", "PARTNER_LEFT")


# --- F6 Boundaries ---

@pytest.mark.asyncio
async def test_t2_f6_http_non_secure_gif_blocked(make_client):
    """Non-HTTPS (insecure http://) GIF URL is rejected and not relayed."""
    client_a, client_b = await _pair_two_clients(make_client)

    insecure_url = "http://media.giphy.com/media/test/giphy.gif"
    await client_a.send_gif(insecure_url)

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_MEDIA_URL", "UNTRUSTED_DOMAIN", "INVALID_PAYLOAD", "MODERATION_BLOCKED")

    # Partner receives no GIF
    drained = await client_b.drain(timeout=0.3)
    assert not any(m.get("type") == "gif" for m in drained)


@pytest.mark.asyncio
async def test_t2_f6_untrusted_gif_domain_blocked(make_client):
    """GIF URL from untrusted domain is rejected by allowlist validation."""
    client_a, client_b = await _pair_two_clients(make_client)

    untrusted_url = "https://untrusted-image-host.com/secret/animation.gif"
    await client_a.send_gif(untrusted_url)

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("UNTRUSTED_DOMAIN", "INVALID_MEDIA_URL", "INVALID_PAYLOAD", "FORBIDDEN_DOMAIN")

    # Partner receives no GIF
    drained = await client_b.drain(timeout=0.3)
    assert not any(m.get("type") == "gif" for m in drained)


@pytest.mark.asyncio
async def test_t2_f6_malformed_url_string(make_client):
    """Malformed URL strings (e.g. javascript URI or arbitrary text) are rejected."""
    client_a, client_b = await _pair_two_clients(make_client)

    malformed_url = "javascript:alert(document.cookie)"
    await client_a.send_gif(malformed_url)

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_MEDIA_URL", "INVALID_PAYLOAD", "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_t2_f6_oversized_url_length(make_client):
    """Excessively long URL (>2048 characters) is rejected with validation error."""
    client_a, client_b = await _pair_two_clients(make_client)

    oversized_url = "https://media.giphy.com/" + ("a" * 2050) + ".gif"
    await client_a.send_gif(oversized_url)

    err = await client_a.recv_type("error", timeout=3.0)
    assert err["type"] == "error"
    code = err.get("payload", {}).get("code", "")
    assert code in ("INVALID_PAYLOAD", "VALIDATION_ERROR", "URL_TOO_LONG", "INVALID_MEDIA_URL")


@pytest.mark.asyncio
async def test_t2_f6_negative_dimensions_in_metadata(make_client):
    """Negative dimensions in GIF metadata are rejected or sanitized to non-negative values."""
    client_a, client_b = await _pair_two_clients(make_client)

    gif_url = "https://media.giphy.com/media/valid/giphy.gif"
    await client_a.send_gif(url=gif_url, width=-640, height=-480)

    # Server must either reject invalid dimensions with error or sanitize them on relay
    try:
        err = await client_a.recv_type("error", timeout=2.0)
        code = err.get("payload", {}).get("code", "")
        assert code in ("INVALID_PAYLOAD", "VALIDATION_ERROR", "INVALID_DIMENSIONS")
    except (TimeoutError, asyncio.TimeoutError):
        msg = await client_b.recv_type("gif", timeout=3.0)
        payload = msg.get("payload", {})
        width = payload.get("width")
        height = payload.get("height")
        assert width is None or width >= 0
        assert height is None or height >= 0
