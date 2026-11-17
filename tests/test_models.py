"""Unit tests for domain models, envelopes, and timezone/validation behavior."""

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from app.models.envelopes import (
    ChatMessagePayload,
    ClientMessageType,
    GifMessagePayload,
    ServerMessageType,
    WebSocketEnvelope,
)
from app.models.session import (
    BanRecord,
    ClientState,
    QueueTicket,
    Session,
    SessionStatus,
    _to_utc,
)


def test_to_utc_helper_with_naive_datetime():
    dt = datetime(2026, 11, 16, 12, 0, 0)
    res = _to_utc(dt)
    assert res.tzinfo == timezone.utc
    assert res.year == 2026 and res.hour == 12


def test_to_utc_helper_with_aware_datetime():
    tz_plus_5 = timezone(timedelta(hours=5))
    dt = datetime(2026, 11, 16, 17, 0, 0, tzinfo=tz_plus_5)
    res = _to_utc(dt)
    assert res.tzinfo == timezone.utc
    assert res.hour == 12


def test_ban_record_is_expired_with_naive_datetime():
    # Test local naive datetime (e.g. from datetime.now())
    local_future = datetime.now() + timedelta(minutes=5)
    active_local = BanRecord(ip="1.1.1.1", expires_at=local_future, reason="test")
    assert active_local.is_expired() is False

    # Test UTC naive datetime (e.g. from deserialized ISO or naive UTC timestamp)
    utc_naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
    future_utc_naive = utc_naive_now + timedelta(minutes=5)
    past_utc_naive = utc_naive_now - timedelta(minutes=5)

    active_ban = BanRecord(ip="1.1.1.1", expires_at=future_utc_naive, reason="test")
    assert active_ban.is_expired() is False

    expired_ban = BanRecord(ip="1.1.1.1", expires_at=past_utc_naive, reason="test")
    assert expired_ban.is_expired() is True


def test_ban_record_remaining_seconds_with_naive_datetime():
    # Test local naive datetime (ensures no TypeError and returns float >= 0.0)
    local_future = datetime.now() + timedelta(minutes=5)
    active_local = BanRecord(ip="1.1.1.1", expires_at=local_future, reason="test")
    rem_local = active_local.remaining_seconds()
    assert isinstance(rem_local, float) and rem_local >= 0.0

    # Test UTC naive datetime with bounded comparison
    utc_naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
    future_utc_naive = utc_naive_now + timedelta(seconds=60)
    past_utc_naive = utc_naive_now - timedelta(seconds=60)

    active_ban = BanRecord(ip="1.1.1.1", expires_at=future_utc_naive, reason="test")
    rem = active_ban.remaining_seconds()
    assert 55.0 <= rem <= 61.0

    expired_ban = BanRecord(ip="1.1.1.1", expires_at=past_utc_naive, reason="test")
    assert expired_ban.remaining_seconds() == 0.0


def test_ban_record_deserialization_from_naive_iso_json():
    json_data = '{"ip": "10.0.0.1", "expires_at": "2026-11-16T23:59:59", "reason": "test"}'
    ban = BanRecord.model_validate_json(json_data)
    assert isinstance(ban.is_expired(), bool)
    assert ban.remaining_seconds() >= 0.0


def test_queue_ticket_elapsed_seconds_with_naive_datetime():
    # Test local naive datetime (ensures no TypeError and returns float >= 0.0)
    ticket_local = QueueTicket(client_id="user_1", queued_at=datetime.now())
    elapsed_local = ticket_local.elapsed_seconds()
    assert isinstance(elapsed_local, float) and elapsed_local >= 0.0

    # Test UTC naive datetime with bounded range
    utc_naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
    past_utc_naive = utc_naive_now - timedelta(seconds=15)
    ticket = QueueTicket(client_id="user_1", queued_at=past_utc_naive)
    elapsed = ticket.elapsed_seconds()
    assert 14.0 <= elapsed <= 17.0


def test_queue_ticket_elapsed_seconds_clamps_future_timestamp_to_zero():
    future_aware = datetime.now(timezone.utc) + timedelta(seconds=30)
    ticket = QueueTicket(client_id="user_1", queued_at=future_aware)
    assert ticket.elapsed_seconds() == 0.0


def test_queue_ticket_deserialization_from_naive_iso_json():
    json_data = '{"client_id": "u1", "queued_at": "2026-11-16T12:00:00"}'
    ticket = QueueTicket.model_validate_json(json_data)
    assert ticket.elapsed_seconds() >= 0.0


def test_session_rejects_self_pairing_with_identical_clients():
    with pytest.raises(ValidationError) as exc_info:
        Session(client_a="user_self", client_b="user_self")
    assert "Session participants client_a and client_b must be distinct" in str(exc_info.value)


def test_session_valid_distinct_clients():
    session = Session(client_a="user_1", client_b="user_2")
    assert session.client_a == "user_1"
    assert session.client_b == "user_2"
    assert session.get_partner_id("user_1") == "user_2"
    assert session.get_partner_id("user_2") == "user_1"
    assert session.get_partner_id("unknown") is None
    assert session.contains_client("user_1") is True
    assert session.contains_client("unknown") is False
    assert session.is_active() is True


def test_gif_message_payload_rejects_empty_url():
    with pytest.raises(ValidationError):
        GifMessagePayload(url="")


def test_gif_message_payload_accepts_valid_url():
    payload = GifMessagePayload(url="https://media.giphy.com/media/test/giphy.gif")
    assert payload.url == "https://media.giphy.com/media/test/giphy.gif"


def test_websocket_envelope_create_and_serialization():
    env = WebSocketEnvelope.create(
        msg_type=ClientMessageType.CHAT,
        payload=ChatMessagePayload(text="Hello world"),
        session_id="sess_123",
    )
    assert env.type == "chat"
    assert env.session_id == "sess_123"
    assert env.payload["text"] == "Hello world"
    json_str = env.to_json()
    assert "sess_123" in json_str
    assert "Hello world" in json_str
