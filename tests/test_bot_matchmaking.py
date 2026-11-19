"""Security and lifecycle regression tests for authenticated standby bots."""

import pytest

from app.matchmaker.queue import MatchmakerQueue


@pytest.mark.asyncio
async def test_standby_bot_lookup_returns_when_every_bot_is_blocked():
    queue = MatchmakerQueue()
    await queue.register_standby_bot("bot-a")

    # This must return promptly and preserve the bot for another human.
    assert await queue.pop_standby_bot(
        human_client_id="human-a",
        blocked_pairs={("bot-a", "human-a")},
        active_connection_ids={"bot-a"},
    ) is None
    assert queue.standby_bot_count() == 1


@pytest.mark.asyncio
async def test_standby_bot_lookup_discards_disconnected_bot():
    queue = MatchmakerQueue()
    await queue.register_standby_bot("offline-bot")

    assert await queue.pop_standby_bot(
        human_client_id="human-a",
        active_connection_ids=set(),
    ) is None
    assert queue.standby_bot_count() == 0


@pytest.mark.asyncio
async def test_client_cannot_register_as_bot_using_payload_flag(monkeypatch):
    import app.main as main

    queue = MatchmakerQueue()
    sent = []
    monkeypatch.setattr(main, "matchmaker_queue", queue)
    monkeypatch.setattr(main, "bot_connections", set())

    async def fake_send(client_id, envelope):
        sent.append((client_id, envelope))
        return True

    monkeypatch.setattr(main, "send_to_client", fake_send)

    await main._handle_join_queue("spoofed-bot", "192.0.2.10", {"is_bot": True})

    assert queue.standby_bot_count() == 0
    assert queue.queue_size() == 0
    assert sent
    assert sent[-1][1].payload["code"] == "BOT_AUTH_REQUIRED"


@pytest.mark.asyncio
async def test_authenticated_bot_enters_standby_pool_not_human_queue(monkeypatch):
    import app.main as main

    queue = MatchmakerQueue()
    sent = []
    monkeypatch.setattr(main, "matchmaker_queue", queue)
    monkeypatch.setattr(main, "bot_connections", {"trusted-bot"})

    async def fake_send(client_id, envelope):
        sent.append((client_id, envelope))
        return True

    monkeypatch.setattr(main, "send_to_client", fake_send)

    await main._handle_join_queue("trusted-bot", "192.0.2.11", {})

    assert queue.queue_size() == 0
    assert queue.standby_bot_count() == 1
    assert sent[-1][1].type == main.ServerMessageType.QUEUED


@pytest.mark.asyncio
async def test_human_skip_partner_bot_is_returned_to_standby(monkeypatch):
    import app.main as main
    from app.models.session import SessionManager

    queue = MatchmakerQueue()
    sessions = SessionManager()
    session = await sessions.create_session("human-a", "trusted-bot")
    sent = []
    monkeypatch.setattr(main, "matchmaker_queue", queue)
    monkeypatch.setattr(main, "session_manager", sessions)
    monkeypatch.setattr(main, "bot_connections", {"trusted-bot"})
    monkeypatch.setattr(main, "active_connections", {"trusted-bot": object()})

    async def fake_send(client_id, envelope):
        sent.append((client_id, envelope))
        return True

    monkeypatch.setattr(main, "send_to_client", fake_send)

    await main._handle_skip("human-a", "192.0.2.12", session.session_id)

    assert queue.standby_bot_count() == 1
    assert queue.queue_size() == 1
    assert any(client_id == "trusted-bot" and envelope.type == main.ServerMessageType.QUEUED for client_id, envelope in sent)
