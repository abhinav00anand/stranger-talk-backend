"""Regression tests for user safety reports, blocking, and GIF metadata validation."""

import asyncio

import httpx
import pytest

from app.safety.report_store import ReportStore


def test_report_store_persists_only_selected_evidence(tmp_path):
    store_path = tmp_path / "reports.sqlite3"
    store = ReportStore(str(store_path), retention_days=90)
    store.initialize()

    created = store.create_report(
        report_id="report-1",
        session_id="session-1",
        reporter_client_id="reporter-1",
        reported_client_id="reported-1",
        category="harassment",
        details="Repeated harassment",
        message_excerpt="An excerpt explicitly selected by the reporter",
    )

    reopened = ReportStore(str(store_path), retention_days=90)
    reports = reopened.list_reports()
    assert created["report_id"] == "report-1"
    assert len(reports) == 1
    assert reports[0]["message_excerpt"] == "An excerpt explicitly selected by the reporter"
    assert "ip_address" not in reports[0]
    assert "transcript" not in reports[0]


@pytest.mark.asyncio
async def test_queue_never_pairs_a_blocked_live_connection_pair():
    from app.matchmaker.queue import MatchmakerQueue

    queue = MatchmakerQueue()
    await queue.enqueue("reporter")
    await queue.enqueue("reported")
    blocked = {tuple(sorted(("reporter", "reported")))}
    assert await queue.try_pop_pair(blocked_pairs=blocked) is None

    await queue.enqueue("new-stranger")
    assert await queue.try_pop_pair(blocked_pairs=blocked) == ("reporter", "new-stranger")


@pytest.mark.asyncio
async def test_report_is_stored_blocks_pair_and_requeues_reporter(make_client, monkeypatch, tmp_path):
    import app.main as main
    from app.safety.report_store import ReportStore as SQLiteReportStore

    store = SQLiteReportStore(str(tmp_path / "safety.sqlite3"), retention_days=90)
    store.initialize()
    monkeypatch.setattr(main, "report_store", store)
    monkeypatch.setattr(main, "reports_ready", True)
    monkeypatch.setattr(main, "blocked_pairs", set())
    monkeypatch.setattr(main, "reported_sessions", set())
    monkeypatch.setattr(main, "report_rate_limiter", main.IPRateLimiter(3, 3600.0))
    monkeypatch.setattr(main, "report_ip_rate_limiter", main.IPRateLimiter(20, 3600.0))

    reporter = await make_client()
    stranger = await make_client()
    await reporter.join_queue()
    await stranger.join_queue()
    matched_reporter = await reporter.recv_type("matched", timeout=5.0)
    await stranger.recv_type("matched", timeout=5.0)

    await reporter.send_envelope(
        "report",
        {
            "category": "harassment",
            "details": "Repeated targeted abuse",
            "message_excerpt": "An example message",
        },
        session_id=matched_reporter["session_id"],
    )

    saved = await reporter.recv_type("report_submitted", timeout=5.0)
    await reporter.recv_type("queued", timeout=5.0)
    partner_event = await stranger.recv_type("peer_disconnected", timeout=5.0)

    assert saved["payload"]["report_id"]
    assert partner_event["payload"]["reason"] == "left"
    assert main.session_manager.get_session_by_client(
        next(iter(main.active_connections.keys()))
    ) is None or main.session_manager.active_session_count() == 0

    records = store.list_reports()
    assert len(records) == 1
    assert records[0]["category"] == "harassment"
    assert records[0]["details"] == "Repeated targeted abuse"
    assert records[0]["message_excerpt"] == "An example message"
    assert records[0]["reported_client_id"] != records[0]["reporter_client_id"]


@pytest.mark.asyncio
async def test_block_action_skips_blocked_pair_and_finds_other_user(make_client):
    reporter = await make_client()
    blocked_user = await make_client()
    await reporter.join_queue()
    await blocked_user.join_queue()
    match = await reporter.recv_type("matched", timeout=5.0)
    await blocked_user.recv_type("matched", timeout=5.0)

    await reporter.send_envelope("block", {}, session_id=match["session_id"])
    blocked = await reporter.recv_type("blocked", timeout=5.0)
    await reporter.recv_type("queued", timeout=5.0)
    await blocked_user.recv_type("peer_disconnected", timeout=5.0)
    assert "blocked" in blocked["payload"]["message"].lower()

    # Rejoining the blocked stranger must not pair them back with the reporter.
    await blocked_user.join_queue()
    replacement = await make_client()
    await replacement.join_queue()
    report_match = await reporter.recv_type("matched", timeout=5.0)
    replacement_match = await replacement.recv_type("matched", timeout=5.0)
    assert report_match["session_id"] == replacement_match["session_id"]



def test_gif_dimension_sanitizer_rejects_hostile_values():
    from app.main import _sanitize_gif_dimension

    assert _sanitize_gif_dimension(640) == 640
    assert _sanitize_gif_dimension("320") == 320
    assert _sanitize_gif_dimension("0") is None
    assert _sanitize_gif_dimension("-10") is None
    assert _sanitize_gif_dimension(True) is None
    assert _sanitize_gif_dimension("9" * 5000) is None
    assert _sanitize_gif_dimension("20000") == 10_000


@pytest.mark.asyncio
async def test_admin_reports_endpoint_requires_configured_token(monkeypatch, tmp_path):
    import app.main as main

    store = ReportStore(str(tmp_path / "admin-reports.sqlite3"), retention_days=90)
    store.initialize()
    monkeypatch.setattr(main, "report_store", store)
    monkeypatch.setattr(main, "reports_ready", True)
    monkeypatch.setattr(main.settings, "REPORTS_ADMIN_TOKEN", "test-only-secret")

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/admin/reports")
        wrong = await client.get("/admin/reports", headers={"X-Admin-Token": "wrong"})
        valid = await client.get("/admin/reports", headers={"X-Admin-Token": "test-only-secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert valid.status_code == 200
    assert valid.json() == {"reports": []}
