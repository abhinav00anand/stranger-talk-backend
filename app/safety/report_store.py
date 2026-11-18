"""SQLite-backed storage for user-submitted safety reports."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ReportStore:
    """SQLite report store; the configured path controls persistence and storage medium."""

    def __init__(self, db_path: str, retention_days: int = 90) -> None:
        self.db_path = db_path
        self.retention_days = max(1, int(retention_days))

    def _connect(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    reporter_client_id TEXT NOT NULL,
                    reported_client_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '',
                    message_excerpt TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at)"
            )
            self._prune(connection)
            connection.commit()
        finally:
            connection.close()

    def _prune(self, connection: sqlite3.Connection) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        connection.execute("DELETE FROM reports WHERE created_at < ?", (cutoff,))

    def create_report(
        self,
        *,
        report_id: str,
        session_id: str,
        reporter_client_id: str,
        reported_client_id: str,
        category: str,
        details: str = "",
        message_excerpt: str = "",
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO reports (
                        report_id, created_at, session_id, reporter_client_id,
                        reported_client_id, category, details, message_excerpt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report_id,
                        created_at,
                        session_id,
                        reporter_client_id,
                        reported_client_id,
                        category,
                        details[:1000],
                        message_excerpt[:800],
                    ),
                )
                self._prune(connection)
            return {
                "report_id": report_id,
                "created_at": created_at,
                "session_id": session_id,
                "category": category,
                "details": details[:1000],
                "message_excerpt": message_excerpt[:800],
                "status": "pending",
            }
        finally:
            connection.close()

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._prune(connection)
            rows = connection.execute(
                """
                SELECT report_id, created_at, session_id, reporter_client_id,
                       reported_client_id, category, details, message_excerpt, status
                FROM reports
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 200)),),
            ).fetchall()
            connection.commit()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def clear_all(self) -> None:
        """Clear all stored reports; intended for isolated tests and maintenance."""
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    reporter_client_id TEXT NOT NULL,
                    reported_client_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '',
                    message_excerpt TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            connection.execute("DELETE FROM reports")
            connection.commit()
        finally:
            connection.close()
