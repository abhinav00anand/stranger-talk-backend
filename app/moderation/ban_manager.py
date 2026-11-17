"""In-memory thread-safe IP ban store and gatekeeper with TTL expiration and auto-pruning."""

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.models.session import BanRecord

logger = logging.getLogger(__name__)


class BanManager:
    """Thread-safe in-memory IP ban manager.

    Provides TTL expiration, auto-pruning on lookup, batch cleanup,
    and cross-test isolation for moderation and WebSocket gatekeeping.
    Supports both synchronous thread-safe operations (via threading.RLock)
    and asynchronous coroutine wrappers (via asyncio.Lock).
    """

    def __init__(self) -> None:
        self._bans: dict[str, BanRecord] = {}
        self._thread_lock = threading.RLock()
        self._async_lock: asyncio.Lock | None = None
        self._async_lock_loop: asyncio.AbstractEventLoop | None = None

    @property
    def async_lock(self) -> asyncio.Lock:
        """Lazily initialize an asyncio.Lock bound to the current running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._async_lock is None or (loop is not None and self._async_lock_loop != loop):
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    @property
    def _lock(self) -> asyncio.Lock:
        """Alias for async_lock matching sub-orchestrator contract."""
        return self.async_lock

    @property
    def bans(self) -> dict[str, BanRecord]:
        """Direct access to ban storage dict for test fixtures and inspection."""
        return self._bans

    @property
    def banned_ips(self) -> dict[str, BanRecord]:
        """Alias property for backward compatibility with conftest.py cleanup."""
        return self._bans

    def _normalize_ip(self, ip: str) -> str:
        """Validate and normalize an IP address or identifier."""
        if not ip or not isinstance(ip, str) or not ip.strip():
            raise ValueError("ip must be a non-empty string")
        return ip.strip().lower()

    # ---------------------------------------------------------------------------
    # Synchronous Thread-Safe Core API
    # ---------------------------------------------------------------------------

    def record_ban(
        self,
        ip: str,
        reason: str,
        duration_seconds: int | None = None,
    ) -> BanRecord:
        """Record a moderation ban for an IP address.

        Args:
            ip: Target IP address or identifier.
            reason: Explanation of the ban policy violation.
            duration_seconds: Validity period in seconds. Defaults to settings.BAN_DURATION_SECONDS.

        Returns:
            The created BanRecord instance.
        """
        norm_ip = self._normalize_ip(ip)
        norm_reason = (
            reason.strip()
            if (reason and isinstance(reason, str) and reason.strip())
            else "Policy violation"
        )

        eff_duration = (
            duration_seconds
            if duration_seconds is not None
            else settings.BAN_DURATION_SECONDS
        )
        eff_duration = max(0, eff_duration)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=eff_duration)

        record = BanRecord(
            ip=norm_ip,
            banned_at=now,
            expires_at=expires_at,
            reason=norm_reason,
        )

        with self._thread_lock:
            self._bans[norm_ip] = record

        logger.warning(
            "Recorded IP ban: ip=%s, reason='%s', duration=%ds, expires_at=%s",
            norm_ip,
            norm_reason,
            eff_duration,
            expires_at.isoformat(),
        )
        return record

    def is_banned(self, ip: str) -> bool:
        """Check if an IP is currently banned. Auto-purges expired records.

        Args:
            ip: IP address to check.

        Returns:
            True if ban is active; False if not banned or expired.
        """
        if not ip or not isinstance(ip, str) or not ip.strip():
            return False

        norm_ip = ip.strip().lower()
        with self._thread_lock:
            record = self._bans.get(norm_ip)
            if record is None:
                return False

            if record.is_expired():
                self._bans.pop(norm_ip, None)
                logger.info("Auto-pruned expired ban on lookup: ip=%s", norm_ip)
                return False

            return True

    def get_ban_record(self, ip: str) -> BanRecord | None:
        """Retrieve active BanRecord for an IP if present and unexpired.

        Args:
            ip: IP address to look up.

        Returns:
            BanRecord if active; None if expired or not found.
        """
        if not ip or not isinstance(ip, str) or not ip.strip():
            return None

        norm_ip = ip.strip().lower()
        with self._thread_lock:
            record = self._bans.get(norm_ip)
            if record is None:
                return None

            if record.is_expired():
                self._bans.pop(norm_ip, None)
                logger.info("Auto-pruned expired ban on get_ban_record: ip=%s", norm_ip)
                return None

            return record

    def lift_ban(self, ip: str) -> bool:
        """Explicitly unban an IP address.

        Args:
            ip: IP address to unban.

        Returns:
            True if a ban record existed and was removed; False otherwise.
        """
        if not ip or not isinstance(ip, str) or not ip.strip():
            return False

        norm_ip = ip.strip().lower()
        with self._thread_lock:
            removed = self._bans.pop(norm_ip, None)
            if removed is not None:
                logger.info("Lifted ban for ip=%s", norm_ip)
                return True
            return False

    def clear_expired(self) -> int:
        """Batch prune all expired ban records from memory.

        Returns:
            Count of expired records pruned.
        """
        with self._thread_lock:
            expired_ips = [k for k, v in self._bans.items() if v.is_expired()]
            for k in expired_ips:
                self._bans.pop(k, None)

            if expired_ips:
                logger.info("Pruned %d expired ban records", len(expired_ips))
            return len(expired_ips)

    def clear_all(self) -> None:
        """Reset and remove all ban records completely (used for test isolation)."""
        with self._thread_lock:
            count = len(self._bans)
            self._bans.clear()
            if count > 0:
                logger.debug("Cleared all %d IP bans from memory", count)

    def clear(self) -> None:
        """Alias for clear_all() providing full compatibility with conftest.py cleanup."""
        self.clear_all()

    def count(self) -> int:
        """Return total count of stored records (including unpruned expired ones)."""
        with self._thread_lock:
            return len(self._bans)

    def get_active_bans(self) -> list[BanRecord]:
        """Return a snapshot list of all currently active (unexpired) ban records."""
        with self._thread_lock:
            self.clear_expired()
            return list(self._bans.values())

    # ---------------------------------------------------------------------------
    # Asynchronous Coroutine Wrappers (with asyncio.Lock)
    # ---------------------------------------------------------------------------

    async def async_record_ban(
        self,
        ip: str,
        reason: str,
        duration_seconds: int | None = None,
    ) -> BanRecord:
        """Asynchronously record a ban under asyncio.Lock."""
        async with self.async_lock:
            return self.record_ban(ip, reason, duration_seconds)

    async def async_is_banned(self, ip: str) -> bool:
        """Asynchronously check if an IP is banned under asyncio.Lock."""
        async with self.async_lock:
            return self.is_banned(ip)

    async def async_get_ban_record(self, ip: str) -> BanRecord | None:
        """Asynchronously retrieve a ban record under asyncio.Lock."""
        async with self.async_lock:
            return self.get_ban_record(ip)

    async def async_lift_ban(self, ip: str) -> bool:
        """Asynchronously lift a ban under asyncio.Lock."""
        async with self.async_lock:
            return self.lift_ban(ip)

    async def async_clear_expired(self) -> int:
        """Asynchronously prune expired bans under asyncio.Lock."""
        async with self.async_lock:
            return self.clear_expired()

    async def async_clear_all(self) -> None:
        """Asynchronously clear all bans under asyncio.Lock."""
        async with self.async_lock:
            self.clear_all()


# Singleton instance for app-wide and test fixture usage
ban_manager = BanManager()

__all__ = ["BanManager", "ban_manager"]
