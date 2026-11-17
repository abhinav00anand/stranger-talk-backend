"""Two-stage content moderation pipeline, media URL validator, and partner notification."""

import asyncio
import inspect
import ipaddress
import logging
import math
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

from app.config import settings
from app.models.envelopes import (
    PeerDisconnectedPayload,
    ServerMessageType,
    WebSocketEnvelope,
)
from app.moderation.normalizer import TextNormalizer, text_normalizer

logger = logging.getLogger(__name__)

DEFAULT_TOXICITY_THRESHOLD = 0.8
DEFAULT_ALLOWED_GIF_DOMAINS = [
    "giphy.com",
    "*.giphy.com",
    "media.giphy.com",
    "i.giphy.com",
    "tenor.com",
    "*.tenor.com",
    "media.tenor.com",
    "c.tenor.com",
]


class ModerationResult(BaseModel):
    """Result of content moderation evaluation."""

    is_flagged: bool = False
    reason: str | None = None
    confidence: float = 0.0
    action: Literal["allow", "block", "ban"] = "allow"
    stage: str | None = None
    normalized_text: str | None = None

    @property
    def is_clean(self) -> bool:
        """Return True if content is clean and approved."""
        return not self.is_flagged

    @property
    def is_bannable(self) -> bool:
        """Return True if content violates policy and warrants a ban."""
        return self.is_flagged and self.action == "ban"

    @property
    def is_blocked(self) -> bool:
        """Return True when content must be withheld without banning the sender."""
        return self.is_flagged and self.action == "block"

    @classmethod
    def allow(
        cls,
        stage: str = "allow",
        normalized_text: str | None = None,
    ) -> "ModerationResult":
        """Factory for an allowed moderation result."""
        return cls(
            is_flagged=False,
            reason=None,
            confidence=0.0,
            action="allow",
            stage=stage,
            normalized_text=normalized_text,
        )

    @classmethod
    def block(
        cls,
        reason: str,
        confidence: float = 0.0,
        stage: str = "block",
        normalized_text: str | None = None,
    ) -> "ModerationResult":
        """Factory for a blocked message that does not penalize the sender."""
        return cls(
            is_flagged=True,
            reason=reason,
            confidence=confidence,
            action="block",
            stage=stage,
            normalized_text=normalized_text,
        )

    @classmethod
    def ban(
        cls,
        reason: str,
        confidence: float = 1.0,
        stage: str = "ban",
        normalized_text: str | None = None,
    ) -> "ModerationResult":
        """Factory for a bannable moderation result."""
        return cls(
            is_flagged=True,
            reason=reason,
            confidence=confidence,
            action="ban",
            stage=stage,
            normalized_text=normalized_text,
        )


class ModerationPipeline:
    """Two-stage content moderation pipeline and media URL validator.

    Stage 1: In-process fast-path normalizer & blocklist (<1ms).
    Stage 2: Asynchronous AI toxicity classifier scored against TOXICITY_THRESHOLD.
    """

    def __init__(
        self,
        normalizer: TextNormalizer | None = None,
        toxicity_threshold: float | None = None,
        allowed_gif_domains: list[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.normalizer = normalizer or text_normalizer
        self.toxicity_threshold = (
            toxicity_threshold
            if toxicity_threshold is not None
            else getattr(settings, "TOXICITY_THRESHOLD", DEFAULT_TOXICITY_THRESHOLD)
        )
        self.allowed_gif_domains = (
            allowed_gif_domains
            if allowed_gif_domains is not None
            else getattr(settings, "ALLOWED_GIF_DOMAINS", DEFAULT_ALLOWED_GIF_DOMAINS)
        )
        self.enabled = (
            enabled
            if enabled is not None
            else getattr(settings, "MODERATION_ENABLED", True)
        )

    def moderate_text_fastpath(self, text: str) -> ModerationResult:
        """Synchronously moderate text using only Stage 1 fast-path (<1ms)."""
        if not self.enabled:
            return ModerationResult.allow(stage="moderation_disabled")

        if not text or not isinstance(text, str):
            return ModerationResult.allow(stage="empty_text")

        is_toxic, reason = self.normalizer.check_toxic(text)
        normalized = (
            self.normalizer.normalize(text)
            if hasattr(self.normalizer, "normalize")
            else None
        )
        if is_toxic:
            return ModerationResult.ban(
                reason=reason or "toxic_language",
                confidence=1.0,
                stage="stage1_fastpath",
                normalized_text=normalized,
            )

        return ModerationResult.allow(
            stage="stage1_clean",
            normalized_text=normalized,
        )

    async def moderate_text(
        self,
        text: str,
        provider: Any | None = None,
    ) -> ModerationResult:
        """Moderate text message through Stage 1 and Stage 2 pipelines."""
        if not self.enabled:
            return ModerationResult.allow(stage="moderation_disabled")

        if not text or not isinstance(text, str):
            return ModerationResult.allow(stage="empty_text")

        # --- Stage 1: Fast-Path Normalizer & Blocklist (<1ms) ---
        is_toxic, reason = self.normalizer.check_toxic(text)
        if is_toxic:
            logger.info("Stage 1 fast-path flagged toxic message: reason=%s", reason)
            normalized = (
                self.normalizer.normalize(text)
                if hasattr(self.normalizer, "normalize")
                else None
            )
            return ModerationResult.ban(
                reason=reason or "toxic_language",
                confidence=1.0,
                stage="stage1_fastpath",
                normalized_text=normalized,
            )

        # --- Stage 2: Deep AI Toxicity Classifier ---
        if provider is not None:
            try:
                score, ai_flagged, ai_reason = await self._call_ai_provider(provider, text)
                if ai_flagged or score >= self.toxicity_threshold:
                    logger.info(
                        "Stage 2 AI provider flagged text: score=%.3f, threshold=%.3f",
                        score,
                        self.toxicity_threshold,
                    )
                    return ModerationResult.ban(
                        reason=ai_reason or "ai_detected_toxicity",
                        confidence=score,
                        stage="stage2_deep",
                    )
            except Exception as e:  # noqa: BLE001 - provider failures must not silently bypass moderation
                logger.warning("Stage 2 moderation provider error: %s", e)
                if getattr(settings, "MODERATION_FAIL_OPEN", False):
                    return ModerationResult.allow(stage="stage2_error_fallback")
                return ModerationResult.block(
                    reason="moderation_unavailable",
                    stage="stage2_error_block",
                )

        normalized = (
            self.normalizer.normalize(text)
            if hasattr(self.normalizer, "normalize")
            else None
        )
        return ModerationResult.allow(
            stage="stage1_clean" if provider is None else "stage2_clean",
            normalized_text=normalized,
        )

    async def _call_ai_provider(
        self,
        provider: Any,
        text: str,
    ) -> tuple[float, bool, str | None]:
        """Safely invoke external or mock AI moderation provider."""
        moderate_fn = getattr(provider, "moderate_text", None)
        if not callable(moderate_fn):
            moderate_fn = provider if callable(provider) else None

        if moderate_fn is None:
            raise TypeError("Moderation provider must expose moderate_text() or be callable")

        if asyncio.iscoroutinefunction(moderate_fn) or inspect.iscoroutinefunction(moderate_fn):
            result = await moderate_fn(text)
        else:
            result = moderate_fn(text)
            if asyncio.iscoroutine(result):
                result = await result

        score = 0.0
        flagged = False
        reason = None

        if isinstance(result, (float, int)):
            score = float(result)
            flagged = score >= self.toxicity_threshold
        elif isinstance(result, dict):
            reason = result.get("reason")
            raw_flagged = result.get("is_flagged")
            score_val = result.get("score")
            if score_val is None:
                score_val = result.get("toxicity_score")
            if score_val is not None:
                score = float(score_val)
                flagged = bool(raw_flagged) if raw_flagged is not None else (score >= self.toxicity_threshold)
            else:
                flagged = bool(raw_flagged) if raw_flagged is not None else False
                score = 1.0 if flagged else 0.0
        elif isinstance(result, bool):
            flagged = result
            score = 1.0 if flagged else 0.0
        elif hasattr(result, "is_flagged") or hasattr(result, "score"):
            reason = getattr(result, "reason", None)
            raw_flagged = getattr(result, "is_flagged", None)
            score_val = getattr(result, "score", None)
            if score_val is None:
                score_val = getattr(result, "toxicity_score", None)
            if score_val is not None:
                score = float(score_val)
                flagged = bool(raw_flagged) if raw_flagged is not None else (score >= self.toxicity_threshold)
            else:
                flagged = bool(raw_flagged) if raw_flagged is not None else False
                score = 1.0 if flagged else 0.0
        else:
            raise TypeError("Moderation provider returned an unsupported result type")

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Moderation provider returned a score outside the 0.0-1.0 range")

        return score, flagged, reason

    def moderate_gif_url(self, url: str) -> ModerationResult:
        """Validate GIF URL protocol, domain allowlist, and formatting safety rules."""
        if not self.enabled:
            return ModerationResult.allow(stage="moderation_disabled")

        if not url or not isinstance(url, str):
            return ModerationResult.ban(reason="empty_url", stage="gif_validator")

        trimmed_url = url.strip()
        if not trimmed_url:
            return ModerationResult.ban(reason="empty_url", stage="gif_validator")

        if len(trimmed_url) > 2048:
            return ModerationResult.ban(reason="url_too_long", stage="gif_validator")

        try:
            parsed = urllib.parse.urlparse(trimmed_url)
        except ValueError:
            return ModerationResult.ban(reason="malformed_url", stage="gif_validator")

        # 1. Protocol check: must be strictly https
        if not parsed.scheme or parsed.scheme.lower() != "https":
            return ModerationResult.ban(reason="insecure_protocol", stage="gif_validator")

        # 2. Hostname extraction & validation
        hostname = parsed.hostname
        if not hostname:
            return ModerationResult.ban(reason="missing_hostname", stage="gif_validator")

        # Reject userinfo credentials on approved domains (RFC 3986 / WHATWG safety)
        if parsed.username is not None or parsed.password is not None or "@" in (parsed.netloc or ""):
            return ModerationResult.ban(reason="unapproved_domain", stage="gif_validator")

        hostname = hostname.lower()

        # Reject raw IP addresses (IPv4 / IPv6)
        try:
            ipaddress.ip_address(hostname)
            return ModerationResult.ban(reason="unapproved_domain", stage="gif_validator")
        except ValueError:
            pass

        # Reject localhost
        if hostname in ("localhost", "localhost.localdomain"):
            return ModerationResult.ban(reason="unapproved_domain", stage="gif_validator")

        # 3. Port check: standard HTTPS only (None or 443)
        try:
            port = parsed.port
        except ValueError:
            return ModerationResult.ban(reason="unapproved_port", stage="gif_validator")

        if port is not None and port != 443:
            return ModerationResult.ban(reason="unapproved_port", stage="gif_validator")

        # 4. Domain allowlist check
        if not self._is_allowed_domain(hostname):
            return ModerationResult.ban(reason="unapproved_domain", stage="gif_validator")

        return ModerationResult.allow(stage="gif_validator")

    def _is_allowed_domain(self, hostname: str) -> bool:
        """Check if hostname matches any allowed domain pattern."""
        for pattern in self.allowed_gif_domains:
            pattern = pattern.strip().lower()
            if not pattern:
                continue

            if pattern.startswith("*."):
                base_domain = pattern[2:]
                if hostname == base_domain or hostname.endswith("." + base_domain):
                    return True
            else:
                if hostname == pattern or hostname.endswith("." + pattern):
                    return True
        return False

    def create_partner_banned_envelope(
        self,
        session_id: str | None = None,
        reason: str = "banned",
        can_requeue: bool = True,
        message: str = "Your partner was removed for violating community guidelines.",
        envelope_type: str | None = None,
    ) -> WebSocketEnvelope:
        """Construct WebSocket notification envelope for partner of banned user."""
        if envelope_type is not None and envelope_type != ServerMessageType.PEER_DISCONNECTED.value:
            return WebSocketEnvelope(
                type=envelope_type,
                session_id=session_id,
                payload={
                    "reason": reason,
                    "can_requeue": can_requeue,
                    "message": message,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        if reason not in ("skipped", "left", "disconnected", "banned"):
            return WebSocketEnvelope(
                type="partner_banned" if envelope_type == "partner_banned" else ServerMessageType.PEER_DISCONNECTED.value,
                session_id=session_id,
                payload={
                    "reason": reason,
                    "can_requeue": can_requeue,
                    "message": message,
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        payload = PeerDisconnectedPayload(
            reason="banned",
            can_requeue=can_requeue,
            message=message,
        )
        return WebSocketEnvelope.create(
            msg_type=ServerMessageType.PEER_DISCONNECTED,
            payload=payload,
            session_id=session_id,
        )


moderation_pipeline = ModerationPipeline()


async def moderate_text(
    text: str,
    provider: Any | None = None,
) -> ModerationResult:
    """Convenience helper to moderate text using singleton pipeline."""
    return await moderation_pipeline.moderate_text(text, provider=provider)


def moderate_gif_url(url: str) -> ModerationResult:
    """Convenience helper to moderate GIF URL using singleton pipeline."""
    return moderation_pipeline.moderate_gif_url(url)


def create_partner_banned_envelope(
    session_id: str | None = None,
    reason: str = "banned",
    can_requeue: bool = True,
    message: str = "Your partner was removed for violating community guidelines.",
    envelope_type: str | None = None,
) -> WebSocketEnvelope:
    """Convenience helper to create partner banned notification envelope."""
    return moderation_pipeline.create_partner_banned_envelope(
        session_id=session_id,
        reason=reason,
        can_requeue=can_requeue,
        message=message,
        envelope_type=envelope_type,
    )


__all__ = [
    "ModerationPipeline",
    "ModerationResult",
    "create_partner_banned_envelope",
    "moderate_gif_url",
    "moderate_text",
    "moderation_pipeline",
]
