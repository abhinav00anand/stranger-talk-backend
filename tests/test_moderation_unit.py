"""Comprehensive unit test suite for Milestone 3 moderation subsystem.

Covers:
1. ModerationResult model, serialization, and factory constructors.
2. TextNormalizer & Stage 1 Fast-Path:
   - Unicode NFKD decomposition
   - Zero-width character stripping
   - Cyrillic and Greek homoglyph mapping
   - Leetspeak translation
   - Pre-compiled regex patterns (self-harm, threats, slurs, harassment, sexual violence)
   - Benign word false-positive immunity (skills, whiskeys, grapes, spooky, pharmacist)
   - Latency benchmark (<1ms per check)
3. ModerationPipeline Stage 2 Deep AI Moderation:
   - Short-circuiting when Stage 1 flags
   - Provider scoring against TOXICITY_THRESHOLD
   - Async provider support
   - Sync provider support
   - Provider exception fail-closed default with explicit fail-open override
   - MODERATION_ENABLED=False bypass
4. Media/GIF URL Validator:
   - HTTPS enforcement
   - Allowed domain & wildcard matching (giphy.com, tenor.com)
   - Prefix/suffix domain spoofing rejection
   - Malicious scheme rejection (javascript:, data:, file:)
   - Userinfo injection rejection
   - Raw IP address rejection
   - Localhost rejection
   - Custom port rejection (port 443 permitted)
   - URL length boundary checks (2048 vs 2049)
   - Empty/whitespace URL rejection
5. BanManager:
   - Synchronous thread-safe recording & lookups
   - Default vs custom duration TTL
   - Zero TTL instant expiration
   - Auto-pruning on is_banned and get_ban_record
   - lift_ban, clear_expired, clear_all, clear
   - Concurrency safety with threading and asyncio wrappers
6. Partner Notification Envelopes:
   - Canonical peer_disconnected with reason="banned"
   - Envelope override options
   - JSON serialization
7. Concurrency & Performance Benchmarks
8. Module Exports & Convenience Wrappers
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.config import settings
from app.models.envelopes import WebSocketEnvelope
from app.models.session import BanRecord
from app.moderation import (
    BanManager,
    ModerationPipeline,
    ModerationResult,
    TextNormalizer,
    create_partner_banned_envelope,
    moderate_gif_url,
    moderate_text,
)

# ============================================================================
# 1. ModerationResult Model Tests
# ============================================================================

def test_moderation_result_defaults():
    """Default ModerationResult is clean, allowed, and not flagged."""
    res = ModerationResult()
    assert res.is_flagged is False
    assert res.reason is None
    assert res.confidence == 0.0
    assert res.action == "allow"
    assert res.stage is None
    assert res.normalized_text is None
    assert res.is_clean is True
    assert res.is_bannable is False


def test_moderation_result_allow_factory():
    """ModerationResult.allow() sets correct fields and properties."""
    res = ModerationResult.allow(stage="stage1_clean", normalized_text="hello world")
    assert res.is_flagged is False
    assert res.reason is None
    assert res.confidence == 0.0
    assert res.action == "allow"
    assert res.stage == "stage1_clean"
    assert res.normalized_text == "hello world"
    assert res.is_clean is True
    assert res.is_bannable is False


def test_moderation_result_ban_factory():
    """ModerationResult.ban() sets flagged, reason, confidence, and action."""
    res = ModerationResult.ban(
        reason="self_harm",
        confidence=1.0,
        stage="stage1_fastpath",
        normalized_text="kys",
    )
    assert res.is_flagged is True
    assert res.reason == "self_harm"
    assert res.confidence == 1.0
    assert res.action == "ban"
    assert res.stage == "stage1_fastpath"
    assert res.normalized_text == "kys"
    assert res.is_clean is False
    assert res.is_bannable is True


def test_moderation_result_serialization():
    """ModerationResult cleanly serializes to dict and JSON."""
    res = ModerationResult.ban(reason="violent_threat", confidence=0.95, stage="stage2_deep")
    data = res.model_dump()
    assert data["is_flagged"] is True
    assert data["reason"] == "violent_threat"
    assert data["confidence"] == 0.95
    assert data["action"] == "ban"

    json_str = res.model_dump_json()
    assert '"action":"ban"' in json_str or '"action": "ban"' in json_str


# ============================================================================
# 2. TextNormalizer & Stage 1 Fast-Path Tests
# ============================================================================

def test_normalizer_clean_messages():
    """Standard polite and benign messages are not flagged."""
    clean_messages = [
        "Hello! How are you doing today?",
        "What is your favorite video game?",
        "The quick brown fox jumps over the lazy dog.",
        "Nice to meet you! I love programming.",
        "Have a great evening and take care!",
    ]
    for msg in clean_messages:
        is_toxic, reason = TextNormalizer.check_toxic(msg)
        assert is_toxic is False, f"Clean message falsely flagged: '{msg}'"
        assert reason is None


def test_normalizer_toxic_self_harm():
    """Self-harm incitement phrases are flagged under 'self_harm'."""
    phrases = [
        "kys",
        "kill yourself",
        "hang yourself",
        "slit your wrists",
        "commit suicide",
        "drink bleach",
        "KYS RIGHT NOW",
        "go kill yourself you trash",
    ]
    for p in phrases:
        is_toxic, reason = TextNormalizer.check_toxic(p)
        assert is_toxic is True, f"Failed to flag self-harm phrase: '{p}'"
        assert reason in ("self_harm", "harassment")


def test_normalizer_toxic_violent_threats():
    """Violent threats and direct harm incitement are flagged under 'violent_threat'."""
    phrases = [
        "I will hunt you down and kill you",
        "i will murder you",
        "im gonna slaughter you",
        "slit your throat",
        "i will doxx you",
        "kill you right now",
    ]
    for p in phrases:
        is_toxic, reason = TextNormalizer.check_toxic(p)
        assert is_toxic is True, f"Failed to flag threat phrase: '{p}'"
        assert reason == "violent_threat"


def test_normalizer_toxic_slurs():
    """Hate speech slurs and their plurals/inflections are flagged under 'hate_speech'."""
    phrases = [
        "nigger",
        "niggers",
        "nigga",
        "niggas",
        "faggot",
        "faggots",
        "retard",
        "retards",
        "retarded",
        "retarding",
        "kike",
        "kikes",
        "spic",
        "spics",
        "chink",
        "chinks",
    ]
    for p in phrases:
        is_toxic, reason = TextNormalizer.check_toxic(p)
        assert is_toxic is True, f"Failed to flag slur: '{p}'"
        assert reason == "hate_speech"


def test_normalizer_toxic_harassment():
    """Harassment keywords and sentinel test phrases are flagged."""
    phrases = [
        "toxic abuse",
        "abuse violation",
        "worthless trash",
        "worthless piece of trash",
    ]
    for p in phrases:
        is_toxic, reason = TextNormalizer.check_toxic(p)
        assert is_toxic is True, f"Failed to flag harassment: '{p}'"
        assert reason == "harassment"


def test_normalizer_leetspeak_decoding():
    """Leetspeak obfuscated toxic words are properly decoded and flagged."""
    leetspeak_cases = [
        ("k!ll y0urs3lf", "self_harm"),
        ("k1ll y0urs3lf", "self_harm"),
        ("k|ll y0urs3lf", "self_harm"),
        ("h@ng y0urs3lf", "self_harm"),
        ("k.y.s", "self_harm"),
        ("k_y_s", "self_harm"),
        ("k y s", "self_harm"),
    ]
    for obfuscated, expected_cat in leetspeak_cases:
        is_toxic, reason = TextNormalizer.check_toxic(obfuscated)
        assert is_toxic is True, f"Failed to catch leetspeak: '{obfuscated}'"
        assert reason == expected_cat


def test_normalizer_homoglyphs_decoding():
    """Cyrillic and Greek homoglyphs are translated and flagged."""
    # Cyrillic: \u0456=i, \u043e=o, \u0455=s, \u0435=e
    homoglyph_str = "k\u0456ll y\u043eur\u0455\u0435lf"
    is_toxic, reason = TextNormalizer.check_toxic(homoglyph_str)
    assert is_toxic is True
    assert reason == "self_harm"

    # Cyrillic а (\u0430) and о (\u043e) in hate speech or threats
    threat_homoglyphs = "i will \u043aill y\u043eu"  # \u043a=k, \u043e=o
    is_toxic_threat, reason_threat = TextNormalizer.check_toxic(threat_homoglyphs)
    assert is_toxic_threat is True
    assert reason_threat == "violent_threat"

    # Greek capital Upsilon (\u03a5 -> y) in self harm (kys)
    assert TextNormalizer.check_toxic("k\u03a5s") == (True, "self_harm")

    # Cyrillic straight u (\u04af -> y, \u04ae -> y) in self harm (kys)
    assert TextNormalizer.check_toxic("k\u04afs") == (True, "self_harm")
    assert TextNormalizer.check_toxic("k\u04aes") == (True, "self_harm")


def test_normalizer_zero_width_character_stripping():
    """Zero-width spaces and joiners inside toxic words are stripped and caught."""
    # Inserting ZWSP (\u200b), ZWNJ (\u200c), ZWJ (\u200d), BOM (\ufeff)
    zw_text = "k\u200bi\u200cl\u200dl\ufeff yourself"
    is_toxic, reason = TextNormalizer.check_toxic(zw_text)
    assert is_toxic is True
    assert reason == "self_harm"


def test_normalizer_benign_word_false_positive_immunity():
    """Benign English words containing toxic substrings MUST NOT trigger false positives."""
    benign_sentences = [
        "I am practicing my programming skills.",  # contains 'kill' inside 'skills'
        "He ordered two fine whiskeys at the bar.",  # contains 'kys' inside 'whiskeys'
        "I really enjoy eating fresh grapes.",  # contains 'rape' inside 'grapes'
        "We listened to spooky ghost stories.",  # contains 'spook' / 'kys' lookalike
        "She visited the licensed pharmacist.",  # contains 'ph'
        "Please answer the telephone promptly.",  # contains 'ph'
        "We took a photograph of the sunset.",  # contains 'ph'
        "Document psychology and mental health.",  # contains 'ch'
        # Benign phrases containing 'cp' (Unix cp, Pokemon CP, competitive programming)
        "Run cp file1.txt file2.txt to duplicate the file.",
        "My Vaporeon has 2400 cp in Pokemon Go!",
        "I practice competitive programming (cp) every weekend.",
        "Just cp the code snippet from the documentation.",
        "cp source dest",
        "cp",
    ]
    for sentence in benign_sentences:
        is_toxic, reason = TextNormalizer.check_toxic(sentence)
        assert is_toxic is False, f"False positive triggered on benign sentence: '{sentence}' (reason: {reason})"
        assert reason is None


def test_normalizer_empty_and_invalid_inputs():
    """Empty strings, None, and non-string types return safe defaults."""
    assert TextNormalizer.normalize("") == ""
    assert TextNormalizer.normalize("   ") == ""
    assert TextNormalizer.normalize(None) == ""  # type: ignore

    assert TextNormalizer.check_toxic("") == (False, None)
    assert TextNormalizer.check_toxic("   ") == (False, None)
    assert TextNormalizer.check_toxic(None) == (False, None)  # type: ignore


def test_normalizer_latency_under_1ms():
    """Single normalization check executes well under 1ms (sub-millisecond benchmark)."""
    sample = "Hello there my friend, this is a normal conversational text message checking speed."
    # Warmup
    for _ in range(10):
        TextNormalizer.check_toxic(sample)

    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        TextNormalizer.check_toxic(sample)
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / iterations) * 1000.0
    assert avg_ms < 1.0, f"Average check latency {avg_ms:.4f}ms exceeds 1.0ms limit"


def test_normalizer_redos_resilience():
    """Verify 4000 characters of repeated non-word punctuation evaluates in < 5ms."""
    payload = "k" + ("_" * 2000) + "y" + ("_" * 2000) + "x"
    # Warmup
    TextNormalizer.check_toxic(payload)

    t0 = time.perf_counter()
    is_toxic, reason = TextNormalizer.check_toxic(payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert is_toxic is False
    assert reason is None
    assert elapsed_ms < 5.0, f"ReDoS evaluation took {elapsed_ms:.2f}ms (threshold 5.0ms)"


# ============================================================================
# 3. ModerationPipeline Stage 2 Deep AI Moderation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_stage2_skipped_when_stage1_flags():
    """Stage 2 AI provider is never called if Stage 1 fast-path catches toxic content."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock()

    pipeline = ModerationPipeline()
    res = await pipeline.moderate_text("kys kill yourself", provider=mock_provider)

    assert res.is_flagged is True
    assert res.action == "ban"
    assert res.stage == "stage1_fastpath"
    # Stage 2 should not have been called
    mock_provider.moderate_text.assert_not_called()


@pytest.mark.asyncio
async def test_stage2_called_when_stage1_clean():
    """Stage 2 AI provider is called when Stage 1 is clean."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(return_value={"score": 0.2, "is_flagged": False})

    pipeline = ModerationPipeline(toxicity_threshold=0.8)
    res = await pipeline.moderate_text("This is clean text.", provider=mock_provider)

    assert res.is_flagged is False
    assert res.action == "allow"
    assert res.stage == "stage2_clean"
    mock_provider.moderate_text.assert_called_once_with("This is clean text.")


@pytest.mark.asyncio
async def test_stage2_score_above_threshold_bans():
    """AI provider returning score >= toxicity_threshold results in a ban."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(return_value={"score": 0.85, "reason": "subtle_hate"})

    pipeline = ModerationPipeline(toxicity_threshold=0.8)
    res = await pipeline.moderate_text("Passive aggressive borderline message", provider=mock_provider)

    assert res.is_flagged is True
    assert res.action == "ban"
    assert res.confidence == 0.85
    assert res.reason == "subtle_hate"
    assert res.stage == "stage2_deep"


@pytest.mark.asyncio
async def test_stage2_score_below_threshold_allows():
    """AI provider returning score < toxicity_threshold allows the message."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(return_value=0.45)

    pipeline = ModerationPipeline(toxicity_threshold=0.8)
    res = await pipeline.moderate_text("Slightly spicy but clean text", provider=mock_provider)

    assert res.is_flagged is False
    assert res.action == "allow"
    assert res.stage == "stage2_clean"


@pytest.mark.asyncio
async def test_stage2_sync_provider_support():
    """Pipeline seamlessly supports synchronous provider functions."""
    def sync_moderate(text: str) -> float:
        return 0.95

    pipeline = ModerationPipeline(toxicity_threshold=0.8)
    res = await pipeline.moderate_text("Evaluating with sync provider", provider=sync_moderate)

    assert res.is_flagged is True
    assert res.action == "ban"
    assert res.confidence == 0.95


@pytest.mark.asyncio
async def test_stage2_provider_exception_blocks_without_banning():
    """Provider failures withhold the message without issuing an unjustified user ban."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(side_effect=RuntimeError("AI Provider Service Unavailable"))

    pipeline = ModerationPipeline()
    res = await pipeline.moderate_text("Valid text during AI outage", provider=mock_provider)

    assert res.is_flagged is True
    assert res.action == "block"
    assert res.is_bannable is False
    assert res.stage == "stage2_error_block"


@pytest.mark.asyncio
async def test_stage2_provider_exception_can_fail_open_when_explicitly_enabled(monkeypatch):
    """The fail-open exception is an explicit deployment choice, not the default."""
    from app.config import settings

    monkeypatch.setattr(settings, "MODERATION_FAIL_OPEN", True)
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(side_effect=RuntimeError("AI Provider Service Unavailable"))

    pipeline = ModerationPipeline()
    res = await pipeline.moderate_text("Valid text during AI outage", provider=mock_provider)

    assert res.is_flagged is False
    assert res.action == "allow"
    assert res.stage == "stage2_error_fallback"


@pytest.mark.asyncio
async def test_stage2_provider_score_none_flagged_bans():
    """AI provider returning dict with score=None and is_flagged=True triggers a ban."""
    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(
        return_value={"score": None, "is_flagged": True, "reason": "toxic_intent"}
    )

    pipeline = ModerationPipeline()
    res = await pipeline.moderate_text("Borderline toxic statement", provider=mock_provider)

    assert res.is_flagged is True
    assert res.action == "ban"
    assert res.reason == "toxic_intent"
    assert res.confidence == 1.0
    assert res.stage == "stage2_deep"


@pytest.mark.asyncio
async def test_stage2_provider_object_score_none_flagged_bans():
    """AI provider returning an object with score=None and is_flagged=True triggers a ban."""
    class CustomAIResult:
        score = None
        is_flagged = True
        reason = "object_flagged"

    mock_provider = MagicMock()
    mock_provider.moderate_text = AsyncMock(return_value=CustomAIResult())

    pipeline = ModerationPipeline()
    res = await pipeline.moderate_text("Borderline toxic statement", provider=mock_provider)

    assert res.is_flagged is True
    assert res.action == "ban"
    assert res.reason == "object_flagged"
    assert res.confidence == 1.0


@pytest.mark.asyncio
async def test_stage2_moderation_disabled_setting():
    """When moderation is disabled, even toxic content is allowed."""
    pipeline = ModerationPipeline(enabled=False)
    res = await pipeline.moderate_text("kys kill yourself")
    assert res.is_flagged is False
    assert res.action == "allow"
    assert res.stage == "moderation_disabled"


# ============================================================================
# 4. GIF URL Validator Tests
# ============================================================================

def test_gif_valid_giphy_urls():
    """Approved Giphy URLs on root and subdomains are allowed."""
    valid_urls = [
        "https://giphy.com/gifs/funny-cat-12345",
        "https://media.giphy.com/media/xyz123/giphy.gif",
        "https://i.giphy.com/media/abc/giphy.webp",
        "https://media0.giphy.com/media/cat/200.gif",
    ]
    pipeline = ModerationPipeline()
    for url in valid_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_clean is True, f"Valid Giphy URL blocked: '{url}'"
        assert res.action == "allow"


def test_gif_valid_tenor_urls():
    """Approved Tenor URLs on root and subdomains are allowed."""
    valid_urls = [
        "https://tenor.com/view/happy-dance-gif-9876",
        "https://media.tenor.com/images/12345/tenor.gif",
        "https://c.tenor.com/abc/tenor.gif",
    ]
    pipeline = ModerationPipeline()
    for url in valid_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_clean is True, f"Valid Tenor URL blocked: '{url}'"
        assert res.action == "allow"


def test_gif_insecure_http_rejected():
    """Non-HTTPS URLs are banned with reason='insecure_protocol'."""
    pipeline = ModerationPipeline()
    res = pipeline.moderate_gif_url("http://media.giphy.com/media/xyz/giphy.gif")
    assert res.is_bannable is True
    assert res.reason == "insecure_protocol"


def test_gif_unapproved_domain_rejected():
    """Untrusted image host domains are banned with reason='unapproved_domain'."""
    untrusted_urls = [
        "https://untrusted-image-host.com/cat.gif",
        "https://imgur.com/gallery/xyz",
        "https://evil.com/giphy.gif",
    ]
    pipeline = ModerationPipeline()
    for url in untrusted_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True
        assert res.reason == "unapproved_domain"


def test_gif_domain_spoofing_attempts_rejected():
    """Prefix and suffix domain spoofing tricks are strictly rejected."""
    spoofed_urls = [
        "https://fakegiphy.com/cat.gif",
        "https://notgiphy.com/cat.gif",
        "https://giphy.com.attacker.com/cat.gif",
        "https://tenor.com.evil.org/dog.gif",
        "https://giphycom/cat.gif",
    ]
    pipeline = ModerationPipeline()
    for url in spoofed_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True
        assert res.reason == "unapproved_domain", f"Spoofed domain not blocked: '{url}'"


def test_gif_userinfo_injection_rejected():
    """Userinfo auth injection (e.g., https://giphy.com@evil.com) is rejected."""
    pipeline = ModerationPipeline()
    res = pipeline.moderate_gif_url("https://giphy.com@evil.com/cat.gif")
    assert res.is_bannable is True
    assert res.reason == "unapproved_domain"


def test_gif_userinfo_on_approved_domain_rejected():
    """Userinfo auth credentials embedded in approved domains are rejected."""
    pipeline = ModerationPipeline()
    userinfo_urls = [
        "https://user:pass@giphy.com/cat.gif",
        "https://user@giphy.com/cat.gif",
        "https://:pass@giphy.com/cat.gif",
        "https://admin:secret@media.giphy.com/cat.gif",
        "https://foo:bar@tenor.com/view/cat-123",
        "https://user:pass@c.tenor.com/cat.gif",
    ]
    for url in userinfo_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True, f"Userinfo URL allowed: '{url}'"
        assert res.reason in ("unapproved_domain", "userinfo_injection")


def test_gif_malicious_schemes_rejected():
    """Malicious URL schemes like javascript:, data:, and file: are banned."""
    malicious = [
        "javascript:alert(document.cookie)",
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        "file:///etc/passwd",
        "ftp://media.giphy.com/cat.gif",
    ]
    pipeline = ModerationPipeline()
    for url in malicious:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True
        assert res.reason == "insecure_protocol"


def test_gif_raw_ip_and_localhost_rejected():
    """Raw IP addresses and localhost destinations are rejected."""
    targets = [
        "https://127.0.0.1/cat.gif",
        "https://192.168.1.1/cat.gif",
        "https://[::1]/cat.gif",
        "https://localhost/cat.gif",
        "https://localhost.localdomain/cat.gif",
    ]
    pipeline = ModerationPipeline()
    for url in targets:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True
        assert res.reason == "unapproved_domain"


def test_gif_custom_port_rejected():
    """Custom ports other than standard 443 are rejected."""
    pipeline = ModerationPipeline()
    # Non-standard port
    res_bad = pipeline.moderate_gif_url("https://media.giphy.com:8443/cat.gif")
    assert res_bad.is_bannable is True
    assert res_bad.reason == "unapproved_port"

    # Standard port 443 explicitly provided is accepted
    res_good = pipeline.moderate_gif_url("https://media.giphy.com:443/cat.gif")
    assert res_good.is_clean is True


def test_gif_malformed_out_of_range_port_graceful_rejection():
    """Malformed and out-of-range ports are rejected without raising unhandled ValueError."""
    pipeline = ModerationPipeline()
    malformed_port_urls = [
        "https://media.giphy.com:65536/cat.gif",
        "https://media.giphy.com:999999/cat.gif",
        "https://media.giphy.com:-1/cat.gif",
        "https://media.giphy.com:abc/cat.gif",
        "https://media.giphy.com:443:443/cat.gif",
    ]
    for url in malformed_port_urls:
        res = pipeline.moderate_gif_url(url)
        assert res.is_bannable is True, f"Malformed port URL was not banned: '{url}'"
        assert res.reason == "unapproved_port"


def test_gif_url_length_limits():
    """URLs <= 2048 characters are permitted; URLs > 2048 are rejected."""
    pipeline = ModerationPipeline()

    base = "https://media.giphy.com/media/cat.gif?pad="
    pad_len = 2048 - len(base)
    valid_2048 = base + ("a" * pad_len)
    assert len(valid_2048) == 2048
    assert pipeline.moderate_gif_url(valid_2048).is_clean is True

    too_long = valid_2048 + "b"
    assert len(too_long) == 2049
    res = pipeline.moderate_gif_url(too_long)
    assert res.is_bannable is True
    assert res.reason == "url_too_long"


def test_gif_empty_url_rejected():
    """Empty strings, whitespace, and None are rejected with reason='empty_url'."""
    pipeline = ModerationPipeline()
    for empty_val in ("", "   ", None):
        res = pipeline.moderate_gif_url(empty_val)  # type: ignore
        assert res.is_bannable is True
        assert res.reason == "empty_url"


def test_gif_moderation_disabled():
    """When moderation is disabled, even unapproved domains pass through."""
    pipeline = ModerationPipeline(enabled=False)
    res = pipeline.moderate_gif_url("http://unapproved.com/cat.gif")
    assert res.is_clean is True
    assert res.stage == "moderation_disabled"


# ============================================================================
# 5. BanManager Tests
# ============================================================================

def test_ban_manager_record_and_is_banned():
    """Record an IP ban with duration and verify active status."""
    manager = BanManager()
    manager.clear_all()

    record = manager.record_ban("192.168.1.10", reason="Spamming", duration_seconds=60)
    assert isinstance(record, BanRecord)
    assert record.ip == "192.168.1.10"
    assert record.reason == "Spamming"
    assert manager.is_banned("192.168.1.10") is True
    assert manager.is_banned("192.168.1.11") is False


def test_ban_manager_default_duration():
    """When duration_seconds is omitted, defaults to settings.BAN_DURATION_SECONDS."""
    manager = BanManager()
    record = manager.record_ban("10.0.0.1", reason="Policy violation")
    # Remaining seconds should be approximately settings.BAN_DURATION_SECONDS
    rem = record.remaining_seconds()
    assert 0 < rem <= settings.BAN_DURATION_SECONDS


def test_ban_manager_zero_ttl_instant_expiration():
    """Zero TTL bans expire immediately and is_banned returns False."""
    manager = BanManager()
    manager.record_ban("10.0.0.99", reason="Zero TTL", duration_seconds=0)
    assert manager.is_banned("10.0.0.99") is False
    assert manager.get_ban_record("10.0.0.99") is None


def test_ban_manager_auto_pruning_on_lookup():
    """Expired ban records are auto-pruned from the dictionary when queried."""
    manager = BanManager()
    manager.clear_all()

    # Record zero-ttl ban
    manager.record_ban("10.0.0.2", reason="Auto-prune test", duration_seconds=0)
    assert "10.0.0.2" in manager.bans

    # Lookup triggers auto-prune
    assert manager.is_banned("10.0.0.2") is False
    assert "10.0.0.2" not in manager.bans


def test_ban_manager_get_ban_record():
    """get_ban_record returns BanRecord for active ban, None for non-existent or expired."""
    manager = BanManager()
    manager.clear_all()

    manager.record_ban("172.16.0.5", reason="Abuse", duration_seconds=120)
    rec = manager.get_ban_record("172.16.0.5")
    assert rec is not None
    assert rec.ip == "172.16.0.5"
    assert rec.reason == "Abuse"

    assert manager.get_ban_record("nonexistent_ip") is None


def test_ban_manager_lift_ban():
    """lift_ban removes ban and returns True if existed, False otherwise."""
    manager = BanManager()
    manager.clear_all()

    manager.record_ban("1.2.3.4", reason="Testing", duration_seconds=60)
    assert manager.is_banned("1.2.3.4") is True

    assert manager.lift_ban("1.2.3.4") is True
    assert manager.is_banned("1.2.3.4") is False
    assert manager.lift_ban("1.2.3.4") is False


def test_ban_manager_clear_expired_and_clear_all():
    """clear_expired prunes only expired records; clear_all purges everything."""
    manager = BanManager()
    manager.clear_all()

    manager.record_ban("active.ip", reason="Active", duration_seconds=300)
    manager.record_ban("expired.ip1", reason="Expired 1", duration_seconds=0)
    manager.record_ban("expired.ip2", reason="Expired 2", duration_seconds=0)

    pruned = manager.clear_expired()
    assert pruned == 2
    assert manager.count() == 1
    assert manager.is_banned("active.ip") is True

    manager.clear()
    assert manager.count() == 0
    assert len(manager.bans) == 0
    assert len(manager.banned_ips) == 0


def test_ban_manager_ip_normalization():
    """BanManager normalizes IP addresses with casing and whitespace trimming."""
    manager = BanManager()
    manager.clear_all()

    manager.record_ban("  192.168.1.50  ", reason="Trim test")
    assert manager.is_banned("192.168.1.50") is True
    assert manager.is_banned("  192.168.1.50  ") is True

    # Empty string raises ValueError on record_ban
    with pytest.raises(ValueError):
        manager.record_ban("", reason="Invalid")


@pytest.mark.asyncio
async def test_ban_manager_async_wrappers():
    """Asynchronous wrapper methods work correctly under asyncio.Lock."""
    manager = BanManager()
    manager.clear_all()

    rec = await manager.async_record_ban("10.10.10.10", reason="Async ban", duration_seconds=60)
    assert rec.ip == "10.10.10.10"
    assert await manager.async_is_banned("10.10.10.10") is True

    rec_get = await manager.async_get_ban_record("10.10.10.10")
    assert rec_get is not None
    assert rec_get.ip == "10.10.10.10"

    assert await manager.async_lift_ban("10.10.10.10") is True
    assert await manager.async_is_banned("10.10.10.10") is False

    await manager.async_record_ban("10.10.10.20", reason="Zero TTL", duration_seconds=0)
    pruned = await manager.async_clear_expired()
    assert pruned == 1

    await manager.async_clear_all()
    assert manager.count() == 0


def test_ban_manager_multithreaded_safety():
    """Concurrent threads recording and querying bans operate without race conditions."""
    import concurrent.futures

    manager = BanManager()
    manager.clear_all()

    def worker(idx: int):
        ip = f"192.168.2.{idx % 10}"
        manager.record_ban(ip, reason=f"Thread-{idx}", duration_seconds=30)
        manager.is_banned(ip)
        manager.get_ban_record(ip)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert manager.count() <= 10
    manager.clear_all()


# ============================================================================
# 6. Partner Notification Envelope Tests
# ============================================================================

def test_partner_notification_envelope_default():
    """Default notification envelope produces canonical peer_disconnected with reason='banned'."""
    envelope = create_partner_banned_envelope(session_id="session-xyz-123")
    assert isinstance(envelope, WebSocketEnvelope)
    assert envelope.type == "peer_disconnected"
    assert envelope.session_id == "session-xyz-123"

    payload = envelope.payload
    assert payload.get("reason") == "banned"
    assert payload.get("can_requeue") is True
    assert "violating community guidelines" in payload.get("message", "")


def test_partner_notification_envelope_custom_message():
    """Partner notification supports customized message strings."""
    custom_msg = "Your chat partner has been disconnected due to misconduct."
    envelope = create_partner_banned_envelope(
        session_id="sess-456",
        message=custom_msg,
    )
    assert envelope.payload.get("message") == custom_msg


def test_partner_notification_envelope_type_override():
    """Envelope supports explicit envelope_type and custom reason overrides."""
    envelope = create_partner_banned_envelope(
        session_id="sess-789",
        reason="peer_banned",
        envelope_type="partner_banned",
    )
    assert envelope.type == "partner_banned"
    assert envelope.payload.get("reason") == "peer_banned"


def test_partner_notification_envelope_json_serialization():
    """Notification envelope serializes to valid JSON matching transport schema."""
    envelope = create_partner_banned_envelope(session_id="sess-abc")
    json_str = envelope.to_json()
    assert '"type":"peer_disconnected"' in json_str or '"type": "peer_disconnected"' in json_str
    assert '"reason":"banned"' in json_str or '"reason": "banned"' in json_str


# ============================================================================
# 7. Concurrency & Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_text_moderation():
    """50 concurrent text moderation calls process safely and return correct classifications."""
    pipeline = ModerationPipeline()

    clean_text = "Good morning everyone!"
    toxic_text = "kys kill yourself"

    tasks = []
    for i in range(50):
        t = clean_text if i % 2 == 0 else toxic_text
        tasks.append(pipeline.moderate_text(t))

    results = await asyncio.gather(*tasks)
    assert len(results) == 50

    for i, res in enumerate(results):
        if i % 2 == 0:
            assert res.is_clean is True
            assert res.action == "allow"
        else:
            assert res.is_bannable is True
            assert res.action == "ban"


def test_concurrent_gif_moderation():
    """50 concurrent GIF URL moderation calls run without contention or errors."""
    import concurrent.futures

    pipeline = ModerationPipeline()
    valid_url = "https://media.giphy.com/media/xyz/cat.gif"
    invalid_url = "https://unapproved.com/dog.gif"

    def moderate_call(idx: int):
        target = valid_url if idx % 2 == 0 else invalid_url
        return pipeline.moderate_gif_url(target)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(moderate_call, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 50


# ============================================================================
# 8. Module Exports & Convenience Wrappers
# ============================================================================

def test_module_exports_and_singletons():
    """Top-level module exports and singleton aliases are present and consistent."""
    import app.moderation as mod

    assert hasattr(mod, "TextNormalizer")
    assert hasattr(mod, "text_normalizer")
    assert hasattr(mod, "normalizer")
    assert mod.normalizer is mod.text_normalizer

    assert hasattr(mod, "BanManager")
    assert hasattr(mod, "ban_manager")

    assert hasattr(mod, "ModerationPipeline")
    assert hasattr(mod, "moderation_pipeline")

    assert hasattr(mod, "ModerationResult")
    assert hasattr(mod, "moderate_text")
    assert hasattr(mod, "moderate_gif_url")
    assert hasattr(mod, "create_partner_banned_envelope")


@pytest.mark.asyncio
async def test_convenience_functions():
    """Top-level convenience functions route to singleton pipeline."""
    # moderate_text
    res_text = await moderate_text("Hello clean world!")
    assert res_text.is_clean is True

    # moderate_gif_url
    res_gif = moderate_gif_url("https://media.giphy.com/media/abc/giphy.gif")
    assert res_gif.is_clean is True

    # create_partner_banned_envelope
    env = create_partner_banned_envelope(session_id="test-session")
    assert env.type == "peer_disconnected"
    assert env.payload["reason"] == "banned"
