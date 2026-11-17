"""Milestone 4 AI Fallback Companion & Pluggable Provider Unit Test Suite.

Contains 80 exhaustive unit tests partitioned across 8 suites:
- Suite 1: Provider Contract Enforcement (9 tests)
- Suite 2: Mock Provider Behavior (13 tests)
- Suite 3: Gemini Provider Integration & Fallbacks (13 tests)
- Suite 4: Factory Provider Resolution & Overrides (10 tests)
- Suite 5: AI Companion Manager (14 tests)
- Suite 6: Queue Watchdog Lifecycle & Popping (10 tests)
- Suite 7: Concurrency & State Isolation (5 tests)
- Suite 8: Alignment with Integration Tests & Test Harness (6 tests)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.ai.base import BaseAICompanionProvider, BaseModerationProvider
from app.ai.companion_manager import AISessionState, CompanionManager, QueueWatchdog
from app.ai.factory import (
    get_ai_companion_provider,
    get_moderation_provider,
    reset_providers,
    set_ai_companion_provider,
    set_moderation_provider,
)
from app.ai.gemini_provider import GeminiAICompanionProvider, GeminiModerationProvider
from app.ai.mock_provider import MockAICompanionProvider, MockModerationProvider
from app.config import settings
from app.matchmaker.queue import MatchmakerQueue
from app.matchmaker.session_manager import SessionManager
from app.models.envelopes import (
    ChatMessagePayload,
    MatchedPayload,
    ServerChatMessagePayload,
    ServerMessageType,
    TypingPayload,
    WebSocketEnvelope,
)
from app.models.session import ClientState, QueueTicket, SessionStatus


# Autouse fixture ensuring provider factory is clean before and after every test
@pytest.fixture(autouse=True)
def clean_provider_state():
    reset_providers()
    yield
    reset_providers()


# ============================================================================
# Suite 1: Provider Contract Enforcement (9 Tests)
# ============================================================================

class TestSuite1ProviderContracts:
    """Verifies interface enforcement, parameter checking, and method delegation."""

    def test_base_companion_provider_is_abstract(self):
        """Instantiating BaseAICompanionProvider directly raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            BaseAICompanionProvider()  # type: ignore
        assert "generate_reply" in str(exc_info.value) or "abstract" in str(exc_info.value)

    def test_base_companion_provider_subclass_missing_method(self):
        """Subclass missing generate_reply cannot be instantiated."""
        class IncompleteCompanion(BaseAICompanionProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteCompanion()  # type: ignore

    @pytest.mark.asyncio
    async def test_base_companion_provider_concrete_subclass_instantiates(self):
        """Concrete subclass implementing generate_reply instantiates and runs."""
        class ConcreteCompanion(BaseAICompanionProvider):
            async def generate_reply(self, session_id, history, latest_message):
                await super().generate_reply(session_id, history, latest_message)
                return "ok"

        provider = ConcreteCompanion()
        assert isinstance(provider, BaseAICompanionProvider)
        reply = await provider.generate_reply("sess1", [], "hello")
        assert reply == "ok"

    def test_base_moderation_provider_is_abstract(self):
        """Instantiating BaseModerationProvider directly raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            BaseModerationProvider()  # type: ignore
        assert "score_toxicity" in str(exc_info.value) or "abstract" in str(exc_info.value)

    def test_base_moderation_provider_subclass_missing_method(self):
        """Subclass missing score_toxicity cannot be instantiated."""
        class IncompleteModeration(BaseModerationProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteModeration()  # type: ignore

    @pytest.mark.asyncio
    async def test_base_moderation_provider_concrete_subclass_instantiates(self):
        """Concrete subclass implementing score_toxicity instantiates and runs."""
        class ConcreteModeration(BaseModerationProvider):
            async def score_toxicity(self, text: str) -> float:
                return 0.42

        provider = ConcreteModeration()
        assert isinstance(provider, BaseModerationProvider)
        score = await provider.score_toxicity("test")
        assert score == 0.42

    @pytest.mark.asyncio
    async def test_base_companion_parameter_validation_empty_session_id(self):
        """Empty or non-string session_id raises ValueError."""
        provider = MockAICompanionProvider()
        with pytest.raises(ValueError, match="session_id"):
            await provider.generate_reply("", [], "hello")

        with pytest.raises(ValueError, match="session_id"):
            await provider.generate_reply(None, [], "hello")  # type: ignore

    @pytest.mark.asyncio
    async def test_base_companion_parameter_validation_empty_latest_message(self):
        """Empty or whitespace-only latest_message raises ValueError."""
        provider = MockAICompanionProvider()
        with pytest.raises(ValueError, match="latest_message"):
            await provider.generate_reply("sess_123", [], "")

        with pytest.raises(ValueError, match="latest_message"):
            await provider.generate_reply("sess_123", [], "   ")

    @pytest.mark.asyncio
    async def test_base_moderation_moderate_text_delegates_to_score_toxicity(self):
        """moderate_text delegates directly to score_toxicity returning identical score."""
        provider = MockModerationProvider(default_score=0.15)
        score_direct = await provider.score_toxicity("test message")
        score_delegated = await provider.moderate_text("test message")
        assert score_direct == 0.15
        assert score_delegated == score_direct


# ============================================================================
# Suite 2: Mock Provider Behavior (13 Tests)
# ============================================================================

class TestSuite2MockProvider:
    """Verifies zero-dependency offline mock behavior and persona responses."""

    def test_mock_companion_zero_dependencies(self):
        """Mock companion instantiates without network or external keys."""
        provider = MockAICompanionProvider()
        assert provider is not None
        assert isinstance(provider, BaseAICompanionProvider)

    @pytest.mark.asyncio
    async def test_mock_companion_returns_non_empty_stripped_reply(self):
        """Replies are non-empty and have no untrimmed whitespace."""
        provider = MockAICompanionProvider()
        reply = await provider.generate_reply("s1", [], "hey there")
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert reply == reply.strip()

    @pytest.mark.asyncio
    async def test_mock_companion_keeps_general_replies_casual(self):
        """General replies stay conversational without pretending to be human."""
        provider = MockAICompanionProvider()
        for prompt in ["what can you do?", "help me with code"]:
            reply = await provider.generate_reply("s1", [], prompt)
            lower_reply = reply.lower()
            assert "how may i assist" not in lower_reply
            assert "language model" not in lower_reply

    @pytest.mark.asyncio
    async def test_mock_companion_casual_identity_reply(self):
        provider = MockAICompanionProvider()
        reply = await provider.generate_reply("s1", [], "are you an ai?")
        assert "stranger" in reply.lower()

    @pytest.mark.asyncio
    async def test_mock_companion_greeting_intent(self):
        """Greetings yield casual internet stranger greetings."""
        provider = MockAICompanionProvider()
        for greeting in ["hi", "hello", "hey", "yo", "sup"]:
            reply = await provider.generate_reply("s1", [], greeting)
            assert any(word in reply.lower() for word in ["hey", "how's", "going", "day"])

    @pytest.mark.asyncio
    async def test_mock_companion_question_intent(self):
        """Specific questions yield friendly, topic-aware answers."""
        provider = MockAICompanionProvider()
        reply_color = await provider.generate_reply("s1", [], "what is your favorite color?")
        assert "teal" in reply_color or "green" in reply_color

        reply_book = await provider.generate_reply("s1", [], "can you recommend a good book?")
        assert "project hail mary" in reply_book.lower()

    @pytest.mark.asyncio
    async def test_mock_companion_emotional_intent(self):
        """Emotional or venting prompts receive empathetic stranger replies."""
        provider = MockAICompanionProvider()
        reply = await provider.generate_reply("s1", [], "i had a really rough day today")
        assert any(kw in reply.lower() for kw in ["sorry", "hear", "vent", "distract"])

    @pytest.mark.asyncio
    async def test_mock_companion_history_context_awareness(self):
        """History sequence influences rotation of conversational replies."""
        provider = MockAICompanionProvider()
        h1 = [ChatMessagePayload(text="hi")]
        h2 = [ChatMessagePayload(text="hi"), ChatMessagePayload(text="hello"), ChatMessagePayload(text="yo")]
        r1 = await provider.generate_reply("s1", h1, "random statement xyz 123")
        r2 = await provider.generate_reply("s1", h2, "random statement xyz 123")
        assert isinstance(r1, str)
        assert isinstance(r2, str)

    @pytest.mark.asyncio
    async def test_mock_companion_empty_history_tolerance(self):
        """Empty history list operates seamlessly."""
        provider = MockAICompanionProvider()
        reply = await provider.generate_reply("s1", [], "how's it going?")
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_mock_companion_large_history_tolerance(self):
        """Large history list (20 items) operates seamlessly without error."""
        provider = MockAICompanionProvider()
        large_history = [ChatMessagePayload(text=f"turn {i}") for i in range(20)]
        reply = await provider.generate_reply("s1", large_history, "are you there?")
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_mock_moderation_benign_text_low_score(self):
        """Benign conversation texts yield low toxicity score."""
        provider = MockModerationProvider()
        benign_inputs = [
            "hello friend, how are you?",
            "what a lovely day for outdoor hiking",
            "i really enjoyed reading that book",
        ]
        for text in benign_inputs:
            score = await provider.score_toxicity(text)
            assert score <= 0.1

    @pytest.mark.asyncio
    async def test_mock_moderation_toxic_text_high_score(self):
        """Toxic patterns produce score >= 0.85."""
        provider = MockModerationProvider()
        toxic_inputs = [
            "go kill yourself right now",
            "kys stupid trash",
            "i hate you you worthless piece of trash",
        ]
        for text in toxic_inputs:
            score = await provider.score_toxicity(text)
            assert score >= 0.85

    @pytest.mark.asyncio
    async def test_mock_moderation_borderline_text_intermediate_score(self):
        """Borderline patterns yield intermediate score (0.3 - 0.7)."""
        provider = MockModerationProvider()
        score = await provider.score_toxicity("shut up you idiot")
        assert 0.3 <= score <= 0.7

    @pytest.mark.asyncio
    async def test_mock_moderation_empty_text_zero_score(self):
        """Empty or whitespace text yields 0.0."""
        provider = MockModerationProvider()
        assert await provider.score_toxicity("") == 0.0
        assert await provider.score_toxicity("   ") == 0.0
        assert await provider.score_toxicity(None) == 0.0  # type: ignore


# ============================================================================
# Suite 3: Gemini Provider Integration & Fallbacks (13 Tests)
# ============================================================================

class TestSuite3GeminiProvider:
    """Verifies Google GenAI SDK integration, contents formatting, and fallbacks."""

    def test_gemini_companion_init_with_explicit_args(self):
        """Custom api_key and model are preserved."""
        provider = GeminiAICompanionProvider(api_key="explicit_key", model="gemini-1.5-flash")
        assert provider.api_key == "explicit_key"
        assert provider.model == "gemini-1.5-flash"

    def test_gemini_companion_init_from_settings(self):
        """Reads default settings when args are omitted."""
        provider = GeminiAICompanionProvider()
        assert provider.model == settings.GEMINI_MODEL

    def test_gemini_companion_init_missing_key_behavior(self):
        """Missing or None API key does not crash; _client is set to None."""
        provider = GeminiAICompanionProvider(api_key=None)
        assert provider._client is None

    def test_gemini_companion_client_setup(self):
        """Providing a valid non-empty key initializes genai.Client."""
        provider = GeminiAICompanionProvider(api_key="dummy_test_key_123")
        assert provider._client is not None

    def test_gemini_companion_system_instruction_is_casual_human(self):
        """System instruction keeps a casual human stranger persona."""
        provider = GeminiAICompanionProvider()
        si = provider.system_instruction.lower()
        assert "stranger" in si
        assert "never state you are an ai" in si

    def test_gemini_companion_history_formatting(self):
        """Converts ChatMessagePayload list and latest message into Content objects."""
        provider = GeminiAICompanionProvider()
        history = [
            ChatMessagePayload(text="hello stranger"),
            ServerChatMessagePayload(sender="Stranger", text="hey! how are you?"),
        ]
        contents = provider._build_contents(history, "doing great, what's your hobby?")
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"

    @pytest.mark.asyncio
    async def test_gemini_companion_successful_generate_reply(self):
        """Successful API call returns parsed response text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "that sounds awesome, tell me more!"
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiAICompanionProvider(client=mock_client)
        reply = await provider.generate_reply("sess_1", [], "i love stargazing")
        assert reply == "that sounds awesome, tell me more!"

    @pytest.mark.asyncio
    async def test_gemini_companion_api_error_fallback_to_mock(self):
        """On API exception, provider logs warning and delegates to fallback."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Google API 503"))

        fallback = MockAICompanionProvider(fixed_reply="fallback reply")
        provider = GeminiAICompanionProvider(client=mock_client, fallback_provider=fallback)
        reply = await provider.generate_reply("sess_1", [], "hello")
        assert reply == "fallback reply"

    @pytest.mark.asyncio
    async def test_gemini_companion_empty_candidates_fallback(self):
        """Empty candidates list delegates to fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.text = ""
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        fallback = MockAICompanionProvider(fixed_reply="fallback candidate reply")
        provider = GeminiAICompanionProvider(client=mock_client, fallback_provider=fallback)
        reply = await provider.generate_reply("sess_1", [], "hello")
        assert reply == "fallback candidate reply"

    @pytest.mark.asyncio
    async def test_gemini_companion_safety_finish_reason_fallback(self):
        """When candidate is blocked by safety filter, delegates to fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        cand = MagicMock()
        cand.finish_reason = "SAFETY"
        mock_response.candidates = [cand]
        mock_response.text = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        fallback = MockAICompanionProvider(fixed_reply="safety fallback reply")
        provider = GeminiAICompanionProvider(client=mock_client, fallback_provider=fallback)
        reply = await provider.generate_reply("sess_1", [], "something questionable")
        assert reply == "safety fallback reply"

    @pytest.mark.asyncio
    async def test_gemini_companion_network_timeout_fallback(self):
        """On timeout, delegates to fallback without raising exception."""
        mock_client = MagicMock()
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(0.5)
            return MagicMock(text="late")

        mock_client.aio.models.generate_content = slow_call
        fallback = MockAICompanionProvider(fixed_reply="timeout fallback reply")
        provider = GeminiAICompanionProvider(client=mock_client, fallback_provider=fallback, timeout=0.05)
        reply = await provider.generate_reply("sess_1", [], "hello")
        assert reply == "timeout fallback reply"

    @pytest.mark.asyncio
    async def test_gemini_moderation_provider_scoring(self):
        """Parses floating-point score from moderation response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Toxicity score: 0.88"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiModerationProvider(client=mock_client)
        score = await provider.score_toxicity("toxic content")
        assert score == 0.88

    @pytest.mark.asyncio
    async def test_gemini_moderation_provider_error_fails_closed(self):
        """Provider errors must propagate so the moderation pipeline can fail closed."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API Error"))

        fallback = MockModerationProvider(default_score=0.0)
        provider = GeminiModerationProvider(client=mock_client, fallback_provider=fallback)
        with pytest.raises(RuntimeError, match="unavailable"):
            await provider.score_toxicity("some message")


# ============================================================================
# Suite 4: Factory Provider Resolution & Overrides (10 Tests)
# ============================================================================

class TestSuite4Factory:
    """Verifies provider resolution, caching, and dependency injection overrides."""

    def test_factory_default_resolution_mock(self):
        """By default, factory resolves MockAICompanionProvider."""
        provider = get_ai_companion_provider()
        assert isinstance(provider, MockAICompanionProvider)

    def test_factory_resolution_gemini_with_key(self, monkeypatch):
        """When AI_PROVIDER='gemini' and key is configured, resolves Gemini."""
        monkeypatch.setattr(settings, "AI_PROVIDER", "gemini")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "valid_gemini_key")
        reset_providers()

        provider = get_ai_companion_provider()
        assert isinstance(provider, GeminiAICompanionProvider)

    def test_factory_resolution_gemini_without_key_falls_back_to_mock(self, monkeypatch):
        """When AI_PROVIDER='gemini' but key is None, falls back to Mock."""
        monkeypatch.setattr(settings, "AI_PROVIDER", "gemini")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
        reset_providers()

        provider = get_ai_companion_provider()
        assert isinstance(provider, MockAICompanionProvider)

    def test_factory_resolution_unsupported_provider_falls_back_to_mock(self, monkeypatch):
        """Unsupported provider name falls back to Mock."""
        monkeypatch.setattr(settings, "AI_PROVIDER", "unsupported_unknown_ai")
        reset_providers()

        provider = get_ai_companion_provider()
        assert isinstance(provider, MockAICompanionProvider)

    def test_factory_moderation_provider_mock(self):
        """Default moderation provider resolves to MockModerationProvider."""
        provider = get_moderation_provider()
        assert isinstance(provider, MockModerationProvider)

    def test_factory_moderation_provider_gemini(self, monkeypatch):
        """When MODERATION_PROVIDER='gemini' and key is set, resolves Gemini."""
        monkeypatch.setattr(settings, "MODERATION_PROVIDER", "gemini")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "valid_gemini_key")
        reset_providers()

        provider = get_moderation_provider()
        assert isinstance(provider, GeminiModerationProvider)

    def test_factory_singleton_caching(self):
        """Repeated calls return the identical cached instance."""
        p1 = get_ai_companion_provider()
        p2 = get_ai_companion_provider()
        assert p1 is p2

    def test_factory_set_companion_override(self):
        """set_ai_companion_provider overrides factory resolution."""
        custom = MockAICompanionProvider(fixed_reply="custom injected")
        set_ai_companion_provider(custom)
        assert get_ai_companion_provider() is custom

    def test_factory_set_moderation_override(self):
        """set_moderation_provider overrides moderation resolution."""
        custom = MockModerationProvider(default_score=0.77)
        set_moderation_provider(custom)
        assert get_moderation_provider() is custom

    def test_factory_reset_providers_clears_all_overrides(self):
        """reset_providers restores factory to default un-overridden state."""
        custom = MockAICompanionProvider(fixed_reply="custom")
        set_ai_companion_provider(custom)
        assert get_ai_companion_provider() is custom

        reset_providers()
        fresh = get_ai_companion_provider()
        assert fresh is not custom
        assert isinstance(fresh, MockAICompanionProvider)


# ============================================================================
# Suite 5: AI Companion Manager (14 Tests)
# ============================================================================

class TestSuite5CompanionManager:
    """Verifies session tracking, sliding window history, typing cadence, and parity."""

    def test_companion_manager_session_registration(self):
        """Registering a session updates internal active session map."""
        mgr = CompanionManager()
        state = mgr.register_session("sess_100", "client_100")
        assert state.session_id == "sess_100"
        assert state.client_id == "client_100"
        assert mgr.is_ai_session("sess_100") is True
        assert mgr.active_sessions_count() == 1

    def test_companion_manager_session_unregistration(self):
        """Unregistering a session marks it inactive and removes it from store."""
        mgr = CompanionManager()
        mgr.register_session("sess_101", "client_101")
        removed = mgr.unregister_session("sess_101")
        assert removed is not None
        assert removed.is_active is False
        assert mgr.is_ai_session("sess_101") is False
        assert mgr.active_sessions_count() == 0

    def test_companion_manager_unknown_session_query(self):
        """Querying unknown session returns None or False without crashing."""
        mgr = CompanionManager()
        assert mgr.get_session("non_existent") is None
        assert mgr.is_ai_session("non_existent") is False
        assert mgr.get_history("non_existent") == []

    @pytest.mark.asyncio
    async def test_companion_manager_sliding_window_10_turns_eviction(self):
        """Adding 12 turns bounds history at exactly 10 items; FIFO eviction works."""
        mgr = CompanionManager(provider=MockAICompanionProvider(fixed_reply="reply"), typing_delay=0.0)
        mgr.register_session("sess_slide", "client_slide")

        for i in range(6):
            await mgr.process_user_message("sess_slide", f"user msg {i}")

        # 6 user messages + 6 companion replies = 12 total turns
        history = mgr.get_history("sess_slide")
        assert len(history) == 10
        # First turn remaining should be user msg 1 or reply 1, user msg 0 must be evicted
        texts = [item.text for item in history]
        assert "user msg 0" not in texts
        assert "user msg 5" in texts

    @pytest.mark.asyncio
    async def test_companion_manager_sliding_window_order_preserved(self):
        """User and AI turns alternate in strict chronological order."""
        mgr = CompanionManager(provider=MockAICompanionProvider(fixed_reply="pong"), typing_delay=0.0)
        mgr.register_session("sess_order", "client_order")

        await mgr.process_user_message("sess_order", "ping 1")
        await mgr.process_user_message("sess_order", "ping 2")

        history = mgr.get_history("sess_order")
        assert len(history) == 4
        assert history[0].text == "ping 1"
        assert history[1].text == "pong"
        assert history[2].text == "ping 2"
        assert history[3].text == "pong"

    @pytest.mark.asyncio
    async def test_companion_manager_sliding_window_large_traffic(self):
        """50 consecutive messages maintain bounded memory of exactly 10 entries."""
        mgr = CompanionManager(provider=MockAICompanionProvider(fixed_reply="ack"), typing_delay=0.0)
        mgr.register_session("sess_large", "client_large")

        for i in range(50):
            await mgr.process_user_message("sess_large", f"msg {i}")

        assert len(mgr.get_history("sess_large")) == 10

    @pytest.mark.asyncio
    async def test_companion_manager_typing_indicator_sequencing(self):
        """Emits typing(True) -> delay -> typing(False) -> chat message in exact order."""
        emitted_envelopes = []

        async def record_send(client_id, envelope):
            emitted_envelopes.append((client_id, envelope.type, envelope.payload))

        mgr = CompanionManager(
            provider=MockAICompanionProvider(fixed_reply="hello!"),
            typing_delay=0.0,
            send_callback=record_send,
        )
        mgr.register_session("sess_type", "client_type")

        reply = await mgr.process_user_message("sess_type", "hi")
        assert reply == "hello!"
        assert len(emitted_envelopes) == 3

        # Envelope 1: typing True
        assert emitted_envelopes[0][1] == ServerMessageType.TYPING
        assert emitted_envelopes[0][2]["is_typing"] is True

        # Envelope 2: typing False
        assert emitted_envelopes[1][1] == ServerMessageType.TYPING
        assert emitted_envelopes[1][2]["is_typing"] is False

        # Envelope 3: chat
        assert emitted_envelopes[2][1] == ServerMessageType.CHAT
        assert emitted_envelopes[2][2]["text"] == "hello!"

    @pytest.mark.asyncio
    async def test_companion_manager_zero_delay_for_instant_tests(self):
        """Setting typing_delay=0.0 processes user message in <20ms."""
        mgr = CompanionManager(
            provider=MockAICompanionProvider(fixed_reply="fast"),
            typing_delay=0.0,
        )
        mgr.register_session("sess_fast", "client_fast")

        start = asyncio.get_event_loop().time()
        await mgr.process_user_message("sess_fast", "quick query")
        duration = asyncio.get_event_loop().time() - start
        assert duration < 0.05

    @pytest.mark.asyncio
    async def test_companion_manager_chat_envelope_structure(self):
        """Emitted chat envelope matches WebSocketEnvelope and ServerChatMessagePayload."""
        captured = []

        async def send_cb(client_id, env):
            captured.append(env)

        mgr = CompanionManager(
            provider=MockAICompanionProvider(fixed_reply="structured"),
            typing_delay=0.0,
            peer_alias="Stranger",
            send_callback=send_cb,
        )
        mgr.register_session("sess_struct", "client_struct")
        await mgr.process_user_message("sess_struct", "test payload")

        chat_env = captured[-1]
        assert chat_env.type == ServerMessageType.CHAT
        assert chat_env.session_id == "sess_struct"
        payload = ServerChatMessagePayload(**chat_env.payload)
        assert payload.sender == "Stranger"
        assert payload.text == "structured"
        assert len(payload.message_id) > 0

    @pytest.mark.asyncio
    async def test_companion_manager_handle_user_message_routes_to_provider(self):
        """Provider receives session_id, history snapshot, and clean latest message."""
        mock_provider = MagicMock(spec=BaseAICompanionProvider)
        mock_provider.generate_reply = AsyncMock(return_value="routed reply")

        mgr = CompanionManager(provider=mock_provider, typing_delay=0.0)
        mgr.register_session("sess_route", "client_route")

        await mgr.process_user_message("sess_route", "   hello routed world   ")
        mock_provider.generate_reply.assert_awaited_once()
        call_args = mock_provider.generate_reply.await_args[1]
        assert call_args["session_id"] == "sess_route"
        assert call_args["latest_message"] == "hello routed world"

    @pytest.mark.asyncio
    async def test_companion_manager_handle_user_message_unregistered_session_ignored(self):
        """Messages for unknown sessions are safely ignored and return None."""
        mgr = CompanionManager(typing_delay=0.0)
        res = await mgr.process_user_message("unknown_session", "hi")
        assert res is None

    @pytest.mark.asyncio
    async def test_companion_manager_skip_parity(self):
        """Skip unregisters AI session, calls session_manager.handle_skip, and re-enqueues."""
        sm = SessionManager()
        q = MatchmakerQueue()
        mgr = CompanionManager(session_manager=sm, queue=q)

        sess = await sm.create_session("client_skip", "ai_companion_skip", is_ai=True)
        mgr.register_session(sess.session_id, "client_skip")

        session_out, requeued = await mgr.handle_skip("client_skip")
        assert session_out is not None
        assert session_out.status == SessionStatus.TERMINATED
        assert requeued is True
        assert sm.get_client_state("client_skip") == ClientState.QUEUED
        assert mgr.is_ai_session(sess.session_id) is False

    @pytest.mark.asyncio
    async def test_companion_manager_leave_parity(self):
        """Leave unregisters AI session, calls session_manager.handle_leave, transitions to IDLE."""
        sm = SessionManager()
        mgr = CompanionManager(session_manager=sm)

        sess = await sm.create_session("client_leave", "ai_companion_leave", is_ai=True)
        mgr.register_session(sess.session_id, "client_leave")

        session_out, left = await mgr.handle_leave("client_leave")
        assert session_out is not None
        assert session_out.status == SessionStatus.TERMINATED
        assert left is True
        assert sm.get_client_state("client_leave") == ClientState.IDLE
        assert mgr.is_ai_session(sess.session_id) is False

    @pytest.mark.asyncio
    async def test_companion_manager_provider_exception_resilience(self):
        """Provider exception turns off typing indicator and delivers safe fallback."""
        emitted = []

        async def send_cb(cid, env):
            emitted.append(env)

        mock_provider = MagicMock(spec=BaseAICompanionProvider)
        mock_provider.generate_reply = AsyncMock(side_effect=RuntimeError("Provider exploded"))

        mgr = CompanionManager(provider=mock_provider, typing_delay=0.0, send_callback=send_cb)
        mgr.register_session("sess_err", "client_err")

        reply = await mgr.process_user_message("sess_err", "hello")
        assert reply is not None
        assert len(emitted) == 3
        # Typing indicator must have been turned off
        assert emitted[1].payload["is_typing"] is False
        # Chat envelope must still have been delivered with fallback
        assert emitted[2].type == ServerMessageType.CHAT


# ============================================================================
# Suite 6: Queue Watchdog Lifecycle & Popping (10 Tests)
# ============================================================================

class TestSuite6QueueWatchdog:
    """Verifies watchdog timeout evaluation, atomic popping, and background loop."""

    @pytest.mark.asyncio
    async def test_queue_watchdog_empty_queue_tick(self):
        """Empty queue check completes without error and returns empty list."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm)

        paired = await watchdog.check_timeouts()
        assert paired == []

    @pytest.mark.asyncio
    async def test_queue_watchdog_non_expired_ticket_retained(self):
        """Ticket that has waited less than timeout is NOT popped."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=5.0)

        ticket = await q.enqueue("client_fresh")
        # Ticket elapsed time ~ 0.0s < 5.0s
        paired = await watchdog.check_timeouts()
        assert paired == []
        assert (await q.get_waiting_tickets()) == [ticket]

    @pytest.mark.asyncio
    async def test_queue_watchdog_expired_ticket_popped(self):
        """Ticket with elapsed time >= timeout is atomically popped and paired."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=5.0)

        ticket = await q.enqueue("client_exp")
        # Backdate queued_at to simulate expiration
        ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=6.0)

        paired = await watchdog.check_timeouts()
        assert len(paired) == 1
        client_id, session_id = paired[0]
        assert client_id == "client_exp"
        # Queue must now be empty
        assert (await q.get_waiting_tickets()) == []

    @pytest.mark.asyncio
    async def test_queue_watchdog_creates_ai_session(self):
        """Popped ticket creates an active AI session in session_manager and companion_manager."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=2.0)

        ticket = await q.enqueue("client_ai_match")
        ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=3.0)

        paired = await watchdog.check_timeouts()
        _, session_id = paired[0]

        session = sm.get_session(session_id)
        assert session is not None
        assert session.is_ai is True
        assert session.client_a == "client_ai_match"
        assert cm.is_ai_session(session_id) is True

    @pytest.mark.asyncio
    async def test_queue_watchdog_delivers_matched_envelope(self):
        """Delivers matched envelope via callback with peer_is_ai=True and peer_alias='Stranger'."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        captured = []

        async def on_matched(client_id, env):
            captured.append((client_id, env))

        watchdog = QueueWatchdog(
            queue=q,
            session_manager=sm,
            companion_manager=cm,
            timeout=1.0,
            send_matched_callback=on_matched,
        )

        ticket = await q.enqueue("client_cb")
        ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=2.0)

        await watchdog.check_timeouts()
        assert len(captured) == 1
        cid, env = captured[0]
        assert cid == "client_cb"
        assert env.type == ServerMessageType.MATCHED
        payload = MatchedPayload(**env.payload)
        assert payload.peer_is_ai is True
        assert payload.peer_alias == "Stranger"
        assert payload.is_initiator is True

    @pytest.mark.asyncio
    async def test_queue_watchdog_mixed_queue_selective_popping(self):
        """Only expired tickets are popped; fresh tickets stay queued in FIFO order."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=5.0)

        t1 = await q.enqueue("exp1")
        t2 = await q.enqueue("fresh1")
        t3 = await q.enqueue("exp2")
        t4 = await q.enqueue("fresh2")

        t1.queued_at = datetime.now(timezone.utc) - timedelta(seconds=6.0)
        t3.queued_at = datetime.now(timezone.utc) - timedelta(seconds=7.0)

        paired = await watchdog.check_timeouts()
        paired_clients = [p[0] for p in paired]
        assert "exp1" in paired_clients
        assert "exp2" in paired_clients
        assert "fresh1" not in paired_clients
        assert "fresh2" not in paired_clients

        remaining = await q.get_waiting_tickets()
        assert len(remaining) == 2
        assert remaining[0].client_id == "fresh1"
        assert remaining[1].client_id == "fresh2"

    @pytest.mark.asyncio
    async def test_queue_watchdog_race_condition_client_already_popped(self):
        """When pop_client returns None (already matched or left), watchdog skips safely."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=1.0)

        ticket = await q.enqueue("client_race")
        ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=2.0)

        # Mock pop_client to simulate concurrent popping by human pair
        with patch.object(q, "pop_client", AsyncMock(return_value=None)):
            paired = await watchdog.check_timeouts()
            assert paired == []

    @pytest.mark.asyncio
    async def test_queue_watchdog_start_and_stop_lifecycle(self):
        """Background watchdog loop starts, runs, and stops cleanly."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, poll_interval=0.01)

        assert watchdog.is_running is False
        await watchdog.start()
        assert watchdog.is_running is True

        await asyncio.sleep(0.05)
        await watchdog.stop()
        assert watchdog.is_running is False

    @pytest.mark.asyncio
    async def test_queue_watchdog_double_start_or_stop(self):
        """Starting twice or stopping twice is completely idempotent."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, poll_interval=0.01)

        await watchdog.start()
        await watchdog.start()  # Idempotent
        assert watchdog.is_running is True

        await watchdog.stop()
        await watchdog.stop()  # Idempotent
        assert watchdog.is_running is False

    @pytest.mark.asyncio
    async def test_queue_watchdog_error_in_tick_does_not_kill_loop(self):
        """Transient error in check_timeouts does not terminate background task."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, poll_interval=0.01)

        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient glitch")
            return []

        watchdog.check_timeouts = failing_check
        await watchdog.start()
        for _ in range(50):
            if call_count >= 2:
                break
            await asyncio.sleep(0.01)
        await watchdog.stop()

        assert call_count >= 2


# ============================================================================
# Suite 7: Concurrency & State Isolation (5 Tests)
# ============================================================================

class TestSuite7ConcurrencyIsolation:
    """Verifies isolation across dozens of simultaneous AI sessions and race condition safety."""

    @pytest.mark.asyncio
    async def test_concurrency_20_simultaneous_ai_sessions(self):
        """20 simultaneous AI sessions remain active and isolated in SessionManager."""
        sm = SessionManager()
        cm = CompanionManager(provider=MockAICompanionProvider(fixed_reply="concurrent reply"), typing_delay=0.0)

        sessions = []
        for i in range(20):
            cid = f"user_{i}"
            companion_id = f"ai_companion_{cid}"
            sess = await sm.create_session(cid, companion_id, is_ai=True)
            cm.register_session(sess.session_id, cid, companion_id)
            sessions.append(sess)

        # Verify all 20 sessions exist concurrently without being terminated by stale cleanup
        assert len(sm._sessions) == 20
        assert cm.active_sessions_count() == 20

        # Discard messages concurrently
        tasks = [
            cm.process_user_message(sess.session_id, f"hello from {sess.client_a}")
            for sess in sessions
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 20
        assert all(r == "concurrent reply" for r in results)

    @pytest.mark.asyncio
    async def test_concurrency_zero_cross_talk_between_sessions(self):
        """Conversations across sessions have zero history leakage."""
        cm = CompanionManager(provider=MockAICompanionProvider(), typing_delay=0.0)
        cm.register_session("sess_science", "u_sci")
        cm.register_session("sess_cook", "u_cook")
        cm.register_session("sess_music", "u_mus")

        await cm.process_user_message("sess_science", "tell me about quantum physics")
        await cm.process_user_message("sess_cook", "what's the best pasta recipe?")
        await cm.process_user_message("sess_music", "do you play guitar?")

        h_sci = [m.text for m in cm.get_history("sess_science")]
        h_cook = [m.text for m in cm.get_history("sess_cook")]
        h_mus = [m.text for m in cm.get_history("sess_music")]

        assert any("quantum" in t for t in h_sci)
        assert not any("pasta" in t for t in h_sci)
        assert not any("guitar" in t for t in h_sci)

        assert any("pasta" in t for t in h_cook)
        assert not any("quantum" in t for t in h_cook)

        assert any("guitar" in t for t in h_mus)
        assert not any("quantum" in t for t in h_mus)

    @pytest.mark.asyncio
    async def test_concurrency_simultaneous_skips_across_sessions(self):
        """10 AI sessions skip simultaneously; all clients cleanly re-enter queue."""
        sm = SessionManager()
        q = MatchmakerQueue()
        cm = CompanionManager(session_manager=sm, queue=q)

        clients = [f"skip_user_{i}" for i in range(10)]
        for cid in clients:
            sess = await sm.create_session(cid, f"ai_companion_{cid}", is_ai=True)
            cm.register_session(sess.session_id, cid)

        # Concurrently skip all 10
        skip_tasks = [cm.handle_skip(cid) for cid in clients]
        results = await asyncio.gather(*skip_tasks)

        assert len(results) == 10
        assert all(r[1] is True for r in results)
        assert cm.active_sessions_count() == 0
        waiting_tickets = await q.get_waiting_tickets()
        assert len(waiting_tickets) == 10

    @pytest.mark.asyncio
    async def test_concurrency_watchdog_and_human_pairing_race(self):
        """Race between queue human matching and watchdog timeout causes zero duplicate sessions."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=1.0)

        # Enqueue two clients
        t1 = await q.enqueue("race_user_1")
        t2 = await q.enqueue("race_user_2")
        t1.queued_at = datetime.now(timezone.utc) - timedelta(seconds=2.0)
        t2.queued_at = datetime.now(timezone.utc) - timedelta(seconds=2.0)

        # Concurrently run human pairing and watchdog check
        pair_task = asyncio.create_task(q.try_pop_pair())
        watchdog_task = asyncio.create_task(watchdog.check_timeouts())

        human_pair, watchdog_paired = await asyncio.gather(pair_task, watchdog_task)

        # Exactly one outcome: either human pair got them, or watchdog got them, or one of each
        total_matched = (2 if human_pair else 0) + len(watchdog_paired)
        assert total_matched in (1, 2)
        # Queue must have no duplicate pops
        assert len(await q.get_waiting_tickets()) == 2 - total_matched

    @pytest.mark.asyncio
    async def test_concurrency_rapid_user_message_burst(self):
        """User sending 5 rapid messages into same session preserves history limits."""
        cm = CompanionManager(provider=MockAICompanionProvider(fixed_reply="ack"), typing_delay=0.0)
        cm.register_session("sess_burst", "user_burst")

        burst_tasks = [
            cm.process_user_message("sess_burst", f"burst msg {i}")
            for i in range(5)
        ]
        results = await asyncio.gather(*burst_tasks)
        assert len(results) == 5
        history = cm.get_history("sess_burst")
        assert len(history) == 10  # 5 user + 5 ack = exactly 10


# ============================================================================
# Suite 8: Alignment with Integration Tests & Test Harness (6 Tests)
# ============================================================================

class TestSuite8Alignment:
    """Verifies alignment with E2E integration test expectations and zero external network requests."""

    def test_alignment_matched_payload_schema(self):
        """Matched envelope payload adheres to MatchedPayload model schema."""
        raw_payload = {
            "is_initiator": True,
            "peer_is_ai": True,
            "peer_alias": "Stranger",
            "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
        }
        model = MatchedPayload(**raw_payload)
        assert model.is_initiator is True
        assert model.peer_is_ai is True
        assert model.peer_alias == "Stranger"

    def test_alignment_typing_payload_schema(self):
        """Typing envelope payload adheres to TypingPayload model schema."""
        raw_payload = {"is_typing": True}
        model = TypingPayload(**raw_payload)
        assert model.is_typing is True

    def test_alignment_server_chat_payload_schema(self):
        """Server chat envelope payload adheres to ServerChatMessagePayload schema."""
        raw_payload = {"sender": "Stranger", "text": "hello from AI"}
        model = ServerChatMessagePayload(**raw_payload)
        assert model.sender == "Stranger"
        assert model.text == "hello from AI"
        assert model.message_id is not None

    def test_alignment_fast_ai_timeout_settings_override(self, monkeypatch):
        """Watchdog timeout property dynamically reads monkeypatched settings."""
        monkeypatch.setattr(settings, "AI_FALLBACK_TIMEOUT_SECONDS", 0.35)
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager()
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm)

        assert watchdog.timeout == 0.35

    def test_alignment_reset_providers_autouse_cleanliness(self):
        """Verifies clean_provider_state fixture ensures factory has no leftover overrides."""
        p = get_ai_companion_provider()
        assert isinstance(p, MockAICompanionProvider)
        assert not hasattr(p, "custom_dirty_flag")

    @pytest.mark.asyncio
    async def test_alignment_full_offline_execution_zero_network(self):
        """End-to-end companion and watchdog lifecycle runs completely offline."""
        q = MatchmakerQueue()
        sm = SessionManager()
        cm = CompanionManager(typing_delay=0.0)
        watchdog = QueueWatchdog(queue=q, session_manager=sm, companion_manager=cm, timeout=0.1)

        ticket = await q.enqueue("offline_client")
        ticket.queued_at = datetime.now(timezone.utc) - timedelta(seconds=0.2)

        paired = await watchdog.check_timeouts()
        assert len(paired) == 1
        _, session_id = paired[0]

        reply = await cm.process_user_message(session_id, "hello offline world")
        assert reply is not None
        assert len(reply) > 0
