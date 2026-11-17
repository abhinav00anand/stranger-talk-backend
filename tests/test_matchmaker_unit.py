"""Unit tests for MatchmakerQueue and SessionManager (FIFO, pairing, concurrency, skip/leave lifecycle)."""

import asyncio
from typing import List, Tuple
import pytest

from app.matchmaker.queue import (
    AlreadyQueuedError,
    MatchmakerQueue,
    MatchmakerQueueError,
)
from app.matchmaker.session_manager import SessionManager
from app.models.session import ClientState, SessionStatus


# ============================================================================
# MatchmakerQueue Unit Tests
# ============================================================================


@pytest.mark.asyncio
async def test_queue_enqueue_and_size():
    """Enqueue client with metadata and verify size, ticket fields, and elapsed seconds."""
    queue = MatchmakerQueue()
    ticket = await queue.enqueue("client_1", ip="192.168.1.100", topics=["gaming", "music"])

    assert queue.queue_size() == 1
    assert ticket.client_id == "client_1"
    assert ticket.ip == "192.168.1.100"
    assert ticket.topics == ["gaming", "music"]
    assert ticket.status == ClientState.QUEUED
    assert ticket.elapsed_seconds() >= 0.0

    retrieved = await queue.get_ticket("client_1")
    assert retrieved is not None
    assert retrieved.ticket_id == ticket.ticket_id

    waiting = await queue.get_waiting_tickets()
    assert len(waiting) == 1
    assert waiting[0].client_id == "client_1"


@pytest.mark.asyncio
async def test_queue_duplicate_enqueue_raises_already_queued():
    """Attempting to enqueue an already queued client raises AlreadyQueuedError and preserves state."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")

    with pytest.raises(AlreadyQueuedError) as exc_info:
        await queue.enqueue("client_1")

    assert "already waiting in the matchmaking queue" in str(exc_info.value)
    assert issubclass(AlreadyQueuedError, ValueError)
    assert issubclass(AlreadyQueuedError, MatchmakerQueueError)
    assert queue.queue_size() == 1


@pytest.mark.asyncio
async def test_queue_invalid_client_id_raises_value_error():
    """Enqueuing with empty string or non-string client_id raises ValueError."""
    queue = MatchmakerQueue()

    with pytest.raises(ValueError):
        await queue.enqueue("")

    with pytest.raises(ValueError):
        await queue.enqueue(None)  # type: ignore


@pytest.mark.asyncio
async def test_queue_dequeue_existing_and_nonexistent():
    """Dequeue removes an active ticket eagerly and returns boolean status."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    assert queue.queue_size() == 1

    # Dequeue existing
    assert await queue.dequeue("client_1") is True
    assert queue.queue_size() == 0
    assert await queue.get_ticket("client_1") is None

    # Dequeue non-existent
    assert await queue.dequeue("client_1") is False
    assert await queue.dequeue("unknown_client") is False


@pytest.mark.asyncio
async def test_queue_pure_fifo_pairing_order():
    """Pure FIFO queue pairs clients in strict arrival order when no topics match."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    await queue.enqueue("client_2")
    await queue.enqueue("client_3")
    await queue.enqueue("client_4")

    assert queue.queue_size() == 4

    pair_1 = await queue.try_pop_pair()
    assert pair_1 == ("client_1", "client_2")
    assert queue.queue_size() == 2

    pair_2 = await queue.try_pop_pair()
    assert pair_2 == ("client_3", "client_4")
    assert queue.queue_size() == 0

    pair_3 = await queue.try_pop_pair()
    assert pair_3 is None


@pytest.mark.asyncio
async def test_queue_odd_client_count_leaves_single_waiting():
    """Odd number of clients leaves the latest client queued safely without pairing."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    await queue.enqueue("client_2")
    await queue.enqueue("client_3")

    pair = await queue.try_pop_pair()
    assert pair == ("client_1", "client_2")
    assert queue.queue_size() == 1

    remaining_ticket = await queue.get_ticket("client_3")
    assert remaining_ticket is not None
    assert remaining_ticket.client_id == "client_3"

    # Cannot form pair with only 1 client
    assert await queue.try_pop_pair() is None


@pytest.mark.asyncio
async def test_queue_ghost_candidate_pruning():
    """Dequeued or cancelled clients are pruned from pairing without affecting active candidates."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    await queue.enqueue("client_2")
    await queue.enqueue("client_3")

    # Client 2 leaves queue
    assert await queue.dequeue("client_2") is True

    # Pairing should bypass Client 2 and pair Client 1 with Client 3
    pair = await queue.try_pop_pair()
    assert pair == ("client_1", "client_3")
    assert queue.queue_size() == 0
    assert await queue.try_pop_pair() is None


@pytest.mark.asyncio
async def test_queue_self_match_prevention():
    """Queue enforces distinct candidates and prevents self-pairing under all circumstances."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")

    # Artificially inject duplicate entry into internal deque to simulate corrupted state
    queue._queue.append("client_1")

    pair = await queue.try_pop_pair()
    assert pair is None  # Only 1 unique client, cannot self-pair


@pytest.mark.asyncio
async def test_queue_topic_matching_priority():
    """Clients sharing a topic are paired with priority over earlier clients with disjoint topics."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1", topics=["python", "ai"])
    await queue.enqueue("client_2", topics=["music"])
    await queue.enqueue("client_3", topics=["gaming", "ai"])

    # client_1 and client_3 share topic "ai"
    pair = await queue.try_pop_pair()
    assert pair == ("client_1", "client_3")
    assert queue.queue_size() == 1

    remaining = await queue.get_ticket("client_2")
    assert remaining is not None
    assert remaining.client_id == "client_2"


@pytest.mark.asyncio
async def test_queue_topic_normalization():
    """Topic matching is case-insensitive and trims whitespace."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1", topics=["  Python  ", ""])
    await queue.enqueue("client_2", topics=["python"])

    pair = await queue.try_pop_pair()
    assert pair == ("client_1", "client_2")
    assert queue.queue_size() == 0


@pytest.mark.asyncio
async def test_queue_disjoint_topics_fallback_to_fifo():
    """When queued clients have no overlapping topics, pairing falls back to pure FIFO arrival order."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1", topics=["cooking"])
    await queue.enqueue("client_2", topics=["space"])

    pair = await queue.try_pop_pair()
    assert pair == ("client_1", "client_2")
    assert queue.queue_size() == 0


@pytest.mark.asyncio
async def test_queue_concurrent_enqueues_and_pairings():
    """Stress test: 30 concurrent enqueues paired concurrently by worker tasks without race bugs."""
    queue = MatchmakerQueue()
    num_clients = 30

    # 30 concurrent enqueues
    await asyncio.gather(*[queue.enqueue(f"client_{i}") for i in range(num_clients)])
    assert queue.queue_size() == num_clients

    pairs: List[Tuple[str, str]] = []
    pairs_lock = asyncio.Lock()

    async def worker():
        pair = await queue.try_pop_pair()
        if pair:
            async with pairs_lock:
                pairs.append(pair)

    # 30 concurrent workers competing to pop pairs
    await asyncio.gather(*[worker() for _ in range(num_clients)])

    # Exactly 15 pairs formed
    assert len(pairs) == 15
    assert queue.queue_size() == 0

    all_matched = []
    for c_a, c_b in pairs:
        assert c_a != c_b  # No self-matching
        all_matched.extend([c_a, c_b])

    # Exactly 30 unique clients paired with zero duplicates
    assert len(set(all_matched)) == num_clients


@pytest.mark.asyncio
async def test_queue_pop_client_watchdog_support():
    """pop_client removes a specific client atomically, leaving other clients undisturbed."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    await queue.enqueue("client_2")

    popped = await queue.pop_client("client_1")
    assert popped is not None
    assert popped.client_id == "client_1"
    assert queue.queue_size() == 1

    # Only client_2 left, cannot form pair
    assert await queue.try_pop_pair() is None

    # Popping non-existent client returns None
    assert await queue.pop_client("client_1") is None


@pytest.mark.asyncio
async def test_queue_clear():
    """Clear resets all queue tickets and internal structures."""
    queue = MatchmakerQueue()
    await queue.enqueue("client_1")
    await queue.enqueue("client_2")
    assert queue.queue_size() == 2

    await queue.clear()
    assert queue.queue_size() == 0
    assert await queue.get_waiting_tickets() == []
    assert await queue.try_pop_pair() is None


# ============================================================================
# SessionManager Unit Tests
# ============================================================================


@pytest.fixture
def sm():
    return SessionManager()


@pytest.mark.asyncio
async def test_session_create_deterministic_initiator(sm):
    """SessionManager creates session with deterministic initiator and bidirectional mappings."""
    session = await sm.create_session("peer_alpha", "peer_beta")

    assert session.session_id is not None
    assert session.client_a == "peer_alpha"
    assert session.client_b == "peer_beta"
    assert session.initiator_id == "peer_alpha"
    assert session.status == SessionStatus.PAIRED
    assert session.is_active() is True

    # Fast synchronous lookups
    assert sm.get_session(session.session_id) == session
    assert sm.get_session_by_client("peer_alpha") == session
    assert sm.get_session_by_client("peer_beta") == session
    assert sm.get_partner_id("peer_alpha") == "peer_beta"
    assert sm.get_partner_id("peer_beta") == "peer_alpha"
    assert sm.get_client_state("peer_alpha") == ClientState.PAIRED
    assert sm.get_client_state("peer_beta") == ClientState.PAIRED
    assert sm.active_session_count() == 1
    assert len(sm.get_active_sessions()) == 1


@pytest.mark.asyncio
async def test_session_create_rejects_self_pairing_and_empty_ids(sm):
    """Attempting to pair a client with themselves or using empty strings raises ValueError."""
    with pytest.raises(ValueError, match="distinct"):
        await sm.create_session("peer_x", "peer_x")

    with pytest.raises(ValueError, match="non-empty"):
        await sm.create_session("", "peer_x")

    with pytest.raises(ValueError, match="non-empty"):
        await sm.create_session("peer_x", "")


@pytest.mark.asyncio
async def test_session_activate_session(sm):
    """activate_session transitions session status from PAIRED to ACTIVE."""
    session = await sm.create_session("peer_1", "peer_2")
    assert session.status == SessionStatus.PAIRED

    activated = await sm.activate_session(session.session_id)
    assert activated is not None
    assert activated.status == SessionStatus.ACTIVE
    assert activated.is_active() is True


@pytest.mark.asyncio
async def test_session_handle_skip_happy_path(sm):
    """handle_skip terminates session, re-queues skipper, sets partner to IDLE, and cleans mappings."""
    session = await sm.create_session("peer_a", "peer_b")

    sess, partner_id = await sm.handle_skip("peer_a")
    assert sess is not None
    assert sess.session_id == session.session_id
    assert sess.status == SessionStatus.TERMINATED
    assert partner_id == "peer_b"

    # State transitions
    assert sm.get_client_state("peer_a") == ClientState.QUEUED
    assert sm.get_client_state("peer_b") == ClientState.IDLE

    # Mappings cleared
    assert sm.get_session_by_client("peer_a") is None
    assert sm.get_session_by_client("peer_b") is None
    assert sm.get_partner_id("peer_a") is None
    assert sm.active_session_count() == 0

    # Retrievable from terminated cache
    assert sm.get_session(session.session_id) is not None
    assert sm.get_session(session.session_id).status == SessionStatus.TERMINATED


@pytest.mark.asyncio
async def test_session_handle_skip_not_in_session(sm):
    """Skipping when client is not in an active session returns (None, None)."""
    sess, partner = await sm.handle_skip("idle_user")
    assert sess is None
    assert partner is None


@pytest.mark.asyncio
async def test_session_handle_skip_expected_session_id_mismatch(sm):
    """Skipping with mismatched expected_session_id returns (None, None)."""
    session = await sm.create_session("peer_a", "peer_b")
    sess, partner = await sm.handle_skip("peer_a", expected_session_id="wrong_id")

    assert sess is None
    assert partner is None
    assert sm.active_session_count() == 1


@pytest.mark.asyncio
async def test_session_handle_skip_simultaneous_race(sm):
    """Simultaneous skips by both peers resolve atomically: exactly one terminates, second returns (None, None)."""
    session = await sm.create_session("peer_x", "peer_y")

    res_x, res_y = await asyncio.gather(
        sm.handle_skip("peer_x"),
        sm.handle_skip("peer_y"),
    )

    results = [res_x, res_y]
    successful = [r for r in results if r[0] is not None]
    already_handled = [r for r in results if r[0] is None]

    assert len(successful) == 1
    assert len(already_handled) == 1
    assert successful[0][0].session_id == session.session_id
    assert already_handled[0] == (None, None)
    assert sm.active_session_count() == 0


@pytest.mark.asyncio
async def test_session_handle_leave_happy_path(sm):
    """handle_leave terminates session, sets leaving client to IDLE and partner to IDLE."""
    session = await sm.create_session("user_1", "user_2")

    sess, partner_id = await sm.handle_leave("user_1")
    assert sess is not None
    assert sess.session_id == session.session_id
    assert partner_id == "user_2"
    assert sess.status == SessionStatus.TERMINATED

    assert sm.get_client_state("user_1") == ClientState.IDLE
    assert sm.get_client_state("user_2") == ClientState.IDLE
    assert sm.get_session_by_client("user_1") is None
    assert sm.get_session_by_client("user_2") is None


@pytest.mark.asyncio
async def test_session_handle_leave_simultaneous_race(sm):
    """Simultaneous leaves by both peers resolve atomically without exceptions or deadlock."""
    session = await sm.create_session("user_1", "user_2")

    res_1, res_2 = await asyncio.gather(
        sm.handle_leave("user_1"),
        sm.handle_leave("user_2"),
    )

    successful = [r for r in (res_1, res_2) if r[0] is not None]
    already_handled = [r for r in (res_1, res_2) if r[0] is None]

    assert len(successful) == 1
    assert len(already_handled) == 1
    assert sm.get_client_state("user_1") == ClientState.IDLE
    assert sm.get_client_state("user_2") == ClientState.IDLE
    assert sm.active_session_count() == 0


@pytest.mark.asyncio
async def test_session_handle_disconnect(sm):
    """Disconnect records the session outcome and releases the disconnected client's state."""
    session = await sm.create_session("user_1", "user_2")

    sess, partner_id = await sm.handle_disconnect("user_1")
    assert sess is not None
    assert partner_id == "user_2"
    assert sess.status == SessionStatus.DISCONNECTED
    assert sm.get_client_state("user_1") == ClientState.IDLE
    assert sm.get_client_state("user_2") == ClientState.IDLE

    # Redundant disconnect returns (None, None) safely
    assert await sm.handle_disconnect("user_1") == (None, None)


@pytest.mark.asyncio
async def test_session_terminate_session_reasons(sm):
    """terminate_session correctly maps termination reasons to SessionStatus."""
    reasons_map = {
        "banned": SessionStatus.BANNED,
        "disconnected": SessionStatus.DISCONNECTED,
        "leaving": SessionStatus.LEAVING,
        "skipping": SessionStatus.SKIPPING,
        "custom_reason": SessionStatus.TERMINATED,
    }

    for reason, expected_status in reasons_map.items():
        session = await sm.create_session(f"c_a_{reason}", f"c_b_{reason}")
        terminated = await sm.terminate_session(session.session_id, reason=reason)
        assert terminated is not None
        assert terminated.status == expected_status
        assert sm.get_session_by_client(f"c_a_{reason}") is None

        # Terminating already terminated session returns None
        assert await sm.terminate_session(session.session_id) is None


@pytest.mark.asyncio
async def test_session_stale_mapping_cleanup_on_new_session(sm):
    """Creating a new session for a client with a dangling prior session cleans up stale mapping."""
    session_1 = await sm.create_session("peer_1", "peer_2")
    assert sm.active_session_count() == 1

    # Re-pairing peer_1 with peer_3 without explicit skip cleans up session_1
    session_2 = await sm.create_session("peer_1", "peer_3")
    assert sm.active_session_count() == 1
    assert sm.get_session_by_client("peer_1") == session_2
    assert sm.get_session_by_client("peer_3") == session_2
    assert sm.get_session_by_client("peer_2") is None
    assert sm.get_client_state("peer_2") == ClientState.IDLE


@pytest.mark.asyncio
async def test_session_history_bounded(sm):
    """Terminated sessions cache is bounded by max_history."""
    small_sm = SessionManager(max_history=3)
    created_ids = []

    for i in range(5):
        sess = await small_sm.create_session(f"a_{i}", f"b_{i}")
        created_ids.append(sess.session_id)
        await small_sm.handle_skip(f"a_{i}")

    # Oldest 2 terminated sessions should be evicted, last 3 retained
    assert small_sm.get_session(created_ids[0]) is None
    assert small_sm.get_session(created_ids[1]) is None
    assert small_sm.get_session(created_ids[2]) is not None
    assert small_sm.get_session(created_ids[3]) is not None
    assert small_sm.get_session(created_ids[4]) is not None


def test_session_reset(sm):
    """Reset clears all active sessions, mappings, and terminated caches."""
    sm.set_client_state("user_1", ClientState.QUEUED)
    sm.reset()
    assert sm.active_session_count() == 0
    assert sm.get_client_state("user_1") == ClientState.IDLE
