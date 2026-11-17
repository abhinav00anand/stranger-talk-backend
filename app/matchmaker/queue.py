"""Async in-memory matchmaking queue with FIFO ordering and topic matching."""

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from app.models.session import ClientState, QueueTicket


class MatchmakerQueueError(Exception):
    """Base exception for matchmaker queue operations."""
    pass


class AlreadyQueuedError(MatchmakerQueueError, ValueError):
    """Raised when attempting to enqueue a client that is already in the queue."""
    pass


class MatchmakerQueue:
    """Async thread-safe in-memory matchmaking queue.

    Provides FIFO ordering, atomic pairing, topic matching with pure FIFO
    fallback, ghost candidate pruning, and self-match prevention.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tickets: Dict[str, QueueTicket] = {}
        self._queue: deque[str] = deque()

    def queue_size(self) -> int:
        """Return the current number of active waiting tickets (O(1))."""
        return len(self._tickets)

    async def get_ticket(self, client_id: str) -> Optional[QueueTicket]:
        """Retrieve a client's active queue ticket if present."""
        async with self._lock:
            return self._tickets.get(client_id)

    async def get_waiting_tickets(self) -> List[QueueTicket]:
        """Return a snapshot list of all active waiting tickets."""
        async with self._lock:
            return list(self._tickets.values())

    async def enqueue(
        self,
        client_id: str,
        ip: Optional[str] = None,
        topics: Optional[List[str]] = None,
    ) -> QueueTicket:
        """Enqueue a client into the matchmaking pool.

        Raises:
            ValueError: If client_id is empty or invalid.
            AlreadyQueuedError: If client_id is already waiting in the queue.
        """
        if not client_id or not isinstance(client_id, str):
            raise ValueError("client_id must be a non-empty string")

        # Sanitize topics
        cleaned_topics: Optional[List[str]] = None
        if topics:
            cleaned_topics = [
                t.strip()
                for t in topics
                if isinstance(t, str) and t.strip()
            ] or None

        async with self._lock:
            if client_id in self._tickets:
                raise AlreadyQueuedError(
                    f"Client '{client_id}' is already waiting in the matchmaking queue"
                )

            ticket = QueueTicket(
                client_id=client_id,
                ip=ip,
                topics=cleaned_topics,
                status=ClientState.QUEUED,
                queued_at=datetime.now(timezone.utc),
            )
            self._tickets[client_id] = ticket
            self._queue.append(client_id)
            return ticket

    async def dequeue(self, client_id: str) -> bool:
        """Remove a client from the queue if present.

        Returns:
            True if the client was queued and removed, False otherwise.
        """
        async with self._lock:
            ticket = self._tickets.pop(client_id, None)
            if ticket is None:
                return False

            try:
                self._queue.remove(client_id)
            except ValueError:
                pass

            return True

    async def pop_client(self, client_id: str) -> Optional[QueueTicket]:
        """Atomically pop a specific client's ticket from the queue.

        Useful for watchdog timeouts or direct cancel transitions.
        """
        async with self._lock:
            ticket = self._tickets.pop(client_id, None)
            if ticket is not None:
                try:
                    self._queue.remove(client_id)
                except ValueError:
                    pass
            return ticket

    async def clear(self) -> None:
        """Clear all queued tickets and reset internal queues."""
        async with self._lock:
            self._tickets.clear()
            self._queue.clear()

    async def try_pop_pair(self) -> Optional[Tuple[str, str]]:
        """Atomically find and pop the best matching pair of clients.

        Selection strategy:
        1. Fast-path exit if fewer than 2 active tickets exist.
        2. Prune leading ghost/stale entries from the queue.
        3. Topic matching pass: If two distinct waiting clients share at least
           one common topic (case-insensitive), pair them.
        4. Fallback pass: If no topic match exists, pair the oldest two
           distinct clients in pure FIFO arrival order.
        5. Atomically remove both participants from queue and ticket registry.

        Returns:
            Tuple of (client_a, client_b) where client_a is the senior peer,
            or None if no valid pair can be formed.
        """
        async with self._lock:
            # 1. Fast-path check
            if len(self._tickets) < 2:
                return None

            # 2. Prune leading ghosts
            while self._queue and self._queue[0] not in self._tickets:
                self._queue.popleft()

            # 3. Collect unique active candidates in arrival order
            active_client_ids: List[str] = []
            seen: Set[str] = set()
            for cid in self._queue:
                if cid in self._tickets and cid not in seen:
                    seen.add(cid)
                    active_client_ids.append(cid)

            if len(active_client_ids) < 2:
                return None

            # 4. Topic Matching Pass: O(N * T)
            matched_pair: Optional[Tuple[str, str]] = None
            topic_index: Dict[str, str] = {}  # topic -> client_id

            for cid in active_client_ids:
                ticket = self._tickets[cid]
                if ticket.topics:
                    for topic in ticket.topics:
                        norm_t = topic.lower()
                        if norm_t in topic_index:
                            partner_id = topic_index[norm_t]
                            if partner_id != cid:
                                matched_pair = (partner_id, cid)
                                break
                        else:
                            topic_index[norm_t] = cid
                if matched_pair is not None:
                    break

            # 5. Pure FIFO Fallback if no topic match
            if matched_pair is None:
                matched_pair = (active_client_ids[0], active_client_ids[1])

            # 6. Atomic extraction
            client_a, client_b = matched_pair

            self._tickets.pop(client_a, None)
            self._tickets.pop(client_b, None)

            try:
                self._queue.remove(client_a)
            except ValueError:
                pass

            try:
                self._queue.remove(client_b)
            except ValueError:
                pass

            # Optional deque compaction if stale tombstones exceed active tickets
            if len(self._queue) > 2 * len(self._tickets) + 10:
                self._queue = deque([cid for cid in self._queue if cid in self._tickets])

            return (client_a, client_b)
