"""Process-lifecycle wiring: exit flush and fork reset for live reporters.

Never touches prompt content. Registered reporters are held WEAKLY, so a closed
or garbage-collected reporter drops out automatically. The atexit hook and the
weakref finalizers below are the last chance to deliver acknowledged provider
spend that a normally-exiting process still has queued — a duplicate send is
harmless because the server dedups via its idempotency ledger.

It also owns fork repair: ``register_fork_reset`` installs a single
``os.register_at_fork(after_in_child=...)`` hook that lets reporters, budget
enforcers, and circuit breakers rebuild the locks / clients / threads that do
not survive ``fork()``. Every fork entry point is guarded for platforms without
fork, so this module stays Windows-safe.
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import httpx

from solwyn._types import CircuitState

if TYPE_CHECKING:
    from solwyn.circuit_breaker import CircuitBreaker
    from solwyn.reporter import (
        AsyncMetadataReporter,
        MetadataReporter,
        _PendingConfirm,
        _PendingEvent,
        _PendingSettlement,
    )

logger = logging.getLogger(__name__)


class _ForkResettable(Protocol):
    """An object that repairs its cross-fork-unsafe state in a forked child."""

    def _reset_after_fork_in_child(self) -> None: ...


# Live reporters, held weakly. A running sync reporter is kept alive by its flush
# thread's bound-method target; a running async reporter by its flush task — so a
# reporter drops out of these sets only once it is closed or unreferenced.
_LIVE_SYNC_REPORTERS: weakref.WeakSet[MetadataReporter] = weakref.WeakSet()
_LIVE_ASYNC_REPORTERS: weakref.WeakSet[AsyncMetadataReporter] = weakref.WeakSet()

_registry_lock = threading.Lock()
_atexit_registered = False

# Objects whose locks / clients / threads must be repaired in a forked child.
_FORK_RESETTABLE: weakref.WeakSet[_ForkResettable] = weakref.WeakSet()
_fork_registered = False

_EXIT_BATCH_SIZE = 50


def register_fork_reset(obj: _ForkResettable) -> None:
    """Track an object so its cross-fork-unsafe state is repaired after a fork.

    Lazily installs a single ``os.register_at_fork(after_in_child=...)`` hook
    (guarded for platforms without fork, e.g. Windows). Objects are held weakly.
    """
    global _fork_registered
    with _registry_lock:
        if not _fork_registered and hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_run_fork_resets)
            _fork_registered = True
    _FORK_RESETTABLE.add(obj)


def _run_fork_resets() -> None:
    """Run every registered object's child-side reset; guard each individually."""
    for obj in list(_FORK_RESETTABLE):
        try:
            obj._reset_after_fork_in_child()
        except Exception as exc:
            logger.warning("lifecycle.fork_reset_failed: exc_type=%s", type(exc).__name__)


def _ensure_atexit_registered() -> None:
    """Register the single exit hook lazily, exactly once per process."""
    global _atexit_registered
    with _registry_lock:
        if not _atexit_registered:
            atexit.register(_exit_flush_all)
            _atexit_registered = True


def register_sync_reporter(reporter: MetadataReporter) -> None:
    """Track a sync reporter so its queued spend is flushed at interpreter exit."""
    _ensure_atexit_registered()
    _LIVE_SYNC_REPORTERS.add(reporter)


def register_async_reporter(reporter: AsyncMetadataReporter) -> None:
    """Track an async reporter for exit flush AND arm a GC finalizer.

    The finalizer covers the reporter-GC'd-before-exit case (constructed, events
    queued, never started — so no flush task holds it alive). It captures the
    QUEUES + config (never ``reporter``, which would defeat the weak reference),
    so a collected reporter still drains over a temporary sync client. A running
    reporter cannot be GC'd (its flush task holds a strong ref), so only the exit
    hook flushes it. ``close()`` detaches the finalizer.
    """
    _ensure_atexit_registered()
    _LIVE_ASYNC_REPORTERS.add(reporter)
    reporter._finalizer = weakref.finalize(
        reporter,
        _drain_queues_blocking,
        reporter._confirm_queue,
        reporter._settlement_queue,
        reporter._queue,
        reporter.api_url,
        reporter.api_key,
        reporter._control_plane_breaker,
        reporter.shutdown_deadline,
        None,  # drop_counter: a GC'd reporter's dropped_counts is unobservable
    )


def blocking_exit_flush(base: AsyncMetadataReporter) -> None:
    """Drain a (loop-less) async reporter's queues over a temporary sync client.

    Called from the atexit hook: no event loop exists at interpreter exit, so a
    synchronous ``httpx.Client`` posts the same sans-I/O payloads the async path
    would, using the reporter's own URL and auth.
    """
    _drain_queues_blocking(
        base._confirm_queue,
        base._settlement_queue,
        base._queue,
        base.api_url,
        base.api_key,
        base._control_plane_breaker,
        base.shutdown_deadline,
        base._count_drop,
    )


def _drain_queues_blocking(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    api_url: str,
    api_key: str,
    breaker: CircuitBreaker | None,
    budget: float,
    drop_counter: Callable[[str, str, int], None] | None,
) -> None:
    """Best-effort, single-attempt, deadline-bounded exit flush of the queues.

    A known-down control plane (breaker OPEN and not recovery-eligible) is
    skipped entirely: hammering it while exiting is pointless. Anything still
    queued when the deadline is reached is abandoned. Both paths count the loss
    (when a counter is supplied) and never raise out of the exit hook.
    """
    if not (confirm_q or settlement_q or event_q):
        return

    if breaker is not None:
        state = breaker.get_state()
        if state.state is CircuitState.OPEN and not state.recovery_eligible:
            _drop_all(confirm_q, settlement_q, event_q, drop_counter, "exit_breaker_open")
            return

    deadline = time.monotonic() + budget
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = api_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            while confirm_q and time.monotonic() < deadline:
                pending = confirm_q.popleft()
                _post_json(
                    client,
                    f"{base_url}/api/v1/budgets/confirm",
                    pending.request.model_dump(mode="json"),
                    headers,
                    deadline,
                )
            while settlement_q and time.monotonic() < deadline:
                settlement = settlement_q.popleft()
                # Confirm-before-metadata order per item is load-bearing.
                _post_json(
                    client,
                    f"{base_url}/api/v1/budgets/confirm",
                    settlement.confirm.request.model_dump(mode="json"),
                    headers,
                    deadline,
                )
                _post_json(
                    client,
                    f"{base_url}/api/v1/metadata/ingest",
                    [settlement.event.model_dump(mode="json")],
                    headers,
                    deadline,
                )
            while event_q and time.monotonic() < deadline:
                batch: list[object] = []
                while event_q and len(batch) < _EXIT_BATCH_SIZE:
                    batch.append(event_q.popleft().event.model_dump(mode="json"))
                _post_json(client, f"{base_url}/api/v1/metadata/ingest", batch, headers, deadline)
    except Exception as exc:
        logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)

    # Whatever the deadline left behind is abandoned — count it.
    _drop_all(confirm_q, settlement_q, event_q, drop_counter, "shutdown_deadline")


def _post_json(
    client: httpx.Client,
    url: str,
    payload: object,
    headers: dict[str, str],
    deadline: float,
) -> None:
    """POST one payload best-effort; never raise, never retry (exit is one shot)."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        response = client.post(url, json=payload, headers=headers, timeout=min(5.0, remaining))
        response.raise_for_status()
    except Exception as exc:
        # A failed post at exit is dropped: the server dedups a later duplicate,
        # and an exit hook must never raise. Log the type name only.
        logger.warning("lifecycle.exit_post_failed: exc_type=%s", type(exc).__name__)


def _drop_all(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    drop_counter: Callable[[str, str, int], None] | None,
    reason: str,
) -> None:
    """Clear every queue, counting the loss by kind and (for a skip) logging once."""
    n_confirm, n_settlement, n_event = len(confirm_q), len(settlement_q), len(event_q)
    total = n_confirm + n_settlement + n_event
    if total == 0:
        return
    if drop_counter is not None:
        if n_confirm:
            drop_counter("confirm", reason, n_confirm)
        if n_settlement:
            drop_counter("settlement_confirm", reason, n_settlement)
        if n_event:
            drop_counter("event", reason, n_event)
    confirm_q.clear()
    settlement_q.clear()
    event_q.clear()
    if reason == "exit_breaker_open":
        logger.error("lifecycle.exit_flush_skipped_breaker_open: dropped=%d", total)


def _exit_flush_all() -> None:
    """The single atexit hook: flush every still-open live reporter."""
    for sync_reporter in list(_LIVE_SYNC_REPORTERS):
        try:
            if not sync_reporter._shutdown.is_set():
                sync_reporter.close()
        except Exception as exc:
            logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
    for async_reporter in list(_LIVE_ASYNC_REPORTERS):
        try:
            if not async_reporter._closed:
                blocking_exit_flush(async_reporter)
        except Exception as exc:
            logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
