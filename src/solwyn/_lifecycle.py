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
from typing import TYPE_CHECKING, Protocol, TypeVar

import httpx

from solwyn._read_only_key import handle_read_only_key_error

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

_T = TypeVar("_T")

# Patchable clock seam (mirrors reporter._monotonic): the exit-drain deadline
# arithmetic reads through this so tests can drive time deterministically.
_monotonic = time.monotonic


def _is_retryable_exc(exc: Exception) -> bool:
    """Whether a send failure is transient and worth retrying.

    Transport errors (connect/read/write/timeout) and the transient HTTP
    statuses 408/429/5xx are retryable; every other HTTP status is a terminal
    rejection (a poison item that would reject identically forever).

    Lives here (not in reporter.py) because the blocking exit drain below needs
    it to classify one-shot exit failures and reporter.py imports this module.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in (408, 429) or code >= 500
    return False


def _drain_count(queue: collections.deque[_T]) -> int:
    """Atomically pop-and-count everything in ``queue``.

    A ``len()`` + ``clear()`` pair races a concurrent producer (the appended
    item is cleared but never counted); popping one item at a time cannot lose
    an item uncounted.
    """
    n = 0
    while True:
        try:
            queue.popleft()
        except IndexError:
            return n
        n += 1


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
            # _registry_lock is the ONE lock the fork-repair machinery cannot
            # repair (it guards the repair registry itself). Hold it across the
            # fork instead: the before-hook waits out any in-flight
            # registration, and both sides release their copy — otherwise a
            # child forked while another thread constructs an SDK object
            # inherits the lock held forever and deadlocks on its own next
            # construction (P1 review finding). Hooks registered here run
            # before _run_fork_resets (after_in_child hooks run in
            # registration order).
            os.register_at_fork(
                before=_registry_lock.acquire,
                after_in_parent=_registry_lock.release,
                after_in_child=_registry_lock.release,
            )
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
    hook flushes it. A COMPLETED ``close()`` detaches the finalizer; a cancelled
    close leaves it armed as the last-chance delivery path.
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

    Every item popped here stays accountable until it has a disposition: each
    POST reports sent / failed / deadline-expired, failures are counted by
    kind, and anything still queued at the deadline is swept by the final
    ``shutdown_deadline`` count. Confirms ride the control-plane breaker's
    admission state machine — a known-down control plane refuses them (counted
    ``exit_breaker_open``) after at most one recovery probe — but metadata
    ingest is deliberately NOT breaker-gated (it is the durable spend truth),
    so events, including the events of breaker-held settlements, still get
    their deadline-bounded attempt. Never raises out of the exit hook.
    """
    if not (confirm_q or settlement_q or event_q):
        return

    deadline = _monotonic() + budget
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = api_url.rstrip("/")
    confirm_url = f"{base_url}/api/v1/budgets/confirm"
    ingest_url = f"{base_url}/api/v1/metadata/ingest"
    held_drops = 0
    try:
        with httpx.Client(timeout=5.0) as client:
            while confirm_q:
                try:
                    pending = confirm_q.popleft()
                except IndexError:
                    break
                outcome = _post_confirm_blocking(
                    client,
                    confirm_url,
                    pending.request.model_dump(mode="json"),
                    headers,
                    breaker,
                    deadline,
                )
                if outcome == "expired":
                    # Keep the in-hand item accountable: the final sweep
                    # below counts it with the rest.
                    confirm_q.appendleft(pending)
                    break
                if outcome == "held":
                    # Control plane known-down: the whole confirm backlog is
                    # undeliverable — count it, then still drain events.
                    n = 1 + _drain_count(confirm_q)
                    _count(drop_counter, "confirm", "exit_breaker_open", n)
                    held_drops += n
                    break
                if outcome != "sent":
                    _count(drop_counter, "confirm", outcome)
            while settlement_q:
                try:
                    settlement = settlement_q.popleft()
                except IndexError:
                    break
                # Confirm-before-metadata order per item is load-bearing.
                outcome = _post_confirm_blocking(
                    client,
                    confirm_url,
                    settlement.confirm.request.model_dump(mode="json"),
                    headers,
                    breaker,
                    deadline,
                )
                if outcome == "expired":
                    settlement_q.appendleft(settlement)
                    break
                if outcome == "held":
                    _count(drop_counter, "settlement_confirm", "exit_breaker_open")
                    held_drops += 1
                elif outcome != "sent":
                    _count(drop_counter, "settlement_confirm", outcome)
                # Ingest is NOT breaker-gated: the event still gets its exit
                # attempt even when its confirm was held or failed.
                ev_outcome = _post_json(
                    client,
                    ingest_url,
                    [settlement.event.model_dump(mode="json")],
                    headers,
                    deadline,
                )
                if ev_outcome == "expired":
                    _count(drop_counter, "event", "shutdown_deadline")
                    break
                if ev_outcome != "sent":
                    _count(drop_counter, "event", ev_outcome)
            while event_q:
                batch: list[_PendingEvent] = []
                while event_q and len(batch) < _EXIT_BATCH_SIZE:
                    try:
                        batch.append(event_q.popleft())
                    except IndexError:
                        break
                if not batch:
                    break
                outcome = _post_json(
                    client,
                    ingest_url,
                    [p.event.model_dump(mode="json") for p in batch],
                    headers,
                    deadline,
                )
                if outcome == "expired":
                    event_q.extendleft(reversed(batch))
                    break
                if outcome != "sent":
                    _count(drop_counter, "event", outcome, len(batch))
    except Exception as exc:
        logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)

    if held_drops:
        logger.error("lifecycle.exit_flush_skipped_breaker_open: dropped=%d", held_drops)
    # Whatever the deadline left behind is abandoned — count it.
    _drop_all(confirm_q, settlement_q, event_q, drop_counter, "shutdown_deadline")


def _post_confirm_blocking(
    client: httpx.Client,
    url: str,
    payload: object,
    headers: dict[str, str],
    breaker: CircuitBreaker | None,
    deadline: float,
) -> str:
    """POST one confirm through the breaker's admission state machine.

    Mirrors ``_send_confirm``: a refused admission is ``"held"`` (no HTTP), an
    OPEN-but-recovery-eligible breaker gets exactly one HALF_OPEN probe (a
    failed probe re-opens it, so the rest of the backlog is refused instead of
    hammering a down endpoint), and success/failure verdicts feed the breaker.
    Returns ``_post_json``'s outcome, or ``"held"``.
    """
    if _monotonic() >= deadline:
        return "expired"
    admission = breaker.admit() if breaker is not None else None
    if admission is not None and not admission.allowed:
        return "held"
    try:
        outcome = _post_json(client, url, payload, headers, deadline)
        if breaker is not None:
            if outcome == "sent":
                breaker.record_success()
            elif outcome != "expired":
                breaker.record_failure()
        return outcome
    finally:
        # "expired" produced no verdict — free a consumed probe slot so the
        # breaker is not wedged. No-op once a verdict has released it.
        if breaker is not None:
            breaker.release_probe(admission)


def _post_json(
    client: httpx.Client,
    url: str,
    payload: object,
    headers: dict[str, str],
    deadline: float,
) -> str:
    """POST one payload best-effort and classify the outcome (exit is one shot).

    Returns ``"sent"``, ``"expired"`` (deadline reached before an attempt), or
    the drop reason for a failed attempt: ``"retry_exhausted"`` for a transient
    failure (the exit budget IS the whole retry budget) or
    ``"terminal_status"``. A read-only-key rejection is a policy skip, not a
    loss. Never raises out of the exit hook.
    """
    remaining = deadline - _monotonic()
    if remaining <= 0:
        return "expired"
    try:
        response = client.post(url, json=payload, headers=headers, timeout=min(5.0, remaining))
        response.raise_for_status()
    except Exception as exc:
        if handle_read_only_key_error(exc):
            return "sent"
        # The server dedups a later duplicate, and an exit hook must never
        # raise. Log the type name only.
        logger.warning("lifecycle.exit_post_failed: exc_type=%s", type(exc).__name__)
        return "retry_exhausted" if _is_retryable_exc(exc) else "terminal_status"
    return "sent"


def _count(
    drop_counter: Callable[[str, str, int], None] | None,
    kind: str,
    reason: str,
    n: int = 1,
) -> None:
    """Invoke the drop counter when present (a GC'd reporter has none)."""
    if drop_counter is not None and n > 0:
        drop_counter(kind, reason, n)


def _drop_all(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    drop_counter: Callable[[str, str, int], None] | None,
    reason: str,
) -> None:
    """Clear every queue, counting the loss by kind.

    Each settlement pair loses its confirm AND its event — count both or
    ``dropped_counts`` understates real event loss.
    """
    n_confirm = _drain_count(confirm_q)
    n_settlement = _drain_count(settlement_q)
    n_event = _drain_count(event_q)
    _count(drop_counter, "confirm", reason, n_confirm)
    _count(drop_counter, "settlement_confirm", reason, n_settlement)
    _count(drop_counter, "event", reason, n_settlement + n_event)


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
            # _close_completed (NOT _closed): a cancelled or partial close()
            # leaves queued spend behind — this hook stays armed until a close
            # has actually finished its flush.
            if not async_reporter._close_completed:
                blocking_exit_flush(async_reporter)
        except Exception as exc:
            logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
