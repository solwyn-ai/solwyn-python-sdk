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
    """Track an async reporter for exit flush AND arm a GC-only finalizer.

    The finalizer covers the reporter-GC'd-before-exit case (constructed, events
    queued, never started — so no flush task holds it alive). It captures the
    QUEUES + config (never ``reporter``, which would defeat the weak reference),
    so a collected reporter still drains over a temporary sync client, with its
    losses accounted the only way left — loudly, via ``_gc_drop_counter``. A
    running reporter cannot be GC'd (its flush task holds a strong ref). A
    COMPLETED ``close()`` detaches the finalizer; a cancelled close leaves it
    armed as the last-chance delivery path.

    ``atexit`` is disabled on the finalizer: weakref's own exit hook is
    atexit-registered when the process's FIRST finalizer is created — here,
    AFTER ``_exit_flush_all`` — so LIFO ordering would run it first at exit and
    drain still-LIVE reporters with logger-only accounting while their
    ``dropped_counts`` is fully observable. ``_exit_flush_all`` is the sole
    exit owner for live reporters; the finalizer covers genuine pre-exit GC.
    """
    _ensure_atexit_registered()
    _LIVE_ASYNC_REPORTERS.add(reporter)
    finalizer = weakref.finalize(
        reporter,
        _drain_queues_blocking,
        reporter._confirm_queue,
        reporter._settlement_queue,
        reporter._queue,
        reporter.api_url,
        reporter.api_key,
        reporter._control_plane_breaker,
        reporter.shutdown_deadline,
        _gc_drop_counter,
    )
    # Documented writable property; typeshed models finalize with __slots__
    # only, so mypy rejects the assignment it cannot see.
    finalizer.atexit = False  # type: ignore[misc]
    reporter._finalizer = finalizer


def _gc_drop_counter(kind: str, reason: str, n: int = 1) -> None:
    """Drop accounting for the GC-finalizer drain.

    A collected reporter's ``dropped_counts`` no longer exists, so a loss on
    this path is reported the only way left: a WARNING per counted drop.
    """
    logger.warning("lifecycle.gc_flush_dropped: kind=%s reason=%s n=%d", kind, reason, n)


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


class _ExitOwnership:
    """Pop-and-claim ownership shared by the exit-drain worker and its joiner.

    Mirrors the reporter's ``_seal_delivery`` machinery for the blocking exit
    drain: the worker registers every popped item in-hand ATOMICALLY with the
    pop, so the deadline seal can never observe an item in neither a queue nor
    a hand. After the seal, ``resolve`` returns owned=0 and ``requeue`` is
    refused — the seal already counted those items ``shutdown_deadline`` — so
    a worker unblocking late no-ops instead of double-counting. A settlement
    pair is held as its two halves (``settlement_confirm`` +
    ``settlement_event``) because each half reaches its disposition
    separately. A send stuck at the seal that later succeeds leaves a
    conservative overcount for that one item; drops are never UNDERstated.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._sealed = False

    def pop_in_hand(self, queue: collections.deque[_T], *kinds: str) -> _T | None:
        """Pop the head and register it under each ``kinds`` atomically."""
        with self._lock:
            if self._sealed:
                return None
            try:
                item = queue.popleft()
            except IndexError:
                return None
            for kind in kinds:
                self._counts[kind] = self._counts.get(kind, 0) + 1
            return item

    def pop_batch_in_hand(self, queue: collections.deque[_T], kind: str, limit: int) -> list[_T]:
        """Batched ``pop_in_hand`` for the event queue."""
        with self._lock:
            if self._sealed:
                return []
            batch: list[_T] = []
            while len(batch) < limit:
                try:
                    batch.append(queue.popleft())
                except IndexError:
                    break
            if batch:
                self._counts[kind] = self._counts.get(kind, 0) + len(batch)
            return batch

    def resolve(self, kind: str, n: int = 1) -> int:
        """Release up to ``n`` in-hand items; returns how many the caller
        still owns (0 after the seal claimed them — those must not be
        counted again)."""
        with self._lock:
            held = self._counts.get(kind, 0)
            owned = min(n, held)
            self._counts[kind] = held - owned
            return owned

    def requeue(self, queue: collections.deque[_T], items: list[_T], kinds: dict[str, int]) -> bool:
        """Return still-owned items to the queue head (deadline-expired
        outcome) so the final sweep counts them; refused once sealed (the
        seal already counted them)."""
        with self._lock:
            if self._sealed:
                return False
            for kind, n in kinds.items():
                self._counts[kind] = max(0, self._counts.get(kind, 0) - n)
            queue.extendleft(reversed(items))
            return True

    def seal(self) -> dict[str, int]:
        """Claim every in-hand item; further pops/resolves/requeues refuse."""
        with self._lock:
            self._sealed = True
            counts = {kind: n for kind, n in self._counts.items() if n}
            self._counts.clear()
            return counts


def _drain_queues_blocking(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    api_url: str,
    api_key: str,
    breaker: CircuitBreaker | None,
    budget: float,
    drop_counter: Callable[[str, str, int], None],
) -> None:
    """Best-effort, single-attempt, deadline-bounded exit flush of the queues.

    The deadline is a true wall-clock bound: the drain runs on a daemon worker
    JOINED at the deadline. httpx timeouts cap socket operations, not total
    response time, and closing a sync client does not reliably interrupt a
    blocked read — so no close()-based abort is trusted; the join is the
    bound. ``_ExitOwnership`` keeps the timeout honest: every pop registers
    the item in-hand atomically, a timed-out join SEALS delivery (counting
    in-hand items ``shutdown_deadline`` and sweeping the queues), and a worker
    unblocking after the seal resolves to owned=0 instead of double-counting.

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
    own = _ExitOwnership()
    client = httpx.Client(timeout=5.0)

    def _run() -> None:
        held_drops = 0
        try:
            while True:
                pending = own.pop_in_hand(confirm_q, "confirm")
                if pending is None:
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
                    # Keep the in-hand item accountable: requeued for the
                    # final sweep (or already claimed by the seal).
                    own.requeue(confirm_q, [pending], {"confirm": 1})
                    break
                owned = own.resolve("confirm")
                if outcome == "held":
                    # Control plane known-down: each queued confirm is
                    # refused per item (admit() short-circuits, no HTTP) so
                    # every pop stays ownership-tracked.
                    _count(drop_counter, "confirm", "exit_breaker_open", owned)
                    held_drops += owned
                    continue
                if outcome != "sent":
                    _count(drop_counter, "confirm", outcome, owned)
            while True:
                settlement = own.pop_in_hand(settlement_q, "settlement_confirm", "settlement_event")
                if settlement is None:
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
                    own.requeue(
                        settlement_q,
                        [settlement],
                        {"settlement_confirm": 1, "settlement_event": 1},
                    )
                    break
                owned = own.resolve("settlement_confirm")
                if outcome == "held":
                    _count(drop_counter, "settlement_confirm", "exit_breaker_open", owned)
                    held_drops += owned
                elif outcome != "sent":
                    _count(drop_counter, "settlement_confirm", outcome, owned)
                # Ingest is NOT breaker-gated: the event still gets its exit
                # attempt even when its confirm was held or failed.
                ev_outcome, ev_response = _post_json_response(
                    client,
                    ingest_url,
                    [settlement.event.model_dump(mode="json")],
                    headers,
                    deadline,
                )
                owned_ev = own.resolve("settlement_event")
                if ev_outcome == "expired":
                    _count(drop_counter, "event", "shutdown_deadline", owned_ev)
                    break
                if ev_outcome != "sent":
                    _count(drop_counter, "event", ev_outcome, owned_ev)
                elif ev_response is not None:
                    # A 202 can still reject the event terminally (#9 pin).
                    _count(
                        drop_counter,
                        "event",
                        "ingest_rejected",
                        min(_ingest_rejected_count(ev_response, 1), owned_ev),
                    )
            while True:
                batch = own.pop_batch_in_hand(event_q, "event", _EXIT_BATCH_SIZE)
                if not batch:
                    break
                outcome, response = _post_json_response(
                    client,
                    ingest_url,
                    [p.event.model_dump(mode="json") for p in batch],
                    headers,
                    deadline,
                )
                if outcome == "expired":
                    own.requeue(event_q, batch, {"event": len(batch)})
                    break
                owned = own.resolve("event", len(batch))
                if outcome != "sent":
                    _count(drop_counter, "event", outcome, owned)
                elif response is not None:
                    # A 202 can still reject individual events terminally (#9).
                    _count(
                        drop_counter,
                        "event",
                        "ingest_rejected",
                        min(_ingest_rejected_count(response, len(batch)), owned),
                    )
        except Exception as exc:
            logger.warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
        if held_drops:
            logger.error("lifecycle.exit_flush_skipped_breaker_open: dropped=%d", held_drops)

    # The join below is the wall-clock bound (#2 re-review pin). At
    # interpreter shutdown new threads can be refused; the inline fallback
    # keeps the per-request deadline clamps as the only bound there.
    try:
        worker = threading.Thread(target=_run, daemon=True, name="solwyn-exit-flush")
        worker.start()
    except RuntimeError:
        _run()
    else:
        worker.join(timeout=max(0.0, deadline - _monotonic()))
    # Seal: claim a stuck worker's in-hand items, then sweep the queues.
    in_hand = own.seal()
    _count(drop_counter, "confirm", "shutdown_deadline", in_hand.get("confirm", 0))
    _count(
        drop_counter,
        "settlement_confirm",
        "shutdown_deadline",
        in_hand.get("settlement_confirm", 0),
    )
    _count(
        drop_counter,
        "event",
        "shutdown_deadline",
        in_hand.get("settlement_event", 0) + in_hand.get("event", 0),
    )
    # Whatever the deadline left behind is abandoned — count it.
    _drop_all(confirm_q, settlement_q, event_q, drop_counter, "shutdown_deadline")
    # Best-effort transport teardown for a still-stuck worker (NOT the bound).
    client.close()


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
    outcome, _ = _post_json_response(client, url, payload, headers, deadline)
    return outcome


def _post_json_response(
    client: httpx.Client,
    url: str,
    payload: object,
    headers: dict[str, str],
    deadline: float,
) -> tuple[str, httpx.Response | None]:
    """``_post_json`` plus the 2xx response, for callers that parse the body."""
    remaining = deadline - _monotonic()
    if remaining <= 0:
        return "expired", None
    try:
        response = client.post(url, json=payload, headers=headers, timeout=min(5.0, remaining))
        response.raise_for_status()
    except Exception as exc:
        if handle_read_only_key_error(exc):
            return "sent", None
        if _monotonic() >= deadline:
            # Deadline passed mid-request (e.g. the joiner's best-effort
            # client teardown surfaced here). The loss reason is the deadline,
            # not the transport artifact: "expired" makes the caller requeue
            # so the final sweep counts it shutdown_deadline.
            return "expired", None
        # The server dedups a later duplicate, and an exit hook must never
        # raise. Log the type name only.
        logger.warning("lifecycle.exit_post_failed: exc_type=%s", type(exc).__name__)
        return "retry_exhausted" if _is_retryable_exc(exc) else "terminal_status", None
    return "sent", response


def _ingest_rejected_count(response: httpx.Response, batch_size: int) -> int:
    """Best-effort count of per-event rejections in a 202 ingest body.

    A 202 can still reject individual events (terminal — they reject
    identically on every resubmission), and the whole-batch ``"sent"`` outcome
    would silently understate that loss. Capped at ``batch_size`` so a
    compromised/misrouted server cannot inflate drop counts. Fail-open: an
    unparseable body counts nothing (the exit hook must never raise).
    """
    try:
        rejected = response.json()["rejected"]
        if not isinstance(rejected, list):
            return 0
        return min(len(rejected), batch_size)
    except Exception:
        return 0


def _count(
    drop_counter: Callable[[str, str, int], None],
    kind: str,
    reason: str,
    n: int = 1,
) -> None:
    """Invoke the drop counter (a GC'd reporter gets ``_gc_drop_counter``)."""
    if n > 0:
        drop_counter(kind, reason, n)


def _drop_all(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    drop_counter: Callable[[str, str, int], None],
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
