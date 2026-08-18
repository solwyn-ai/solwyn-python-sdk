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
import dataclasses
import enum
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol, TypeVar

import httpx

from solwyn._read_only_key import handle_read_only_key_error

if TYPE_CHECKING:
    from solwyn.budget import _BudgetEnforcerBase
    from solwyn.circuit_breaker import CircuitBreaker
    from solwyn.reporter import (
        AsyncMetadataReporter,
        MetadataReporter,
        _PendingConfirm,
        _PendingEvent,
        _PendingSettlement,
        _ReceiptFoldState,
        _UntrackedReportState,
    )

logger = logging.getLogger(__name__)


def _log_warning(message: str, *args: object) -> None:
    """Invoke host logging without allowing teardown handlers to abort work."""
    with suppress(Exception):
        logger.warning(message, *args)


def _log_error(message: str, *args: object) -> None:
    with suppress(Exception):
        logger.error(message, *args)


def _log_debug(message: str, *args: object) -> None:
    with suppress(Exception):
        logger.debug(message, *args)


_T = TypeVar("_T")

# Patchable clock seam (mirrors reporter._monotonic): the exit-drain deadline
# arithmetic reads through this so tests can drive time deterministically.
_monotonic = time.monotonic


class _ExitIngestRejectionKind(enum.Enum):
    """Identity precision available from a successful exit ingest response."""

    CLEAN = "clean"
    EXACT = "exact"
    LEGACY = "legacy"
    MALFORMED = "malformed"


@dataclasses.dataclass(frozen=True)
class _ExitIngestRejections:
    kind: _ExitIngestRejectionKind
    indexes: frozenset[int] = frozenset()
    count: int = 0


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

# Budget enforcers holding PJ-2 leases. Held weakly, like the reporters: a
# closed enforcer has already surrendered and drops out on its own.
_LIVE_LEASE_HOLDERS: weakref.WeakSet[_BudgetEnforcerBase] = weakref.WeakSet()

_EXIT_BATCH_SIZE = 50

# Whole-process budget for handing leases back at exit. A surrender is a
# courtesy — the server reclaims the float at the lease deadline anyway — so it
# gets a small, hard bound AFTER the reporters have flushed real spend.
_EXIT_SURRENDER_BUDGET_S = 2.0


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
            _log_warning("lifecycle.fork_reset_failed: exc_type=%s", type(exc).__name__)


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
    QUEUES + reporter-owned advisory state + config (never ``reporter``, which
    would defeat the weak reference), so a collected reporter still drains over
    a temporary sync client. Spend losses are accounted the only way left —
    loudly, via ``_gc_drop_counter`` — while advisory failures remain silent. A
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
        reporter._untracked_state,
        reporter._receipt_fold_state,
        reporter._sdk_instance_id,
    )
    # Documented writable property; typeshed models finalize with __slots__
    # only, so mypy rejects the assignment it cannot see.
    finalizer.atexit = False  # type: ignore[misc]
    reporter._finalizer = finalizer


def register_lease_holder(enforcer: _BudgetEnforcerBase) -> None:
    """Track a budget enforcer so its held leases are released at exit."""
    _ensure_atexit_registered()
    _LIVE_LEASE_HOLDERS.add(enforcer)


def _exit_surrender_all() -> None:
    """Hand every still-held lease back (spec §5: release, not just flush).

    Runs AFTER the reporter flush — settled spend is the durable truth and owns
    the shutdown deadline; this is the DHCPRELEASE-style courtesy that lets the
    server re-lend the float immediately instead of waiting out the lease. It
    is breaker-admission gated exactly like exit confirms. Both enforcer
    flavours are drained over a temporary SYNC client: no event loop exists at
    interpreter exit.

    The budget is a TRUE WALL-CLOCK bound, the same way the reporter's exit
    flush gets one: the releases run on a daemon worker JOINED at the deadline.
    httpx timeouts cap socket operations, not total response time, so a
    trickling server would otherwise hold exit well past the budget — and
    unlike spend, nothing here is worth waiting for (an unsurrendered lease is
    simply reclaimed at its deadline). Nothing needs counting when the join
    times out: the payloads were already drained out of the ledger, and the
    server reclaims what it never heard about.
    """
    holders = list(_LIVE_LEASE_HOLDERS)
    if not holders:
        return
    deadline = _monotonic() + _EXIT_SURRENDER_BUDGET_S

    def _run() -> None:
        client: httpx.Client | None = None
        try:
            for holder in holders:
                try:
                    payloads = holder.lease_surrender_payloads()
                except Exception as exc:
                    _log_warning("lifecycle.exit_surrender_failed: exc_type=%s", type(exc).__name__)
                    continue
                for request in payloads:
                    if _monotonic() >= deadline:
                        _log_debug("lifecycle.exit_surrender_expired")
                        return
                    if client is None:
                        client = httpx.Client(timeout=5.0)
                    _post_confirm_blocking(
                        client,
                        f"{holder.api_url}/api/v1/budgets/lease/surrender",
                        request.model_dump(mode="json"),
                        holder._auth_headers(),
                        holder._control_plane_breaker,
                        deadline,
                    )
        except Exception as exc:  # the exit hook must never raise
            _log_warning("lifecycle.exit_surrender_failed: exc_type=%s", type(exc).__name__)
        finally:
            if client is not None:
                # Best-effort transport teardown (NOT the bound — the join is).
                client.close()

    # At interpreter shutdown new threads can be refused; the inline fallback
    # keeps the per-request deadline clamps as the only bound there.
    try:
        worker = threading.Thread(target=_run, daemon=True, name="solwyn-exit-surrender")
        worker.start()
    except RuntimeError:
        _run()
    else:
        worker.join(timeout=max(0.0, deadline - _monotonic()))


def _gc_drop_counter(kind: str, reason: str, n: int = 1) -> None:
    """Drop accounting for the GC-finalizer drain.

    A collected reporter's ``dropped_counts`` no longer exists, so a loss on
    this path is reported the only way left: a WARNING per counted drop.
    """
    # Finalizers must continue accounting the remaining items even when a
    # host-installed logging handler raises during interpreter teardown.
    _log_warning("lifecycle.gc_flush_dropped: kind=%s reason=%s n=%d", kind, reason, n)


class _ExitDropState:
    """Mutation-only exit accounting, emitted outside ownership locks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}

    def record(self, kind: str, reason: str, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            key = (kind, reason)
            self._counts[key] = self._counts.get(key, 0) + n

    def emit(self, drop_counter: Callable[[str, str, int], None]) -> None:
        with self._lock:
            counts = self._counts
            self._counts = {}
        for (kind, reason), n in counts.items():
            try:
                drop_counter(kind, reason, n)
            except Exception:
                # The durable disposition has already been recorded here and
                # ownership released. A hostile logger/callback cannot abort
                # emission of the remaining kinds.
                continue


def _exit_event_drop_weight(pending: object) -> int:
    """Receipt count one queued event's terminal loss represents.

    Mirrors ``_ReporterBase._event_drop_weight``: only an aggregate replay
    stands for more than one receipt, and only denied receipts ever carry
    ``receipt_aggregate_count``.
    """
    event = getattr(pending, "event", pending)
    return getattr(event, "receipt_aggregate_count", None) or 1


def _dispose_exit_event(
    event: object,
    reason: str,
    drop_counter: Callable[[str, str, int], None],
    receipt_fold_state: _ReceiptFoldState | None,
) -> None:
    """Fold one denied exit event, or retain ordinary counted-drop behavior."""
    if receipt_fold_state is not None:
        outcome = receipt_fold_state.fold(event)  # type: ignore[arg-type]
        if outcome == "folded":
            return
        if outcome == "overflow":
            drop_counter(
                "event",
                "receipt_fold_overflow",
                getattr(event, "receipt_aggregate_count", None) or 1,
            )
            return
        if outcome == "terminal":
            drop_counter("event", reason, getattr(event, "receipt_aggregate_count", None) or 1)
            return
    drop_counter("event", reason, 1)


def blocking_exit_flush(base: AsyncMetadataReporter) -> None:
    """Drain a (loop-less) async reporter's queues over a temporary sync client.

    Called from the atexit hook: no event loop exists at interpreter exit, so a
    synchronous ``httpx.Client`` posts the same sans-I/O spend and due advisory
    payloads the async path would, using the reporter's own URL and auth.
    """
    base._begin_blocking_exit()
    _drain_queues_blocking(
        base._confirm_queue,
        base._settlement_queue,
        base._queue,
        base.api_url,
        base.api_key,
        base._control_plane_breaker,
        base.shutdown_deadline,
        base._record_drop,
        base._untracked_state,
        base._receipt_fold_state,
        base._sdk_instance_id,
        base._maybe_log_drops,
    )


class _ExitOwnership:
    """Pop-and-claim ownership shared by the exit-drain worker and its joiner.

    Mirrors the reporter's ``_seal_delivery`` machinery for the blocking exit
    drain: the worker registers every popped item in-hand ATOMICALLY with the
    pop, so the deadline seal can never observe an item in neither a queue nor
    a hand — and a drop disposition is PUBLISHED under the same lock
    (``resolve_counted``), so the seal can never land in a release→publish gap
    and return while an item is accounted nowhere. After the seal,
    ``resolve``/``resolve_counted`` return owned=0 and ``requeue`` is
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
        self._items: dict[str, collections.deque[object]] = {}
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
                self._items.setdefault(kind, collections.deque()).append(item)
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
                self._items.setdefault(kind, collections.deque()).extend(batch)
            return batch

    def _release_items(self, kind: str, n: int) -> list[object]:
        items = self._items.setdefault(kind, collections.deque())
        released: list[object] = []
        for _ in range(min(n, len(items))):
            released.append(items.popleft())
        return released

    def resolve(self, kind: str, n: int = 1) -> int:
        """Release up to ``n`` in-hand items WITHOUT a drop publication.

        For SENT dispositions only — a lossy disposition must ride
        ``resolve_counted`` so its publication is atomic with the release.
        Returns how many the caller still owned (0 after the seal claimed
        them)."""
        with self._lock:
            held = self._counts.get(kind, 0)
            owned = min(n, held)
            self._counts[kind] = held - owned
            self._release_items(kind, owned)
            return owned

    def resolve_counted(
        self,
        kind: str,
        drop_counter: Callable[[str, str, int], None],
        publish_kind: str,
        reason: str,
        n: int = 1,
        cap: int | None = None,
        weigher: Callable[[object], int] | None = None,
    ) -> int:
        """Atomically release up to ``n`` in-hand items AND publish their drop.

        The mutation-only publication runs while the ownership lock is held:
        if the seal
        could land between the release and the ``drop_counter`` call, the
        drain would return with the item accounted NOWHERE — and at
        interpreter exit the daemon worker may never run again to record it
        (P1 re-review pin). Either this call publishes the disposition (and a
        later seal finds nothing in hand), or the seal already claimed the
        items as ``shutdown_deadline`` (owned=0, nothing published here).
        ``cap`` bounds the published count below ``owned`` (per-event 202
        rejections inside an otherwise-sent batch). ``weigher`` turns that item
        count into RECEIPT weight: an index-less body proves how many events
        were rejected but not which, so the ``cap`` HEAVIEST released items are
        charged — a bounded overcount instead of understating an aggregate
        receipt's cardinality. Callers pass the local ``_ExitDropState.record``
        primitive here; logging/user callbacks are emitted only after the seal.
        Returns owned.
        """
        with self._lock:
            held = self._counts.get(kind, 0)
            owned = min(n, held)
            self._counts[kind] = held - owned
            released = self._release_items(kind, owned)
            if weigher is None:
                publish_n = owned if cap is None else min(owned, cap)
            else:
                weights = sorted((weigher(item) for item in released), reverse=True)
                publish_n = sum(weights if cap is None else weights[:cap])
            if publish_n > 0:
                drop_counter(publish_kind, reason, publish_n)
            return owned

    def resolve_disposed(
        self,
        kind: str,
        disposer: Callable[[object], None],
        n: int = 1,
        indexes: set[int] | None = None,
    ) -> int:
        """Release items and publish selected event dispositions atomically."""
        with self._lock:
            held = self._counts.get(kind, 0)
            owned = min(n, held)
            self._counts[kind] = held - owned
            items = self._release_items(kind, owned)
            selected = range(len(items)) if indexes is None else sorted(indexes)
            for index in selected:
                if 0 <= index < len(items):
                    disposer(items[index])
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
                self._release_items(kind, n)
            queue.extendleft(reversed(items))
            return True

    def seal(self) -> dict[str, list[object]]:
        """Claim every in-hand item; further pops/resolves/requeues refuse."""
        with self._lock:
            self._sealed = True
            items = {kind: list(values) for kind, values in self._items.items() if values}
            self._counts.clear()
            self._items.clear()
            return items


def _drain_queues_blocking(
    confirm_q: collections.deque[_PendingConfirm],
    settlement_q: collections.deque[_PendingSettlement],
    event_q: collections.deque[_PendingEvent],
    api_url: str,
    api_key: str,
    breaker: CircuitBreaker | None,
    budget: float,
    drop_counter: Callable[[str, str, int], None],
    untracked_state: _UntrackedReportState | None = None,
    receipt_fold_state: _ReceiptFoldState | None = None,
    sdk_instance_id: str | None = None,
    drop_emitter: Callable[[], None] | None = None,
) -> None:
    """Best-effort, single-attempt, deadline-bounded exit flush of the queues.

    The deadline is a true wall-clock bound: the drain runs on a daemon worker
    JOINED at the deadline. httpx timeouts cap socket operations, not total
    response time, and closing a sync client does not reliably interrupt a
    blocked read — so no close()-based abort is trusted; the join is the
    bound. ``_ExitOwnership`` keeps the timeout honest: every pop registers
    the item in-hand atomically, every lossy disposition publishes atomically
    with its ownership release (``resolve_counted``), a timed-out join SEALS
    delivery (counting in-hand items ``shutdown_deadline`` and sweeping the
    queues), and a worker unblocking after the seal resolves to owned=0
    instead of double-counting.

    Every item popped here stays accountable until it has a disposition: each
    POST reports sent / failed / deadline-expired, failures are counted by
    kind, and anything still queued at the deadline is swept by the final
    ``shutdown_deadline`` count. Confirms ride the control-plane breaker's
    admission state machine — a known-down control plane refuses them (counted
    ``exit_breaker_open``) after at most one recovery probe — but metadata
    ingest is deliberately NOT breaker-gated (it is the durable spend truth),
    so events, including the events of breaker-held settlements, still get
    their deadline-bounded attempt. Due untracked-surface reports use the same
    deadline but remain fire-and-forget: their failures never enter spend-drop
    accounting. Never raises out of the exit hook.
    """
    drop_state = _ExitDropState()

    # Final/GC/atexit gets one unconditional aggregate attempt. Local imports
    # avoid the reporter -> lifecycle import cycle during module initialization.
    if receipt_fold_state is not None:
        from solwyn.reporter import _build_receipt_replay_events, _PendingEvent

        for key, fold in receipt_fold_state.take_for_cycle(final=True):
            event_q.extend(
                _PendingEvent(event)
                for event in _build_receipt_replay_events(key, fold, sdk_instance_id)
            )

    if not (
        confirm_q
        or settlement_q
        or event_q
        or (untracked_state is not None and untracked_state.reports_due(_monotonic()))
    ):
        return

    deadline = _monotonic() + budget
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = api_url.rstrip("/")
    confirm_url = f"{base_url}/api/v1/budgets/confirm"
    ingest_url = f"{base_url}/api/v1/metadata/ingest"
    untracked_url = f"{base_url}/api/v1/untracked-surfaces"
    own = _ExitOwnership()
    client = httpx.Client(timeout=5.0)

    def _dispose_pending(item: object, reason: str) -> None:
        _dispose_exit_event(
            item.event,  # type: ignore[attr-defined]
            reason,
            drop_state.record,
            receipt_fold_state,
        )

    def _dispose_for(reason: str) -> Callable[[object], None]:
        def dispose(item: object) -> None:
            _dispose_pending(item, reason)

        return dispose

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
                if outcome == "held":
                    # Control plane known-down: each queued confirm is
                    # refused per item (admit() short-circuits, no HTTP) so
                    # every pop stays ownership-tracked.
                    held_drops += own.resolve_counted(
                        "confirm", drop_state.record, "confirm", "exit_breaker_open"
                    )
                    continue
                if outcome != "sent":
                    own.resolve_counted("confirm", drop_state.record, "confirm", outcome)
                else:
                    own.resolve("confirm")
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
                if outcome == "held":
                    held_drops += own.resolve_counted(
                        "settlement_confirm",
                        drop_state.record,
                        "settlement_confirm",
                        "exit_breaker_open",
                    )
                elif outcome != "sent":
                    own.resolve_counted(
                        "settlement_confirm", drop_state.record, "settlement_confirm", outcome
                    )
                else:
                    own.resolve("settlement_confirm")
                # Ingest is NOT breaker-gated: the event still gets its exit
                # attempt even when its confirm was held or failed.
                ev_outcome, ev_response = _post_json_response(
                    client,
                    ingest_url,
                    [settlement.event.model_dump(mode="json")],
                    headers,
                    deadline,
                )
                if ev_outcome == "expired":
                    own.resolve_disposed(
                        "settlement_event",
                        lambda item: _dispose_pending(item, "shutdown_deadline"),
                    )
                    break
                if ev_outcome != "sent":
                    own.resolve_disposed(
                        "settlement_event",
                        _dispose_for(ev_outcome),
                    )
                elif ev_response is not None:
                    # A 202 can still reject the event terminally (#9 pin).
                    rejections = _parse_exit_ingest_rejections(ev_response, 1)
                    if rejections.kind is _ExitIngestRejectionKind.EXACT:
                        own.resolve_disposed(
                            "settlement_event",
                            lambda item: _dispose_pending(item, "ingest_rejected"),
                            indexes=set(rejections.indexes),
                        )
                    elif rejections.kind is _ExitIngestRejectionKind.LEGACY:
                        if rejections.count == 1:
                            own.resolve_disposed(
                                "settlement_event",
                                lambda item: _dispose_pending(item, "ingest_rejected"),
                            )
                        else:
                            own.resolve_counted(
                                "settlement_event",
                                drop_state.record,
                                "event",
                                "ingest_rejected",
                                cap=rejections.count,
                                weigher=_exit_event_drop_weight,
                            )
                    else:
                        own.resolve("settlement_event")
                else:
                    own.resolve("settlement_event")
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
                if outcome != "sent":
                    own.resolve_disposed(
                        "event",
                        _dispose_for(outcome),
                        n=len(batch),
                    )
                elif response is not None:
                    # A 202 can still reject individual events terminally (#9).
                    rejections = _parse_exit_ingest_rejections(response, len(batch))
                    if rejections.kind is _ExitIngestRejectionKind.EXACT:
                        own.resolve_disposed(
                            "event",
                            lambda item: _dispose_pending(item, "ingest_rejected"),
                            n=len(batch),
                            indexes=set(rejections.indexes),
                        )
                    elif rejections.kind is _ExitIngestRejectionKind.LEGACY:
                        if rejections.count == len(batch):
                            own.resolve_disposed(
                                "event",
                                lambda item: _dispose_pending(item, "ingest_rejected"),
                                n=len(batch),
                            )
                        else:
                            own.resolve_counted(
                                "event",
                                drop_state.record,
                                "event",
                                "ingest_rejected",
                                n=len(batch),
                                cap=rejections.count,
                                weigher=_exit_event_drop_weight,
                            )
                    else:
                        own.resolve("event", len(batch))
                else:
                    own.resolve("event", len(batch))
            if untracked_state is not None:
                advisory_reports = untracked_state.build_reports(_monotonic())
                for offset in range(0, len(advisory_reports), 100):
                    if _monotonic() >= deadline:
                        break
                    advisory_batch = advisory_reports[offset : offset + 100]
                    if not advisory_batch:
                        continue
                    untracked_state.mark_attempted(advisory_batch, _monotonic())
                    try:
                        response = client.post(
                            untracked_url,
                            json=[built.report.model_dump(mode="json") for built in advisory_batch],
                            headers=headers,
                            timeout=max(0.001, deadline - _monotonic()),
                        )
                        response.raise_for_status()
                    except Exception:
                        continue
                    untracked_state.mark_sent(advisory_batch)
        except Exception as exc:
            _log_warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
        if held_drops:
            _log_error("lifecycle.exit_flush_skipped_breaker_open: dropped=%d", held_drops)

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
    _count(drop_state.record, "confirm", "shutdown_deadline", len(in_hand.get("confirm", [])))
    _count(
        drop_state.record,
        "settlement_confirm",
        "shutdown_deadline",
        len(in_hand.get("settlement_confirm", [])),
    )
    for item in in_hand.get("settlement_event", []):
        _dispose_pending(item, "shutdown_deadline")
    for item in in_hand.get("event", []):
        _dispose_pending(item, "shutdown_deadline")
    # Whatever the deadline left behind is abandoned — count it.
    _drop_all(
        confirm_q,
        settlement_q,
        event_q,
        drop_state.record,
        "shutdown_deadline",
        receipt_fold_state,
    )
    # No logger or user callback has run under _ExitOwnership. Flush the
    # mutation-only ledger after seal/sweep, then let a live reporter emit its
    # rate-limited aggregate warning outside all lifecycle ownership locks.
    drop_state.emit(drop_counter)
    if drop_emitter is not None:
        with suppress(Exception):
            drop_emitter()
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
        _log_warning("lifecycle.exit_post_failed: exc_type=%s", type(exc).__name__)
        return "retry_exhausted" if _is_retryable_exc(exc) else "terminal_status", None
    return "sent", response


def _parse_exit_ingest_rejections(
    response: httpx.Response, batch_size: int
) -> _ExitIngestRejections:
    """Distinguish clean, exact, legacy count-only, and malformed 202 bodies.

    A 202 can still reject individual events (terminal — they reject
    identically on every resubmission), and the whole-batch ``"sent"`` outcome
    would silently understate that loss. A partial legacy index-less list
    proves only the count and must never guess which event to fold; a full
    legacy rejection proves every identity. Malformed responses retain the
    exit hook's fail-open behavior — but only bodies whose rejection COUNT is
    untrustworthy qualify: an index-shape violation (non-integer, duplicate,
    out-of-range) still proves the count, so it degrades to LEGACY rather than
    zeroing the loss accounting. Mirrors
    ``_ReporterBase._parse_ingest_rejections`` rule for rule: the two parsers
    must reach the SAME disposition for the same body.
    """
    try:
        rejected = response.json()["rejected"]
        if not isinstance(rejected, list):
            return _ExitIngestRejections(_ExitIngestRejectionKind.MALFORMED)
        if not rejected:
            return _ExitIngestRejections(_ExitIngestRejectionKind.CLEAN)
        if len(rejected) > batch_size:
            return _ExitIngestRejections(_ExitIngestRejectionKind.MALFORMED)
        indexes: list[int] = []
        indexes_complete = True
        shape_violation = False
        for item in rejected:
            try:
                raw_index = item["index"]
            except KeyError:
                indexes_complete = False
                continue
            if type(raw_index) is not int or raw_index < 0 or raw_index >= batch_size:
                shape_violation = True
                continue
            indexes.append(raw_index)
        unique_indexes = frozenset(indexes)
        if len(unique_indexes) != len(indexes):
            shape_violation = True
        if shape_violation or not indexes_complete:
            return _ExitIngestRejections(_ExitIngestRejectionKind.LEGACY, count=len(rejected))
        return _ExitIngestRejections(
            _ExitIngestRejectionKind.EXACT,
            indexes=unique_indexes,
            count=len(rejected),
        )
    except Exception:
        return _ExitIngestRejections(_ExitIngestRejectionKind.MALFORMED)


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
    receipt_fold_state: _ReceiptFoldState | None = None,
) -> None:
    """Clear every queue, counting the loss by kind.

    Each settlement pair loses its confirm AND its event — count both or
    ``dropped_counts`` understates real event loss.
    """
    n_confirm = _drain_count(confirm_q)
    settlements: list[_PendingSettlement] = []
    while settlement_q:
        settlements.append(settlement_q.popleft())
    events: list[_PendingEvent] = []
    while event_q:
        events.append(event_q.popleft())
    _count(drop_counter, "confirm", reason, n_confirm)
    _count(drop_counter, "settlement_confirm", reason, len(settlements))
    for settlement in settlements:
        _dispose_exit_event(settlement.event, reason, drop_counter, receipt_fold_state)
    for pending in events:
        _dispose_exit_event(pending.event, reason, drop_counter, receipt_fold_state)


def _exit_flush_all() -> None:
    """The single atexit hook: flush every still-open live reporter."""
    for sync_reporter in list(_LIVE_SYNC_REPORTERS):
        try:
            # _shutdown means stop REQUESTED, _delivery_closed is the EARLY
            # enqueue-refusal gate, and _delivery_completed covers only spend
            # sealing. Lifecycle rescue stays armed until transport cleanup is
            # also complete, so a false synchronized read joins/retries the
            # close path instead of skipping a partial transition.
            if not sync_reporter._close_is_completed():
                sync_reporter.close()
        except Exception as exc:
            _log_warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
    for async_reporter in list(_LIVE_ASYNC_REPORTERS):
        try:
            # _close_completed (NOT _closed): a cancelled or partial close()
            # leaves queued spend behind — this hook stays armed until a close
            # has actually finished its flush.
            if not async_reporter._close_completed:
                blocking_exit_flush(async_reporter)
        except Exception as exc:
            _log_warning("lifecycle.exit_flush_failed: exc_type=%s", type(exc).__name__)
    try:
        _exit_surrender_all()
    except Exception as exc:
        _log_warning("lifecycle.exit_surrender_failed: exc_type=%s", type(exc).__name__)
