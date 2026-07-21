"""Async metadata and provider-health reporter.

MetadataReporter (sync, background thread queue) and AsyncMetadataReporter
(asyncio.create_task) batch and flush metadata events plus current circuit-breaker
snapshots to the Solwyn cloud API. Neither blocks the LLM call path.

Delivery is at-least-once: confirms, settlements, and metadata batches are
retried with bounded backoff, a control-plane-breaker-open flush HOLDS work
instead of dropping it, and every unavoidable drop is counted (``dropped_counts``)
and logged at a bounded rate. The server dedups duplicate sends via an
idempotency ledger, so retries (and fork/shutdown races) are safe.

Events contain cost/latency metadata only -- never prompts or responses.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import functools
import logging
import re
import threading
import time
import weakref
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

import httpx

from solwyn._lifecycle import (
    _drain_count,
    _is_retryable_exc,
    register_async_reporter,
    register_fork_reset,
    register_sync_reporter,
)
from solwyn._read_only_key import handle_read_only_key_error
from solwyn._types import BreakerStateReport, BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState

logger = logging.getLogger(__name__)

# Patchable clock seam: all backoff/deadline arithmetic reads through this so
# tests can drive time deterministically.
_monotonic = time.monotonic

# At most one aggregated drop WARNING per this many seconds (the first drop ever
# logs immediately).
_DROP_LOG_INTERVAL = 60.0

# Bound for the standalone confirm + settlement queues. Distinct from the
# (larger) event queue bound so a control-plane outage cannot let pending
# settlements grow without limit.
_MAX_PENDING_CONTROL = 1000

_T = TypeVar("_T")

# Raw C0/DEL/C1 control bytes. A compliant server repr-escapes these before
# they reach the wire, so the substitution below is a no-op on compliant
# bodies and verbatim logging is preserved.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _escape_control(value: str) -> str:
    """Repr-escape raw control bytes in a server-echoed value.

    Defense-in-depth: the server should have escaped these already — a
    server-side escaping regression (or a misrouted/compromised endpoint)
    must not be able to inject forged log lines or ANSI sequences into
    customer logs via echoed model names or rejection messages.
    """
    return _CONTROL_CHARS.sub(lambda m: repr(m.group())[1:-1], value)


class _SendOutcome(enum.Enum):
    """The disposition of one send attempt."""

    SENT = "sent"  # delivered, or a read-only-key policy skip (terminal, not a loss)
    HELD = "held"  # control-plane breaker refused admission — not an attempt
    RETRY = "retry"  # transient failure — retry after backoff
    DROPPED = "dropped"  # terminal status / poison item — never retry


@dataclasses.dataclass
class _PendingConfirm:
    """A standalone confirm awaiting delivery, with retry bookkeeping."""

    request: BudgetConfirmRequest
    attempts: int = 0
    next_attempt_at: float = 0.0


@dataclasses.dataclass
class _PendingEvent:
    """A metadata event awaiting delivery, with retry bookkeeping."""

    event: MetadataEvent
    attempts: int = 0
    next_attempt_at: float = 0.0


@dataclasses.dataclass
class _PendingSettlement:
    """A reservation settlement: its confirm (retryable) paired with its event."""

    confirm: _PendingConfirm
    event: MetadataEvent


class _ReporterBase:
    """Sans-I/O base class for metadata reporting.

    Manages the pending queues, retry/backoff arithmetic, and drop accounting.
    Subclasses add the I/O layer (threading or asyncio) and HTTP transport.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        self.max_in_flight = max_in_flight
        # At-least-once delivery bounds (see SolwynConfig.reporter_*): retries
        # per item before a counted drop, exponential backoff base/cap, and the
        # single wall-clock budget the shutdown/exit flush chain may spend.
        self.max_send_attempts = max_send_attempts
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_cap = retry_backoff_cap
        self.shutdown_deadline = shutdown_deadline
        self._breaker_snapshots = breaker_snapshots
        self._sdk_instance_id = sdk_instance_id
        self._breaker_reporting_enabled = breaker_reporting_enabled
        # Shared with the budget enforcer's check path: a streak of confirm
        # failures opens this breaker so a known-down confirm is HELD (not
        # dropped) without paying the timeout. Never a provider — excluded from
        # _build_breaker_reports (that is provider health only).
        self._control_plane_breaker = control_plane_breaker
        self._breaker_project_id: str | None = None
        self._breaker_project_lock = threading.Lock()

        # Plain deque (NO maxlen): bounds are enforced by _enqueue_bounded so an
        # overflow is COUNTED, not silently dropped by the deque itself.
        self._queue: collections.deque[_PendingEvent] = collections.deque()
        self._in_flight = 0
        self._consecutive_confirm_failures = 0
        self._confirm_failure_threshold = 10
        self._consecutive_unparseable_responses = 0
        self._unparseable_response_threshold = 10

        # Drop accounting. Guarded by its own lock so enqueues from non-loop
        # threads and the flush loop never race the counters.
        self._drop_counts: dict[str, int] = {}
        self._drop_lock = threading.Lock()
        self._drops_since_last_log = 0
        self._last_drop_log_at: float | None = None
        self._logged_first_drop = False

    def observe_project_id(self, project_id: str | None) -> None:
        """Remember the non-empty project resolved by a real/cached budget check."""
        if not project_id:
            return
        with self._breaker_project_lock:
            self._breaker_project_id = project_id

    def _breaker_reporting_project_id(self) -> str | None:
        """Return the learned project only when breaker reporting can run."""
        if (
            not self._breaker_reporting_enabled
            or self._breaker_snapshots is None
            or self._sdk_instance_id is None
        ):
            return None
        with self._breaker_project_lock:
            return self._breaker_project_id

    def _build_breaker_reports(self) -> list[tuple[str, BreakerStateReport]]:
        """Eagerly build one timestamped current-state report cycle."""
        reports: list[tuple[str, BreakerStateReport]] = []
        project_id = self._breaker_reporting_project_id()
        if project_id is None or self._breaker_snapshots is None or self._sdk_instance_id is None:
            return reports
        try:
            snapshots = self._breaker_snapshots()
            reported_at = datetime.now(UTC)
        except Exception as exc:
            logger.warning(
                "reporter.breaker_snapshot_failed: exc_type=%s",
                type(exc).__name__,
            )
            return reports
        for provider, snapshot in snapshots:
            try:
                reports.append(
                    (
                        project_id,
                        BreakerStateReport(
                            provider=provider,
                            state=snapshot.state,
                            failure_count=snapshot.failure_count,
                            success_count=snapshot.success_count,
                            reported_at=reported_at,
                            sdk_instance_id=self._sdk_instance_id,
                        ),
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "reporter.breaker_snapshot_invalid: provider=%s exc_type=%s",
                    provider.value,
                    type(exc).__name__,
                )
        return reports

    # ------------------------------------------------------------------
    # Queueing + bounds
    # ------------------------------------------------------------------

    def _enqueue(self, event: MetadataEvent) -> None:
        """Add an event to the queue.  Counted drop-oldest on overflow."""
        self._enqueue_bounded(self._queue, _PendingEvent(event), self.max_queue_size, "event")

    def _enqueue_bounded(
        self,
        queue: collections.deque[_T],
        item: _T,
        maxlen: int,
        kind: str,
        on_evict: Callable[[_T], None] | None = None,
    ) -> None:
        """Append with a drop-oldest bound that COUNTS the dropped item.

        The len-check + append is not atomic across threads; the bound is
        approximate by design — CPython deque ops are individually thread-safe,
        so a small transient overshoot is acceptable and never corrupts state.
        ``on_evict`` sees the evicted item so a compound item can salvage or
        account its other half (a settlement pair's metadata event).
        """
        if len(queue) >= maxlen:
            try:
                evicted = queue.popleft()
            except IndexError:
                pass
            else:
                self._count_drop(kind, "overflow")
                if on_evict is not None:
                    on_evict(evicted)
        queue.append(item)

    def _move_event_to_queue(self, event: MetadataEvent) -> None:
        """Move a settlement's metadata event onto the main event queue.

        cost_events ingest is the durable spend truth, so the event must never
        be lost because its confirm failed — it rides the normal event path.
        """
        self._enqueue_bounded(self._queue, _PendingEvent(event), self.max_queue_size, "event")

    def _drain_batch(self) -> list[MetadataEvent]:
        """Drain up to batch_size events from the front of the queue."""
        batch: list[MetadataEvent] = []
        for _ in range(min(self.batch_size, len(self._queue))):
            try:
                batch.append(self._queue.popleft().event)
            except IndexError:
                break
        return batch

    # ------------------------------------------------------------------
    # Retry / backoff / deadline arithmetic
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempts: int) -> float:
        """Exponential backoff for the ``attempts``-th failed try (1-indexed)."""
        return min(self.retry_backoff_cap, self.retry_backoff_base * 2.0 ** (attempts - 1))

    def _resolve_retryable(self, *, attempts: int) -> tuple[bool, float]:
        """Return (finished, next_attempt_at) for a RETRY outcome.

        ``finished`` is True when the retry budget is spent (the item must be
        dropped); otherwise ``next_attempt_at`` is when it may be retried.
        """
        if attempts >= self.max_send_attempts:
            return True, 0.0
        return False, _monotonic() + self._backoff_delay(attempts)

    def _deadline_expired(self, deadline: float | None) -> bool:
        """Whether the shared shutdown deadline (if any) has been reached."""
        return deadline is not None and _monotonic() >= deadline

    def _send_timeout(self, deadline: float | None) -> float:
        """Per-request timeout, clamped into the remaining deadline window."""
        if deadline is None:
            return 5.0
        return max(0.05, min(5.0, deadline - _monotonic()))

    # ------------------------------------------------------------------
    # Drop accounting
    # ------------------------------------------------------------------

    def _count_drop(self, kind: str, reason: str, n: int = 1) -> None:
        """Record ``n`` dropped spend items keyed ``"{kind}.{reason}"``.

        Logging is driven from HERE (rate-limited by ``_maybe_log_drops``), not
        only from flush cycles: the first drop ever must warn immediately, and
        a post-close drop has no later flush cycle to surface it.
        """
        if n <= 0:
            return
        with self._drop_lock:
            key = f"{kind}.{reason}"
            self._drop_counts[key] = self._drop_counts.get(key, 0) + n
            self._drops_since_last_log += n
        self._maybe_log_drops()

    def _count_settlement_drop(self, reason: str, n: int = 1) -> None:
        """Count a lost settlement PAIR: its confirm AND its metadata event.

        Counting only the confirm half understates real event loss.
        """
        self._count_drop("settlement_confirm", reason, n)
        self._count_drop("event", reason, n)

    def _maybe_log_drops(self, *, force: bool = False) -> None:
        """Emit an aggregated drop WARNING, rate-limited to one per interval.

        The first drop ever logs immediately; afterwards at most one aggregate
        line per ``_DROP_LOG_INTERVAL`` (or immediately when ``force`` is set,
        e.g. from ``close()``).
        """
        with self._drop_lock:
            if self._drops_since_last_log == 0:
                return
            now = _monotonic()
            first = not self._logged_first_drop
            due = (
                first
                or force
                or self._last_drop_log_at is None
                or (now - self._last_drop_log_at) >= _DROP_LOG_INTERVAL
            )
            if not due:
                return
            since = self._drops_since_last_log
            totals = dict(self._drop_counts)
            self._drops_since_last_log = 0
            self._last_drop_log_at = now
            self._logged_first_drop = True
        logger.warning("reporter.spend_events_dropped: new=%d totals=%s", since, totals)

    @property
    def dropped_counts(self) -> dict[str, int]:
        """A snapshot of counted, undeliverable spend items by ``kind.reason``."""
        with self._drop_lock:
            return dict(self._drop_counts)

    # ------------------------------------------------------------------
    # Auth + failure accounting
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for cloud API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _confirm_failure_outcome(self, exc: Exception) -> _SendOutcome:
        """Map a confirm send failure to RETRY (transient) or DROPPED (terminal)."""
        if _is_retryable_exc(exc):
            return _SendOutcome.RETRY
        if isinstance(exc, httpx.HTTPStatusError):
            # Status code only — never a body (privacy).
            logger.warning(
                "reporter.confirm_terminal_status: status=%d",
                exc.response.status_code,
            )
        return _SendOutcome.DROPPED

    def _batch_failure_outcome(self, exc: Exception) -> _SendOutcome:
        """Map an ingest send failure to RETRY (transient) or DROPPED (terminal)."""
        return _SendOutcome.RETRY if _is_retryable_exc(exc) else _SendOutcome.DROPPED

    def _record_confirm_success(self) -> None:
        """Reset the confirm failure counter after a successful confirm POST."""
        self._consecutive_confirm_failures = 0

    def _record_confirm_failure(self, exc: Exception) -> None:
        """Track confirm send failures and escalate persistent outages."""
        self._consecutive_confirm_failures += 1
        count = self._consecutive_confirm_failures
        if count >= self._confirm_failure_threshold:
            logger.error(
                "reporter.confirm_send_persistent_failure: exc_type=%s consecutive_failures=%d",
                type(exc).__name__,
                count,
            )
        else:
            logger.warning(
                "reporter.confirm_send_failed: exc_type=%s",
                type(exc).__name__,
            )

    def _record_parseable_response(self) -> None:
        """Reset the unparseable counter after a successfully parsed ingest body."""
        self._consecutive_unparseable_responses = 0

    def _record_unparseable_response(self, exc: Exception) -> None:
        """Track unparseable ingest bodies and escalate persistent contract drift."""
        self._consecutive_unparseable_responses += 1
        count = self._consecutive_unparseable_responses
        if count >= self._unparseable_response_threshold:
            logger.error(
                "reporter.ingest_response_unparseable_persistent: "
                "exc_type=%s consecutive_failures=%d",
                type(exc).__name__,
                count,
            )
        else:
            logger.warning(
                "reporter.ingest_response_unparseable: exc_type=%s",
                type(exc).__name__,
            )

    def _log_ingest_rejections(self, response: httpx.Response, batch_size: int) -> None:
        """Surface per-event rejection dispositions from the 202 ingest body.

        The API returns 202 for every well-formed request; accepted events are
        already durable and rejected events are terminal — they reject
        identically on every resubmission until a pricing entry lands
        server-side — so rejections are logged and dropped, never re-queued.
        One WARNING per distinct (code, model) per batch keeps a fleet stuck
        on a single unpriced model from flooding logs. The server's message is
        logged verbatim (repr-escaped server-side) and never parsed.

        Fail-open: a malformed body must never raise into the flush loop —
        fall back to the pre-v0.1.7 count-only acknowledgment and log once.
        """
        try:
            rejected = response.json()["rejected"]
            if not isinstance(rejected, list):
                # A falsy non-list ("rejected": null) must take the fail-open
                # path, not masquerade as a clean batch.
                raise TypeError(f"rejected is {type(rejected).__name__}, expected list")
            if not rejected:
                self._record_parseable_response()
                return
            if len(rejected) > batch_size:
                # Contract violation — and a cap: a compromised/misrouted
                # server cannot flood logs or the groups dict with more
                # distinct (code, model) pairs than events were submitted.
                raise ValueError("server rejected more events than were submitted")
            # Aggregate before logging so a malformed entry can never leave a
            # half-logged flush behind. First message per group wins (the
            # server's message is keyed by the same (code, model) inputs).
            groups: dict[tuple[str, str], tuple[int, str]] = {}
            for rejection in rejected:
                key = (
                    _escape_control(str(rejection["code"])),
                    _escape_control(str(rejection["model"])),
                )
                count, message = groups.get(key, (0, _escape_control(str(rejection["message"]))))
                groups[key] = (count + 1, message)
            self._record_parseable_response()
        except Exception as exc:
            self._record_unparseable_response(exc)
            return
        try:
            for (code, model), (count, message) in groups.items():
                logger.warning(
                    "reporter.ingest_events_rejected: code=%s model=%s count=%d message=%s",
                    code,
                    model,
                    count,
                    message,
                )
        except Exception:
            # The body parsed fine and the batch IS durable server-side — only
            # the host's logging stack raised (e.g. a user-installed Filter).
            # A fallback log would raise again; rejection detail is
            # best-effort, but letting this propagate would mislabel the batch
            # as a send failure in _send_batch's accounting.
            return


class MetadataReporter(_ReporterBase):
    """Synchronous metadata reporter with a background daemon thread.

    Usage::

        reporter = MetadataReporter(api_url, api_key)
        reporter.report(event)  # non-blocking
        # ...
        reporter.close()  # flush remaining events

    Or as a context manager::

        with MetadataReporter(api_url, api_key) as reporter:
            reporter.report(event)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
            breaker_snapshots=breaker_snapshots,
            sdk_instance_id=sdk_instance_id,
            breaker_reporting_enabled=breaker_reporting_enabled,
            control_plane_breaker=control_plane_breaker,
            max_send_attempts=max_send_attempts,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_cap=retry_backoff_cap,
            shutdown_deadline=shutdown_deadline,
        )
        self._http = httpx.Client(timeout=10.0)
        self._shutdown = threading.Event()
        self._in_flight_lock = threading.Lock()
        self._breaker_worker_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        # Set by the fork handler so the next enqueue relaunches the (fork-killed)
        # flush thread. False otherwise, so a never-forked reporter never spawns
        # an unexpected thread from report().
        self._needs_thread_restart = False
        self._breaker_worker: threading.Thread | None = None
        # Separate queues for standalone confirms and settlements. Fire-and-
        # forgets onto these so the user's thread is not blocked on an
        # httpx.post to Solwyn. Bounds are enforced by _enqueue_bounded.
        self._confirm_queue: collections.deque[_PendingConfirm] = collections.deque()
        self._settlement_queue: collections.deque[_PendingSettlement] = collections.deque()
        # Shutdown ownership. Every enqueue and every popped-but-unresolved
        # ("in hand") drain item is tracked under this lock so a timed-out
        # close() can take final ownership of ALL undelivered spend instead of
        # letting a stuck flush thread requeue it (or a racing producer append
        # it) into a queue nothing will ever drain — see _seal_delivery.
        self._ownership_lock = threading.Lock()
        self._in_hand: dict[str, int] = {}
        self._delivery_closed = False
        self._thread = self._launch_thread()
        # Exit flush: if the process exits without close(), the atexit hook runs
        # close() so queued spend is still delivered. The live flush thread keeps
        # this reporter alive (its bound-method target), so no finalizer is
        # needed on the sync path — close() is the whole story.
        register_sync_reporter(self)
        register_fork_reset(self)

    def _launch_thread(self) -> threading.Thread:
        """Create and start a fresh flush thread."""
        thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="solwyn-reporter",
        )
        thread.start()
        return thread

    def _ensure_thread(self) -> None:
        """Relaunch the flush thread if it is not running (e.g. after a fork).

        The flush thread starts in ``__init__``; a forked child inherits a DEAD
        thread. Restarting it lazily on the next enqueue — rather than inside the
        ``os.register_at_fork`` handler, where starting a thread is fragile —
        keeps delivery alive in the child. Guarded by ``_needs_thread_restart``
        (set only by the fork handler) so a never-forked reporter never spawns an
        unexpected thread; a no-op for a closed reporter.
        """
        if not self._needs_thread_restart or self._shutdown.is_set():
            return
        with self._thread_lock:
            if not self._needs_thread_restart or self._shutdown.is_set():
                return
            self._needs_thread_restart = False
            self._thread = self._launch_thread()

    def _reset_after_fork_in_child(self) -> None:
        """Repair a forked child: fresh locks + client; defer the thread relaunch.

        Threads do not survive ``fork()``, so the inherited flush thread is dead
        in the child. Starting a replacement INSIDE this fork handler is fragile
        (the child is in a delicate post-fork state), so the thread is instead
        relaunched lazily by ``_ensure_thread`` on the next enqueue. Locks
        possibly held by a now-absent thread are replaced, and the inherited
        client is abandoned (never closed — the parent owns those sockets).
        Queued items duplicated into the child by fork are deliberately KEPT: the
        server dedups. A closed reporter stays closed (fresh shutdown Event kept
        set — see below).
        """
        self._breaker_project_lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._in_flight_lock = threading.Lock()
        self._breaker_worker_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._ownership_lock = threading.Lock()
        # In-hand items live on the parent's (now-absent) flush thread stack;
        # the parent delivers them. The child's view starts clean.
        self._in_hand = {}
        self._in_flight = 0
        self._breaker_worker = None
        self._http = httpx.Client(timeout=10.0)
        if not self._shutdown.is_set():
            # Replace the inherited Event and arm the lazy relaunch; a closed
            # reporter keeps its set Event and never relaunches.
            self._shutdown = threading.Event()
            self._needs_thread_restart = True

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        if self._shutdown.is_set():
            # Nothing will ever drain a post-close enqueue — count it like
            # report_confirm/report_settlement do instead of retaining it
            # silently (a late streaming on_complete can land here).
            self._count_drop("event", "closed_enqueue")
            return
        self._ensure_thread()
        if not self._enqueue_owned(self._queue, _PendingEvent(event), self.max_queue_size, "event"):
            self._count_drop("event", "closed_enqueue")

    def close(self, timeout: float | None = None) -> None:
        """Flush remaining events and shut down within a single deadline.

        ``timeout`` (default ``self.shutdown_deadline``) bounds the WHOLE
        shutdown chain — thread join, final flush, and breaker report cycle all
        share one monotonic ``deadline``. Work still queued when it is reached is
        counted ``shutdown_deadline`` and dropped rather than paying a serial
        per-request timeout chain against a black-holed control plane.
        """
        budget = self.shutdown_deadline if timeout is None else timeout
        deadline = _monotonic() + budget
        # Serialize shutdown with cadence-triggered breaker launches. If a
        # breaker cycle is active now, it is adopted as the final cycle.
        with self._breaker_worker_lock:
            active_breaker_worker = self._breaker_worker
            if active_breaker_worker is not None and not active_breaker_worker.is_alive():
                active_breaker_worker = None
            self._shutdown.set()

        # Join the ingest loop within the remaining budget. If the join times
        # out, the final flush below STILL runs: deque ops are thread-safe and a
        # duplicate send is server-deduped via the idempotency ledger.
        self._thread.join(timeout=max(0.0, deadline - _monotonic()))
        self._flush_remaining(deadline=deadline, final=True)
        self._seal_delivery()

        if active_breaker_worker is None:
            active_breaker_worker = self._start_breaker_cycle(
                during_shutdown=True, deadline=deadline
            )
        if active_breaker_worker is not None:
            active_breaker_worker.join(timeout=max(0.0, deadline - _monotonic()))
        self._http.close()

    def __enter__(self) -> MetadataReporter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _flush_loop(self) -> None:
        """Background thread: periodically flush batches to the cloud."""
        while not self._shutdown.is_set():
            if self._shutdown.wait(timeout=self.flush_interval):
                break
            try:
                self._flush_remaining()
            except Exception as exc:
                # The flush loop must survive anything a drain raises (a
                # transport edge, a concurrent-close artifact): a dead flush
                # thread strands ALL queued spend until overflow, silently.
                logger.warning("reporter.flush_cycle_failed: exc_type=%s", type(exc).__name__)
            self._start_breaker_cycle()

    def _flush_remaining(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Flush queued confirms, settlements, then metadata events in batches.

        ``final`` is the shutdown/exit mode: backoff gates are ignored, a RETRY
        outcome drops (there is no later cycle), and a HELD outcome drops the
        remainder (never hammer a known-down control plane while exiting).
        ``deadline`` (a monotonic instant) caps the whole chain; work still
        queued when it is reached is counted ``shutdown_deadline`` and dropped.
        """
        self._drain_confirms(deadline=deadline, final=final)
        self._drain_settlements(deadline=deadline, final=final)
        self._drain_event_batches(deadline=deadline, final=final)
        self._maybe_log_drops(force=final)

    # ------------------------------------------------------------------
    # Shutdown ownership: enqueue admission + in-hand drain items
    # ------------------------------------------------------------------

    def _enqueue_owned(
        self,
        queue: collections.deque[_T],
        item: _T,
        maxlen: int,
        kind: str,
        on_evict: Callable[[_T], None] | None = None,
    ) -> bool:
        """Append under the ownership lock; refuse once delivery has sealed.

        Returns False when ``_seal_delivery`` already ran — the caller counts
        the refusal. Without this gate an enqueue could pass the ``_shutdown``
        check, lose the race with close()'s final drain, and append into a
        queue nothing will ever drain. The drop-oldest bound is enforced
        OUTSIDE the lock (the bound is approximate by design; deque ops are
        individually thread-safe) so eviction accounting never logs while
        holding the ownership lock.
        """
        with self._ownership_lock:
            if self._delivery_closed:
                return False
            queue.append(item)
        if len(queue) > maxlen:
            try:
                evicted = queue.popleft()
            except IndexError:
                pass
            else:
                self._count_drop(kind, "overflow")
                if on_evict is not None:
                    on_evict(evicted)
        return True

    def _take_in_hand(self, kind: str, n: int = 1) -> None:
        """Mark ``n`` just-popped drain items as in hand (owned by this drain)."""
        with self._ownership_lock:
            self._in_hand[kind] = self._in_hand.get(kind, 0) + n

    def _resolve_in_hand(
        self,
        kind: str,
        n: int = 1,
        requeue: Callable[[], None] | None = None,
    ) -> tuple[int, bool]:
        """Release ``n`` in-hand items with a terminal disposition.

        Returns ``(owned, requeued)``. ``owned`` is how many of the items
        close()'s seal has NOT already claimed-and-counted — the caller
        counts any drop for ``owned`` items only, so a seal never double
        counts. ``requeue`` runs under the ownership lock and only while
        delivery is open; once sealed, requeueing would strand the items in a
        queue with no owner, so the caller counts them instead.
        """
        with self._ownership_lock:
            held = self._in_hand.get(kind, 0)
            owned = min(held, n)
            if owned:
                self._in_hand[kind] = held - owned
            if requeue is not None and not self._delivery_closed:
                requeue()
                return owned, True
            return owned, False

    def _park_confirm(self, pending: _PendingConfirm) -> None:
        """Requeue a still-retryable confirm head — or count it once sealed."""
        owned, requeued = self._resolve_in_hand(
            "confirm", requeue=lambda: self._confirm_queue.appendleft(pending)
        )
        if not requeued and owned:
            self._count_drop("confirm", "shutdown_deadline", n=owned)

    def _park_settlement(self, pending: _PendingSettlement) -> None:
        """Requeue a still-retryable settlement head — or count the pair."""
        owned, requeued = self._resolve_in_hand(
            "settlement", requeue=lambda: self._settlement_queue.appendleft(pending)
        )
        if not requeued and owned:
            self._count_settlement_drop("shutdown_deadline", n=owned)

    def _ship_settlement_event(self, event: MetadataEvent, owned: int = 1) -> None:
        """Hand a resolved settlement's event to the event queue.

        cost_events ingest is the durable spend truth, so the event must never
        be lost because its confirm failed. Once delivery has sealed nothing
        will drain the event queue — count the loss instead, unless close()'s
        seal already claimed and counted the pair (``owned == 0``).
        """
        enqueued = self._enqueue_owned(
            self._queue, _PendingEvent(event), self.max_queue_size, "event"
        )
        if not enqueued and owned:
            self._count_drop("event", "shutdown_deadline", n=owned)

    def _move_event_to_queue(self, event: MetadataEvent) -> None:
        """Sync override: settlement events route through the ownership gate."""
        self._ship_settlement_event(event)

    def _seal_delivery(self) -> None:
        """Take final ownership of every queued or in-hand item at close().

        Runs after close()'s final flush. A join-timeout-stranded flush thread
        may still hold popped items mid-POST (and would otherwise requeue them
        into a dead queue), and a racing producer may have appended after the
        final drain passed its queue. Atomically: seal delivery (enqueues and
        requeues are refused from here on), claim whatever re-appeared in the
        queues, and count the flush thread's in-hand items — their owner can
        no longer deliver them within the deadline. A stuck send that later
        succeeds leaves a conservative overcount for that one item; drops are
        never UNDERstated.
        """
        with self._ownership_lock:
            self._delivery_closed = True
            in_hand = dict(self._in_hand)
            self._in_hand.clear()
            n_confirm = _drain_count(self._confirm_queue)
            n_settlement = _drain_count(self._settlement_queue)
            n_event = _drain_count(self._queue)
        self._count_drop("confirm", "shutdown_deadline", n=n_confirm + in_hand.get("confirm", 0))
        self._count_settlement_drop(
            "shutdown_deadline", n=n_settlement + in_hand.get("settlement", 0)
        )
        self._count_drop("event", "shutdown_deadline", n=n_event + in_hand.get("event", 0))
        self._maybe_log_drops(force=True)

    # ------------------------------------------------------------------
    # Drains
    # ------------------------------------------------------------------

    def _drain_confirms(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Send due confirms in FIFO order; a retrying head parks the queue."""
        while self._confirm_queue:
            if self._deadline_expired(deadline):
                self._count_drop(
                    "confirm", "shutdown_deadline", n=_drain_count(self._confirm_queue)
                )
                break
            try:
                # Atomic pop, never peek-then-pop: once close()'s bounded join
                # times out, close/atexit drains CONCURRENTLY with the flush
                # thread — two drainers peeking one head would double-send it
                # and IndexError on the second popleft.
                pending = self._confirm_queue.popleft()
            except IndexError:
                break
            self._take_in_hand("confirm")
            if not final and pending.next_attempt_at > _monotonic():
                self._park_confirm(pending)
                break  # head still backing off — nothing behind is due earlier
            outcome = self._send_confirm(pending.request, timeout=self._send_timeout(deadline))
            if outcome is _SendOutcome.SENT:
                self._resolve_in_hand("confirm")
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    owned, _ = self._resolve_in_hand("confirm")
                    self._count_drop(
                        "confirm", "exit_breaker_open", n=owned + _drain_count(self._confirm_queue)
                    )
                else:
                    self._park_confirm(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.attempts)
                if finished or final:
                    owned, _ = self._resolve_in_hand("confirm")
                    self._count_drop("confirm", "retry_exhausted", n=owned)
                    continue
                pending.next_attempt_at = next_at
                self._park_confirm(pending)
                break  # FIFO: nothing behind a backing-off head may jump it
            owned, _ = self._resolve_in_hand("confirm")
            self._count_drop("confirm", "terminal_status", n=owned)

    def _drain_settlements(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Resolve each settlement's confirm first, then hand its event to the
        event queue (confirm-before-metadata order per item is load-bearing).
        A retrying head parks the queue behind it (FIFO)."""
        while self._settlement_queue:
            if self._deadline_expired(deadline):
                self._count_settlement_drop(
                    "shutdown_deadline", n=_drain_count(self._settlement_queue)
                )
                break
            try:
                # Atomic pop — see _drain_confirms for the concurrent-drain
                # rationale.
                pending = self._settlement_queue.popleft()
            except IndexError:
                break
            self._take_in_hand("settlement")
            if not final and pending.confirm.next_attempt_at > _monotonic():
                self._park_settlement(pending)
                break
            outcome = self._send_confirm(
                pending.confirm.request, timeout=self._send_timeout(deadline)
            )
            if outcome is _SendOutcome.SENT:
                owned, _ = self._resolve_in_hand("settlement")
                self._ship_settlement_event(pending.event, owned)
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    # The confirms are undeliverable (control plane known-down),
                    # but ingest is NOT breaker-gated: hand every event to the
                    # event drain so the durable spend truth still gets its
                    # deadline-bounded exit attempt.
                    owned, _ = self._resolve_in_hand("settlement")
                    stranded: list[_PendingSettlement] = []
                    while True:
                        try:
                            stranded.append(self._settlement_queue.popleft())
                        except IndexError:
                            break
                    self._count_drop(
                        "settlement_confirm", "exit_breaker_open", n=owned + len(stranded)
                    )
                    self._ship_settlement_event(pending.event, owned)
                    for orphan in stranded:
                        self._ship_settlement_event(orphan.event)
                else:
                    self._park_settlement(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.confirm.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.confirm.attempts)
                if finished or final:
                    owned, _ = self._resolve_in_hand("settlement")
                    self._count_drop("settlement_confirm", "retry_exhausted", n=owned)
                    self._ship_settlement_event(pending.event, owned)
                    continue
                pending.confirm.next_attempt_at = next_at
                self._park_settlement(pending)
                break
            # Terminal confirm: the event is still the durable spend truth.
            owned, _ = self._resolve_in_hand("settlement")
            self._count_drop("settlement_confirm", "terminal_status", n=owned)
            self._ship_settlement_event(pending.event, owned)

    def _drain_event_batches(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Send due metadata events in batches; requeue a failed batch to front."""
        while self._queue:
            if self._deadline_expired(deadline):
                self._count_drop("event", "shutdown_deadline", n=_drain_count(self._queue))
                break
            with self._in_flight_lock:
                if self._in_flight >= self.max_in_flight:
                    break
            now = _monotonic()
            prefix: list[_PendingEvent] = []
            while len(prefix) < self.batch_size and self._queue:
                try:
                    head = self._queue[0]
                except IndexError:
                    break
                if not final and head.next_attempt_at > now:
                    break
                try:
                    prefix.append(self._queue.popleft())
                except IndexError:
                    break
            if not prefix:
                break
            self._take_in_hand("event", n=len(prefix))
            outcome = self._send_batch([p.event for p in prefix], deadline=deadline)
            if outcome is _SendOutcome.SENT:
                self._resolve_in_hand("event", n=len(prefix))
                continue
            if outcome is _SendOutcome.RETRY:
                keep: list[_PendingEvent] = []
                dropped = 0
                for p in prefix:
                    p.attempts += 1
                    finished, next_at = self._resolve_retryable(attempts=p.attempts)
                    if finished or final:
                        dropped += 1
                    else:
                        p.next_attempt_at = next_at
                        keep.append(p)
                owned, requeued = self._resolve_in_hand(
                    "event",
                    n=len(prefix),
                    requeue=functools.partial(self._queue.extendleft, tuple(reversed(keep))),
                )
                if requeued:
                    self._count_drop("event", "retry_exhausted", n=dropped)
                else:
                    # Sealed while in hand: whatever the seal did not claim is
                    # abandoned here.
                    self._count_drop("event", "shutdown_deadline", n=owned)
                break
            owned, _ = self._resolve_in_hand("event", n=len(prefix))
            self._count_drop("event", "terminal_status", n=owned)

    def _start_breaker_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
    ) -> threading.Thread | None:
        """Start one tracked breaker cycle, or coalesce into the active one."""
        with self._breaker_worker_lock:
            if self._shutdown.is_set() and not during_shutdown:
                return None
            worker = self._breaker_worker
            if worker is not None and worker.is_alive():
                return worker
            worker = threading.Thread(
                target=self._flush_breaker_reports,
                kwargs={"deadline": deadline},
                daemon=True,
                name="solwyn-breaker-reporter",
            )
            self._breaker_worker = worker
            worker.start()
            return worker

    def _flush_breaker_reports(self, deadline: float | None = None) -> None:
        """POST current breaker snapshots independently and drop every failure.

        Bounded by the shared shutdown ``deadline`` when set: remaining providers
        are skipped once it is reached. Breaker snapshots are advisory — dropping
        them is fine and is NOT counted as a spend drop.
        """
        for project_id, report in self._build_breaker_reports():
            if self._deadline_expired(deadline):
                return
            try:
                response = self._http.post(
                    f"{self.api_url}/api/v1/projects/{project_id}/providers/breaker-reports",
                    json=report.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=self._send_timeout(deadline),
                )
                response.raise_for_status()
            except Exception as exc:
                if handle_read_only_key_error(exc):
                    # A read-only key denies every write: end the cycle instead
                    # of posting the remaining doomed snapshots.
                    return
                logger.warning(
                    "reporter.breaker_send_failed: provider=%s exc_type=%s",
                    report.provider.value,
                    type(exc).__name__,
                )

    def _send_confirm(
        self, confirm_request: BudgetConfirmRequest, *, timeout: float = 5.0
    ) -> _SendOutcome:
        """Send one confirm request and return its delivery outcome.

        Breaker-refused admission returns HELD (the caller keeps the item for a
        later cycle); a transient failure returns RETRY; a terminal status or
        poison item returns DROPPED. Read-only-key is a policy skip, not a loss,
        so it returns SENT.
        """
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Control plane known-down: hold this confirm for a later cycle
            # without paying the timeout. Does not touch the confirm-failure
            # counter or the breaker.
            logger.debug("reporter.confirm_held_breaker_open")
            return _SendOutcome.HELD
        try:
            resp = self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            if breaker is not None:
                breaker.record_success()
            self._record_confirm_success()
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
                return _SendOutcome.SENT
            if breaker is not None:
                breaker.record_failure()
            self._record_confirm_failure(exc)
            return self._confirm_failure_outcome(exc)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    def _send_batch(
        self, batch: list[MetadataEvent], *, deadline: float | None = None
    ) -> _SendOutcome:
        """Send a batch of events to the cloud API and return its outcome.

        Ingest is deliberately NOT control-plane-breaker-guarded: opening the
        enforcement breaker (which flips budget checks to their fail-open
        posture) on an ingest blip would be a worse failure mode than a delayed
        batch. Ingest self-paces via the retry backoff instead.

        ``deadline`` clamps the request into the shutdown window; without it a
        black-holed control plane made close() overrun its budget by the full
        10s client timeout (P0 review finding).
        """
        timeout = 10.0 if deadline is None else self._send_timeout(deadline)
        with self._in_flight_lock:
            self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            self._log_ingest_rejections(resp, len(batch))
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                return _SendOutcome.SENT
            # Log only the exception's class name — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            logger.warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
            return self._batch_failure_outcome(exc)
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm request onto the flush queue.

        Called from stream completion callbacks so the user's thread
        never blocks on Solwyn HTTP. The flush loop picks up confirm
        requests alongside metadata events.
        """
        if self._shutdown.is_set():
            self._count_drop("confirm", "closed_enqueue")
            return
        self._ensure_thread()
        enqueued = self._enqueue_owned(
            self._confirm_queue, _PendingConfirm(request), _MAX_PENDING_CONTROL, "confirm"
        )
        if not enqueued:
            self._count_drop("confirm", "closed_enqueue")

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._shutdown.is_set():
            self._count_settlement_drop("closed_enqueue")
            return
        self._ensure_thread()
        enqueued = self._enqueue_owned(
            self._settlement_queue,
            _PendingSettlement(_PendingConfirm(request), event),
            _MAX_PENDING_CONTROL,
            "settlement_confirm",
            # An overflow-evicted pair drops only its confirm; the event is
            # the durable spend truth and still rides the event queue.
            on_evict=lambda evicted: self._ship_settlement_event(evicted.event),
        )
        if not enqueued:
            self._count_settlement_drop("closed_enqueue")


class AsyncMetadataReporter(_ReporterBase):
    """Asynchronous metadata reporter using asyncio.create_task.

    Usage::

        async with AsyncMetadataReporter(api_url, api_key) as reporter:
            reporter.report(event)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 10_000,
        max_in_flight: int = 3,
        breaker_snapshots: Callable[[], list[tuple[ProviderName, CircuitBreakerState]]]
        | None = None,
        sdk_instance_id: str | None = None,
        breaker_reporting_enabled: bool = True,
        control_plane_breaker: CircuitBreaker | None = None,
        max_send_attempts: int = 5,
        retry_backoff_base: float = 1.0,
        retry_backoff_cap: float = 60.0,
        shutdown_deadline: float = 5.0,
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
            breaker_snapshots=breaker_snapshots,
            sdk_instance_id=sdk_instance_id,
            breaker_reporting_enabled=breaker_reporting_enabled,
            control_plane_breaker=control_plane_breaker,
            max_send_attempts=max_send_attempts,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_cap=retry_backoff_cap,
            shutdown_deadline=shutdown_deadline,
        )
        self._http = httpx.AsyncClient(timeout=10.0)
        self._shutdown_event: asyncio.Event | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._breaker_task: asyncio.Task[None] | None = None
        # Set by close(); once closed, enqueues are dropped and start() fails
        # loud. Distinct from _shutdown_event, which only exists once a flush
        # loop has run — a never-started reporter has no shutdown event but can
        # still be closed.
        self._closed = False
        # Set only when close() FINISHES its flush chain. _closed alone must
        # not disarm the lifecycle rescue paths: a close() cancelled at its
        # first await has flushed nothing, so the atexit hook keys on this.
        self._close_completed = False
        # Latches the one-per-instance "enqueued with no running loop" warning
        # so a caller that never enters an event loop is warned once, not per
        # event.
        self._warned_no_loop = False
        self._confirm_queue: collections.deque[_PendingConfirm] = collections.deque()
        self._settlement_queue: collections.deque[_PendingSettlement] = collections.deque()
        # Set by register_async_reporter: a GC finalizer covering the case where
        # this reporter is dropped before its flush loop ever ran. close()
        # detaches it so nothing double-flushes.
        self._finalizer: weakref.finalize[..., AsyncMetadataReporter] | None = None
        register_async_reporter(self)
        register_fork_reset(self)

    def _reset_after_fork_in_child(self) -> None:
        """Repair a forked child: fresh locks/client and cleared loop state.

        The parent's event loop, flush task, and breaker task do not exist in the
        child; clear them so ``_ensure_started`` relaunches the flush loop in the
        child's own loop on the next enqueue. The inherited client is abandoned
        (never closed — the parent owns those sockets). Queued items duplicated
        into the child by fork are deliberately KEPT: the server dedups.
        """
        self._breaker_project_lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._in_flight = 0
        self._flush_task = None
        self._breaker_task = None
        self._shutdown_event = None
        self._http = httpx.AsyncClient(timeout=10.0)

    def start(self) -> None:
        """Start the background flush loop.  Must be called within an event loop.

        Idempotent: if a flush task is already running, this is a no-op and the
        existing ``_shutdown_event`` is left untouched (a second call must not
        orphan the live task or reset its shutdown signal). Restarting a closed
        reporter is a programming error and raises — the sync reporter is live
        from construction, so a closed async reporter has no valid restart.
        """
        if self._closed:
            raise RuntimeError("cannot start a closed AsyncMetadataReporter")
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._shutdown_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

    def _ensure_started(self) -> None:
        """Auto-start the flush loop on first enqueue, when a loop is running.

        The sync reporter starts its flush thread in ``__init__``; the async
        reporter cannot (there may be no event loop yet), so it starts lazily on
        the first enqueue instead of only via ``start()`` / ``__aenter__``.
        Without this, a reporter constructed outside ``async with`` queues
        events AND budget-confirm settlements silently until ``close()``, so
        server-side spend tracking drifts.

        Never raises (called on the enqueue path): a closed reporter, an
        already-running flush task, or the absence of a running loop each
        short-circuit. With no running loop the event stays queued and a single
        warning per reporter instance is logged.
        """
        if self._closed:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not self._warned_no_loop:
                self._warned_no_loop = True
                logger.warning(
                    "reporter.enqueue_without_event_loop: no running event loop; "
                    "events stay queued until start() or close() runs inside a loop"
                )
            return
        self.start()

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        if self._closed:
            self._count_drop("event", "closed_enqueue")
            return
        self._ensure_started()
        self._enqueue(event)

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm request onto the async flush queue."""
        if self._closed:
            self._count_drop("confirm", "closed_enqueue")
            return
        self._ensure_started()
        self._enqueue_bounded(
            self._confirm_queue, _PendingConfirm(request), _MAX_PENDING_CONTROL, "confirm"
        )

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._closed:
            # The pair loses its confirm AND its event — count both halves.
            self._count_settlement_drop("closed_enqueue")
            return
        self._ensure_started()
        self._enqueue_bounded(
            self._settlement_queue,
            _PendingSettlement(_PendingConfirm(request), event),
            _MAX_PENDING_CONTROL,
            "settlement_confirm",
            # An overflow-evicted pair drops only its confirm; the event is
            # the durable spend truth and still rides the event queue.
            on_evict=lambda evicted: self._move_event_to_queue(evicted.event),
        )

    async def close(self, timeout: float | None = None) -> None:
        """Flush remaining events and shut down within a single deadline.

        See ``MetadataReporter.close`` — one monotonic ``deadline`` bounds the
        flush-task await, the final flush, and the breaker report cycle.

        ``_closed`` (no new work) is set before the first await; the exit
        rescue state (``_close_completed`` + the GC finalizer) is touched only
        AFTER the flush chain finishes. A close() cancelled mid-await
        propagates the cancellation but leaves both lifecycle rescue paths
        armed for whatever spend is still queued.
        """
        self._closed = True
        budget = self.shutdown_deadline if timeout is None else timeout
        deadline = _monotonic() + budget
        active_breaker_task = self._breaker_task
        if active_breaker_task is not None and active_breaker_task.done():
            active_breaker_task = None
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._flush_task is not None:
            await self._await_within(self._flush_task, deadline)
        await self._flush_remaining(deadline=deadline, final=True)

        if active_breaker_task is None:
            active_breaker_task = self._start_breaker_cycle(during_shutdown=True, deadline=deadline)
        if active_breaker_task is not None:
            await self._await_within(active_breaker_task, deadline)
        await self._http.aclose()
        # A COMPLETED close supersedes the exit-flush safety nets: drop the GC
        # finalizer so it can never double-flush drained queues, and tell the
        # atexit hook this reporter needs no rescue.
        self._close_completed = True
        if self._finalizer is not None:
            self._finalizer.detach()

    async def _await_within(self, task: asyncio.Task[None], deadline: float) -> None:
        """Await ``task`` but never past the shared shutdown deadline.

        On timeout the task is cancelled — during close we would rather abandon a
        stuck flush/breaker task than exceed the deadline; a duplicate send is
        server-deduped.
        """
        remaining = deadline - _monotonic()
        if remaining <= 0:
            task.cancel()
            return
        try:
            await asyncio.wait_for(task, timeout=remaining)
        except TimeoutError:
            pass  # wait_for already cancelled the stuck task
        except asyncio.CancelledError:
            # Two sources are conflated here: close() ITSELF being cancelled
            # (must propagate — a cancelled close must not silently keep
            # running) vs the awaited task having been cancelled elsewhere
            # (safe to continue shutting down).
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise

    async def __aenter__(self) -> AsyncMetadataReporter:
        self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _flush_loop(self) -> None:
        """Background task: periodically flush batches to the cloud."""
        if self._shutdown_event is None:
            raise RuntimeError("_flush_loop called before reporter was started")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.flush_interval,
                )
            except TimeoutError:
                try:
                    await self._flush_remaining()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # The flush task must survive anything a drain raises: a
                    # dead flush task strands ALL queued spend until overflow.
                    logger.warning("reporter.flush_cycle_failed: exc_type=%s", type(exc).__name__)
                self._start_breaker_cycle()
            else:
                break

    async def _flush_remaining(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Flush queued confirms, settlements, then metadata events in batches.

        See ``MetadataReporter._flush_remaining`` for the ``final`` / ``deadline``
        semantics.
        """
        await self._drain_confirms(deadline=deadline, final=final)
        await self._drain_settlements(deadline=deadline, final=final)
        await self._drain_event_batches(deadline=deadline, final=final)
        self._maybe_log_drops(force=final)

    async def _drain_confirms(self, *, deadline: float | None = None, final: bool = False) -> None:
        """Send due confirms in FIFO order; a retrying head parks the queue."""
        while self._confirm_queue:
            if self._deadline_expired(deadline):
                self._count_drop(
                    "confirm", "shutdown_deadline", n=_drain_count(self._confirm_queue)
                )
                break
            try:
                # Atomic pop, never peek-then-pop: a cancelled close() and the
                # cooperatively-scheduled flush task can interleave drains.
                pending = self._confirm_queue.popleft()
            except IndexError:
                break
            if not final and pending.next_attempt_at > _monotonic():
                self._confirm_queue.appendleft(pending)
                break
            try:
                outcome = await self._send_confirm(
                    pending.request, timeout=self._send_timeout(deadline)
                )
            except asyncio.CancelledError:
                # close() cancelled a stuck drain at the deadline — the in-hand
                # item would otherwise vanish uncounted.
                self._count_drop("confirm", "shutdown_deadline")
                raise
            if outcome is _SendOutcome.SENT:
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    self._count_drop(
                        "confirm", "exit_breaker_open", n=1 + _drain_count(self._confirm_queue)
                    )
                else:
                    self._confirm_queue.appendleft(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.attempts)
                if finished or final:
                    self._count_drop("confirm", "retry_exhausted")
                    continue
                pending.next_attempt_at = next_at
                self._confirm_queue.appendleft(pending)
                break  # FIFO: nothing behind a backing-off head may jump it
            self._count_drop("confirm", "terminal_status")

    async def _drain_settlements(
        self, *, deadline: float | None = None, final: bool = False
    ) -> None:
        """Resolve each settlement's confirm first, then hand its event to the
        event queue (confirm-before-metadata order per item is load-bearing).
        A retrying head parks the queue behind it (FIFO)."""
        while self._settlement_queue:
            if self._deadline_expired(deadline):
                self._count_settlement_drop(
                    "shutdown_deadline", n=_drain_count(self._settlement_queue)
                )
                break
            try:
                # Atomic pop — see _drain_confirms for the interleaving
                # rationale.
                pending = self._settlement_queue.popleft()
            except IndexError:
                break
            if not final and pending.confirm.next_attempt_at > _monotonic():
                self._settlement_queue.appendleft(pending)
                break
            try:
                outcome = await self._send_confirm(
                    pending.confirm.request, timeout=self._send_timeout(deadline)
                )
            except asyncio.CancelledError:
                # In-hand pair vanishes with the cancellation — count both.
                self._count_settlement_drop("shutdown_deadline")
                raise
            if outcome is _SendOutcome.SENT:
                self._move_event_to_queue(pending.event)
                continue
            if outcome is _SendOutcome.HELD:
                if final:
                    # The confirms are undeliverable (control plane known-down),
                    # but ingest is NOT breaker-gated: hand every event to the
                    # event drain so the durable spend truth still gets its
                    # deadline-bounded exit attempt.
                    stranded = [pending, *self._settlement_queue]
                    self._settlement_queue.clear()
                    self._count_drop("settlement_confirm", "exit_breaker_open", n=len(stranded))
                    for orphan in stranded:
                        self._move_event_to_queue(orphan.event)
                else:
                    self._settlement_queue.appendleft(pending)
                break
            if outcome is _SendOutcome.RETRY:
                pending.confirm.attempts += 1
                finished, next_at = self._resolve_retryable(attempts=pending.confirm.attempts)
                if finished or final:
                    self._count_drop("settlement_confirm", "retry_exhausted")
                    self._move_event_to_queue(pending.event)
                    continue
                pending.confirm.next_attempt_at = next_at
                self._settlement_queue.appendleft(pending)
                break
            self._count_drop("settlement_confirm", "terminal_status")
            self._move_event_to_queue(pending.event)

    async def _drain_event_batches(
        self, *, deadline: float | None = None, final: bool = False
    ) -> None:
        """Send due metadata events in batches; requeue a failed batch to front."""
        while self._queue:
            if self._deadline_expired(deadline):
                self._count_drop("event", "shutdown_deadline", n=_drain_count(self._queue))
                break
            if self._in_flight >= self.max_in_flight:
                break
            now = _monotonic()
            prefix: list[_PendingEvent] = []
            while len(prefix) < self.batch_size and self._queue:
                head = self._queue[0]
                if not final and head.next_attempt_at > now:
                    break
                prefix.append(self._queue.popleft())
            if not prefix:
                break
            try:
                outcome = await self._send_batch([p.event for p in prefix], deadline=deadline)
            except asyncio.CancelledError:
                self._count_drop("event", "shutdown_deadline", n=len(prefix))
                raise
            if outcome is _SendOutcome.SENT:
                continue
            if outcome is _SendOutcome.RETRY:
                keep: list[_PendingEvent] = []
                dropped = 0
                for p in prefix:
                    p.attempts += 1
                    finished, next_at = self._resolve_retryable(attempts=p.attempts)
                    if finished or final:
                        dropped += 1
                    else:
                        p.next_attempt_at = next_at
                        keep.append(p)
                if dropped:
                    self._count_drop("event", "retry_exhausted", n=dropped)
                if keep:
                    self._queue.extendleft(reversed(keep))
                break
            self._count_drop("event", "terminal_status", n=len(prefix))

    def _start_breaker_cycle(
        self,
        *,
        during_shutdown: bool = False,
        deadline: float | None = None,
    ) -> asyncio.Task[None] | None:
        """Start one tracked breaker task, or coalesce into the active task."""
        if (
            self._shutdown_event is not None
            and self._shutdown_event.is_set()
            and not during_shutdown
        ):
            return None
        task = self._breaker_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(
            self._flush_breaker_reports(deadline=deadline),
            name="solwyn-breaker-reporter",
        )
        self._breaker_task = task
        return task

    async def _flush_breaker_reports(self, deadline: float | None = None) -> None:
        """POST current breaker snapshots independently and drop every failure.

        Bounded by the shared shutdown ``deadline`` when set (advisory snapshots
        are dropped, not counted as spend).
        """
        for project_id, report in self._build_breaker_reports():
            if self._deadline_expired(deadline):
                return
            try:
                response = await self._http.post(
                    f"{self.api_url}/api/v1/projects/{project_id}/providers/breaker-reports",
                    json=report.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=self._send_timeout(deadline),
                )
                response.raise_for_status()
            except Exception as exc:
                if handle_read_only_key_error(exc):
                    # A read-only key denies every write: end the cycle instead
                    # of posting the remaining doomed snapshots.
                    return
                logger.warning(
                    "reporter.breaker_send_failed: provider=%s exc_type=%s",
                    report.provider.value,
                    type(exc).__name__,
                )

    async def _send_confirm(
        self, confirm_request: BudgetConfirmRequest, *, timeout: float = 5.0
    ) -> _SendOutcome:
        """Send one confirm request and return its delivery outcome.

        See ``MetadataReporter._send_confirm``.
        """
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            logger.debug("reporter.confirm_held_breaker_open")
            return _SendOutcome.HELD
        try:
            resp = await self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            if breaker is not None:
                breaker.record_success()
            self._record_confirm_success()
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
                return _SendOutcome.SENT
            if breaker is not None:
                breaker.record_failure()
            self._record_confirm_failure(exc)
            return self._confirm_failure_outcome(exc)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    async def _send_batch(
        self, batch: list[MetadataEvent], *, deadline: float | None = None
    ) -> _SendOutcome:
        """Send a batch of events to the cloud API and return its outcome.

        Ingest is deliberately NOT control-plane-breaker-guarded: opening the
        enforcement breaker (which flips budget checks to their fail-open
        posture) on an ingest blip would be a worse failure mode than a delayed
        batch. Ingest self-paces via the retry backoff instead.

        ``deadline`` clamps the request into the shutdown window; without it a
        black-holed control plane made close() overrun its budget by the full
        10s client timeout (P0 review finding).
        """
        timeout = 10.0 if deadline is None else self._send_timeout(deadline)
        self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = await self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            # httpx.Response.json() is sync on both clients — shared helper.
            self._log_ingest_rejections(resp, len(batch))
            return _SendOutcome.SENT
        except Exception as exc:
            if handle_read_only_key_error(exc):
                return _SendOutcome.SENT
            # Log only the exception's class name — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            logger.warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
            return self._batch_failure_outcome(exc)
        finally:
            self._in_flight -= 1
