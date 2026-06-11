"""Async metadata reporter.

MetadataReporter (sync, background thread queue) and AsyncMetadataReporter
(asyncio.create_task) batch and flush metadata events to the Solwyn cloud API.
Neither blocks the LLM call path.

Events contain cost/latency metadata only -- never prompts or responses.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
import threading

import httpx

from solwyn._types import BudgetConfirmRequest, MetadataEvent

logger = logging.getLogger(__name__)

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


class _ReporterBase:
    """Sans-I/O base class for metadata reporting.

    Manages a bounded deque and batching logic.  Subclasses add the
    I/O layer (threading or asyncio) and HTTP transport.
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
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_in_flight = max_in_flight

        # Bounded deque: drops oldest events when full
        self._queue: collections.deque[MetadataEvent] = collections.deque(maxlen=max_queue_size)
        self._in_flight = 0
        self._consecutive_confirm_failures = 0
        self._confirm_failure_threshold = 10
        self._consecutive_unparseable_responses = 0
        self._unparseable_response_threshold = 10

    def _enqueue(self, event: MetadataEvent) -> None:
        """Add an event to the queue.  Drop-oldest semantics on overflow."""
        self._queue.append(event)

    def _drain_batch(self) -> list[MetadataEvent]:
        """Drain up to batch_size events from the front of the queue."""
        batch: list[MetadataEvent] = []
        for _ in range(min(self.batch_size, len(self._queue))):
            try:
                batch.append(self._queue.popleft())
            except IndexError:
                break
        return batch

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for cloud API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
        )
        self._http = httpx.Client(timeout=10.0)
        self._shutdown = threading.Event()
        self._in_flight_lock = threading.Lock()
        # Separate queue for confirm_cost requests. Stream completion
        # fire-and-forgets onto this queue so the user's thread is not
        # blocked on an httpx.post to Solwyn.
        self._confirm_queue: collections.deque[BudgetConfirmRequest] = collections.deque(
            maxlen=1000
        )
        self._settlement_queue: collections.deque[tuple[BudgetConfirmRequest, MetadataEvent]] = (
            collections.deque(maxlen=1000)
        )
        self._thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="solwyn-reporter",
        )
        self._thread.start()

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        self._enqueue(event)

    def close(self) -> None:
        """Flush remaining events and shut down the background thread."""
        self._shutdown.set()
        self._thread.join(timeout=10.0)
        # Final flush of anything remaining
        self._flush_remaining()
        self._http.close()

    def __enter__(self) -> MetadataReporter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _flush_loop(self) -> None:
        """Background thread: periodically flush batches to the cloud."""
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=self.flush_interval)
            self._flush_remaining()

    def _flush_remaining(self) -> None:
        """Flush queued confirms, settlements, then metadata events in batches."""
        while self._confirm_queue:
            self._send_confirm(self._confirm_queue.popleft())
        self._flush_settlements()
        while len(self._queue) > 0:
            with self._in_flight_lock:
                if self._in_flight >= self.max_in_flight:
                    break
            batch = self._drain_batch()
            if not batch:
                break
            self._send_batch(batch)

    def _send_confirm(self, confirm_request: BudgetConfirmRequest) -> None:
        """Send one confirm request and update confirm failure accounting."""
        try:
            resp = self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=5.0,
            )
            resp.raise_for_status()
            self._record_confirm_success()
        except Exception as exc:
            self._record_confirm_failure(exc)

    def _flush_settlements(self) -> None:
        """Send reservation settlements in confirm-before-metadata order."""
        batch: list[MetadataEvent] = []
        while self._settlement_queue:
            confirm_request, event = self._settlement_queue.popleft()
            self._send_confirm(confirm_request)
            batch.append(event)
            if len(batch) >= self.batch_size:
                self._send_batch(batch)
                batch = []
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: list[MetadataEvent]) -> None:
        """Send a batch of events to the cloud API."""
        with self._in_flight_lock:
            self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            self._log_ingest_rejections(resp, len(batch))
        except Exception as exc:
            # Log only the exception's class name (fix [D]) — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            logger.warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm_cost request onto the flush queue.

        Called from stream completion callbacks so the user's thread
        never blocks on Solwyn HTTP. The flush loop picks up confirm
        requests alongside metadata events.
        """
        if self._shutdown.is_set():
            return
        try:
            self._confirm_queue.append(request)
        except Exception as exc:
            logger.warning(
                "reporter.confirm_enqueue_failed: exc_type=%s",
                type(exc).__name__,
            )

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._shutdown.is_set():
            return
        try:
            self._settlement_queue.append((request, event))
        except Exception as exc:
            logger.warning(
                "reporter.settlement_enqueue_failed: exc_type=%s",
                type(exc).__name__,
            )


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
    ) -> None:
        super().__init__(
            api_url,
            api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
            max_in_flight=max_in_flight,
        )
        self._http = httpx.AsyncClient(timeout=10.0)
        self._shutdown_event: asyncio.Event | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._confirm_queue: collections.deque[BudgetConfirmRequest] = collections.deque(
            maxlen=1000
        )
        self._settlement_queue: collections.deque[tuple[BudgetConfirmRequest, MetadataEvent]] = (
            collections.deque(maxlen=1000)
        )

    def start(self) -> None:
        """Start the background flush loop.  Must be called within an event loop."""
        self._shutdown_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())

    def report(self, event: MetadataEvent) -> None:
        """Enqueue a metadata event for async reporting.  Non-blocking."""
        self._enqueue(event)

    def report_confirm(self, request: BudgetConfirmRequest) -> None:
        """Fire-and-forget a confirm_cost request onto the async flush queue."""
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            return
        try:
            self._confirm_queue.append(request)
        except Exception as exc:
            logger.warning(
                "reporter.confirm_enqueue_failed: exc_type=%s",
                type(exc).__name__,
            )

    def report_settlement(self, request: BudgetConfirmRequest, event: MetadataEvent) -> None:
        """Fire-and-forget a stream settlement as one ordered queue item."""
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            return
        try:
            self._settlement_queue.append((request, event))
        except Exception as exc:
            logger.warning(
                "reporter.settlement_enqueue_failed: exc_type=%s",
                type(exc).__name__,
            )

    async def close(self) -> None:
        """Flush remaining events and shut down."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._flush_task is not None:
            await self._flush_task
        # Final flush
        await self._flush_remaining()
        await self._http.aclose()

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
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.flush_interval,
                )
            await self._flush_remaining()

    async def _flush_remaining(self) -> None:
        """Flush queued confirms, settlements, then metadata events in batches."""
        while self._confirm_queue:
            await self._send_confirm(self._confirm_queue.popleft())
        await self._flush_settlements()
        while len(self._queue) > 0:
            if self._in_flight >= self.max_in_flight:
                break
            batch = self._drain_batch()
            if not batch:
                break
            await self._send_batch(batch)

    async def _send_confirm(self, confirm_request: BudgetConfirmRequest) -> None:
        """Send one confirm request and update confirm failure accounting."""
        try:
            resp = await self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=confirm_request.model_dump(mode="json"),
                headers=self._auth_headers(),
                timeout=5.0,
            )
            resp.raise_for_status()
            self._record_confirm_success()
        except Exception as exc:
            self._record_confirm_failure(exc)

    async def _flush_settlements(self) -> None:
        """Send reservation settlements in confirm-before-metadata order."""
        batch: list[MetadataEvent] = []
        while self._settlement_queue:
            confirm_request, event = self._settlement_queue.popleft()
            await self._send_confirm(confirm_request)
            batch.append(event)
            if len(batch) >= self.batch_size:
                await self._send_batch(batch)
                batch = []
        if batch:
            await self._send_batch(batch)

    async def _send_batch(self, batch: list[MetadataEvent]) -> None:
        """Send a batch of events to the cloud API."""
        self._in_flight += 1
        try:
            payload = [e.model_dump(mode="json") for e in batch]
            resp = await self._http.post(
                f"{self.api_url}/api/v1/metadata/ingest",
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            # httpx.Response.json() is sync on both clients — shared helper.
            self._log_ingest_rejections(resp, len(batch))
        except Exception as exc:
            # Log only the exception's class name (fix [D]) — the type-name-only
            # convention every other except-block follows. Safe here even though
            # the batch is content-free; keeps the privacy contract uniform.
            logger.warning(
                "Failed to send metadata batch (%d events): %s",
                len(batch),
                type(exc).__name__,
            )
        finally:
            self._in_flight -= 1
