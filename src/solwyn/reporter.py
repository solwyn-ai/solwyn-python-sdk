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
import threading

import httpx

from solwyn._types import BudgetConfirmRequest, MetadataEvent

logger = logging.getLogger(__name__)


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
        """Flush all queued events in batches."""
        while len(self._queue) > 0:
            with self._in_flight_lock:
                if self._in_flight >= self.max_in_flight:
                    break
            batch = self._drain_batch()
            if not batch:
                break
            self._send_batch(batch)
        # Drain any queued confirm-cost requests
        while self._confirm_queue:
            confirm_request = self._confirm_queue.popleft()
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
        """Flush all queued events in batches."""
        while len(self._queue) > 0:
            if self._in_flight >= self.max_in_flight:
                break
            batch = self._drain_batch()
            if not batch:
                break
            await self._send_batch(batch)
        while self._confirm_queue:
            confirm_request = self._confirm_queue.popleft()
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
