"""Exit flush (R2): a normally-exiting process still delivers queued spend.

Scenario-1 pin: a sync reporter that queues a settlement and exits WITHOUT
close() must flush it via the atexit hook. Plus in-process coverage of the async
reporter's weakref finalizer, breaker admission at exit (confirms held, events
still shipped), per-item exit accounting, and close() detaching the finalizer
only once it COMPLETES.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, call_uuid

from solwyn._lifecycle import _drain_queues_blocking, blocking_exit_flush
from solwyn._surfaces import SurfaceContext
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import AsyncMetadataReporter

_URL = "https://api.test.solwyn.ai"


def _observe_untracked(reporter: AsyncMetadataReporter) -> None:
    reporter.observe_untracked_surface(
        context=SurfaceContext(
            provider="openai",
            dialect="openai",
            client_shape="openai_sdk",
            mode="async",
        ),
        surface="responses.create",
        rule_kind="unmetered_spend",
        capability_scope="operation",
        posture="warn",
        seen_at=datetime.now(UTC),
    )


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": call_uuid("call_exit_confirm"),
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


def _make_event(**overrides) -> MetadataEvent:
    defaults = {
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 1.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "exit-instance",
        "timestamp": datetime.now(UTC),
        "call_id": call_uuid("call_exit_event"),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


# ---------------------------------------------------------------------------
# A tiny recording server for the subprocess pin
# ---------------------------------------------------------------------------


class _RecordingServer:
    """A localhost HTTP server that records POST paths; 200s every request."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        recorder = self.paths

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                recorder.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ingested":1,"rejected":[]}')

            def log_message(self, *_args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# The child runs in a fresh interpreter with no test helpers, and the confirm
# wire pins call_id to the canonical UUID text form — so its id is a literal.
EXIT_CALL_ID = "33333333-3333-4333-8333-333333333333"

_CHILD = """
from datetime import UTC, datetime

import solwyn.reporter as r
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName

confirm = BudgetConfirmRequest(
    reservation_id="res_exit",
    model="gpt-5.5",
    provider=ProviderName.OPENAI,
    call_id="{call_id}",
    token_details=TokenDetails(input_tokens=10, output_tokens=5),
)
event = MetadataEvent(
    model="gpt-5.5",
    provider=ProviderName.OPENAI,
    input_tokens=100,
    output_tokens=50,
    latency_ms=1.0,
    status="success",
    is_model_fallback=False,
    sdk_instance_id="child-instance",
    timestamp=datetime.now(UTC),
    call_id="{call_id}",
)
rep = r.MetadataReporter("http://127.0.0.1:{port}", "sk_test_key", flush_interval=60.0)
rep.report_settlement(confirm, event)
# NO close(): a normal interpreter exit must flush via the atexit hook.
"""


@pytest.mark.unit
def test_interpreter_exit_without_close_flushes_settlements() -> None:
    server = _RecordingServer()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD.format(port=server.port, call_id=EXIT_CALL_ID)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"child failed: {result.stderr[-800:]}"

        # The exit hook posts synchronously before the child exits, but poll a
        # moment for the server thread to record both.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any("budgets/confirm" in p for p in server.paths) and any(
                "metadata/ingest" in p for p in server.paths
            ):
                break
            time.sleep(0.05)

        assert any("/api/v1/budgets/confirm" in p for p in server.paths), (
            f"exit hook did not flush the confirm; got {server.paths}"
        )
        assert any("/api/v1/metadata/ingest" in p for p in server.paths), (
            f"exit hook did not flush the metadata event; got {server.paths}"
        )
    finally:
        server.close()


# ---------------------------------------------------------------------------
# In-process: async finalizer, breaker-open skip, close() detach
# ---------------------------------------------------------------------------


class _RecordingResponse:
    def raise_for_status(self) -> None:
        return None


def _make_recording_client(sink: list[str], handler: Callable[[str], object] | None = None) -> type:
    """A fake ``httpx.Client``: records POST urls; ``handler`` scripts outcomes
    (return a response or raise) — default 200s everything."""

    class _Client:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_a: object) -> bool:
            return False

        def post(
            self,
            url: str,
            *,
            json: object = None,
            headers: object = None,
            timeout: object = None,
        ) -> object:
            sink.append(url)
            if handler is not None:
                return handler(url)
            return _RecordingResponse()

        def close(self) -> None:
            # The exit drain's deadline watchdog holds client.close.
            return None

    return _Client


def _terminal_response(status_code: int) -> MagicMock:
    """A response stand-in whose raise_for_status raises HTTPStatusError."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock(
        spec=httpx.Response.raise_for_status,
        side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(spec=httpx.Request), response=resp
        ),
    )
    return resp


def _eligible_open_breaker() -> CircuitBreaker:
    """An OPEN control-plane breaker whose recovery window has elapsed."""
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=60.0, success_threshold=1, name="control-plane"
    )
    breaker.record_failure()  # OPEN
    breaker.last_failure_time = time.monotonic() - 120.0  # window elapsed -> eligible
    return breaker


@pytest.mark.unit
def test_async_finalizer_flushes_queued_confirm_on_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))

    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    # No running loop -> the confirm stays queued (never started).
    reporter.report_confirm(_make_confirm_request())
    assert len(reporter._confirm_queue) == 1

    del reporter
    gc.collect()

    assert any("budgets/confirm" in u for u in sink), (
        f"GC finalizer did not flush the queued confirm; got {sink}"
    )


@pytest.mark.unit
def test_async_finalizer_flushes_due_untracked_observation_on_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))

    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, sdk_instance_id="gc-instance")
    _observe_untracked(reporter)

    del reporter
    gc.collect()

    assert f"{_URL}/api/v1/untracked-surfaces" in sink


@pytest.mark.unit
def test_blocking_exit_flush_initiates_due_untracked_cycle_without_spend_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, sdk_instance_id="exit-instance")
    _observe_untracked(reporter)

    blocking_exit_flush(reporter)

    assert f"{_URL}/api/v1/untracked-surfaces" in sink
    assert reporter.dropped_counts == {}
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_finalizer_is_gc_only_never_exit() -> None:
    """Re-review #1 pin: weakref's own atexit hook is registered AFTER
    _exit_flush_all (first-finalizer creation) and so runs FIRST at exit —
    it would drain still-LIVE reporters with logger-only accounting while
    their dropped_counts is fully observable. The finalizer must be GC-only;
    _exit_flush_all is the sole live-reporter exit owner."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    assert reporter._finalizer is not None
    assert reporter._finalizer.atexit is False
    reporter._finalizer.detach()
    reporter._close_completed = True  # nothing queued; disarm the exit hook


@pytest.mark.unit
def test_gc_finalizer_losses_are_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Re-review #1 pin: a genuine pre-exit GC drain has no dropped_counts left
    to increment — a failed delivery must still be accounted, loudly, in the
    logs instead of vanishing."""
    sink: list[str] = []

    def _down(_url: str) -> object:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink, _down))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report_confirm(_make_confirm_request())

    with caplog.at_level("WARNING", logger="solwyn._lifecycle"):
        del reporter
        gc.collect()

    assert "lifecycle.gc_flush_dropped" in caplog.text
    assert "kind=confirm" in caplog.text


@pytest.mark.unit
def test_exit_flush_skips_when_breaker_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))

    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1e9, success_threshold=1, name="control-plane"
    )
    breaker.record_failure()  # OPEN, not recovery-eligible
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, control_plane_breaker=breaker)
    reporter.report_confirm(_make_confirm_request())

    with caplog.at_level("ERROR"):
        blocking_exit_flush(reporter)

    assert sink == []  # never hammer a known-down control plane while exiting
    assert reporter.dropped_counts["confirm.exit_breaker_open"] == 1
    assert "lifecycle.exit_flush_skipped_breaker_open" in caplog.text

    # Detach so the process-exit hook doesn't retry against the dead endpoint.
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_detaches_finalizer() -> None:
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=3600.0)
    reporter.start()
    assert reporter._finalizer is not None
    assert reporter._finalizer.alive

    with patch.object(reporter._http, "post", new=_async_noop_post):
        await reporter.close()

    # A COMPLETED close() supersedes the safety nets: the finalizer is
    # detached and the atexit hook skips this reporter.
    assert not reporter._finalizer.alive
    assert reporter._close_completed is True


async def _async_noop_post(*_a: object, **_k: object) -> _RecordingResponse:
    return _RecordingResponse()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_close_keeps_exit_rescue_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1 review pin: a close() cancelled mid-flush must NOT disarm the atexit
    hook or the GC finalizer — they are the last delivery path for whatever
    the cancelled close left queued."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=3600.0)
    reporter.start()
    reporter.report_confirm(_make_confirm_request(call_id=call_uuid("in_hand")))
    reporter.report_confirm(_make_confirm_request(call_id=call_uuid("queued_behind")))
    started = asyncio.Event()

    async def hung_post(url: str, **_kw: object) -> _RecordingResponse:
        started.set()
        await asyncio.sleep(3600)
        return _RecordingResponse()

    with patch.object(reporter._http, "post", new=hung_post):
        close_task = asyncio.create_task(reporter.close(timeout=30.0))
        await started.wait()
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

    # No new work — but shutdown did NOT complete, so both rescue paths stay
    # armed and BOTH confirms are retained: the cancelled drain REQUEUES its
    # in-hand item (re-review #4 pin — rescue can only retry what is in the
    # queues) instead of writing off deliverable spend as dropped.
    assert reporter._closed is True
    assert reporter._close_completed is False
    assert reporter._finalizer is not None and reporter._finalizer.alive
    assert len(reporter._confirm_queue) == 2
    assert reporter._confirm_queue[0].request.call_id == call_uuid("in_hand")  # FIFO kept
    assert "confirm.shutdown_deadline" not in reporter.dropped_counts

    # The atexit-style blocking flush still delivers the WHOLE backlog.
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    blocking_exit_flush(reporter)
    assert len([u for u in sink if "budgets/confirm" in u]) == 2
    assert len(reporter._confirm_queue) == 0
    reporter._finalizer.detach()
    await reporter._http.aclose()


# ---------------------------------------------------------------------------
# Blocking exit drain: per-item accounting + breaker admission
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exit_flush_breaker_open_still_ships_events(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """P1 review pin: a breaker-OPEN exit refuses the confirms, but metadata
    ingest is NOT breaker-gated — settlement events and standalone events must
    still get their deadline-bounded ingest attempt, never a breaker drop."""
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    breaker = CircuitBreaker(
        failure_threshold=1, recovery_timeout=1e9, success_threshold=1, name="control-plane"
    )
    breaker.record_failure()  # OPEN, not recovery-eligible
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, control_plane_breaker=breaker)
    reporter.report_confirm(_make_confirm_request())
    reporter.report_settlement(_make_confirm_request(), _make_event(call_id=call_uuid("pair")))
    reporter.report(_make_event(call_id=call_uuid("standalone")))

    with caplog.at_level("ERROR"):
        blocking_exit_flush(reporter)

    assert not any("budgets/confirm" in u for u in sink)
    assert len([u for u in sink if "metadata/ingest" in u]) == 2
    counts = reporter.dropped_counts
    assert counts.get("confirm.exit_breaker_open") == 1
    assert counts.get("settlement_confirm.exit_breaker_open") == 1
    assert not any(key.startswith("event.") for key in counts)
    assert "lifecycle.exit_flush_skipped_breaker_open" in caplog.text
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_counts_failed_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1 review pin: an exit POST that fails must count the popped item — it
    can no longer hide behind the swallow-and-log path."""
    sink: list[str] = []

    def _down(_url: str) -> object:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink, _down))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report_confirm(_make_confirm_request())
    reporter.report_settlement(_make_confirm_request(), _make_event())
    reporter.report(_make_event())

    blocking_exit_flush(reporter)

    counts = reporter.dropped_counts
    assert counts.get("confirm.retry_exhausted") == 1
    assert counts.get("settlement_confirm.retry_exhausted") == 1
    # The settlement's event AND the standalone batch both failed ingest.
    assert counts.get("event.retry_exhausted") == 2
    assert not reporter._confirm_queue
    assert not reporter._settlement_queue
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_counts_terminal_status_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    monkeypatch.setattr(
        httpx, "Client", _make_recording_client(sink, lambda _url: _terminal_response(400))
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report_confirm(_make_confirm_request())
    reporter.report(_make_event())

    blocking_exit_flush(reporter)

    counts = reporter.dropped_counts
    assert counts.get("confirm.terminal_status") == 1
    assert counts.get("event.terminal_status") == 1
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_counts_202_rejections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-review coverage gap: the EXIT drain parses 202 bodies too — a
    per-event rejection inside an accepted batch is a counted terminal loss on
    both the settlement-event and standalone-batch ingest paths."""
    sink: list[str] = []

    class _PartialRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"ingested": 0, "rejected": [{"index": 0, "reason": "invalid"}]}

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client(sink, lambda _url: _PartialRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report_settlement(_make_confirm_request(), _make_event(call_id=call_uuid("pair")))
    reporter.report(_make_event(call_id=call_uuid("standalone")))

    blocking_exit_flush(reporter)

    # One rejection per ingest POST: the settlement's event and the batch.
    assert reporter.dropped_counts.get("event.ingest_rejected") == 2
    assert not reporter.dropped_counts.get("settlement_confirm.terminal_status")
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_partial_deadline_counts_unsent_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1 review pin: when the exit budget dies mid-drain, the popped-but-unsent
    item goes back to its queue and the final sweep counts it — nothing
    silently vanishes between popleft and POST."""
    fake = {"t": 1000.0}
    monkeypatch.setattr("solwyn._lifecycle._monotonic", lambda: fake["t"])
    sink: list[str] = []

    def _consume_budget(_url: str) -> _RecordingResponse:
        fake["t"] += 10.0  # the first POST eats the whole exit budget
        return _RecordingResponse()

    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink, _consume_budget))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, shutdown_deadline=5.0)
    reporter.report_confirm(_make_confirm_request(call_id=call_uuid("sent")))
    reporter.report_confirm(_make_confirm_request(call_id=call_uuid("expired")))

    blocking_exit_flush(reporter)

    assert len(sink) == 1
    assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    assert not reporter._confirm_queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_open_breaker_gets_single_recovery_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 review pin: a recovery-eligible OPEN breaker admits exactly ONE
    HALF_OPEN probe at exit; a failed probe re-opens the breaker, so the rest
    of the backlog is refused instead of hammering a down endpoint."""
    sink: list[str] = []

    def _down(url: str) -> object:
        raise httpx.ConnectError("still down")

    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink, _down))
    breaker = _eligible_open_breaker()
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, control_plane_breaker=breaker)
    for i in range(3):
        reporter.report_confirm(_make_confirm_request(call_id=call_uuid(f"c{i}")))
    reporter.report_settlement(_make_confirm_request(), _make_event())

    blocking_exit_flush(reporter)

    assert len([u for u in sink if "budgets/confirm" in u]) == 1  # the single probe
    counts = reporter.dropped_counts
    assert counts.get("confirm.retry_exhausted") == 1  # the failed probe itself
    assert counts.get("confirm.exit_breaker_open") == 2  # the refused remainder
    assert counts.get("settlement_confirm.exit_breaker_open") == 1
    # The settlement's event still got its (failed, counted) ingest attempt.
    assert any("metadata/ingest" in u for u in sink)
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_recovered_breaker_drains_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recovery-eligible OPEN breaker whose probe SUCCEEDS closes and lets
    the whole backlog drain at exit."""
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    breaker = _eligible_open_breaker()
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, control_plane_breaker=breaker)
    for i in range(3):
        reporter.report_confirm(_make_confirm_request(call_id=call_uuid(f"c{i}")))

    blocking_exit_flush(reporter)

    assert len([u for u in sink if "budgets/confirm" in u]) == 3
    assert reporter.dropped_counts == {}
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_drain_publishes_disposition_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-review P1 pin: releasing ownership and publishing the drop must be
    ATOMIC under the ownership lock. If the seal can land in the gap between
    them, the drain returns with the item accounted NOWHERE — and at
    interpreter exit the daemon worker may never run again to record it. A
    gated counter holds the worker's publication open across the join
    deadline; the drain must wait for it, never return empty-handed."""
    sink: list[str] = []
    monkeypatch.setattr(
        httpx, "Client", _make_recording_client(sink, lambda _url: _terminal_response(400))
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(_make_event())

    gate = threading.Event()
    published: list[tuple[str, str, int]] = []

    def gated_counter(kind: str, reason: str, n: int = 1) -> None:
        if reason != "shutdown_deadline":
            gate.wait(timeout=5.0)  # hold the publication open past the join deadline
        published.append((kind, reason, n))

    releaser = threading.Timer(0.6, gate.set)
    releaser.daemon = True
    releaser.start()
    try:
        _drain_queues_blocking(
            reporter._confirm_queue,
            reporter._settlement_queue,
            reporter._queue,
            reporter.api_url,
            reporter.api_key,
            reporter._control_plane_breaker,
            0.15,  # join deadline expires while the publication is gated
            gated_counter,
        )
    finally:
        releaser.cancel()
        gate.set()

    assert published == [("event", "terminal_status", 1)], (
        f"drain returned before the disposition was published: {published}"
    )
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True  # queues empty; disarm the exit hook


class _SlowDripServer:
    """A REAL localhost server that answers a POST with an endless drip.

    Sends valid response headers with a huge Content-Length, then one body
    byte every 0.1s — under httpx's per-socket-op read timeout, so the read
    never times out and only an external wall-clock bound can end the drain.
    """

    def __init__(self) -> None:
        stop = self._stop = threading.Event()

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    self.rfile.read(length)
                    self.send_response(202)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "1000000")
                    self.end_headers()
                    while not stop.is_set():
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.1)
                except OSError:
                    pass  # client torn down mid-drip

            def log_message(self, *_args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.mark.unit
def test_exit_flush_bounded_despite_slow_drip_post() -> None:
    """Re-review #2 pin, REAL httpx client against a REAL slow-drip server:
    httpx timeouts bound socket operations, not total response time, and
    closing a sync client does not reliably interrupt a blocked read — so the
    deadline must come from joining the drain worker, with the in-hand item
    claimed by the seal."""
    server = _SlowDripServer()
    try:
        reporter = AsyncMetadataReporter(
            f"http://127.0.0.1:{server.port}", VALID_API_KEY, shutdown_deadline=0.3
        )
        reporter.report(_make_event())

        start = time.monotonic()
        blocking_exit_flush(reporter)
        elapsed = time.monotonic() - start

        assert elapsed < 0.55, f"exit drain outlived its deadline: {elapsed:.3f}s"
        assert reporter.dropped_counts.get("event.shutdown_deadline") == 1
        assert not reporter._queue
        if reporter._finalizer is not None:
            reporter._finalizer.detach()
    finally:
        server.close()
        # Tear down cleanly: the drip's end errors the worker's read promptly.
        for worker in threading.enumerate():
            if worker.name == "solwyn-exit-flush":
                worker.join(timeout=2.0)
