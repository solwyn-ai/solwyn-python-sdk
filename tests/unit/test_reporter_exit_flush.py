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
import weakref
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, call_uuid

import solwyn._lifecycle as lifecycle
from solwyn._lifecycle import (
    _drain_queues_blocking,
    _ExitIngestRejectionKind,
    _gc_drop_counter,
    _parse_exit_ingest_rejections,
    blocking_exit_flush,
)
from solwyn._surfaces import SurfaceContext
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.client import AsyncSolwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter, _IngestRejectionKind

_URL = "https://api.test.solwyn.ai"


class _UntrackedAsyncOpenAIClient:
    async def post(self) -> str:
        return "posted"

    async def close(self) -> None:
        return None


_UntrackedAsyncOpenAIClient.__module__ = "openai._client"
_UntrackedAsyncOpenAIClient.__name__ = "AsyncOpenAI"


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
def test_async_finalizer_has_no_untracked_delivery_state_when_reporting_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))

    reporter = AsyncMetadataReporter(
        _URL,
        VALID_API_KEY,
        sdk_instance_id="gc-disabled-instance",
        report_untracked_surfaces=False,
    )
    _observe_untracked(reporter)

    assert reporter._untracked_state is None
    del reporter
    gc.collect()

    assert f"{_URL}/api/v1/untracked-surfaces" not in sink


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
def test_no_loop_advisory_access_is_silent_and_still_exit_flushes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    client = AsyncSolwyn(
        _UntrackedAsyncOpenAIClient(),
        api_key=VALID_API_KEY,
        api_url=_URL,
        on_unmetered="allow",
    )

    try:
        with caplog.at_level("WARNING"):
            operation = client.post

        assert callable(operation)
        assert client._solwyn_reporter._warned_no_loop is False
        assert not any(
            record.name in {"solwyn._base", "solwyn.reporter"} for record in caplog.records
        )
        state = client._solwyn_reporter._untracked_state
        assert state is not None
        assert ("openai", "openai_sdk", "async", "post") in state.observations

        blocking_exit_flush(client._solwyn_reporter)

        assert f"{_URL}/api/v1/untracked-surfaces" in sink
    finally:
        asyncio.run(client.close())


@pytest.mark.unit
def test_failing_exit_advisory_post_is_silent_and_cadence_throttled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink: list[str] = []

    def down(_url: str) -> object:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink, down))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, sdk_instance_id="exit-failure")
    _observe_untracked(reporter)
    state = reporter._untracked_state
    assert state is not None
    key = ("openai", "openai_sdk", "async", "responses.create")

    with caplog.at_level("WARNING"):
        blocking_exit_flush(reporter)

    assert f"{_URL}/api/v1/untracked-surfaces" in sink
    assert key not in state.last_sent_occurrences
    attempted_at = state.last_attempted_at[key]
    assert state.reports_due(attempted_at + 899.999) is False
    assert state.reports_due(attempted_at + 900.0) is True
    assert reporter.dropped_counts == {}
    assert "lifecycle.exit_flush_failed" not in caplog.text
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True
    asyncio.run(reporter._http.aclose())


@pytest.mark.unit
def test_blocking_exit_flush_skips_untracked_delivery_when_reporting_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: list[str] = []
    monkeypatch.setattr(httpx, "Client", _make_recording_client(sink))
    reporter = AsyncMetadataReporter(
        _URL,
        VALID_API_KEY,
        sdk_instance_id="exit-disabled-instance",
        report_untracked_surfaces=False,
    )
    _observe_untracked(reporter)

    blocking_exit_flush(reporter)

    assert f"{_URL}/api/v1/untracked-surfaces" not in sink
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
def test_gc_finalizer_failed_aggregate_logs_underlying_receipt_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A terminal aggregate replay represents every folded denial, not one."""

    def _down(_url: str) -> object:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", _make_recording_client([], _down))
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter._fold_or_count_event_drop(
        _make_event(
            status="budget_denied",
            output_tokens=0,
            agent_run_id="gc-fold-run",
            deny_source="server",
            deny_reason="structural",
            estimated_output_bound=20,
            receipt_aggregate_count=5,
        ),
        "retry_exhausted",
    )

    with caplog.at_level("WARNING", logger="solwyn._lifecycle"):
        del reporter
        gc.collect()

    assert "lifecycle.gc_flush_dropped" in caplog.text
    assert "kind=event" in caplog.text
    assert "reason=retry_exhausted" in caplog.text
    assert "n=5" in caplog.text


@pytest.mark.unit
def test_gc_finalizer_full_legacy_rejection_logs_aggregate_cardinality(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _LegacyRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ingested": 0,
                "rejected": [
                    {
                        "code": "unknown_model",
                        "model": "gpt-5.5",
                        "message": "structural",
                    }
                ],
            }

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _LegacyRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(
        _make_event(
            status="budget_denied",
            output_tokens=0,
            agent_run_id="gc-legacy-run",
            deny_source="aggregate_replay",
            deny_reason=None,
            estimated_output_bound=20,
            receipt_aggregate_count=100,
        )
    )

    with caplog.at_level("WARNING", logger="solwyn._lifecycle"):
        del reporter
        gc.collect()

    assert "lifecycle.gc_flush_dropped" in caplog.text
    assert "reason=ingest_rejected" in caplog.text
    assert "n=100" in caplog.text


@pytest.mark.unit
def test_gc_finalizer_exact_index_rejection_disposes_only_the_named_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The GC/atexit parser resolves EXACT indexes against its own batch: only
    the named member is terminal, and it is counted at full receipt weight."""

    class _ExactRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ingested": 1,
                "rejected": [
                    {
                        "index": 1,
                        "code": "unknown_model",
                        "model": "gpt-5.5",
                        "message": "structural",
                    }
                ],
            }

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _ExactRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(_make_event(call_id=call_uuid("gc-exact-accepted")))
    reporter.report(
        _make_event(
            call_id=call_uuid("gc-exact-rejected"),
            status="budget_denied",
            output_tokens=0,
            agent_run_id="gc-exact-run",
            deny_source="aggregate_replay",
            deny_reason=None,
            estimated_output_bound=20,
            receipt_aggregate_count=4,
        )
    )

    with caplog.at_level("WARNING", logger="solwyn._lifecycle"):
        del reporter
        gc.collect()

    drops = [
        record.getMessage()
        for record in caplog.records
        if "lifecycle.gc_flush_dropped" in record.getMessage()
    ]
    assert len(drops) == 1
    assert "kind=event" in drops[0]
    assert "reason=ingest_rejected" in drops[0]
    assert "n=4" in drops[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rejected", "expected_kind", "expected_count"),
    [
        # Exact indexes agree.
        ([{"index": 0, "code": "c", "model": "m", "message": "x"}], "exact", 1),
        # Missing (code, model, message) costs the WARNING, never the identity.
        ([{"index": 0}], "exact", 1),
        # Index-shape violations degrade to the count the legacy parser proved.
        (
            [
                {"index": 0, "code": "c", "model": "m", "message": "x"},
                {"index": 0, "code": "c", "model": "m", "message": "x"},
            ],
            "legacy",
            2,
        ),
        ([{"index": 5, "code": "c", "model": "m", "message": "x"}], "legacy", 1),
        ([{"index": -1, "code": "c", "model": "m", "message": "x"}], "legacy", 1),
        ([{"index": 0.5, "code": "c", "model": "m", "message": "x"}], "legacy", 1),
        ([{"code": "c", "model": "m", "message": "x"}], "legacy", 1),
        # Only an untrustworthy COUNT stays malformed.
        (["not-a-dict"], "malformed", 0),
    ],
)
def test_both_ingest_rejection_parsers_reach_the_same_disposition(
    rejected: list[object], expected_kind: str, expected_count: int
) -> None:
    """Same server body, same accounting: the in-process reporter and the exit
    twin must never disagree about how much loss a 202 proves."""
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = {"ingested": 0, "rejected": rejected}
    reporter = MetadataReporter(
        _URL,
        VALID_API_KEY,
        flush_interval=60.0,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    try:
        in_process = reporter._parse_ingest_rejections(response, 2)
    finally:
        reporter.close(timeout=1.0)
    at_exit = _parse_exit_ingest_rejections(response, 2)

    assert in_process.kind is _IngestRejectionKind(expected_kind)
    assert at_exit.kind is _ExitIngestRejectionKind(expected_kind)
    assert in_process.count == at_exit.count == expected_count
    assert set(in_process.indexes) == set(at_exit.indexes)


@pytest.mark.unit
def test_gc_drop_logging_exceptions_do_not_abort_remaining_exit_dispositions() -> None:
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report_confirm(_make_confirm_request())
    reporter.report(_make_event())

    with patch("solwyn._lifecycle.logger.warning", side_effect=RuntimeError("handler")) as warning:
        _drain_queues_blocking(
            reporter._confirm_queue,
            reporter._settlement_queue,
            reporter._queue,
            reporter.api_url,
            reporter.api_key,
            reporter._control_plane_breaker,
            0.0,
            _gc_drop_counter,
            reporter._new_exit_http_client,
            reporter._untracked_state,
            reporter._receipt_fold_state,
            reporter._sdk_instance_id,
        )

    assert warning.call_count == 2
    assert not reporter._confirm_queue
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True


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
def test_exit_flush_counts_legacy_indexless_rejections_without_guessing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy 202 bodies prove a loss count but not which event was rejected."""
    sink: list[str] = []

    class _LegacyRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ingested": 0,
                "rejected": [
                    {
                        "code": "unknown_model",
                        "model": "gpt-5.5",
                        "message": "structural",
                    }
                ],
            }

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client(sink, lambda _url: _LegacyRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    denied = _make_event(
        call_id=call_uuid("legacy-pair-denied"),
        status="budget_denied",
        output_tokens=0,
        agent_run_id="legacy-run",
        deny_source="server",
        deny_reason="structural",
        estimated_output_bound=20,
    )
    reporter.report_settlement(_make_confirm_request(), denied)
    reporter.report(_make_event(call_id=call_uuid("legacy-standalone-success")))

    blocking_exit_flush(reporter)
    blocking_exit_flush(reporter)  # ownership was released; no shutdown double count

    assert reporter.dropped_counts.get("event.ingest_rejected") == 2
    assert reporter._receipt_fold_snapshot() == {}
    assert not reporter._settlement_queue
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_full_legacy_rejection_counts_aggregate_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ingested": 0,
                "rejected": [
                    {
                        "code": "unknown_model",
                        "model": "gpt-5.5",
                        "message": "structural",
                    }
                ],
            }

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _LegacyRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(
        _make_event(
            status="budget_denied",
            output_tokens=0,
            agent_run_id="exit-legacy-run",
            deny_source="aggregate_replay",
            deny_reason=None,
            estimated_output_bound=20,
            receipt_aggregate_count=100,
        )
    )

    blocking_exit_flush(reporter)

    assert reporter.dropped_counts == {"event.ingest_rejected": 100}
    assert reporter._receipt_fold_snapshot() == {}
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_partial_legacy_rejection_charges_the_heaviest_receipt_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyRejectResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ingested": 1,
                "rejected": [
                    {
                        "code": "unknown_model",
                        "model": "gpt-5.5",
                        "message": "structural",
                    }
                ],
            }

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _LegacyRejectResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(
        _make_event(
            call_id=call_uuid("exit-partial-aggregate"),
            status="budget_denied",
            output_tokens=0,
            agent_run_id="exit-partial-run",
            deny_source="aggregate_replay",
            deny_reason=None,
            estimated_output_bound=20,
            receipt_aggregate_count=100,
        )
    )
    reporter.report(_make_event(call_id=call_uuid("exit-partial-success")))

    blocking_exit_flush(reporter)

    # One proven rejection, no proven identity: the heaviest candidate in the
    # batch is charged so an aggregate receipt can never be understated.
    assert reporter.dropped_counts == {"event.ingest_rejected": 100}
    assert reporter._receipt_fold_snapshot() == {}
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
@pytest.mark.parametrize("invalid_index", [0.9, True])
def test_exit_flush_non_integer_rejection_index_degrades_to_legacy_count_only(
    monkeypatch: pytest.MonkeyPatch, invalid_index: object
) -> None:
    class _MalformedIndexResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"ingested": 1, "rejected": [{"index": invalid_index}]}

    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _MalformedIndexResponse())
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(
        _make_event(
            call_id=call_uuid("exit-malformed-denied"),
            status="budget_denied",
            output_tokens=0,
            agent_run_id="exit-malformed-run",
            deny_source="server",
            deny_reason="structural",
            estimated_output_bound=20,
        )
    )
    reporter.report(_make_event(call_id=call_uuid("exit-malformed-success")))

    blocking_exit_flush(reporter)

    # Same rule as the in-process parser: an unusable index costs identity,
    # never the count. The denied event stays unfolded (identity unproven).
    assert reporter.dropped_counts == {"event.ingest_rejected": 1}
    assert reporter._receipt_fold_snapshot() == {}
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()


@pytest.mark.unit
def test_exit_flush_refuses_reentrant_async_report_after_final_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        httpx, "Client", _make_recording_client([], lambda _url: _terminal_response(400))
    )
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter.report(_make_event(call_id=call_uuid("exit-before-reentrant")))
    reentrant = _make_event(call_id=call_uuid("exit-reentrant"))
    reentered = threading.Event()

    monkeypatch.setattr(lifecycle, "_LIVE_SYNC_REPORTERS", [])
    monkeypatch.setattr(lifecycle, "_LIVE_ASYNC_REPORTERS", [reporter])
    monkeypatch.setattr(lifecycle, "_exit_surrender_all", lambda: None)

    def warning(message: str, *_args: object, **_kwargs: object) -> None:
        if message.startswith("reporter.spend_events_dropped") and not reentered.is_set():
            reentered.set()
            reporter.report(reentrant)

    with patch("solwyn.reporter.logger.warning", side_effect=warning):
        lifecycle._exit_flush_all()

    assert reentered.is_set()
    assert reporter._closed is True
    assert reporter.dropped_counts == {
        "event.terminal_status": 1,
        "event.closed_enqueue": 1,
    }
    assert reporter._receipt_fold_snapshot() == {}
    assert not reporter._queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True
    asyncio.run(reporter._http.aclose())


@pytest.mark.unit
def test_exit_flush_waits_for_sync_seal_dispositions_before_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enqueue gate is not a completion marker until seal publishes loss."""
    reporter = MetadataReporter(
        _URL,
        VALID_API_KEY,
        flush_interval=60.0,
        shutdown_deadline=0.5,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    reporter.report(_make_event(call_id=call_uuid("mid-seal-exit")))
    mutation_entered = threading.Event()
    release_mutation = threading.Event()
    seal_finished = threading.Event()
    exit_finished = threading.Event()
    original_record = reporter._record_drop

    def gated_record(kind: str, reason: str, n: int = 1) -> None:
        if (
            threading.current_thread().name == "sync-seal-race"
            and kind == "event"
            and reason == "shutdown_deadline"
        ):
            mutation_entered.set()
            assert release_mutation.wait(timeout=5.0)
        original_record(kind, reason, n)

    def seal() -> None:
        reporter._seal_delivery()
        seal_finished.set()

    def exit_flush() -> None:
        lifecycle._exit_flush_all()
        exit_finished.set()

    monkeypatch.setattr(lifecycle, "_LIVE_SYNC_REPORTERS", [reporter])
    monkeypatch.setattr(lifecycle, "_LIVE_ASYNC_REPORTERS", [])
    monkeypatch.setattr(lifecycle, "_exit_surrender_all", lambda: None)
    sealer = threading.Thread(target=seal, name="sync-seal-race", daemon=True)
    exiter = threading.Thread(target=exit_flush, name="sync-exit-race", daemon=True)
    try:
        with patch.object(reporter, "_record_drop", side_effect=gated_record):
            sealer.start()
            assert mutation_entered.wait(timeout=2.0)
            assert reporter._delivery_closed is True
            assert getattr(reporter, "_delivery_completed", False) is False
            assert not reporter._queue
            assert reporter.dropped_counts == {}

            exiter.start()
            assert not exit_finished.wait(timeout=0.1), (
                "exit returned while seal still owned an unpublished event disposition"
            )
            release_mutation.set()
            sealer.join(timeout=2.0)
            exiter.join(timeout=2.0)

        assert seal_finished.is_set()
        assert exit_finished.is_set()
        assert reporter._delivery_completed is True
        assert reporter.dropped_counts == {"event.shutdown_deadline": 1}
        assert not reporter._queue
    finally:
        release_mutation.set()
        sealer.join(timeout=2.0)
        exiter.join(timeout=2.0)
        if not reporter._shutdown.is_set():
            reporter.close(timeout=1.0)


@pytest.mark.unit
def test_worker_reentrant_close_finalizes_before_last_sync_reporter_reference_drops() -> None:
    """A weak registry cannot rescue spend after the worker releases its last ref."""
    reporter = MetadataReporter(
        _URL,
        VALID_API_KEY,
        batch_size=1,
        flush_interval=0.01,
        max_send_attempts=5,
        retry_backoff_base=60.0,
        retry_backoff_cap=60.0,
        shutdown_deadline=1.0,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    current = _make_event(call_id=call_uuid("worker-finalize-current"))
    queued = _make_event(
        call_id=call_uuid("worker-finalize-queued"),
        status="budget_denied",
        output_tokens=0,
        agent_run_id="queued-weighted-run",
        deny_source="server",
        deny_reason="structural",
        estimated_output_bound=30,
        receipt_aggregate_count=5,
    )
    folded = _make_event(
        call_id=call_uuid("worker-finalize-folded"),
        status="budget_denied",
        input_tokens=70,
        output_tokens=0,
        agent_run_id="folded-weighted-run",
        deny_source="server",
        deny_reason="structural",
        estimated_output_bound=90,
        receipt_aggregate_count=7,
    )
    request = httpx.Request("POST", f"{_URL}/api/v1/metadata/ingest")
    healthy = MagicMock(spec=httpx.Response)
    healthy.raise_for_status.return_value = None
    healthy.json.return_value = {"ingested": 1, "rejected": []}
    failed_once = False
    delivered: list[dict[str, object]] = []

    def post(
        _url: str,
        *,
        json: object,
        headers: object,
        timeout: object,
    ) -> httpx.Response:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise httpx.ConnectError("control plane unavailable", request=request)
        assert isinstance(json, list)
        delivered.extend(json)
        return healthy

    close_entered = threading.Event()
    close_finished = threading.Event()
    close_errors: list[BaseException] = []
    reporter_ref = weakref.ref(reporter)

    def warning(message: str, *_args: object, **_kwargs: object) -> None:
        if message.startswith("Failed to send metadata batch") and not close_entered.is_set():
            close_entered.set()
            live = reporter_ref()
            assert live is not None
            try:
                live.close(timeout=1.0)
            except BaseException as exc:
                close_errors.append(exc)
                raise
            finally:
                close_finished.set()

    worker = reporter._thread
    try:
        with (
            patch.object(reporter._http, "post", side_effect=post),
            patch("solwyn.reporter.logger.warning", side_effect=warning),
        ):
            reporter._fold_or_count_event_drop(folded, "retry_exhausted")
            reporter.report(current)
            reporter.report(queued)
            assert close_entered.wait(timeout=2.0)
            assert close_finished.wait(timeout=2.0)
            worker.join(timeout=2.0)
            assert not worker.is_alive()

        assert close_errors == []
        assert reporter._shutdown.is_set()
        assert reporter._delivery_closed is True
        assert reporter._delivery_completed is True
        assert not reporter._queue
        assert reporter._receipt_fold_snapshot() == {}
        assert reporter.dropped_counts == {}
        by_call = {str(item["call_id"]): item for item in delivered}
        assert by_call[str(current.call_id)]["status"] == "success"
        assert by_call[str(queued.call_id)]["receipt_aggregate_count"] == 5
        replay = [
            item
            for item in delivered
            if item.get("deny_source") == "aggregate_replay"
            and item.get("agent_run_id") == "folded-weighted-run"
        ]
        assert len(replay) == 1
        assert replay[0]["receipt_aggregate_count"] == 7
        assert replay[0]["input_tokens"] == 70
        assert replay[0]["estimated_output_bound"] == 90

        del reporter
        gc.collect()
        assert reporter_ref() is None
    finally:
        live = reporter_ref()
        if live is not None and not getattr(live, "_delivery_completed", False):
            live._shutdown.set()
            live._seal_delivery()
            live._http.close()


@pytest.mark.unit
def test_reentrant_close_from_final_flush_worker_sends_instead_of_sealing() -> None:
    """A close-calling log handler on close()'s OWN final-flush worker must not
    deadlock into the expired force-seal: the worker is reporter-owned, so it
    takes the same finalize-request path as the flush/breaker/advisory threads
    and keeps draining."""
    reporter = MetadataReporter(
        _URL,
        VALID_API_KEY,
        batch_size=10,
        flush_interval=60.0,
        max_send_attempts=1,
        shutdown_deadline=1.0,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    request = httpx.Request("POST", f"{_URL}/api/v1/budgets/confirm")
    healthy = MagicMock(spec=httpx.Response)
    healthy.raise_for_status.return_value = None
    healthy.json.return_value = {"ingested": 1, "rejected": []}
    delivered: list[dict[str, object]] = []

    def post(url: str, *, json: object, headers: object, timeout: object) -> httpx.Response:
        if url.endswith("/budgets/confirm"):
            raise httpx.ConnectError("control plane unavailable", request=request)
        assert isinstance(json, list)
        delivered.extend(json)
        return healthy

    reentrant_thread: list[str] = []
    close_errors: list[BaseException] = []

    def warning(message: str, *_args: object, **_kwargs: object) -> None:
        if not message.startswith("reporter.confirm_send_failed") or reentrant_thread:
            return
        reentrant_thread.append(threading.current_thread().name)
        try:
            reporter.close(timeout=1.0)
        except BaseException as exc:  # pragma: no cover - failure detail only
            close_errors.append(exc)

    reporter.report_confirm(_make_confirm_request())
    reporter.report(_make_event(call_id=call_uuid("final-flush-reentrant")))
    try:
        with (
            patch.object(reporter._http, "post", side_effect=post),
            patch("solwyn.reporter.logger.warning", side_effect=warning),
        ):
            reporter.close(timeout=2.0)

        assert close_errors == []
        assert reentrant_thread == ["solwyn-final-flush"]
        # The queued event was DELIVERED, not force-sealed at a burnt deadline.
        assert [str(item["call_id"]) for item in delivered] == [
            str(call_uuid("final-flush-reentrant"))
        ]
        assert reporter.dropped_counts == {"confirm.retry_exhausted": 1}
        assert reporter._delivery_completed is True
    finally:
        if not getattr(reporter, "_delivery_completed", False):
            reporter._shutdown.set()
            reporter._seal_delivery()
            reporter._http.close()


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
            reporter._new_exit_http_client,
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
