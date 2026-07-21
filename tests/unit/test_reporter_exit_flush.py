"""Exit flush (R2): a normally-exiting process still delivers queued spend.

Scenario-1 pin: a sync reporter that queues a settlement and exits WITHOUT
close() must flush it via the atexit hook. Plus in-process coverage of the async
reporter's weakref finalizer, the breaker-open exit skip, and close() detaching
the finalizer.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn._lifecycle import blocking_exit_flush
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import AsyncMetadataReporter

_URL = "https://api.test.solwyn.ai"


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": "call_exit_confirm",
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


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


_CHILD = """
from datetime import UTC, datetime

import solwyn.reporter as r
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName

confirm = BudgetConfirmRequest(
    reservation_id="res_exit",
    model="gpt-5.5",
    provider=ProviderName.OPENAI,
    call_id="call_exit",
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
    call_id="call_exit",
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
            [sys.executable, "-c", _CHILD.format(port=server.port)],
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


def _make_recording_client(sink: list[str]) -> type:
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
        ) -> _RecordingResponse:
            sink.append(url)
            return _RecordingResponse()

    return _Client


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

    # close() supersedes the safety net: the finalizer is detached.
    assert not reporter._finalizer.alive


async def _async_noop_post(*_a: object, **_k: object) -> _RecordingResponse:
    return _RecordingResponse()
