"""One deadline bounds the whole shutdown/flush/breaker chain (R13).

``close(timeout=...)`` (default: ``reporter_shutdown_deadline``) shares a single
monotonic deadline across the thread/task join, the final flush, and the breaker
report cycle. Work still queued when the deadline is reached is counted
``shutdown_deadline`` and dropped, so a black-holed control plane can never make
shutdown hang on a serial per-request timeout chain.

These use REAL small sleeps (not the fake clock) to pin wall-clock boundedness.
Every constant stays <= 1.5s so the suite stays fast.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _PendingConfirm,
    _PendingEvent,
    _PendingSettlement,
)

_URL = "https://api.test.solwyn.ai"


def _make_event(**overrides) -> MetadataEvent:
    defaults = {
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 200.0,
        "status": "success",
        "is_model_fallback": False,
        "sdk_instance_id": "test-instance-001",
        "timestamp": datetime.now(UTC),
        "call_id": "call_shutdown_event",
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": "call_shutdown_confirm",
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


def _unstarted(**kwargs) -> MetadataReporter:
    """A sync reporter whose thread exited but whose _shutdown stays UNSET."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._thread.join(timeout=2.0)
    return reporter


def _load_work(reporter: MetadataReporter | AsyncMetadataReporter, n: int = 5) -> None:
    for _ in range(n):
        reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
        reporter._settlement_queue.append(
            _PendingSettlement(_PendingConfirm(_make_confirm_request()), _make_event())
        )
        reporter._queue.append(_PendingEvent(_make_event()))


@pytest.mark.unit
class TestSyncShutdownDeadline:
    def test_close_bounded_by_single_deadline(self, caplog: pytest.LogCaptureFixture) -> None:
        reporter = _unstarted(flush_interval=3600.0)
        _load_work(reporter)

        def slow_post(url: str, **_kw: object) -> httpx.Response:
            time.sleep(1.0)
            raise httpx.ConnectError("black hole")

        with (
            patch.object(reporter._http, "post", side_effect=slow_post),
            caplog.at_level("WARNING"),
        ):
            start = time.monotonic()
            reporter.close(timeout=1.5)
            elapsed = time.monotonic() - start

        # Pre-fix: a serial 5s-per-request chain across ~15 queued items. Post-fix:
        # bounded by the single 1.5s deadline (a couple of in-flight posts).
        assert elapsed < 3.0, f"close took {elapsed:.2f}s — the shutdown deadline must bound it"
        counts = reporter.dropped_counts
        assert counts.get("settlement_confirm.shutdown_deadline", 0) >= 1
        assert counts.get("event.shutdown_deadline", 0) >= 1
        assert "reporter.spend_events_dropped" in caplog.text

    def test_close_default_uses_shutdown_deadline_knob(self) -> None:
        reporter = _unstarted(flush_interval=3600.0, shutdown_deadline=0.2)
        for _ in range(5):
            reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))

        def slow_post(url: str, **_kw: object) -> httpx.Response:
            time.sleep(1.0)
            raise httpx.ConnectError("black hole")

        with patch.object(reporter._http, "post", side_effect=slow_post):
            start = time.monotonic()
            reporter.close()  # no arg -> reporter_shutdown_deadline (0.2)
            elapsed = time.monotonic() - start

        assert elapsed < 1.5, f"close took {elapsed:.2f}s — must use the 0.2s knob deadline"


@pytest.mark.unit
class TestAsyncShutdownDeadline:
    @pytest.mark.asyncio
    async def test_async_close_bounded_by_single_deadline(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=3600.0)
        _load_work(reporter)

        async def slow_post(url: str, **_kw: object) -> httpx.Response:
            await asyncio.sleep(1.0)
            raise httpx.ConnectError("black hole")

        with patch.object(reporter._http, "post", new=slow_post):
            start = time.monotonic()
            await reporter.close(timeout=1.5)
            elapsed = time.monotonic() - start

        assert elapsed < 3.0, f"async close took {elapsed:.2f}s — deadline must bound it"
        counts = reporter.dropped_counts
        assert counts.get("settlement_confirm.shutdown_deadline", 0) >= 1
        assert counts.get("event.shutdown_deadline", 0) >= 1


@pytest.mark.unit
def test_close_deadline_bounds_metadata_batch_send() -> None:
    """P0 review pin: the ingest POST must be clamped into the shutdown window.

    The mock HONORS the timeout kwarg (a mock that ignores it cannot tell a
    bounded request from an unbounded one — the gap that hid this). Pre-fix,
    _send_batch passed no timeout, so a black-holed control plane held close()
    for the full 10s client default regardless of close(timeout=...).
    """
    reporter = _unstarted()
    reporter.report(_make_event())

    def _blackholed_post(url: str, **kwargs: object) -> None:
        timeout = kwargs.get("timeout", 10.0)
        time.sleep(min(float(timeout), 10.0))  # type: ignore[arg-type]
        raise httpx.ConnectTimeout("simulated blackhole")

    start = time.monotonic()
    with patch.object(reporter._http, "post", side_effect=_blackholed_post):
        reporter.close(timeout=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"close(timeout=1.0) took {elapsed:.1f}s — ingest POST unbounded"
    reporter._http.close()


@pytest.mark.unit
async def test_async_close_deadline_bounds_metadata_batch_send() -> None:
    """Async twin of the P0 ingest-deadline pin (timeout-honoring mock)."""
    reporter = AsyncMetadataReporter("https://api.test.solwyn.ai", VALID_API_KEY)
    reporter.start()
    reporter.report(_make_event())

    async def _blackholed_post(url: str, **kwargs: object) -> None:
        timeout = kwargs.get("timeout", 10.0)
        await asyncio.sleep(min(float(timeout), 10.0))  # type: ignore[arg-type]
        raise httpx.ConnectTimeout("simulated blackhole")

    start = time.monotonic()
    with patch.object(reporter._http, "post", side_effect=_blackholed_post):
        await reporter.close(timeout=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"close(timeout=1.0) took {elapsed:.1f}s — ingest POST unbounded"


@pytest.mark.unit
def test_settlement_drop_at_deadline_counts_event_loss() -> None:
    """P2 review pin: a settlement dropped at the deadline loses its confirm
    AND its event — dropped_counts must say so for both kinds."""
    reporter = _unstarted()
    reporter.report_settlement(_make_confirm_request(), _make_event())
    reporter._flush_remaining(deadline=time.monotonic() - 1.0, final=True)
    assert reporter.dropped_counts["settlement_confirm.shutdown_deadline"] == 1
    assert reporter.dropped_counts["event.shutdown_deadline"] == 1
    reporter._http.close()


@pytest.mark.unit
def test_exit_breaker_open_settlement_event_still_ships() -> None:
    """P2 review pin: at final flush with the breaker OPEN the confirms are
    undeliverable, but ingest is not breaker-gated — the settlement's event
    must still get its deadline-bounded exit attempt."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=3600, name="control-plane")
    breaker.record_failure()  # OPEN, not recovery-eligible
    reporter = _unstarted(control_plane_breaker=breaker)
    reporter.report_settlement(_make_confirm_request(), _make_event())

    sent_urls: list[str] = []

    def _ok_post(url: str, **kwargs: object) -> MagicMock:
        sent_urls.append(url)
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = {"ingested": 1, "rejected": []}
        return response

    with patch.object(reporter._http, "post", side_effect=_ok_post):
        reporter._flush_remaining(deadline=time.monotonic() + 5.0, final=True)

    assert reporter.dropped_counts.get("settlement_confirm.exit_breaker_open") == 1
    assert any("metadata/ingest" in url for url in sent_urls), (
        "settlement event must ride ingest even when confirms are breaker-held"
    )
    assert not any("budgets/confirm" in url for url in sent_urls)
    reporter._http.close()


@pytest.mark.unit
def test_report_after_close_counts_closed_enqueue() -> None:
    """P2 review pin: report() after close() must count, not silently retain."""
    reporter = _unstarted()
    reporter._shutdown.set()
    reporter.report(_make_event())
    assert reporter.dropped_counts["event.closed_enqueue"] == 1
    assert len(reporter._queue) == 0
    reporter._http.close()


# ---------------------------------------------------------------------------
# Shutdown ownership: in-hand items and enqueue races (P1 review pins)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_timed_out_close_counts_in_hand_confirm_and_blocks_requeue() -> None:
    """P1 review pin: when close()'s bounded join expires while the flush
    thread holds a popped confirm mid-POST, close must count that in-hand item
    before returning, and the worker's later transient outcome must neither
    requeue it into the dead queue nor double count it."""
    reporter = MetadataReporter(_URL, VALID_API_KEY, flush_interval=0.05)
    entered = threading.Event()
    release = threading.Event()

    def blocked_post(url: str, **_kw: object) -> httpx.Response:
        entered.set()
        release.wait(timeout=5.0)
        raise httpx.ConnectError("transient")

    try:
        with patch.object(reporter._http, "post", side_effect=blocked_post):
            reporter.report_confirm(_make_confirm_request())
            assert entered.wait(2.0), "flush thread never picked up the confirm"

            reporter.close(timeout=0.2)

            # The in-hand confirm is counted BEFORE close returns.
            assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
            assert len(reporter._confirm_queue) == 0

            release.set()
            reporter._thread.join(timeout=2.0)
            assert not reporter._thread.is_alive()

        # The worker's RETRY outcome after sealing: no requeue, no double count.
        assert len(reporter._confirm_queue) == 0
        assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    finally:
        release.set()


@pytest.mark.unit
def test_enqueue_racing_close_is_refused_and_counted() -> None:
    """P1 review pin: an enqueue that passes the _shutdown check and then loses
    the race with close()'s final drain must be refused (counted), never
    appended into a queue nothing will ever drain."""
    reporter = _unstarted()
    producer_paused = threading.Event()
    resume_producer = threading.Event()

    def paused_ensure_thread() -> None:
        producer_paused.set()
        resume_producer.wait(timeout=5.0)

    reporter._ensure_thread = paused_ensure_thread  # type: ignore[method-assign]
    producer = threading.Thread(
        target=lambda: reporter.report_confirm(_make_confirm_request()),
        daemon=True,
    )
    producer.start()
    assert producer_paused.wait(2.0)

    reporter.close(timeout=0.5)  # completes fully and seals delivery
    resume_producer.set()
    producer.join(timeout=2.0)

    assert len(reporter._confirm_queue) == 0
    assert reporter.dropped_counts.get("confirm.closed_enqueue") == 1
    reporter._http.close()
