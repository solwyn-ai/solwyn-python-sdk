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
import time
from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
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
