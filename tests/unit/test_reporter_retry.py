"""At-least-once delivery: bounded retry, breaker-open holds, counted drops.

These pin the reporter's PJ-3 hardening: a transient control-plane blip is
retried with backoff instead of dropped, a breaker-open flush HOLDS work (never
drops it), every unavoidable drop is counted under ``dropped_counts`` and logged
at a bounded rate, and the metadata event survives even when its confirm cannot.
"""

from __future__ import annotations

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
        "call_id": "call_retry_event",
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": "call_retry_confirm",
        "token_details": TokenDetails(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return BudgetConfirmRequest(**defaults)


def _error_response(status_code: int) -> MagicMock:
    """A 4xx/5xx httpx.Response stand-in: raise_for_status raises HTTPStatusError."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(spec=httpx.Request), response=resp
        )
    )
    return resp


def _ok_response() -> MagicMock:
    """A 2xx httpx.Response stand-in: raise_for_status is a no-op, clean body."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"ingested": 0, "rejected": []})
    return resp


class _FakeClock:
    """A patchable monotonic clock so backoff windows are deterministic."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _cp_breaker() -> CircuitBreaker:
    """A fresh CLOSED control-plane breaker (success_threshold=1, prod-shaped)."""
    return CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=1e9,
        success_threshold=1,
        name="control-plane",
    )


def _quiet(**kwargs) -> MetadataReporter:
    """A sync reporter with its background flush thread stopped."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    return reporter


def _unstarted(**kwargs) -> MetadataReporter:
    """A sync reporter whose thread exited but whose _shutdown stays UNSET.

    Lets report_confirm / report_settlement enqueue (they gate on _shutdown).
    """
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._thread.join(timeout=2.0)
    return reporter


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncReporterRetry:
    def test_transient_5xx_settlement_retries_then_delivers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = _quiet(max_send_attempts=5, retry_backoff_base=1.0)
        reporter._settlement_queue.append(
            _PendingSettlement(_PendingConfirm(_make_confirm_request()), _make_event())
        )
        confirm_posts: list[str] = []
        responses = [_error_response(503), _error_response(503), _ok_response()]

        def post(url: str, **_kw: object) -> MagicMock:
            if "budgets/confirm" in url:
                confirm_posts.append(url)
                return responses[len(confirm_posts) - 1]
            return _ok_response()

        with patch.object(reporter._http, "post", side_effect=post):
            reporter._flush_remaining()
            clock.advance(2.0)
            reporter._flush_remaining()
            clock.advance(4.0)
            reporter._flush_remaining()

        assert len(confirm_posts) == 3
        assert reporter.dropped_counts == {}
        assert len(reporter._settlement_queue) == 0
        assert len(reporter._queue) == 0
        reporter._http.close()

    def test_retry_exhaustion_drops_and_counts(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = _quiet(max_send_attempts=3, retry_backoff_base=1.0)
        reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))

        with (
            patch.object(reporter._http, "post", side_effect=httpx.ConnectError("down")),
            caplog.at_level("WARNING"),
        ):
            reporter._flush_remaining()
            clock.advance(2.0)
            reporter._flush_remaining()
            clock.advance(4.0)
            reporter._flush_remaining()

        assert len(reporter._confirm_queue) == 0
        assert reporter.dropped_counts["confirm.retry_exhausted"] == 1
        assert "reporter.spend_events_dropped" in caplog.text
        reporter._http.close()

    def test_settlement_event_survives_confirm_drop(self) -> None:
        reporter = _quiet()
        reporter._settlement_queue.append(
            _PendingSettlement(
                _PendingConfirm(_make_confirm_request(call_id="c1")),
                _make_event(call_id="c1"),
            )
        )
        urls: list[str] = []

        def post(url: str, **_kw: object) -> MagicMock:
            urls.append(url)
            if "budgets/confirm" in url:
                return _error_response(404)  # terminal — never retried
            return _ok_response()

        with patch.object(reporter._http, "post", side_effect=post):
            reporter._flush_remaining()

        assert reporter.dropped_counts.get("settlement_confirm.terminal_status") == 1
        # The metadata event is the durable spend truth — it MUST be sent even
        # though its confirm was terminal.
        assert any("metadata/ingest" in u for u in urls)
        assert len(reporter._settlement_queue) == 0
        assert len(reporter._queue) == 0
        reporter._http.close()

    def test_breaker_open_holds_settlement_until_recovery(self) -> None:
        # SCENARIO-2 PIN: pre-fix a breaker-open flush DROPS the confirm (the
        # metadata event is posted and the settlement leaves the queue). Post-fix
        # the settlement is HELD intact and delivered once the breaker recovers.
        breaker = _cp_breaker()
        breaker.record_failure()  # OPEN, recovery window huge -> not eligible
        reporter = _quiet(control_plane_breaker=breaker)
        reporter._settlement_queue.append(
            _PendingSettlement(_PendingConfirm(_make_confirm_request()), _make_event())
        )
        urls: list[str] = []

        def post(url: str, **_kw: object) -> MagicMock:
            urls.append(url)
            return _ok_response()

        with patch.object(reporter._http, "post", side_effect=post):
            reporter._flush_remaining()
            # Breaker open: confirm HELD -> nothing posted, settlement intact.
            assert urls == []
            assert len(reporter._settlement_queue) == 1
            assert reporter.dropped_counts == {}

            # Recover by swapping in a fresh CLOSED breaker.
            reporter._control_plane_breaker = _cp_breaker()
            reporter._flush_remaining()

        assert any("budgets/confirm" in u for u in urls)
        assert any("metadata/ingest" in u for u in urls)
        assert len(reporter._settlement_queue) == 0
        assert len(reporter._queue) == 0
        reporter._http.close()

    def test_confirm_backoff_respects_next_attempt_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = _quiet(retry_backoff_base=1.0, max_send_attempts=5)
        reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
        calls: list[str] = []

        def post(url: str, **_kw: object) -> MagicMock:
            calls.append(url)
            return _error_response(503)

        with patch.object(reporter._http, "post", side_effect=post):
            reporter._flush_remaining()  # attempt 1 -> 503
            assert len(calls) == 1
            assert reporter._confirm_queue[0].next_attempt_at == pytest.approx(1001.0)
            reporter._flush_remaining()  # backing off -> no send
            assert len(calls) == 1
            clock.advance(1.5)
            reporter._flush_remaining()  # due -> send
            assert len(calls) == 2
        reporter._http.close()

    def test_queue_overflow_counts_drops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reporter = _quiet(max_queue_size=3)
        for i in range(5):
            reporter.report(_make_event(input_tokens=i))
        assert reporter.dropped_counts["event.overflow"] == 2
        assert len(reporter._queue) == 3
        # Oldest dropped, newest retained.
        assert reporter._queue[0].event.input_tokens == 2
        assert reporter._queue[-1].event.input_tokens == 4
        reporter._http.close()

        monkeypatch.setattr("solwyn.reporter._MAX_PENDING_CONTROL", 2)
        control = _unstarted()
        for _ in range(4):
            control.report_confirm(_make_confirm_request())
        assert control.dropped_counts["confirm.overflow"] == 2
        assert len(control._confirm_queue) == 2
        for _ in range(4):
            control.report_settlement(_make_confirm_request(), _make_event())
        assert control.dropped_counts["settlement_confirm.overflow"] == 2
        assert len(control._settlement_queue) == 2
        control._http.close()

    def test_batch_5xx_requeues_members_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = _quiet(batch_size=10, max_send_attempts=5)
        for i in range(3):
            reporter.report(_make_event(input_tokens=i))

        with patch.object(reporter._http, "post", return_value=_error_response(500)):
            reporter._flush_remaining()

        assert len(reporter._queue) == 3
        assert [p.event.input_tokens for p in reporter._queue] == [0, 1, 2]
        assert all(p.attempts == 1 for p in reporter._queue)

        clock.advance(2.0)
        with patch.object(reporter._http, "post", return_value=_error_response(422)):
            reporter._flush_remaining()

        assert len(reporter._queue) == 0
        assert reporter.dropped_counts["event.terminal_status"] == 3
        reporter._http.close()

    def test_first_drop_logs_immediately_then_rate_limited(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = _quiet()

        def _drop_logs() -> list[str]:
            return [
                r.getMessage()
                for r in caplog.records
                if "reporter.spend_events_dropped" in r.getMessage()
            ]

        with caplog.at_level("WARNING"):
            reporter._count_drop("event", "terminal_status")
            reporter._count_drop("event", "terminal_status")
            reporter._maybe_log_drops()
            assert len(_drop_logs()) == 1
            assert "new=2" in _drop_logs()[0]

            # Within the interval: another drop + flush logs nothing new.
            reporter._count_drop("confirm", "retry_exhausted")
            reporter._maybe_log_drops()
            assert len(_drop_logs()) == 1

            # After the interval: a second aggregate WARNING carrying totals.
            clock.advance(61.0)
            reporter._maybe_log_drops()
            logs = _drop_logs()
            assert len(logs) == 2
            assert "new=1" in logs[1]
            assert "event.terminal_status" in logs[1]
        reporter._http.close()


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncReporterRetry:
    @pytest.mark.asyncio
    async def test_transient_5xx_settlement_retries_then_delivers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = AsyncMetadataReporter(
            _URL, VALID_API_KEY, max_send_attempts=5, retry_backoff_base=1.0
        )
        reporter._settlement_queue.append(
            _PendingSettlement(_PendingConfirm(_make_confirm_request()), _make_event())
        )
        confirm_posts: list[str] = []
        responses = [_error_response(503), _error_response(503), _ok_response()]

        async def post(url: str, **_kw: object) -> MagicMock:
            if "budgets/confirm" in url:
                confirm_posts.append(url)
                return responses[len(confirm_posts) - 1]
            return _ok_response()

        with patch.object(reporter._http, "post", new=post):
            await reporter._flush_remaining()
            clock.advance(2.0)
            await reporter._flush_remaining()
            clock.advance(4.0)
            await reporter._flush_remaining()

        assert len(confirm_posts) == 3
        assert reporter.dropped_counts == {}
        assert len(reporter._settlement_queue) == 0
        assert len(reporter._queue) == 0
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_retry_exhaustion_drops_and_counts(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr("solwyn.reporter._monotonic", clock)
        reporter = AsyncMetadataReporter(
            _URL, VALID_API_KEY, max_send_attempts=3, retry_backoff_base=1.0
        )
        reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))

        async def post(url: str, **_kw: object) -> MagicMock:
            raise httpx.ConnectError("down")

        with patch.object(reporter._http, "post", new=post), caplog.at_level("WARNING"):
            await reporter._flush_remaining()
            clock.advance(2.0)
            await reporter._flush_remaining()
            clock.advance(4.0)
            await reporter._flush_remaining()

        assert len(reporter._confirm_queue) == 0
        assert reporter.dropped_counts["confirm.retry_exhausted"] == 1
        assert "reporter.spend_events_dropped" in caplog.text
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_settlement_event_survives_confirm_drop(self) -> None:
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
        reporter._settlement_queue.append(
            _PendingSettlement(
                _PendingConfirm(_make_confirm_request(call_id="c1")),
                _make_event(call_id="c1"),
            )
        )
        urls: list[str] = []

        async def post(url: str, **_kw: object) -> MagicMock:
            urls.append(url)
            if "budgets/confirm" in url:
                return _error_response(404)
            return _ok_response()

        with patch.object(reporter._http, "post", new=post):
            await reporter._flush_remaining()

        assert reporter.dropped_counts.get("settlement_confirm.terminal_status") == 1
        assert any("metadata/ingest" in u for u in urls)
        assert len(reporter._settlement_queue) == 0
        assert len(reporter._queue) == 0
        await reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_breaker_open_holds_settlement_until_recovery(self) -> None:
        breaker = _cp_breaker()
        breaker.record_failure()
        reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, control_plane_breaker=breaker)
        reporter._settlement_queue.append(
            _PendingSettlement(_PendingConfirm(_make_confirm_request()), _make_event())
        )
        urls: list[str] = []

        async def post(url: str, **_kw: object) -> MagicMock:
            urls.append(url)
            return _ok_response()

        with patch.object(reporter._http, "post", new=post):
            await reporter._flush_remaining()
            assert urls == []
            assert len(reporter._settlement_queue) == 1
            assert reporter.dropped_counts == {}

            reporter._control_plane_breaker = _cp_breaker()
            await reporter._flush_remaining()

        assert any("budgets/confirm" in u for u in urls)
        assert any("metadata/ingest" in u for u in urls)
        assert len(reporter._settlement_queue) == 0
        await reporter._http.aclose()
