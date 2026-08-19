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
from conftest import VALID_API_KEY, call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import (
    AsyncMetadataReporter,
    MetadataReporter,
    _PendingConfirm,
    _PendingEvent,
    _PendingSettlement,
    _SendOutcome,
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
        "call_id": call_uuid("call_shutdown_event"),
    }
    defaults.update(overrides)
    return MetadataEvent(**defaults)


def _make_confirm_request(**overrides) -> BudgetConfirmRequest:
    defaults = {
        "reservation_id": "res_123",
        "model": "gpt-5.5",
        "provider": ProviderName.OPENAI,
        "call_id": call_uuid("call_shutdown_confirm"),
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

    def test_reentrant_longer_close_cannot_extend_outer_earliest_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 100.0}
        monkeypatch.setattr("solwyn.reporter._monotonic", lambda: clock["now"])
        reporter = _unstarted(
            batch_size=1,
            breaker_reporting_enabled=False,
            report_untracked_surfaces=False,
        )
        first = _make_event(call_id=call_uuid("reentrant-deadline-first"))
        second = _make_event(call_id=call_uuid("reentrant-deadline-second"))
        reporter._queue.extend((_PendingEvent(first), _PendingEvent(second)))
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "ingested": 0,
            "rejected": [
                {
                    "index": 0,
                    "code": "structural",
                    "model": "gpt-5.5",
                    "message": "rejected",
                }
            ],
        }
        sent: list[str] = []
        reentered = threading.Event()
        original_thread_start = threading.Thread.start

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]
            assert isinstance(payload, list)
            sent.extend(str(item["call_id"]) for item in payload)
            return response

        def warning(message: str, *_args: object, **_kwargs: object) -> None:
            if message.startswith("reporter.ingest_events_rejected") and not reentered.is_set():
                reentered.set()
                clock["now"] = 100.03
                reporter.close(timeout=0.5)

        def start_thread(thread: threading.Thread) -> None:
            # Exercise the real documented inline fallback so the host warning
            # handler and recursive close own the same RLock thread.
            if thread.name == "solwyn-final-flush":
                raise RuntimeError("interpreter refuses new thread")
            original_thread_start(thread)

        with (
            patch.object(reporter._http, "post", side_effect=post),
            patch("solwyn.reporter.logger.warning", side_effect=warning),
            patch.object(threading.Thread, "start", new=start_thread),
        ):
            reporter.close(timeout=0.02)

        assert reentered.is_set()
        assert sent == [str(first.call_id)]
        # The same-owner recursive close tightens the deadline but cannot seal
        # beneath the active response owner. Its exact rejection therefore
        # keeps that disposition; only the untouched tail hits the deadline.
        assert reporter.dropped_counts == {
            "event.ingest_rejected": 1,
            "event.shutdown_deadline": 1,
        }
        assert reporter._delivery_completed is True
        assert reporter._close_is_completed() is True

    def test_worker_self_close_deadline_stops_later_ordinary_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 200.0}
        monkeypatch.setattr("solwyn.reporter._monotonic", lambda: clock["now"])
        reporter = MetadataReporter(
            _URL,
            VALID_API_KEY,
            batch_size=1,
            flush_interval=0.01,
            breaker_reporting_enabled=False,
            report_untracked_surfaces=False,
        )
        first = _make_event(call_id=call_uuid("worker-deadline-first"))
        second = _make_event(call_id=call_uuid("worker-deadline-second"))
        rejected = MagicMock(spec=httpx.Response)
        rejected.raise_for_status.return_value = None
        rejected.json.return_value = {
            "ingested": 0,
            "rejected": [
                {
                    "index": 0,
                    "code": "structural",
                    "model": "gpt-5.5",
                    "message": "rejected",
                }
            ],
        }
        accepted = MagicMock(spec=httpx.Response)
        accepted.raise_for_status.return_value = None
        accepted.json.return_value = {"ingested": 1, "rejected": []}
        sent: list[str] = []
        close_requested = threading.Event()

        def post(_url: str, **kwargs: object) -> MagicMock:
            payload = kwargs["json"]
            assert isinstance(payload, list)
            sent.extend(str(item["call_id"]) for item in payload)
            return rejected if len(sent) == 1 else accepted

        def warning(message: str, *_args: object, **_kwargs: object) -> None:
            if (
                message.startswith("reporter.ingest_events_rejected")
                and not close_requested.is_set()
            ):
                reporter.close(timeout=0.02)
                close_requested.set()
                clock["now"] = 200.03

        try:
            with (
                patch.object(reporter._http, "post", side_effect=post),
                patch("solwyn.reporter.logger.warning", side_effect=warning),
            ):
                reporter.report(first)
                reporter.report(second)
                assert close_requested.wait(timeout=2.0)
                reporter._thread.join(timeout=2.0)

            assert not reporter._thread.is_alive()
            assert sent == [str(first.call_id)]
            assert reporter.dropped_counts == {
                "event.ingest_rejected": 1,
                "event.shutdown_deadline": 1,
            }
            assert reporter._delivery_completed is True
            assert reporter._close_is_completed() is True
        finally:
            reporter._shutdown.set()
            reporter._thread.join(timeout=2.0)
            reporter._http.close()

    def test_ordinary_event_send_uses_newly_published_close_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 300.0}
        monkeypatch.setattr("solwyn.reporter._monotonic", lambda: clock["now"])
        reporter = _unstarted(
            breaker_reporting_enabled=False,
            report_untracked_surfaces=False,
        )
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.return_value = None
        response.json.return_value = {"ingested": 1, "rejected": []}
        timeouts: list[float] = []

        def post(_url: str, **kwargs: object) -> MagicMock:
            timeouts.append(float(kwargs["timeout"]))
            return response

        reporter._tighten_close_deadline(300.02)
        with patch.object(reporter._http, "post", side_effect=post):
            reporter._send_batch([_make_event()], deadline=None)

        assert timeouts == [0.05]
        reporter._shutdown.set()
        reporter._http.close()

    def test_competing_transport_close_wait_respects_earliest_deadline(self) -> None:
        reporter = _unstarted(
            breaker_reporting_enabled=False,
            report_untracked_surfaces=False,
        )
        transport_entered = threading.Event()
        release_transport = threading.Event()
        first_finished = threading.Event()
        second_finished = threading.Event()
        close_errors: list[BaseException] = []
        real_close = reporter._http.close

        def blocked_close() -> None:
            transport_entered.set()
            assert release_transport.wait(timeout=2.0)

        def close_first() -> None:
            try:
                reporter.close(timeout=1.0)
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                first_finished.set()

        def close_second() -> None:
            try:
                reporter.close(timeout=0.0)
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                second_finished.set()

        first = threading.Thread(target=close_first, daemon=True)
        second = threading.Thread(target=close_second, daemon=True)
        try:
            with patch.object(reporter._http, "close", side_effect=blocked_close):
                first.start()
                assert transport_entered.wait(timeout=1.0)
                second.start()
                assert second_finished.wait(timeout=0.2), (
                    "a zero-deadline close must not wait behind another transport owner"
                )
                assert not first_finished.is_set()
                release_transport.set()
                first.join(timeout=2.0)
                second.join(timeout=2.0)

            assert close_errors == []
            assert first_finished.is_set()
            assert reporter._close_is_completed() is True
        finally:
            release_transport.set()
            first.join(timeout=2.0)
            second.join(timeout=2.0)
            real_close()

    def test_expired_competing_close_seals_spend_without_stealing_whole_close(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"now": 500.0}
        monkeypatch.setattr("solwyn.reporter._monotonic", lambda: clock["now"])
        reporter = _unstarted(
            breaker_reporting_enabled=False,
            report_untracked_surfaces=False,
        )
        reporter._queue.append(_PendingEvent(_make_event()))
        owner_claimed = threading.Event()
        release_owner = threading.Event()
        loser_tightened_deadline = threading.Event()
        owner_finished = threading.Event()
        loser_finished = threading.Event()
        close_errors: list[BaseException] = []
        real_close = reporter._http.close
        original_finish_owned = reporter._finish_close_owned
        original_tighten = reporter._tighten_close_deadline
        owner: threading.Thread | None = None
        loser: threading.Thread | None = None

        def finish_owned(deadline: float) -> None:
            if threading.current_thread() is owner:
                owner_claimed.set()
                assert release_owner.wait(timeout=2.0)
            original_finish_owned(deadline)

        def tighten(deadline: float) -> float:
            tightened = original_tighten(deadline)
            if threading.current_thread() is loser:
                loser_tightened_deadline.set()
            return tightened

        def close_reporter(timeout: float, finished: threading.Event) -> None:
            try:
                reporter.close(timeout=timeout)
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                finished.set()

        owner = threading.Thread(
            target=close_reporter,
            args=(1.0, owner_finished),
            daemon=True,
        )
        loser = threading.Thread(
            target=close_reporter,
            args=(0.03, loser_finished),
            daemon=True,
        )
        try:
            with (
                patch.object(reporter, "_finish_close_owned", side_effect=finish_owned),
                patch.object(reporter, "_tighten_close_deadline", side_effect=tighten),
                patch.object(reporter._http, "close") as http_close,
            ):
                owner.start()
                assert owner_claimed.wait(timeout=1.0)
                loser.start()
                assert loser_tightened_deadline.wait(timeout=1.0)
                clock["now"] = 500.04
                with reporter._close_state_changed:
                    reporter._close_state_changed.notify_all()
                assert loser_finished.wait(timeout=1.0)

                assert reporter._delivery_completed is True
                assert not reporter._queue
                assert reporter.dropped_counts == {"event.shutdown_deadline": 1}
                assert reporter._close_is_completed() is False
                http_close.assert_not_called()

                release_owner.set()
                assert owner_finished.wait(timeout=2.0)
                owner.join(timeout=2.0)
                loser.join(timeout=2.0)

                assert close_errors == []
                assert reporter.dropped_counts == {"event.shutdown_deadline": 1}
                assert reporter._close_is_completed() is True
                http_close.assert_called_once_with()
        finally:
            release_owner.set()
            owner.join(timeout=2.0)
            loser.join(timeout=2.0)
            real_close()


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


@pytest.mark.unit
def test_pop_in_hand_is_atomic_with_seal() -> None:
    """#5 review pin: a drain's pop and its ownership claim are ONE step under
    the ownership lock — the seal can never observe a popped item in neither
    the queue nor _in_hand, and a sealed reporter refuses further pops."""
    reporter = _unstarted()
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))

    # A drain claims the head: queue and _in_hand always account for both.
    pending = reporter._pop_in_hand(reporter._confirm_queue, "confirm")
    assert pending is not None
    assert reporter._in_hand["confirm"] == 1

    # The seal counts the in-hand item AND the still-queued one.
    reporter._seal_delivery()
    assert reporter.dropped_counts["confirm.shutdown_deadline"] == 2

    # Sealed: pops are refused outright (the seal owns whatever re-appears).
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
    assert reporter._pop_in_hand(reporter._confirm_queue, "confirm") is None
    assert reporter._pop_batch_in_hand(now=time.monotonic(), final=True) is None
    reporter._http.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome,reason,queue_size,deadline_expired",
    [
        (_SendOutcome.DROPPED, "terminal_status", 1, False),
        (_SendOutcome.RETRY, "retry_exhausted", 1, False),
        (_SendOutcome.HELD, "exit_breaker_open", 2, False),
        (_SendOutcome.SENT, "shutdown_deadline", 2, True),
    ],
)
def test_close_publishes_confirm_disposition_before_releasing_ownership(
    outcome: _SendOutcome,
    reason: str,
    queue_size: int,
    deadline_expired: bool,
) -> None:
    """A bounded close must never return while a daemon owns an untracked loss."""
    reporter = _unstarted(max_send_attempts=1)
    for index in range(queue_size):
        reporter._confirm_queue.append(
            _PendingConfirm(_make_confirm_request(call_id=call_uuid(f"atomic-confirm-{index}")))
        )

    disposition_entered = threading.Event()
    disposition_finished = threading.Event()
    release_unsafe_disposition = threading.Event()
    original_record = reporter._record_drop

    def gated_record(kind: str, drop_reason: str, n: int = 1) -> None:
        if (
            threading.current_thread().name == "solwyn-final-flush"
            and kind == "confirm"
            and drop_reason == reason
        ):
            # The vulnerable path releases/drains ownership before mutation.
            # Pause only when the ownership lock is therefore available; the
            # fixed transaction records while holding it and never enters the
            # ownerless interval.
            owns_no_lock = reporter._ownership_lock.acquire(blocking=False)
            if owns_no_lock:
                reporter._ownership_lock.release()
            disposition_entered.set()
            if owns_no_lock:
                assert release_unsafe_disposition.wait(timeout=5.0)
        original_record(kind, drop_reason, n)
        disposition_finished.set()

    close_finished = threading.Event()

    def close() -> None:
        reporter.close(timeout=0.1)
        close_finished.set()

    closer = threading.Thread(target=close, daemon=True)
    try:
        with (
            patch.object(reporter, "_record_drop", side_effect=gated_record),
            patch.object(reporter, "_send_confirm", return_value=outcome),
            patch.object(
                reporter,
                "_deadline_expired",
                side_effect=lambda _deadline: (
                    deadline_expired and threading.current_thread().name == "solwyn-final-flush"
                ),
            ),
            patch.object(reporter, "_start_breaker_cycle", return_value=None),
        ):
            closer.start()
            assert disposition_entered.wait(timeout=2.0)
            closer.join(timeout=2.0)
            assert close_finished.is_set()

            assert not reporter._confirm_queue
            assert reporter._in_hand.get("confirm", 0) == 0
            assert reporter.dropped_counts == {f"confirm.{reason}": queue_size}
            before_late_worker = reporter.dropped_counts

            release_unsafe_disposition.set()
            assert disposition_finished.wait(timeout=2.0)
            assert reporter.dropped_counts == before_late_worker
    finally:
        release_unsafe_disposition.set()
        closer.join(timeout=2.0)
        reporter._shutdown.set()
        reporter._http.close()


@pytest.mark.unit
def test_sent_confirm_releases_ownership_without_a_drop() -> None:
    reporter = _unstarted()
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
    try:
        with patch.object(reporter, "_send_confirm", return_value=_SendOutcome.SENT):
            reporter._drain_confirms(final=True)
        assert not reporter._confirm_queue
        assert reporter._in_hand.get("confirm", 0) == 0
        assert reporter.dropped_counts == {}
    finally:
        reporter._shutdown.set()
        reporter._http.close()


@pytest.mark.unit
def test_close_bounded_despite_slow_drip_send() -> None:
    """#4 review pin: httpx timeouts bound socket operations, not total response
    time — a slow-drip send must not hold close() past the wall-clock deadline.
    The blocked post here ignores its timeout kwarg entirely, standing in for a
    server that keeps making incremental progress."""
    reporter = _unstarted()
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
    entered = threading.Event()
    release = threading.Event()

    def dripping_post(url: str, **_kw: object) -> httpx.Response:
        entered.set()
        release.wait(timeout=10.0)
        raise httpx.ConnectError("released")

    try:
        with patch.object(reporter._http, "post", side_effect=dripping_post):
            start = time.monotonic()
            reporter.close(timeout=0.3)
            elapsed = time.monotonic() - start

        assert entered.is_set(), "the final flush never attempted the confirm"
        assert elapsed < 2.0, f"close took {elapsed:.2f}s — a dripping send must not hold it"
        # The in-hand confirm got its disposition before close returned.
        assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    finally:
        release.set()
        for worker in threading.enumerate():
            if worker.name == "solwyn-final-flush":
                worker.join(timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_close_bounded_despite_slow_drip_send() -> None:
    """Async twin of the slow-drip pin: the final flush rides its own task so
    the deadline can cancel a send whose per-operation timeouts never fire."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=3600.0)
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))

    async def dripping_post(url: str, **_kw: object) -> httpx.Response:
        await asyncio.sleep(30.0)
        raise httpx.ConnectError("unreachable")

    with patch.object(reporter._http, "post", new=dripping_post):
        start = time.monotonic()
        await reporter.close(timeout=0.3)
        elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"async close took {elapsed:.2f}s — a dripping send must not hold it"
    assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    assert reporter._close_completed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_zero_deadline_close_counts_in_hand_before_returning() -> None:
    """#7 review pin: a zero-deadline close cancels the flush task — but must
    still AWAIT its cleanup so the in-hand item is counted (and the exit rescue
    is detached only after every disposition is assigned)."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=0.05)
    entered = asyncio.Event()
    blocker = asyncio.Event()

    async def blocked_post(url: str, **_kw: object) -> httpx.Response:
        entered.set()
        await blocker.wait()
        raise httpx.ConnectError("released")

    with patch.object(reporter._http, "post", new=blocked_post):
        reporter.report_confirm(_make_confirm_request())
        await asyncio.wait_for(entered.wait(), timeout=2.0)

        await reporter.close(timeout=0.0)

    # The cancelled drain's accounting ran BEFORE close returned.
    assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    assert reporter._close_completed is True
    assert reporter._finalizer is None or not reporter._finalizer.alive


@pytest.mark.unit
def test_async_seal_counts_straggler_and_refuses_enqueue() -> None:
    """#10 review pin: the async reporter shares the ownership gate — a
    straggler appended after the final drain is claimed by the seal, and a
    producer that lost the race with close() is refused-and-counted."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    reporter._queue.append(_PendingEvent(_make_event()))

    reporter._seal_delivery()
    assert reporter.dropped_counts["event.shutdown_deadline"] == 1
    assert len(reporter._queue) == 0

    # A producer that passed the _closed check before the seal: refused, counted.
    reporter.report(_make_event())
    reporter.report_confirm(_make_confirm_request())
    reporter.report_settlement(_make_confirm_request(), _make_event())
    assert len(reporter._queue) == 0
    assert len(reporter._confirm_queue) == 0
    assert len(reporter._settlement_queue) == 0
    assert reporter.dropped_counts["event.closed_enqueue"] == 2  # event + pair half
    assert reporter.dropped_counts["confirm.closed_enqueue"] == 1
    assert reporter.dropped_counts["settlement_confirm.closed_enqueue"] == 1
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True  # queues empty; disarm the exit hook


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_drain_after_seal_counts_in_hand() -> None:
    """Re-review #4 pin, seal branch: a cancelled drain requeues its in-hand
    item for the lifecycle rescue — but never into a SEALED queue (nothing will
    ever drain it again). A seal-refused requeue is counted instead."""
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY, flush_interval=3600.0)
    reporter._confirm_queue.append(_PendingConfirm(_make_confirm_request()))
    entered = asyncio.Event()

    async def hung_post(url: str, **_kw: object) -> httpx.Response:
        entered.set()
        await asyncio.sleep(3600)
        raise httpx.ConnectError("unreachable")

    with patch.object(reporter._http, "post", new=hung_post):
        drain = asyncio.create_task(reporter._drain_confirms(final=True))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        reporter._seal_delivery()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain

    assert reporter.dropped_counts.get("confirm.shutdown_deadline") == 1
    assert not reporter._confirm_queue
    if reporter._finalizer is not None:
        reporter._finalizer.detach()
    reporter._close_completed = True  # queues empty; disarm the exit hook
    await reporter._http.aclose()
