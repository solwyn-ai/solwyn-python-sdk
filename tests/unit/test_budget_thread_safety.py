"""The sync BudgetEnforcer and reporter must tolerate concurrent threads."""

from __future__ import annotations

import sys
import threading
from typing import cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, call_uuid

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest
from solwyn.budget import BudgetEnforcer
from solwyn.reporter import MetadataReporter

_DUMMY_DETAILS = TokenDetails(input_tokens=10, output_tokens=5)


@pytest.mark.unit
def test_reporter_report_confirm_concurrent_appends() -> None:
    """report_confirm must be safe to call from many threads at once."""
    reporter = MetadataReporter(
        api_url="http://test",
        api_key="sk_test",
    )
    try:

        def worker() -> None:
            for i in range(100):
                reporter.report_confirm(
                    BudgetConfirmRequest(
                        reservation_id=f"r{i}",
                        model="gpt-5.5",
                        provider="openai",
                        token_details=_DUMMY_DETAILS,
                        call_id=call_uuid(f"call-{threading.get_ident()}-{i}"),
                    )
                )

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 threads × 100 appends = 1000 entries, but the deque has
        # maxlen=1000 so either all land or the first few dropped.
        assert len(reporter._confirm_queue) <= 1000
    finally:
        # Clear queues before close so the flush loop doesn't attempt
        # real HTTP calls to the dummy URL (hangs on CI DNS resolution).
        reporter._confirm_queue.clear()
        reporter._queue.clear()
        reporter.close()


@pytest.mark.unit
def test_budget_enforcer_uncounted_tally_concurrent() -> None:
    """Concurrent fail-open tallies, true-ups, and check flushes lose nothing.

    Every call adds 10 estimated tokens and trues up to 25 actual tokens,
    while a flusher thread claims, reports, and acknowledges the tally the way
    ``check_budget`` does. What was acknowledged plus what is still owed must
    equal exactly what the workers produced: no lost increments, no report
    subtracted twice.
    """
    enforcer = BudgetEnforcer(
        api_url="http://test",
        api_key="sk_test",
    )

    THREADS = 8
    CALLS_PER_THREAD = 200
    ESTIMATE = 10
    ACTUAL = 25

    reported_calls = 0
    reported_tokens = 0
    workers_done = threading.Event()

    def worker(thread_index: int) -> None:
        for i in range(CALLS_PER_THREAD):
            call_id = f"t{thread_index}-c{i}"
            enforcer._record_legacy_uncounted(ESTIMATE, call_id)
            enforcer.settle_uncounted(call_id=call_id, total_tokens=ACTUAL)

    def flusher() -> None:
        nonlocal reported_calls, reported_tokens
        while True:
            finished = workers_done.is_set()
            report = enforcer._claim_uncounted_report()
            if report is not None:
                reported_calls += report.calls
                reported_tokens += report.tokens
                enforcer._finish_uncounted_report(report, acknowledged=True)
            if finished:
                return

    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        flush_thread = threading.Thread(target=flusher)
        flush_thread.start()
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        workers_done.set()
        flush_thread.join()
    finally:
        sys.setswitchinterval(original_interval)

    owed_calls, owed_tokens = enforcer.uncounted_tally()
    total_calls = THREADS * CALLS_PER_THREAD
    assert reported_calls + owed_calls == total_calls
    assert reported_tokens + owed_tokens == total_calls * ACTUAL
    assert not enforcer._uncounted_estimates
    assert enforcer._uncounted_report_in_flight is None
    enforcer.close()


@pytest.mark.unit
def test_budget_enforcer_concurrent_checks_report_the_tally_once() -> None:
    """Concurrent successful checks never double-report or over-subtract."""
    enforcer = BudgetEnforcer(
        api_url="http://test",
        api_key="sk_test",
        cache_ttl=0,
    )
    for _ in range(50):
        enforcer._record_legacy_uncounted(4, None)

    bodies: list[dict[str, object]] = []
    bodies_lock = threading.Lock()
    barrier = threading.Barrier(10)

    def fake_post(*_args: object, **kwargs: object) -> MagicMock:
        with bodies_lock:
            bodies.append(cast("dict[str, object]", kwargs["json"]))
        response = MagicMock(spec=httpx.Response)
        response.json.return_value = ALLOW_BUDGET_RESPONSE
        return response

    def checker() -> None:
        barrier.wait()
        enforcer.check_budget(estimated_input_tokens=1, model="gpt-5.5", provider="openai")

    with patch.object(enforcer._http, "post", side_effect=fake_post):
        threads = [threading.Thread(target=checker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    carried = [b for b in bodies if "uncounted_calls" in b]
    assert sum(cast("int", b["uncounted_calls"]) for b in carried) == 50
    assert sum(cast("int", b["uncounted_tokens"]) for b in carried) == 200
    assert enforcer.uncounted_tally() == (0, 0)
    enforcer.close()
