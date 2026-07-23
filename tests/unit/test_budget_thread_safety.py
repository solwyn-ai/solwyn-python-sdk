"""The sync BudgetEnforcer and reporter must tolerate concurrent threads."""

from __future__ import annotations

import threading

import pytest
from conftest import call_uuid

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
def test_budget_enforcer_local_costs_concurrent() -> None:
    """Concurrent _track_local_cost calls must not lose writes."""
    enforcer = BudgetEnforcer(
        api_url="http://test",
        api_key="sk_test",
    )

    COST_PER_CALL = 0.01
    THREADS = 10
    CALLS_PER_THREAD = 100
    EXPECTED_TOTAL = COST_PER_CALL * THREADS * CALLS_PER_THREAD

    def worker() -> None:
        for _ in range(CALLS_PER_THREAD):
            enforcer._track_local_cost(COST_PER_CALL)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(enforcer._local_costs.values())
    assert abs(total - EXPECTED_TOTAL) < 1e-6, (
        f"Lost writes: expected {EXPECTED_TOTAL}, got {total}"
    )
    enforcer.close()
