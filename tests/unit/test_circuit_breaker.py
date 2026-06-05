"""Tests for circuit breaker state machine."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from solwyn._types import CircuitState
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState


class _SlowTransitionCircuitBreaker(CircuitBreaker):
    """Expose duplicate concurrent transitions without relying on lost increments."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.open_transitions: list[None] = []
        self.closed_transitions: list[None] = []
        self.half_open_transitions: list[None] = []

    def _transition_to_open(self) -> None:
        self.open_transitions.append(None)
        time.sleep(0.02)
        super()._transition_to_open()

    def _transition_to_closed(self) -> None:
        self.closed_transitions.append(None)
        time.sleep(0.02)
        super()._transition_to_closed()

    def _transition_to_half_open(self) -> None:
        self.half_open_transitions.append(None)
        time.sleep(0.02)
        super()._transition_to_half_open()


def _run_concurrently(workers: int, target: Callable[[], object]) -> None:
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=2.0)
            target()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)


def _run_concurrently_collecting(workers: int, target: Callable[[], object]) -> list[object]:
    """Run ``target`` on ``workers`` threads and return each call's result.

    Like ``_run_concurrently`` but captures return values under a lock so
    callers can assert on the distribution of outcomes (e.g. exactly one
    ``admit()`` call wins the single HALF_OPEN slot).
    """
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []
    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait(timeout=2.0)
            result = target()
            with results_lock:
                results.append(result)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    return results


@pytest.mark.unit
class TestClosedToOpen:
    """CLOSED -> OPEN after N consecutive failures."""

    def test_opens_after_failure_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # 2 < 3

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets streak
        assert cb.failure_count == 0

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # only 2 since reset


@pytest.mark.unit
class TestOpenToHalfOpen:
    """OPEN -> HALF_OPEN after recovery_timeout elapses."""

    def test_transitions_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing beyond recovery_timeout
        cb.last_failure_time = time.monotonic() - 15
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        admission = cb.admit()
        assert admission.allowed is False
        assert admission.owns_probe is False
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestHalfOpenToClosed:
    """HALF_OPEN -> CLOSED after N successes."""

    def test_closes_after_success_threshold(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=2,
        )
        cb.record_failure()  # -> OPEN
        assert cb.state == CircuitState.OPEN

        admission = cb.admit()  # -> HALF_OPEN (recovery_timeout=0)
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # 1 < 2

        cb.record_success()
        assert cb.state == CircuitState.CLOSED


@pytest.mark.unit
class TestHalfOpenFailure:
    """HALF_OPEN -> OPEN on any failure."""

    def test_reopens_on_failure(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=3,
        )
        cb.record_failure()  # -> OPEN
        admission = cb.admit()  # -> HALF_OPEN
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()  # -> back to OPEN
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestConcurrentStateMutation:
    """Concurrent sync callers must not duplicate state transitions."""

    def test_concurrent_half_open_successes_close_once(self) -> None:
        cb = _SlowTransitionCircuitBreaker(success_threshold=1)
        cb.state = CircuitState.HALF_OPEN

        _run_concurrently(32, cb.record_success)

        assert cb.state == CircuitState.CLOSED
        assert len(cb.closed_transitions) == 1
        assert cb.success_count == 0

    def test_concurrent_closed_failures_open_once(self) -> None:
        cb = _SlowTransitionCircuitBreaker(failure_threshold=1)

        _run_concurrently(32, cb.record_failure)

        assert cb.state == CircuitState.OPEN
        assert len(cb.open_transitions) == 1
        assert cb.failure_count == 1

    def test_concurrent_recovery_probe_transitions_half_open_once(self) -> None:
        cb = _SlowTransitionCircuitBreaker(recovery_timeout=0)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.monotonic() - 1

        admissions = _run_concurrently_collecting(32, cb.admit)

        assert cb.state == CircuitState.HALF_OPEN
        assert len(cb.half_open_transitions) == 1
        # Single-probe slot: exactly ONE concurrent caller wins the
        # probe; the other 31 are refused — no recovery stampede.
        assert sum(1 for admission in admissions if admission.allowed) == 1
        assert sum(1 for admission in admissions if not admission.allowed) == 31
        assert sum(1 for admission in admissions if admission.owns_probe) == 1


@pytest.mark.unit
class TestAdmit:
    """admit() returns admission and probe-ownership details per state."""

    def test_allows_without_probe_when_closed(self) -> None:
        cb = CircuitBreaker()
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is False

    def test_refuses_without_probe_when_open_before_recovery(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        admission = cb.admit()
        assert admission.allowed is False
        assert admission.owns_probe is False

    def test_half_open_single_probe_slot(self) -> None:
        # HALF_OPEN opens exactly ONE probe slot. The first caller
        # consumes it; a concurrent second caller is refused while the probe is
        # in flight; once the probe reports a non-closing success the slot is
        # freed and a fresh caller may probe again.
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, success_threshold=2)
        cb.record_failure()

        # First caller wins the probe slot and transitions OPEN -> HALF_OPEN.
        first = cb.admit()
        assert first.allowed is True
        assert first.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

        # A second caller while the probe is still in flight is refused.
        second = cb.admit()
        assert second.allowed is False
        assert second.owns_probe is False
        assert cb.state == CircuitState.HALF_OPEN

        # The in-flight probe reports a success that does NOT close the breaker
        # (success_threshold=2), which frees the slot.
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        # A fresh caller may now probe again — the slot was freed on outcome.
        third = cb.admit()
        assert third.allowed is True
        assert third.owns_probe is True


@pytest.mark.unit
class TestReleaseProbe:
    """release_probe() frees a HALF_OPEN probe slot without a health verdict."""

    def test_release_probe_frees_slot_without_state_change(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0, success_threshold=2)
        cb.record_failure()  # -> OPEN
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN
        blocked = cb.admit()
        assert blocked.allowed is False
        assert blocked.owns_probe is False

        cb.release_probe(admission)

        # Neutral outcome: no verdict recorded, breaker stays HALF_OPEN...
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 0
        assert cb.failure_count == 0
        # ...but the slot is freed, so the next caller may probe again.
        fresh = cb.admit()
        assert fresh.allowed is True
        assert fresh.owns_probe is True

    def test_release_probe_is_noop_when_no_probe_active(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        cb.release_probe()  # must not raise or change state
        assert cb.state == CircuitState.CLOSED
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is False

    def test_stale_closed_admission_cannot_release_active_probe(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        closed_admission = cb.admit()
        assert closed_admission.allowed is True
        assert closed_admission.owns_probe is False

        cb.record_failure()  # -> OPEN, recovery-eligible
        half_open_admission = cb.admit()
        assert half_open_admission.allowed is True
        assert half_open_admission.owns_probe is True
        blocked = cb.admit()
        assert blocked.allowed is False
        assert blocked.owns_probe is False

        cb.release_probe(closed_admission)

        still_blocked = cb.admit()
        assert still_blocked.allowed is False
        assert still_blocked.owns_probe is False

    def test_stale_probe_admission_cannot_release_new_active_probe(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()  # -> OPEN, recovery-eligible
        stale_admission = cb.admit()
        assert stale_admission.owns_probe is True
        cb.release_probe(stale_admission)

        active_admission = cb.admit()
        assert active_admission.owns_probe is True
        blocked = cb.admit()
        assert blocked.allowed is False
        assert blocked.owns_probe is False

        cb.release_probe(stale_admission)

        still_blocked = cb.admit()
        assert still_blocked.allowed is False
        assert still_blocked.owns_probe is False


@pytest.mark.unit
class TestGetState:
    """get_state() returns a well-formed dict."""

    def test_returns_correct_dataclass(self) -> None:
        cb = CircuitBreaker()
        state = cb.get_state()

        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.last_failure_time is None
        assert isinstance(state.last_state_change, float)

    def test_reflects_mutations(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()

        state = cb.get_state()
        assert state.state == CircuitState.OPEN
        assert state.failure_count == 1
        assert state.last_failure_time is not None


def test_circuit_breaker_state_is_pydantic_model() -> None:
    """CircuitBreakerState must be a Pydantic BaseModel, not a dataclass."""
    assert issubclass(CircuitBreakerState, BaseModel)


@pytest.mark.unit
def test_can_proceed_boolean_api_is_removed() -> None:
    """Admission must expose probe ownership instead of hiding it behind bool."""
    assert not hasattr(CircuitBreaker, "can_proceed")


@pytest.mark.unit
class TestRecoveryEligible:
    """recovery_eligible is a non-mutating read; the router orders on it."""

    def test_false_when_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.recovery_eligible is False
        assert cb.state == CircuitState.CLOSED

    def test_false_when_open_before_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        assert cb.recovery_eligible is False
        assert cb.state == CircuitState.OPEN  # unchanged

    def test_true_when_open_after_timeout_without_mutating(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing beyond recovery_timeout.
        cb.last_failure_time = time.monotonic() - 15

        # Inspection must report eligibility WITHOUT flipping to HALF_OPEN.
        assert cb.recovery_eligible is True
        assert cb.state == CircuitState.OPEN  # still OPEN — pure read

        # Re-reading is idempotent (still non-mutating).
        assert cb.recovery_eligible is True
        assert cb.state == CircuitState.OPEN

        # admit() is the ONLY consumer permitted to transition.
        admission = cb.admit()
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_false_when_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        admission = cb.admit()  # -> HALF_OPEN
        assert admission.allowed is True
        assert admission.owns_probe is True
        assert cb.state == CircuitState.HALF_OPEN

        assert cb.recovery_eligible is False
        assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.unit
class TestRecoveryTimeoutJitter:
    """recovery_timeout_jitter widens/narrows the effective recovery window."""

    def test_default_jitter_is_deterministic(self) -> None:
        # jitter=0.0 (default) keeps behavior identical to the un-jittered path.
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        cb.record_failure()

        cb.last_failure_time = time.monotonic() - 9.9
        assert cb.recovery_eligible is False  # below window
        cb.last_failure_time = time.monotonic() - 10.1
        assert cb.recovery_eligible is True  # above window

    def test_effective_window_within_jitter_bounds(self) -> None:
        # With ±20% jitter the effective window stays inside [8, 12] for a
        # 10s base across many opens — never outside the bound.
        for _ in range(200):
            cb = CircuitBreaker(
                failure_threshold=1,
                recovery_timeout=10,
                recovery_timeout_jitter=0.2,
            )
            cb.record_failure()  # opens -> samples an effective window

            # Just under the minimum possible window -> never eligible.
            cb.last_failure_time = time.monotonic() - 7.9
            assert cb.recovery_eligible is False

            # Just over the maximum possible window -> always eligible.
            cb.last_failure_time = time.monotonic() - 12.1
            assert cb.recovery_eligible is True
