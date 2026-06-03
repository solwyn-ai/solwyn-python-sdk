"""Tests for circuit breaker state machine."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel

from solwyn._types import CircuitState
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState


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
        assert cb.can_proceed() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_stays_open_before_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_proceed() is False
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

        cb.can_proceed()  # -> HALF_OPEN (recovery_timeout=0)
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
        cb.can_proceed()  # -> HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()  # -> back to OPEN
        assert cb.state == CircuitState.OPEN


@pytest.mark.unit
class TestConcurrentStateMutation:
    """Concurrent sync callers must not lose breaker counter updates."""

    def test_concurrent_half_open_successes_count_exactly(self) -> None:
        workers = 32
        cb = CircuitBreaker(success_threshold=workers + 1)
        cb.state = CircuitState.HALF_OPEN
        barrier = threading.Barrier(workers)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                cb.record_success()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(workers)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == workers

    def test_concurrent_closed_failures_count_exactly(self) -> None:
        workers = 32
        cb = CircuitBreaker(failure_threshold=workers + 1)
        barrier = threading.Barrier(workers)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                cb.record_failure()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(workers)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == workers


@pytest.mark.unit
class TestCanProceed:
    """can_proceed() returns correct value per state."""

    def test_returns_true_when_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.can_proceed() is True

    def test_returns_false_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        assert cb.can_proceed() is False

    def test_returns_true_when_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.can_proceed()  # triggers HALF_OPEN transition
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_proceed() is True


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

        # can_proceed() is the ONLY consumer permitted to transition.
        assert cb.can_proceed() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_false_when_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        cb.can_proceed()  # -> HALF_OPEN
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
