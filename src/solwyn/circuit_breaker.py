"""Local-only circuit breaker state machine.

Tracks provider health per SDK instance. State (CLOSED, OPEN, HALF_OPEN)
is reported to the cloud API for dashboard visibility but never shared
across instances.

Extracted from solwyn-core ``llm_client.py`` with Redis code removed --
all state is process-local and all methods are synchronous.
"""

from __future__ import annotations

import logging
import random
import time

from pydantic import BaseModel, ConfigDict

from solwyn._types import CircuitState

logger = logging.getLogger(__name__)


class CircuitBreakerState(BaseModel):
    """Snapshot of a circuit breaker's current state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None  # monotonic seconds
    last_state_change: float  # monotonic seconds


class CircuitBreaker:
    """Circuit breaker for handling LLM provider failures.

    State transitions:

        CLOSED  --[failure_threshold failures]--> OPEN
        OPEN    --[recovery_timeout elapsed]----> HALF_OPEN
        HALF_OPEN --[success_threshold successes]--> CLOSED
        HALF_OPEN --[any failure]-------------------> OPEN

    All state is process-local.  No Redis, no async.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        recovery_timeout_jitter: float = 0.0,
    ) -> None:
        """Initialise circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening.
            recovery_timeout: Seconds to wait before probing recovery.
            success_threshold: Successes in HALF_OPEN needed to close.
            recovery_timeout_jitter: Anti-stampede jitter fraction (§6.4). When
                ``> 0``, each time the breaker opens it samples an effective
                recovery window of ``recovery_timeout * (1 +
                uniform(-jitter, +jitter))``. ``0.0`` (default) keeps the window
                exactly ``recovery_timeout`` — deterministic.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.recovery_timeout_jitter = recovery_timeout_jitter

        # Authoritative in-process state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_state_change = time.monotonic()
        # Effective recovery window for the current OPEN episode. Re-sampled on
        # each transition to OPEN; equals recovery_timeout when jitter is 0.
        self._effective_recovery_timeout: float = float(recovery_timeout)

    # ------------------------------------------------------------------
    # Public interface (synchronous)
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call from the provider."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure streak on any success while closed
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call from the provider."""
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure during probing immediately re-opens
            self._transition_to_open()

    def can_proceed(self) -> bool:
        """Return ``True`` if the circuit allows a request through."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._transition_to_half_open()
                return True
            return False
        else:  # HALF_OPEN
            return True

    @property
    def recovery_eligible(self) -> bool:
        """Report whether an OPEN breaker is ready to probe — WITHOUT mutating.

        The router orders candidates on this read; it must never transition
        state (§4.2/§6.4: separate INSPECTION from CONSUMPTION). Only
        ``can_proceed()`` may flip an eligible OPEN breaker to HALF_OPEN.

        Returns ``True`` only when the breaker is OPEN and the (possibly
        jittered) recovery window has elapsed; ``False`` in every other state.
        """
        return self.state == CircuitState.OPEN and self._should_attempt_recovery()

    def get_state(self) -> CircuitBreakerState:
        """Return a frozen snapshot of the circuit breaker's internal state."""
        return CircuitBreakerState(
            state=self.state,
            failure_count=self.failure_count,
            success_count=self.success_count,
            last_failure_time=self.last_failure_time,
            last_state_change=self.last_state_change,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.monotonic() - self.last_failure_time) >= self._effective_recovery_timeout

    def _sample_recovery_window(self) -> float:
        """Sample the effective recovery window for a fresh OPEN episode."""
        if self.recovery_timeout_jitter <= 0.0:
            return float(self.recovery_timeout)
        factor = 1.0 + random.uniform(-self.recovery_timeout_jitter, self.recovery_timeout_jitter)
        return self.recovery_timeout * factor

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_state_change = time.monotonic()
        self.success_count = 0
        self._effective_recovery_timeout = self._sample_recovery_window()
        logger.warning("Circuit breaker opened due to failures")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.monotonic()
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker closed, provider recovered")

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.monotonic()
        self.success_count = 0
        self.failure_count = 0
        logger.info("Circuit breaker half-open, testing recovery")
