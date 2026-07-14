"""Process-local circuit breaker state machine with advisory cloud snapshots.

Tracks provider health per SDK instance. State (CLOSED, OPEN, HALF_OPEN)
remains authoritative only in-process; the reporter reads frozen ``get_state()``
snapshots on its periodic flush loop and sends the current state to the cloud API
for dashboard visibility. Cloud snapshots never drive admission or cross-instance
state sharing.

Extracted from solwyn-core ``llm_client.py`` with Redis code removed --
all state is process-local and all methods are synchronous.
"""

from __future__ import annotations

import logging
import random
import threading
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
    recovery_eligible: bool


class CircuitBreakerAdmission(BaseModel):
    """Result of asking a circuit breaker to admit one provider attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    probe_token: int | None = None

    @property
    def owns_probe(self) -> bool:
        """Whether this admission consumed a HALF_OPEN probe slot."""
        return self.probe_token is not None


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
            recovery_timeout_jitter: Anti-stampede jitter fraction. When
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
        # Single-probe slot: while HALF_OPEN, exactly one in-flight
        # probe is permitted. Set True by the caller that opens the slot;
        # freed (False) when that probe reports an outcome so the next call
        # may probe again (matters when success_threshold > 1).
        self._half_open_probe_active: bool = False
        self._half_open_probe_token: int | None = None
        self._next_probe_token = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface (synchronous)
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call from the provider."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # Free the probe slot before the close check so that when
                # success_threshold > 1 the next call can probe again.
                self._half_open_probe_active = False
                self._half_open_probe_token = None
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure streak on any success while closed
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call from the provider."""
        with self._lock:
            self.last_failure_time = time.monotonic()

            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure during probing immediately re-opens. Free the
                # probe slot first (defense-in-depth; _transition_to_open also
                # resets it for a fresh OPEN episode).
                self._half_open_probe_active = False
                self._half_open_probe_token = None
                self._transition_to_open()

    def admit(self) -> CircuitBreakerAdmission:
        """Return an admission result for one provider attempt.

        ``allowed`` indicates whether the attempt may run. When the attempt
        consumes a HALF_OPEN probe slot, ``probe_token`` identifies the slot so
        a later neutral release can prove ownership.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return CircuitBreakerAdmission(allowed=True)
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                    return self._consume_probe_slot_locked()
                return CircuitBreakerAdmission(allowed=False)
            else:  # HALF_OPEN
                # Single-probe slot: if a probe is already in flight, refuse so
                # concurrent callers cannot stampede a just-recovering provider.
                if self._half_open_probe_active:
                    return CircuitBreakerAdmission(allowed=False)
                return self._consume_probe_slot_locked()

    def release_probe(self, admission: CircuitBreakerAdmission | None = None) -> None:
        """Release a consumed HALF_OPEN probe slot WITHOUT a health verdict.

        ``record_success`` / ``record_failure`` are health signals that both
        free the slot AND move state. This is the neutral third outcome: a probe
        that aborted before producing any verdict about the provider — a
        cross-provider translation error or a FAIL_FAST (request-shaped)
        response raised after ``admit()`` consumed the slot. It frees the
        slot only; state, success_count, and failure_count are untouched, so the
        breaker stays HALF_OPEN and the next caller may probe instead of being
        stranded behind a permanently-occupied slot. No-op if ``admission`` did
        not acquire the currently active probe; safe to call in any state.
        """
        if admission is None or not admission.owns_probe:
            return
        with self._lock:
            if (
                self.state == CircuitState.HALF_OPEN
                and self._half_open_probe_active
                and self._half_open_probe_token == admission.probe_token
            ):
                self._half_open_probe_active = False
                self._half_open_probe_token = None

    @property
    def recovery_eligible(self) -> bool:
        """Report whether an OPEN breaker is ready to probe — WITHOUT mutating.

        The router orders candidates on this read; it must never transition
        state (separate INSPECTION from CONSUMPTION). Only
        ``admit()`` may flip an eligible OPEN breaker to HALF_OPEN.

        Returns ``True`` only when the breaker is OPEN and the (possibly
        jittered) recovery window has elapsed; ``False`` in every other state.
        """
        with self._lock:
            return self.state == CircuitState.OPEN and self._should_attempt_recovery()

    def get_state(self) -> CircuitBreakerState:
        """Return a frozen snapshot of the circuit breaker's internal state."""
        with self._lock:
            return CircuitBreakerState(
                state=self.state,
                failure_count=self.failure_count,
                success_count=self.success_count,
                last_failure_time=self.last_failure_time,
                last_state_change=self.last_state_change,
                recovery_eligible=self.state == CircuitState.OPEN
                and self._should_attempt_recovery(),
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

    def _consume_probe_slot_locked(self) -> CircuitBreakerAdmission:
        """Consume the HALF_OPEN probe slot and return its ownership token."""
        self._next_probe_token += 1
        self._half_open_probe_active = True
        self._half_open_probe_token = self._next_probe_token
        return CircuitBreakerAdmission(allowed=True, probe_token=self._half_open_probe_token)

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_state_change = time.monotonic()
        self.success_count = 0
        # A fresh OPEN episode starts with no probe in flight.
        self._half_open_probe_active = False
        self._half_open_probe_token = None
        self._effective_recovery_timeout = self._sample_recovery_window()
        logger.warning("Circuit breaker opened due to failures")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.monotonic()
        self.failure_count = 0
        self.success_count = 0
        # A fresh CLOSED episode starts with no probe in flight.
        self._half_open_probe_active = False
        self._half_open_probe_token = None
        logger.info("Circuit breaker closed, provider recovered")

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.monotonic()
        self.success_count = 0
        self.failure_count = 0
        logger.info("Circuit breaker half-open, testing recovery")
