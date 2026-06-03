"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and candidate selection. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from solwyn._routing import (
    HealthBasedPolicy,
    ProviderCandidate,
    RoutingRequest,
    SelectionPolicy,
)
from solwyn._run import current_run
from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, FailoverReason, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig
from solwyn.tokenizer import TokenizerManager

if TYPE_CHECKING:
    from solwyn._registry import ProviderRuntime

# Bounded rolling window of recent SUCCESS latencies kept per provider, and the
# minimum sample count before observed_p50 reports a value. Until that many
# samples accrue, observed_p50 returns None so an under-sampled provider's
# latency never jumps the LatencyPolicy queue (it sorts AFTER known-p50 peers).
_LATENCY_WINDOW = 50
_LATENCY_MIN_SAMPLES = 3


class _AttemptContext(BaseModel):
    """Per-attempt state for one candidate in the dispatch walk.

    Immutable and prompt-free: deliberately does NOT carry the call kwargs
    (privacy — never hold prompt content here). The dispatch loop builds the
    per-call kwargs as a plain local that never leaves the candidate walk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    start_time: float
    is_provider_fallback: bool
    attempt_index: int

    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since this attempt started."""
        return (time.monotonic() - self.start_time) * 1000


class _SolwynBase:
    """Shared sans-I/O base class for Solwyn sync and async clients.

    Provides:
    - Token estimation seam (via TokenizerManager)
    - Metadata event construction
    - Circuit breaker management and pure candidate selection
    - SDK instance identity
    """

    def __init__(
        self,
        config: SolwynConfig,
        runtimes: list[ProviderRuntime],
        selection_policy: SelectionPolicy | None = None,
    ) -> None:
        self._config = config
        self._runtimes = runtimes
        self._sdk_instance_id = str(uuid.uuid4())
        self._tokenizer = TokenizerManager()
        # Injectable routing policy (P5): defaults to the health-only policy.
        # Swapping in LatencyPolicy/CostPolicy reorders candidates with ZERO
        # changes to dispatch / translation / budget.
        self._policy: SelectionPolicy = selection_policy or HealthBasedPolicy()

        # Per-provider rolling window of recent SUCCESS latencies (ms) for the
        # LatencyPolicy signal. Lock-guarded because the sync client is
        # multi-threaded (the async client is event-loop-serialized; the lock is
        # then uncontended). Pure signal store — no I/O.
        self._latency_lock = threading.Lock()
        self._latency_windows: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_LATENCY_WINDOW)
        )

        # Last server-provided RELATIVE price hints per provider (CostPolicy
        # signal). The client refreshes this after each budget check. The SDK
        # never computes price — this only forwards the server signal.
        self._last_price_hints: dict[str, float] = {}

        # One circuit breaker per DISTINCT provider across ALL runtimes.
        # Additional providers get lazily-created breakers via
        # _get_circuit_breaker (same jitter).
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._breaker_lock = threading.Lock()
        for runtime in runtimes:
            provider = runtime.entry.provider.value
            if provider not in self._circuit_breakers:
                self._circuit_breakers[provider] = self._new_circuit_breaker()

    def _new_circuit_breaker(self) -> CircuitBreaker:
        """Create a circuit breaker from the configured tuning + jitter."""
        return CircuitBreaker(
            failure_threshold=self._config.circuit_breaker_failure_threshold,
            recovery_timeout=self._config.circuit_breaker_recovery_timeout,
            success_threshold=self._config.circuit_breaker_success_threshold,
            recovery_timeout_jitter=self._config.circuit_breaker_recovery_timeout_jitter,
        )

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get the circuit breaker for a provider.

        Lazily creates a circuit breaker if one doesn't exist for this provider.
        """
        with self._breaker_lock:
            if provider not in self._circuit_breakers:
                self._circuit_breakers[provider] = self._new_circuit_breaker()
            return self._circuit_breakers[provider]

    def record_latency(self, provider: str, ms: float) -> None:
        """Record one observed SUCCESS latency (ms) for a provider (LatencyPolicy).

        Appends to the provider's bounded rolling window. Lock-guarded for the
        multi-threaded sync client (uncontended on the async client). Pure signal
        store — no I/O, no breaker mutation.
        """
        with self._latency_lock:
            self._latency_windows[provider].append(ms)

    def observed_p50(self, provider: str) -> float | None:
        """Observed p50 (median) latency (ms) for a provider, or None.

        Returns None until at least ``_LATENCY_MIN_SAMPLES`` samples have
        accrued, so an under-sampled provider never jumps the LatencyPolicy
        queue. Lock-guarded; snapshots the window under the lock and computes the
        median outside it.
        """
        with self._latency_lock:
            window = self._latency_windows.get(provider)
            samples = list(window) if window is not None else []
        if len(samples) < _LATENCY_MIN_SAMPLES:
            return None
        return statistics.median(samples)

    def update_price_hints(self, hints: dict[str, float]) -> None:
        """Replace the last-known server price hints (CostPolicy signal).

        Called by the client after each budget check with the server-provided
        RELATIVE price signal. The SDK never computes price — it only stores and
        forwards this. An empty dict clears the hints (server provided none).
        """
        with self._latency_lock:
            self._last_price_hints = dict(hints)

    def _select_candidates(self, req: RoutingRequest) -> list[ProviderRuntime]:
        """Order runtimes into attempt order via the pure SelectionPolicy.

        Builds one ProviderCandidate per runtime using NON-MUTATING breaker
        reads only (``state`` / ``recovery_eligible``) — never ``can_proceed()``
        (which consumes a probe). Probe consumption happens exactly once, on the
        single candidate actually attempted, in the dispatch loop (§4.2).
        """
        # Snapshot the price hints once under the lock so every candidate in this
        # selection sees a consistent view (the setter may replace the dict
        # concurrently on another thread).
        with self._latency_lock:
            price_hints = dict(self._last_price_hints)
        candidates = [
            ProviderCandidate(
                runtime=runtime,
                breaker_state=self._get_circuit_breaker(runtime.adapter.name).state,
                recovery_eligible=self._get_circuit_breaker(runtime.adapter.name).recovery_eligible,
                translatable=True,  # P1: native passthrough; P2 supplies the real predicate
                # P5 routing signals: observed p50 latency (LatencyPolicy) and the
                # server-provided relative price hint (CostPolicy). Both default to
                # None when unavailable; HealthBasedPolicy ignores them.
                latency_p50=self.observed_p50(runtime.adapter.name),
                price_hint=price_hints.get(runtime.adapter.name),
            )
            for runtime in self._runtimes
        ]
        ordered = self._policy.order(candidates, req)
        # Defensive: a custom (possibly misbehaving) injected policy must not be
        # able to inject a runtime that was never in the configured chain into the
        # dispatch walk. Keep ONLY candidates whose runtime is one of our own
        # runtimes (identity check), preserving the policy's order for that valid
        # subset. Drops any foreign/unknown runtime the policy may have appended.
        chain = set(map(id, self._runtimes))
        return [c.runtime for c in ordered if id(c.runtime) in chain]

    def _build_metadata_event(
        self,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        token_details: TokenDetails | None,
        latency_ms: float,
        status: CallStatus,
        is_model_fallback: bool,
        is_provider_fallback: bool = False,
        requested_provider: ProviderName | None = None,
        requested_model: str | None = None,
        failover_reason: FailoverReason | None = None,
        failover_error_class: str | None = None,
        attempt_index: int = 0,
        call_id: str | None = None,
        possibly_succeeded: bool | None = None,
        service_tier: str | None = None,
        sdk_instance_id: str | None = None,
        timestamp: datetime | None = None,
        agent_run: tuple[str | None, str | None] | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API.

        ``call_id`` is the per-call reconciliation join key (§8.4); when None the
        model's default_factory fills a fresh uuid. ``possibly_succeeded`` is the
        post-send-ambiguous abort flag — left None on every non-abort event.
        """
        agent_run_id, agent_run_name = current_run() if agent_run is None else agent_run
        # When call_id is None let the MetadataEvent default_factory mint one so
        # direct construction keeps working; the client threads an explicit value.
        extra: dict[str, str] = {} if call_id is None else {"call_id": call_id}
        return MetadataEvent(
            model=model,
            provider=ProviderName(provider),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_details=token_details,
            latency_ms=latency_ms,
            status=status,
            is_model_fallback=is_model_fallback,
            is_provider_fallback=is_provider_fallback,
            requested_provider=requested_provider,
            requested_model=requested_model,
            failover_reason=failover_reason,
            failover_error_class=failover_error_class,
            attempt_index=attempt_index,
            possibly_succeeded=possibly_succeeded,
            service_tier=service_tier,
            sdk_instance_id=sdk_instance_id or self._sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
            agent_run_id=agent_run_id,
            agent_run_name=agent_run_name,
            **extra,
        )

    def _build_error_event(
        self,
        *,
        model: str,
        provider: str,
        latency_ms: float,
        is_model_fallback: bool,
        is_provider_fallback: bool = False,
        requested_provider: ProviderName | None = None,
        requested_model: str | None = None,
        failover_error_class: str | None = None,
        attempt_index: int = 0,
        call_id: str | None = None,
        possibly_succeeded: bool | None = None,
        agent_run: tuple[str | None, str | None] | None = None,
    ) -> MetadataEvent:
        """Build an error-status MetadataEvent with zeroed token counts.

        Convenience wrapper for the dispatch-failure paths where
        token_details is unavailable and status is always ERROR. ``call_id``
        threads the reconciliation join key; ``possibly_succeeded`` is True only
        on a correctly-not-failed-over post-send-ambiguous abort (§8.4).
        """
        return self._build_metadata_event(
            model=model,
            provider=provider,
            input_tokens=0,
            output_tokens=0,
            token_details=None,
            latency_ms=latency_ms,
            status=CallStatus.ERROR,
            is_model_fallback=is_model_fallback,
            is_provider_fallback=is_provider_fallback,
            requested_provider=requested_provider,
            requested_model=requested_model,
            failover_error_class=failover_error_class,
            attempt_index=attempt_index,
            call_id=call_id,
            possibly_succeeded=possibly_succeeded,
            agent_run=agent_run,
        )
