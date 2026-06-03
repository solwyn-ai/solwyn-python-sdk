"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and candidate selection. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from solwyn._routing import HealthBasedPolicy, ProviderCandidate, RoutingRequest
from solwyn._run import current_run
from solwyn._token_details import TokenDetails
from solwyn._types import CallStatus, FailoverReason, MetadataEvent, ProviderName
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.config import SolwynConfig
from solwyn.tokenizer import TokenizerManager

if TYPE_CHECKING:
    from solwyn._registry import ProviderRuntime


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

    def __init__(self, config: SolwynConfig, runtimes: list[ProviderRuntime]) -> None:
        self._config = config
        self._runtimes = runtimes
        self._sdk_instance_id = str(uuid.uuid4())
        self._tokenizer = TokenizerManager()
        self._policy = HealthBasedPolicy()

        # One circuit breaker per DISTINCT provider across ALL runtimes.
        # Additional providers get lazily-created breakers via
        # _get_circuit_breaker (same jitter).
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
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
        if provider not in self._circuit_breakers:
            self._circuit_breakers[provider] = self._new_circuit_breaker()
        return self._circuit_breakers[provider]

    def _select_candidates(self, req: RoutingRequest) -> list[ProviderRuntime]:
        """Order runtimes into attempt order via the pure SelectionPolicy.

        Builds one ProviderCandidate per runtime using NON-MUTATING breaker
        reads only (``state`` / ``recovery_eligible``) — never ``can_proceed()``
        (which consumes a probe). Probe consumption happens exactly once, on the
        single candidate actually attempted, in the dispatch loop (§4.2).
        """
        candidates = [
            ProviderCandidate(
                runtime=runtime,
                breaker_state=self._get_circuit_breaker(runtime.adapter.name).state,
                recovery_eligible=self._get_circuit_breaker(runtime.adapter.name).recovery_eligible,
                translatable=True,  # P1: native passthrough; P2 supplies the real predicate
            )
            for runtime in self._runtimes
        ]
        ordered = self._policy.order(candidates, req)
        return [c.runtime for c in ordered]

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
        service_tier: str | None = None,
        sdk_instance_id: str | None = None,
        timestamp: datetime | None = None,
        agent_run: tuple[str | None, str | None] | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API."""
        agent_run_id, agent_run_name = current_run() if agent_run is None else agent_run
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
            service_tier=service_tier,
            sdk_instance_id=sdk_instance_id or self._sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
            agent_run_id=agent_run_id,
            agent_run_name=agent_run_name,
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
        agent_run: tuple[str | None, str | None] | None = None,
    ) -> MetadataEvent:
        """Build an error-status MetadataEvent with zeroed token counts.

        Convenience wrapper for the dispatch-failure paths where
        token_details is unavailable and status is always ERROR.
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
            agent_run=agent_run,
        )
