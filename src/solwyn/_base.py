"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and candidate selection. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from solwyn._routing import (
    CostPolicy,
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

logger = logging.getLogger(__name__)

# Bounded rolling window of recent SUCCESS latencies kept per provider, and the
# minimum sample count before observed_p50 reports a value. Until that many
# samples accrue, observed_p50 returns None so an under-sampled provider's
# latency never jumps the LatencyPolicy queue (it sorts AFTER known-p50 peers).
_LATENCY_WINDOW = 50
_LATENCY_MIN_SAMPLES = 3

# CostPolicy is inert until the server sends a relative price hint: with no hint
# on any candidate it degrades to health-based ordering. Warn ONCE per process
# when that fallback is in effect so the no-op is visible without logging on
# every routed call. Lock-guarded because the sync client is multi-threaded.
_cost_policy_inactive_warned = False
_cost_policy_warn_lock = threading.Lock()


def _warn_cost_policy_inactive_once() -> None:
    """Emit the CostPolicy-inert warning at most once per process."""
    global _cost_policy_inactive_warned
    with _cost_policy_warn_lock:
        if _cost_policy_inactive_warned:
            return
        _cost_policy_inactive_warned = True
    logger.warning("CostPolicy selected but no price hints available; using health-based order")


def _openai_uses_max_completion_tokens(model: str) -> bool:
    """Return whether an OpenAI model rejects the legacy max_tokens key."""
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def _with_openai_completion_token_key(
    provider: str,
    model: str,
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Rewrite OpenAI max_tokens for models that require max_completion_tokens.

    Applies to OpenAI itself and Azure OpenAI (which hosts the same models);
    other OpenAI-compatible providers accept the legacy max_tokens key.
    """
    if provider not in (ProviderName.OPENAI.value, ProviderName.AZURE_OPENAI.value):
        return kwargs
    if not _openai_uses_max_completion_tokens(model):
        return kwargs
    if "max_tokens" not in kwargs:
        return kwargs
    rewritten = dict(kwargs)
    if "max_completion_tokens" not in rewritten:
        rewritten["max_completion_tokens"] = rewritten["max_tokens"]
    del rewritten["max_tokens"]
    return rewritten


def _with_legacy_max_tokens_key(provider: str, kwargs: dict[str, object]) -> dict[str, object]:
    """Rewrite max_completion_tokens back to legacy max_tokens for compat targets.

    The inverse of ``_with_openai_completion_token_key``, applied ONLY on a
    same-dialect CROSS-PROVIDER failover hop: kwargs authored for an OpenAI
    model (which REQUIRES max_completion_tokens for o*/gpt-5 families) would
    otherwise hit a strict compat target as an unknown param, 4xx, and
    FAIL_FAST-abort the whole chain. Never applied to the caller's own
    configured target.

    Pure SINGLE-SOURCE rewrite: within one kwargs dict, max_completion_tokens
    wins over (overwrites) any max_tokens in the same dict — both-present is
    input OpenAI itself rejects, so we keep the modern key's value rather
    than guess. Cross-source precedence (per-call kwargs > per-entry
    default_params > global defaults) is the CALLER's job: ``_build_hop_kwargs``
    normalizes each kwargs layer with this function separately BEFORE the
    precedence merge, so a cap from a defaults layer can never collapse onto
    the per-call key and beat caller intent. Never apply this to an
    already-merged dict.
    """
    if provider in (ProviderName.OPENAI.value, ProviderName.AZURE_OPENAI.value):
        return kwargs
    if "max_completion_tokens" not in kwargs:
        return kwargs
    rewritten = dict(kwargs)
    value = rewritten.pop("max_completion_tokens")
    rewritten["max_tokens"] = value
    return rewritten


@dataclass(frozen=True)
class MediaSurfaceSpec:
    """Per-surface wiring for the ``_media_call`` lifecycle.

    A non-chat media surface (embeddings, images, audio, video) is fully
    described here, so the lifecycle stays surface-agnostic and later batches
    add a surface by constructing one of these — never by editing the lifecycle:

    - ``surface``: the dispatch key handed to ``adapter.prepare_media_call``
      (a ``MediaSurface`` value, e.g. ``"embeddings"``).
    - ``modality``: the billing modality carried through the lifecycle so the
      server's card unit can select it. The vendored wire types do not carry a
      ``modality`` field yet (P1.11 vendors it); until then this attribute rides
      on the spec and is referenced at the confirm/metadata sites, so P1.11 only
      has to add the field and connect ``spec.modality`` to the payloads.
    - ``extract_usage``: pulls the billable quantity from the RESPONSE's usage
      block, or None when the response reports none.
    - ``measure_request``: derives the billable quantity from the REQUEST when
      the response reports none (request-side measurement lands in
      ``solwyn._privacy`` per P1.9). Returns None when the quantity is
      unobservable.

    Both hooks return ``TokenDetails`` (the quantity carrier) or None — never a
    zero-filled default, so an unobservable quantity is never settled as a real
    $0 price.
    """

    surface: str
    modality: str
    extract_usage: Callable[[Any], TokenDetails | None]
    measure_request: Callable[[dict[str, Any]], TokenDetails | None]


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
        # Injectable routing policy: defaults to the health-only policy.
        # Swapping in LatencyPolicy/CostPolicy reorders candidates with ZERO
        # changes to dispatch / translation / budget.
        self._policy: SelectionPolicy = selection_policy or HealthBasedPolicy()

        # Per-provider rolling window of recent SUCCESS latencies (ms) for the
        # LatencyPolicy signal. Lock-guarded because the sync client is
        # multi-threaded (the async client is event-loop-serialized; the lock is
        # then uncontended). Pure signal store — no I/O.
        self._signal_lock = threading.Lock()
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
        with self._signal_lock:
            self._latency_windows[provider].append(ms)

    def observed_p50(self, provider: str) -> float | None:
        """Observed p50 (median) latency (ms) for a provider, or None.

        Returns None until at least ``_LATENCY_MIN_SAMPLES`` samples have
        accrued, so an under-sampled provider never jumps the LatencyPolicy
        queue. Lock-guarded; snapshots the window under the lock and computes the
        median outside it.
        """
        with self._signal_lock:
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
        with self._signal_lock:
            self._last_price_hints = dict(hints)

    def _select_candidates(self, req: RoutingRequest) -> list[ProviderRuntime]:
        """Order runtimes into attempt order via the pure SelectionPolicy.

        Builds one ProviderCandidate per runtime using NON-MUTATING breaker
        reads only (``state`` / ``recovery_eligible``) — never ``admit()``
        (which consumes a probe). Probe consumption happens exactly once, on the
        single candidate actually attempted, in the dispatch loop.
        """
        # Snapshot the price hints once under the lock so every candidate in this
        # selection sees a consistent view (the setter may replace the dict
        # concurrently on another thread).
        with self._signal_lock:
            price_hints = dict(self._last_price_hints)
        candidates: list[ProviderCandidate] = []
        for runtime in self._runtimes:
            breaker = self._get_circuit_breaker(runtime.adapter.name)
            state = breaker.get_state()
            candidates.append(
                ProviderCandidate(
                    runtime=runtime,
                    breaker_state=state.state,
                    recovery_eligible=state.recovery_eligible,
                    translatable=True,  # native passthrough; a later predicate refines this
                    # Routing signals: observed p50 latency (LatencyPolicy) and the
                    # server-provided relative price hint (CostPolicy). Both default to
                    # None when unavailable; HealthBasedPolicy ignores them.
                    latency_p50=self.observed_p50(runtime.adapter.name),
                    price_hint=price_hints.get(runtime.adapter.name),
                )
            )
        ordered = self._policy.order(candidates, req)
        if isinstance(self._policy, CostPolicy) and not any(
            c.price_hint is not None for c in candidates
        ):
            # CostPolicy was selected but no candidate carries a server price
            # hint, so it degraded to health-based order — surface the no-op once.
            _warn_cost_policy_inactive_once()
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
        call_id: str,
        possibly_succeeded: bool | None = None,
        service_tier: str | None = None,
        sdk_instance_id: str | None = None,
        timestamp: datetime | None = None,
        agent_run: tuple[str | None, str | None] | None = None,
        provider_region: str | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API.

        ``call_id`` is the per-call reconciliation join key. ``possibly_succeeded``
        is the post-send-ambiguous abort flag — left None on every non-abort event.
        ``provider_region`` is the served endpoint's cloud region (Bedrock pricing
        is per model AND region); None for providers without regional pricing.
        """
        if not call_id:
            raise RuntimeError("call_id is required for metadata reconciliation")
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
            possibly_succeeded=possibly_succeeded,
            service_tier=service_tier,
            sdk_instance_id=sdk_instance_id or self._sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
            agent_run_id=agent_run_id,
            agent_run_name=agent_run_name,
            call_id=call_id,
            provider_region=provider_region,
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
        call_id: str,
        possibly_succeeded: bool | None = None,
        agent_run: tuple[str | None, str | None] | None = None,
        provider_region: str | None = None,
    ) -> MetadataEvent:
        """Build an error-status MetadataEvent with zeroed token counts.

        Convenience wrapper for the dispatch-failure paths where
        token_details is unavailable and status is always ERROR. ``call_id``
        threads the reconciliation join key; ``possibly_succeeded`` is True only
        on a correctly-not-failed-over post-send-ambiguous abort.
        ``provider_region`` is the FAILED hop's endpoint region — on a
        possibly-succeeded abort the Cloud API needs it to reconcile a
        possibly-landed charge per (model, region); None-skipped otherwise.
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
            provider_region=provider_region,
        )
