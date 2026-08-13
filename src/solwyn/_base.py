"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and candidate selection. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import inspect
import logging
import os
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NoReturn, cast

from pydantic import BaseModel, ConfigDict

from solwyn._lifecycle import register_fork_reset
from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    ProviderCandidate,
    RoutingRequest,
    SelectionPolicy,
)
from solwyn._run import _capture_run_context, _RunContextSnapshot
from solwyn._surface_graph import (
    _descriptor_category,
    _return_shape,
    _static_attribute,
    _static_return_shape,
)
from solwyn._surfaces import (
    SURFACE_RULES,
    AttributeShape,
    CapabilityScope,
    SurfaceCondition,
    SurfaceContext,
    SurfaceKind,
    SurfaceRule,
    SurfaceSource,
    _validate_surface_path,
    context_is_declared,
    resolve_surface_rule,
)
from solwyn._token_details import TokenDetails
from solwyn._types import (
    CallStatus,
    FailoverReason,
    MediaUsage,
    MetadataEvent,
    Modality,
    ProviderName,
)
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    ConfigurationError,
    UnsupportedSurfaceError,
    UntrackedSpendSurfaceError,
)

if TYPE_CHECKING:
    from solwyn._registry import ProviderRuntime

logger = logging.getLogger(__name__)

# Bounded rolling window of recent SUCCESS latencies kept per provider, and the
# minimum sample count before observed_p50 reports a value. Until that many
# samples accrue, observed_p50 returns None so an under-sampled provider's
# latency never jumps the LatencyPolicy queue (it sorts AFTER known-p50 peers).
_LATENCY_WINDOW = 50
_LATENCY_MIN_SAMPLES = 3

# The complete and intentionally narrow set of customer-configurable failover
# tuning governed by the server directive. Provider entries and routing policy
# are deliberately outside this boundary.
_FAILOVER_TUNING_FIELDS = (
    "failover_total_timeout",
    "failover_hop_read_timeout",
    "failover_idempotency",
    "same_provider_retries",
    "circuit_breaker_recovery_timeout_jitter",
    "circuit_breaker_failure_threshold",
    "circuit_breaker_recovery_timeout",
    "circuit_breaker_success_threshold",
)

_GUARDABLE_RETURN_SHAPES = frozenset({"resource", "context_manager", "async_context_manager"})

_CONTEXT_MISMATCH_HINTS = {
    ("bedrock_boto3", "async"): (
        " boto3 clients are synchronous — wrap them with Solwyn, or use an "
        "aioboto3 client with AsyncSolwyn"
    ),
    ("bedrock_aioboto3", "sync"): (
        " aioboto3 clients are asynchronous — wrap them with AsyncSolwyn"
    ),
    ("google_generativeai", "async"): (
        " google-generativeai supports sync only — wrap it with Solwyn"
    ),
}


class FailoverTuning(NamedTuple):
    """One COHERENT per-call snapshot of the server-governed failover tuning.

    The directive writer mutates self._config's tuning fields under
    _breaker_lock (_apply_failover_tuning_directive); an unlocked mid-call
    re-read of self._config can observe a torn mix of old and new tuning
    (PJ-8/R12). The dispatch path therefore consumes tuning ONLY through this
    immutable snapshot, captured once per call under that same lock.
    """

    failover_total_timeout: float
    failover_idempotency: Literal["safe", "never", "always"]
    same_provider_retries: int
    failover_hop_read_timeout: float


_BEDROCK_UNBOUNDED_READ_WARNING = (
    "Bedrock client (model %r) was built with botocore Config(read_timeout=None): "
    "Solwyn cannot bound Bedrock hops per-call (boto3 has no per-call timeout "
    "override) and only checks the failover deadline BETWEEN hops, so one stuck "
    "Converse read can hang the call indefinitely. Set a finite read_timeout on "
    "the client's botocore Config, e.g. Config(read_timeout=60)."
)


def _bedrock_read_timeout_is_unbounded(sdk_client: object) -> bool:
    """True when a bedrock-runtime client carries read_timeout=None (duck-typed).

    botocore's default Config has read_timeout=60, so None is always an
    explicit caller choice - the one shape neither Solwyn (no per-call
    override) nor botocore will ever bound. Never imports botocore: the
    ``client.meta.config.read_timeout`` path is read defensively, and any
    missing attribute (test doubles, exotic wrappers) reads as bounded.
    """
    config = getattr(getattr(sdk_client, "meta", None), "config", None)
    if config is None:
        return False
    sentinel = object()
    return getattr(config, "read_timeout", sentinel) is None


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


# Per-process warn-once state is keyed by the complete contextual rule identity,
# so equal terminal names in different provider graphs cannot suppress each other.
_WARNED_SURFACE_LIMIT = 512
_warned_contextual_surfaces: set[tuple[str, str, str, str]] = set()
_warn_limit_reached = False
_spend_surface_warn_lock = threading.Lock()


def _reset_warn_locks_after_fork_in_child() -> None:
    global _spend_surface_warn_lock, _cost_policy_warn_lock
    _spend_surface_warn_lock = threading.Lock()
    _cost_policy_warn_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_warn_locks_after_fork_in_child)


def _reset_unmetered_spend_warnings() -> None:
    """Clear the per-process warn-once latch. Test-support hook only."""
    global _warn_limit_reached
    with _spend_surface_warn_lock:
        _warned_contextual_surfaces.clear()
        _warn_limit_reached = False


def _warn_contextual_surface_once(
    *,
    context: SurfaceContext,
    rule_id: str,
    surface: str,
    capability_scope: str | None,
    drifted_from_rule_id: str | None = None,
) -> None:
    """Warn once per provider/client-shape/mode/rule identity, bounded in size."""

    global _warn_limit_reached
    warning_key = (context.provider, context.client_shape, context.mode, rule_id)
    warn_limit_reached_now = False
    with _spend_surface_warn_lock:
        if warning_key in _warned_contextual_surfaces:
            return
        if len(_warned_contextual_surfaces) >= _WARNED_SURFACE_LIMIT:
            if _warn_limit_reached:
                return
            _warn_limit_reached = True
            warn_limit_reached_now = True
        else:
            _warned_contextual_surfaces.add(warning_key)
    if warn_limit_reached_now:
        logger.warning(
            "Untracked-surface warning limit (%d) reached; further distinct "
            "surfaces will not be individually reported this process.",
            _WARNED_SURFACE_LIMIT,
        )
        return
    if drifted_from_rule_id is not None:
        drift_message = "Reviewed rule %s no longer matches its shape.".replace(
            "%s", drifted_from_rule_id
        )
        logger.warning(
            "Provider '%s' client shape '%s' exposes untracked surface '%s' "
            "(scope: %s); no budget check and no cost event will be emitted. "
            f"Tracking for this surface is coming. {drift_message}",
            context.provider,
            context.client_shape,
            surface,
            capability_scope,
        )
        return
    logger.warning(
        "Provider '%s' client shape '%s' exposes untracked surface '%s' "
        "(scope: %s); no budget check and no cost event will be emitted. "
        "Tracking for this surface is coming.",
        context.provider,
        context.client_shape,
        surface,
        capability_scope,
    )


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


_OUTPUT_CAP_KEYS = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "config",
        "inferenceConfig",
    }
)
_OUTPUT_CAP_KEYS_BY_DIALECT = {
    ProviderName.OPENAI.value: frozenset({"max_tokens", "max_completion_tokens"}),
    ProviderName.ANTHROPIC.value: frozenset({"max_tokens"}),
    ProviderName.GOOGLE.value: frozenset({"config", "max_output_tokens"}),
    ProviderName.BEDROCK.value: frozenset({"inferenceConfig"}),
}


def _positive_output_cap(value: object) -> int | None:
    """Return a usable token cap without coercing caller-owned objects."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _nested_output_cap(value: object, key: str) -> int | None:
    """Read one cap from a dict or SDK config object without copying config.

    Google config objects may also carry a system instruction, and Bedrock
    request dictionaries carry messages beside ``inferenceConfig``. Reading
    only the named scalar keeps lease sizing outside content-privileged code.
    """
    nested = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
    return _positive_output_cap(nested)


def _dialect_output_cap(dialect: str, kwargs: dict[str, object]) -> int | None:
    """Return the effective native cap in one dialect-shaped kwargs mapping."""
    if dialect == ProviderName.GOOGLE.value:
        nested = _nested_output_cap(kwargs.get("config"), "max_output_tokens")
        return nested or _positive_output_cap(kwargs.get("max_output_tokens"))
    if dialect == ProviderName.BEDROCK.value:
        return _nested_output_cap(kwargs.get("inferenceConfig"), "maxTokens")
    if dialect == ProviderName.OPENAI.value:
        return _positive_output_cap(kwargs.get("max_completion_tokens")) or _positive_output_cap(
            kwargs.get("max_tokens")
        )
    if dialect == ProviderName.ANTHROPIC.value:
        return _positive_output_cap(kwargs.get("max_tokens"))
    return None


def _output_cap_params(params: dict[str, Any]) -> dict[str, object]:
    """Copy only structural cap fields, never request content."""
    return {key: value for key, value in params.items() if key in _OUTPUT_CAP_KEYS}


def _source_output_cap_defaults(
    dialect: str,
    params: dict[str, object],
) -> dict[str, object]:
    """Keep only entry-default cap fields legal in the source dialect."""
    allowed = _OUTPUT_CAP_KEYS_BY_DIALECT.get(dialect, frozenset())
    return {key: value for key, value in params.items() if key in allowed}


def _normalized_openai_output_cap_layer(
    provider: str,
    model: str,
    params: dict[str, object],
) -> dict[str, object]:
    """Normalize cap aliases in one precedence layer to its OpenAI target key."""
    requires_modern_key = provider in (
        ProviderName.OPENAI.value,
        ProviderName.AZURE_OPENAI.value,
    ) and _openai_uses_max_completion_tokens(model)
    if requires_modern_key:
        return _with_openai_completion_token_key(provider, model, params)
    if "max_completion_tokens" not in params:
        return params
    normalized = dict(params)
    value = normalized.pop("max_completion_tokens")
    normalized["max_tokens"] = value
    return normalized


def _effective_output_bound(
    *,
    primary: ProviderRuntime,
    runtimes: list[ProviderRuntime],
    global_defaults: dict[str, Any],
    kwargs: dict[str, object],
    default_bound: int,
) -> int:
    """Largest output reservation any configured provider hop can spend.

    Mirrors dispatch cap precedence without translating or copying request
    content: call kwargs beat entry defaults, which beat global defaults;
    same-dialect compatibility hops normalize completion-token keys per source
    layer; cross-dialect hops retain the source dialect's cap. Every actually
    unbounded hop contributes the configured conservative fallback.
    """
    provider_global_defaults = _output_cap_params(global_defaults)
    provider_kwargs = _output_cap_params(kwargs)
    bounds: list[int] = []

    for runtime in runtimes:
        provider_entry_defaults = _output_cap_params(runtime.entry.default_params)
        is_primary = runtime is primary
        is_provider_fallback = runtime.entry.provider != primary.entry.provider
        target_model = cast(str, kwargs["model"]) if is_primary else runtime.entry.model

        if not is_provider_fallback:
            if runtime.adapter.dialect == ProviderName.OPENAI.value:
                native = {
                    **_normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_global_defaults,
                    ),
                    **_normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_entry_defaults,
                    ),
                    **_normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_kwargs,
                    ),
                }
            else:
                native = {
                    **provider_global_defaults,
                    **provider_entry_defaults,
                    **provider_kwargs,
                }
            cap = _dialect_output_cap(runtime.adapter.dialect, native)
        elif primary.adapter.dialect == runtime.adapter.dialect:
            target_name = runtime.adapter.name
            native = {
                **_normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_global_defaults,
                ),
                **_normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_entry_defaults,
                ),
                **_normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_kwargs,
                ),
            }
            cap = _dialect_output_cap(runtime.adapter.dialect, native)
        else:
            source_defaults = _source_output_cap_defaults(
                primary.adapter.dialect,
                provider_entry_defaults,
            )
            if primary.adapter.dialect == ProviderName.OPENAI.value:
                source_model = cast(str, kwargs["model"])
                source_native = {
                    **_normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        provider_global_defaults,
                    ),
                    **_normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        source_defaults,
                    ),
                    **_normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        provider_kwargs,
                    ),
                }
            else:
                source_native = {
                    **provider_global_defaults,
                    **source_defaults,
                    **provider_kwargs,
                }
            cap = _dialect_output_cap(primary.adapter.dialect, source_native)

        bounds.append(cap if cap is not None else default_bound)

    return max(bounds, default=default_bound)


@dataclass(frozen=True)
class MediaSurfaceSpec:
    """Per-surface wiring for the ``_media_call`` lifecycle.

    A non-chat media surface (embeddings, images, audio, video) is fully
    described here, so the lifecycle stays surface-agnostic and a new surface
    is added by constructing one of these — never by editing the lifecycle:

    - ``surface``: the dispatch key handed to ``adapter.prepare_media_call``
      (a ``MediaSurface`` value, e.g. ``"embeddings"``).
    - ``modality``: the billing modality carried through the lifecycle so the
      server's card unit can select it. The vendored wire types carry a
      ``modality`` field; ``_media_call`` connects ``spec.modality`` onto
      the budget check, the confirm, and the SUCCESS / BUDGET_DENIED metadata
      events. The chat pipeline never sets it and rides the ``"text"`` default.
    - ``extract_usage``: pulls the billable TOKEN quantity from the RESPONSE's
      usage block, or None when the response reports none.
    - ``measure_request``: derives the billable TOKEN quantity from the REQUEST
      when the response reports none (request-side measurement lives in
      ``solwyn._privacy``). Returns None when the quantity is
      unobservable.
    - ``measure_media``: OPTIONAL non-token quantity channel. Derives the
      settled ``MediaUsage`` (image counts, media seconds, character counts,
      variant selectors) from the REQUEST and RESPONSE for a per-unit priced
      surface. None (the default) for token-only surfaces like embeddings, which
      carry no ``MediaUsage``. Kept SEPARATE from the token hooks so media
      quantities are never shoehorned into ``TokenDetails``. Returns None
      when unobservable.
    - ``estimate_media``: OPTIONAL pre-flight non-token quantity. Derives
      the ``estimated_media`` from the REQUEST alone so the budget CHECK carries
      a precise per-unit pre-flight cost. None (the default) for token-only
      surfaces. Returns None when unobservable.

    The token hooks return ``TokenDetails`` and the media hooks ``MediaUsage`` —
    each None (never a zero-filled default) when unobservable, so an unobservable
    quantity is never settled as a real $0 price. BOTH bases ride the confirm
    when both are observable (e.g. native gpt-image sends token usage AND
    request-derived ``MediaUsage``); the server's pricing card unit picks.
    """

    surface: str
    modality: Modality
    extract_usage: Callable[[Any], TokenDetails | None]
    measure_request: Callable[[dict[str, Any]], TokenDetails | None]
    measure_media: Callable[[dict[str, Any], Any], MediaUsage | None] | None = None
    estimate_media: Callable[[dict[str, Any]], MediaUsage | None] | None = None


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


def _client_shape(client: object, dialect: str) -> str:
    """Return the provider-independent structural client-shape identity."""

    module = getattr(type(client), "__module__", "")
    class_name = getattr(type(client), "__name__", "")
    if dialect == "openai":
        if (module == "together" or module.startswith("together.")) and class_name in {
            "Together",
            "AsyncTogether",
        }:
            return "native_together"
        return "openai_sdk"
    if dialect == "anthropic":
        return "anthropic_sdk"
    if dialect == "google":
        if "google.generativeai" in module:
            return "google_generativeai"
        return "google_genai"
    if dialect == "bedrock":
        return "bedrock_aioboto3" if "aiobotocore" in module else "bedrock_boto3"
    raise RuntimeError(f"unsupported provider dialect for surface context: {dialect}")


def _validate_surface_context(context: SurfaceContext) -> None:
    """Reject a provider client/mode pairing absent from the reviewed rules."""

    if context_is_declared(context):
        return
    hint = _CONTEXT_MISMATCH_HINTS.get((context.client_shape, context.mode), "")
    raise ConfigurationError(
        "unsupported provider client/mode pairing: "
        f"{context.provider}/{context.client_shape}/{context.mode}.{hint}",
        field="client",
    )


def _belongs_to_client_shape(value: object, client_shape: str) -> bool:
    """Whether an opaque resource belongs to the detected provider SDK family.

    Context validation prevents undeclared shapes from constructing; the final
    error remains defense-in-depth against future vocabulary drift.
    """

    module = getattr(type(value), "__module__", "")
    if client_shape == "openai_sdk":
        return "openai" in module
    if client_shape == "native_together":
        return module == "together" or module.startswith("together.")
    if client_shape == "anthropic_sdk":
        return "anthropic" in module
    if client_shape == "google_genai":
        return "google.genai" in module
    if client_shape == "google_generativeai":
        return "google.generativeai" in module
    if client_shape == "bedrock_boto3":
        return "botocore" in module and "aiobotocore" not in module
    if client_shape == "bedrock_aioboto3":
        return "aiobotocore" in module
    raise RuntimeError(f"unrecognized client shape: {client_shape}")


class _GuardedResource:
    """Path-qualified provider namespace that re-enters its owner's resolver."""

    def __init__(self, owner: _SolwynBase, raw: object, path: str) -> None:
        self._solwyn_owner = owner
        self._solwyn_raw = raw
        self._solwyn_path = path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return getattr(self._solwyn_raw, name)
        path = f"{self._solwyn_path}.{name}"
        return self._solwyn_owner._resolve_public_attribute(
            self._solwyn_raw,
            name=name,
            path=path,
            source=SurfaceSource.RAW,
        )

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._solwyn_raw)))

    def __repr__(self) -> str:
        return f"<guarded provider resource {self._solwyn_path!r}>"


class _SolwynBase:
    """Shared sans-I/O base class for Solwyn sync and async clients.

    Provides:
    - Metadata event construction
    - Circuit breaker management and pure candidate selection
    - SDK instance identity
    """

    def __init__(
        self,
        config: SolwynConfig,
        runtimes: list[ProviderRuntime],
        selection_policy: SelectionPolicy | None = None,
        *,
        mode: Literal["sync", "async"] = "sync",
    ) -> None:
        runtime_contexts = tuple(
            SurfaceContext(
                provider=runtime.adapter.name,
                dialect=runtime.adapter.dialect,
                client_shape=_client_shape(runtime.sdk_client, runtime.adapter.dialect),
                mode=mode,
            )
            for runtime in runtimes
        )
        for context in runtime_contexts:
            _validate_surface_context(context)

        self._config = config
        self._runtimes = runtimes
        primary = runtimes[0]
        self._surface_context = runtime_contexts[0]
        self._guard_lock = threading.Lock()
        self._guarded_resources: dict[str, _GuardedResource] = {}
        self._validate_acknowledgments(primary.sdk_client)
        self._requested_failover_tuning = {
            name: getattr(config, name) for name in _FAILOVER_TUNING_FIELDS
        }
        self._failover_tuning_suppression_logged = False
        self._sdk_instance_id = str(uuid.uuid4())
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

        # PJ-8/R8: Solwyn's per-hop bound is unenforceable on boto3 (no
        # per-call timeout override; the deadline is only checked between
        # hops). An explicitly unbounded botocore read is therefore the one
        # configuration that can hang a call forever - warn loudly at build.
        for runtime in runtimes:
            if runtime.adapter.dialect == "bedrock" and _bedrock_read_timeout_is_unbounded(
                runtime.sdk_client
            ):
                logger.warning(_BEDROCK_UNBOUNDED_READ_WARNING, runtime.entry.model)

        # ONE control-plane breaker per client instance, shared by the budget
        # enforcer (check) and the reporter (confirm): the SDK discovers a
        # Solwyn outage once, not once per call. Never provider-reported.
        self._control_plane_breaker = CircuitBreaker(
            failure_threshold=config.control_plane_failure_threshold,
            recovery_timeout=config.control_plane_recovery_timeout,
            success_threshold=1,
            name="control-plane",
        )

        # Fork repair: a user thread can hold _breaker_lock (lazy provider-
        # breaker creation) or _signal_lock (latency window append) at fork
        # time; the child would inherit them held by a thread it doesn't have.
        register_fork_reset(self)

    def _validate_acknowledgments(self, raw_client: object) -> None:
        for token in sorted(self._config.acknowledge_untracked):
            applicable = self._applicable_rules_for_token(token)
            eligible = [rule for rule in applicable if rule.kind is SurfaceKind.UNMETERED_SPEND]
            if applicable and not eligible:
                kinds = ", ".join(sorted({rule.kind.value for rule in applicable}))
                self._invalid_acknowledgment(token, f"classified as {kinds}")
            if eligible:
                if any(rule.source is SurfaceSource.SYNTHETIC_POLICY for rule in eligible):
                    continue
                raw_rules = [
                    rule
                    for rule in eligible
                    if rule.source in {SurfaceSource.RAW, SurfaceSource.BOTH}
                ]
                if not raw_rules:
                    continue
                self._validate_live_acknowledgment(raw_client, token, raw_rules[0])
                continue
            if ":" in token:
                self._invalid_acknowledgment(token, "no applicable conditional rule")
            self._validate_live_acknowledgment(raw_client, token, None)

    def _applicable_rules_for_token(self, token: str) -> tuple[SurfaceRule, ...]:
        applicable: list[SurfaceRule] = []
        for rule in SURFACE_RULES:
            if rule.token != token:
                continue
            sources = (
                (SurfaceSource.RAW, SurfaceSource.WRAPPER)
                if rule.source is SurfaceSource.BOTH
                else (rule.source,)
            )
            if any(
                resolve_surface_rule(
                    context=self._surface_context,
                    path=rule.surface,
                    source=source,
                    condition=rule.condition,
                )
                is rule
                for source in sources
            ):
                applicable.append(rule)
        return tuple(applicable)

    def _validate_live_acknowledgment(
        self,
        raw_client: object,
        token: str,
        expected_rule: SurfaceRule | None,
    ) -> None:
        try:
            _validate_surface_path(token)
        except RuntimeError:
            self._invalid_acknowledgment(token, "not an exact public dotted path")

        value = raw_client
        parts = token.split(".")
        for index, name in enumerate(parts):
            path = ".".join(parts[: index + 1])
            static = self._inspect_static_attribute(value, name)
            if static is None:
                self._invalid_acknowledgment(token, f"path {path!r} is not statically visible")
            descriptor_category, static_return_shape = static
            rule = resolve_surface_rule(
                context=self._surface_context,
                path=path,
                source=SurfaceSource.RAW,
            )
            terminal = index == len(parts) - 1
            if terminal:
                if expected_rule is not None:
                    if rule is not expected_rule:
                        self._invalid_acknowledgment(token, "does not resolve to its exact rule")
                    if not self._shape_matches_before_evaluation(
                        expected_rule,
                        descriptor_category,
                        static_return_shape,
                    ):
                        self._invalid_acknowledgment(token, "does not match its reviewed shape")
                    if self._shape_is_intentionally_unevaluated(
                        expected_rule,
                        descriptor_category,
                    ):
                        return
                    returned = getattr(value, name)
                    if not expected_rule.accepts_shape(
                        AttributeShape(descriptor_category, _return_shape(returned))
                    ):
                        self._invalid_acknowledgment(
                            token,
                            "evaluated attribute does not match its reviewed shape",
                        )
                    return
                returned = getattr(value, name)
                if self._is_guardable_provider_resource(returned):
                    self._invalid_acknowledgment(token, "names a resource container")
                return

            if rule is not None and rule.kind is SurfaceKind.NAMESPACE:
                if not self._shape_matches_before_evaluation(
                    rule,
                    descriptor_category,
                    static_return_shape,
                ):
                    self._invalid_acknowledgment(token, f"namespace {path!r} shape drifted")
                returned = getattr(value, name)
                if not rule.accepts_shape(
                    AttributeShape(descriptor_category, _return_shape(returned))
                ):
                    self._invalid_acknowledgment(token, f"namespace {path!r} shape drifted")
                value = returned
                continue
            if rule is not None and rule.kind is SurfaceKind.UNMETERED_SPEND:
                if not self._shape_matches_before_evaluation(
                    rule,
                    descriptor_category,
                    static_return_shape,
                ):
                    self._invalid_acknowledgment(
                        token,
                        f"unmetered prefix {path!r} shape drifted",
                    )
                returned = getattr(value, name)
                if not self._shape_is_intentionally_unevaluated(
                    rule,
                    descriptor_category,
                ) and not rule.accepts_shape(
                    AttributeShape(descriptor_category, _return_shape(returned))
                ):
                    self._invalid_acknowledgment(
                        token,
                        f"unmetered prefix {path!r} shape drifted",
                    )
                if not self._is_guardable_provider_resource(returned):
                    self._invalid_acknowledgment(
                        token,
                        f"unmetered prefix {path!r} is not a guardable provider resource",
                    )
                value = returned
                continue
            if rule is not None:
                self._invalid_acknowledgment(token, f"prefix {path!r} is not a namespace")
            returned = getattr(value, name)
            if not self._is_guardable_provider_resource(returned):
                self._invalid_acknowledgment(token, f"unknown prefix {path!r} is not guardable")
            value = returned

    def _invalid_acknowledgment(self, token: str, reason: str) -> NoReturn:
        raise ConfigurationError(
            f"invalid acknowledgment token {token!r}: {reason}",
            field="acknowledge_untracked",
        )

    def _inspect_static_attribute(self, value: object, name: str) -> tuple[str, str] | None:
        try:
            static_value = _static_attribute(value, name)
        except AttributeError:
            return None
        descriptor_category = _descriptor_category(static_value)
        return descriptor_category, _static_return_shape(static_value, descriptor_category)

    def _shape_matches_before_evaluation(
        self,
        rule: SurfaceRule,
        descriptor_category: str,
        static_return_shape: str,
    ) -> bool:
        static_shape = AttributeShape(descriptor_category, static_return_shape)
        if rule.accepts_shape(static_shape):
            return True
        return static_return_shape == "unevaluated_descriptor" and any(
            shape.descriptor_category == descriptor_category for shape in rule.expected_shapes
        )

    def _shape_is_intentionally_unevaluated(
        self,
        rule: SurfaceRule,
        descriptor_category: str,
    ) -> bool:
        matching_shapes = tuple(
            shape.return_shape
            for shape in rule.expected_shapes
            if shape.descriptor_category == descriptor_category
        )
        return bool(matching_shapes) and set(matching_shapes) == {"unevaluated_descriptor"}

    def _has_dynamic_attribute_hook(self, value: object) -> bool:
        value_type = type(value)
        if self._inspect_static_attribute(value_type, "__getattr__") is not None:
            return True
        return (
            inspect.getattr_static(value_type, "__getattribute__", object.__getattribute__)
            is not object.__getattribute__
        )

    def _is_guardable_provider_resource(self, value: object) -> bool:
        return _return_shape(value) in _GUARDABLE_RETURN_SHAPES and _belongs_to_client_shape(
            value, self._surface_context.client_shape
        )

    def _guard_resource(self, value: object, path: str) -> _GuardedResource:
        with self._guard_lock:
            cached = self._guarded_resources.get(path)
            if cached is not None:
                return cached
            guarded = _GuardedResource(self, value, path)
            self._guarded_resources[path] = guarded
            return guarded

    def _has_acknowledged_descendant(self, path: str) -> bool:
        prefix = f"{path}."
        return any(token.startswith(prefix) for token in self._config.acknowledge_untracked)

    def _is_exact_raw_response_escape(self, path: str, rule: SurfaceRule) -> bool:
        return (
            rule.capability_scope is CapabilityScope.RAW_RESPONSE
            and rule.token == path
            and path in self._config.acknowledge_untracked
        )

    def _apply_untracked_posture(
        self,
        path: str,
        rule: SurfaceRule | None,
        *,
        honor_acknowledgment: bool = True,
        drifted_from: SurfaceRule | None = None,
    ) -> None:
        token = rule.token if rule is not None else path
        if honor_acknowledgment and (
            token in self._config.acknowledge_untracked or self._has_acknowledged_descendant(path)
        ):
            return
        scope = (
            rule.capability_scope.value
            if rule is not None and rule.capability_scope is not None
            else None
        )
        kind = rule.kind.value if rule is not None else SurfaceKind.UNKNOWN.value
        if rule is not None:
            rule_id = rule.rule_id
        elif drifted_from is not None:
            rule_id = drifted_from.rule_id
        else:
            rule_id = (
                f"unknown:{self._surface_context.client_shape}:{self._surface_context.mode}:"
                f"{self._surface_context.provider}:{path}"
            )
        if self._config.on_unmetered == "allow":
            return
        if self._config.on_unmetered == "warn":
            _warn_contextual_surface_once(
                context=self._surface_context,
                rule_id=rule_id,
                surface=path,
                capability_scope=scope,
                drifted_from_rule_id=(drifted_from.rule_id if drifted_from is not None else None),
            )
            return
        raise UntrackedSpendSurfaceError(
            surface=path,
            token=token,
            provider=self._surface_context.provider,
            client_shape=self._surface_context.client_shape,
            kind=kind,
            capability_scope=scope,
            drifted_from_rule_id=(drifted_from.rule_id if drifted_from is not None else None),
        )

    def _resolve_unknown_value(self, value: object, path: str) -> Any:
        if self._is_guardable_provider_resource(value):
            return self._guard_resource(value, path)
        if self._has_acknowledged_descendant(path):
            raise RuntimeError(f"acknowledged descendant has unguardable prefix: {path}")
        return value

    def _enforce_explicit_surface(
        self,
        path: str,
        *,
        source: SurfaceSource,
        condition: SurfaceCondition | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        """Resolve policy before an explicit wrapper method dispatches."""

        rule = resolve_surface_rule(
            context=self._surface_context,
            path=path,
            source=source,
            condition=condition,
        )
        if rule is None:
            self._apply_untracked_posture(path, None)
            return
        if rule.kind is SurfaceKind.METERED:
            return
        if rule.kind is SurfaceKind.UNMETERED_SPEND:
            self._apply_untracked_posture(path, rule)
            return
        if rule.kind is SurfaceKind.BLOCKED:
            raise ConfigurationError(blocked_reason or rule.reason or "blocked surface", field=path)
        if rule.kind is SurfaceKind.UNSUPPORTED:
            raise UnsupportedSurfaceError(
                surface=path,
                provider=self._surface_context.provider,
            )
        raise RuntimeError(f"non-dispatch surface reached explicit resolver: {path}")

    def _resolve_public_attribute(
        self,
        value: object,
        *,
        name: str,
        path: str,
        source: SurfaceSource,
    ) -> Any:
        """Resolve one public provider attribute before evaluating its descriptor."""

        if name.startswith("_"):
            return getattr(value, name)
        with self._guard_lock:
            cached = self._guarded_resources.get(path)
        if cached is not None:
            return cached

        rule = resolve_surface_rule(
            context=self._surface_context,
            path=path,
            source=source,
        )
        static = self._inspect_static_attribute(value, name)
        if static is None:
            try:
                visible = name in dir(value)
            except Exception:
                visible = False
            if not visible and not self._has_dynamic_attribute_hook(value):
                return getattr(value, name)
            if rule is not None and rule.kind is SurfaceKind.BLOCKED:
                raise ConfigurationError(rule.reason or "blocked surface", field=path)
            if rule is not None and rule.kind is SurfaceKind.UNSUPPORTED:
                raise UnsupportedSurfaceError(
                    surface=path,
                    provider=self._surface_context.provider,
                )
            if rule is not None and rule.kind is SurfaceKind.METERED:
                raise RuntimeError(f"metered surface reached generic resolver: {path}")
            effective_rule = (
                rule if rule is not None and rule.kind is SurfaceKind.UNMETERED_SPEND else None
            )
            if self._has_dynamic_attribute_hook(value):
                effective_rule = None
            self._apply_untracked_posture(path, effective_rule)
            return self._resolve_unknown_value(getattr(value, name), path)

        descriptor_category, static_return_shape = static
        if rule is None:
            self._apply_untracked_posture(path, None)
            return self._resolve_unknown_value(getattr(value, name), path)
        if rule.kind is SurfaceKind.BLOCKED:
            raise ConfigurationError(rule.reason or "blocked surface", field=path)
        if rule.kind is SurfaceKind.UNSUPPORTED:
            raise UnsupportedSurfaceError(
                surface=path,
                provider=self._surface_context.provider,
            )
        if rule.kind is SurfaceKind.METERED:
            raise RuntimeError(f"metered surface reached generic resolver: {path}")
        if rule.kind is SurfaceKind.UNMETERED_SPEND:
            if not self._shape_matches_before_evaluation(
                rule,
                descriptor_category,
                static_return_shape,
            ):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(getattr(value, name), path)
            self._apply_untracked_posture(path, rule)
            returned = getattr(value, name)
            if self._shape_is_intentionally_unevaluated(rule, descriptor_category):
                return self._resolve_unknown_value(returned, path)
            if not rule.accepts_shape(AttributeShape(descriptor_category, _return_shape(returned))):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(returned, path)
            if self._is_exact_raw_response_escape(path, rule):
                return returned
            return self._resolve_unknown_value(returned, path)
        if rule.kind in {SurfaceKind.METADATA, SurfaceKind.INFRASTRUCTURE}:
            if not self._shape_matches_before_evaluation(
                rule,
                descriptor_category,
                static_return_shape,
            ):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(getattr(value, name), path)
            returned = getattr(value, name)
            if not rule.accepts_shape(AttributeShape(descriptor_category, _return_shape(returned))):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(returned, path)
            return returned
        if rule.kind is SurfaceKind.NAMESPACE:
            if not self._shape_matches_before_evaluation(
                rule,
                descriptor_category,
                static_return_shape,
            ):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(getattr(value, name), path)
            returned = getattr(value, name)
            if not rule.accepts_shape(AttributeShape(descriptor_category, _return_shape(returned))):
                self._apply_untracked_posture(
                    path,
                    None,
                    honor_acknowledgment=False,
                    drifted_from=rule,
                )
                return self._resolve_unknown_value(returned, path)
            return self._guard_resource(returned, path)
        raise RuntimeError(f"unsupported surface kind in runtime resolver: {rule.kind.value}")

    def _reset_after_fork_in_child(self) -> None:
        """Replace this client's locks in a forked child.

        Only the locks: latency windows, price hints, breakers, and the
        control-plane breaker are plain state the child legitimately inherits
        (the breakers repair their own internal locks via their own
        registration).
        """
        self._signal_lock = threading.Lock()
        self._breaker_lock = threading.Lock()
        self._guard_lock = threading.Lock()

    def _new_circuit_breaker(self) -> CircuitBreaker:
        """Create a circuit breaker from the configured tuning + jitter."""
        return CircuitBreaker(
            failure_threshold=self._config.circuit_breaker_failure_threshold,
            recovery_timeout=self._config.circuit_breaker_recovery_timeout,
            success_threshold=self._config.circuit_breaker_success_threshold,
            recovery_timeout_jitter=self._config.circuit_breaker_recovery_timeout_jitter,
        )

    def _apply_failover_tuning_directive(self, allowed: bool | None) -> FailoverTuning:
        """Apply the server's tuning decision; return the call's effective tuning.

        ``None`` is advisory-delivery failure or a legacy/missing directive:
        no mutation, but the call still receives ONE coherent snapshot read
        under the breaker-management lock (never a torn unlocked re-read).
        Config mutation and retuning every existing breaker share that lock
        with lazy breaker creation, so existing and newly created breakers
        receive coherent tuning before the current call continues into
        post-check routing.
        """
        if allowed is None:
            with self._breaker_lock:
                return self._tuning_snapshot_locked()

        if allowed:
            effective = dict(self._requested_failover_tuning)
        else:
            effective = {
                name: SolwynConfig.model_fields[name].default for name in _FAILOVER_TUNING_FIELDS
            }

        should_log_suppression = False
        with self._breaker_lock:
            tuning_changed = any(
                getattr(self._config, name) != effective[name] for name in _FAILOVER_TUNING_FIELDS
            )
            if tuning_changed:
                for name, value in effective.items():
                    setattr(self._config, name, value)
                for breaker in self._circuit_breakers.values():
                    breaker.replace_tuning(
                        failure_threshold=self._config.circuit_breaker_failure_threshold,
                        recovery_timeout=self._config.circuit_breaker_recovery_timeout,
                        success_threshold=self._config.circuit_breaker_success_threshold,
                        recovery_timeout_jitter=(
                            self._config.circuit_breaker_recovery_timeout_jitter
                        ),
                    )

            suppresses_requested_tuning = any(
                effective[name] != self._requested_failover_tuning[name]
                for name in _FAILOVER_TUNING_FIELDS
            )
            if (
                not allowed
                and suppresses_requested_tuning
                and not self._failover_tuning_suppression_logged
            ):
                self._failover_tuning_suppression_logged = True
                should_log_suppression = True
            snapshot = self._tuning_snapshot_locked()

        if should_log_suppression:
            logger.warning(
                "Custom failover tuning is unavailable for this plan; SDK defaults applied"
            )
        return snapshot

    def _tuning_snapshot_locked(self) -> FailoverTuning:
        """Read one coherent tuning snapshot. Caller must hold _breaker_lock."""
        return FailoverTuning(
            failover_total_timeout=self._config.failover_total_timeout,
            failover_idempotency=self._config.failover_idempotency,
            same_provider_retries=self._config.same_provider_retries,
            failover_hop_read_timeout=self._config.failover_hop_read_timeout,
        )

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get the circuit breaker for a provider.

        Lazily creates a circuit breaker if one doesn't exist for this provider.
        """
        with self._breaker_lock:
            if provider not in self._circuit_breakers:
                self._circuit_breakers[provider] = self._new_circuit_breaker()
            return self._circuit_breakers[provider]

    def _get_breaker_snapshots(self) -> list[tuple[ProviderName, CircuitBreakerState]]:
        """Return one frozen current-state snapshot per distinct provider.

        Copy the provider/breaker mapping under its lock, then release that lock
        before each breaker acquires its own state lock. This keeps the two lock
        domains independent while allowing providers to be added concurrently.
        """
        with self._breaker_lock:
            breakers = list(self._circuit_breakers.items())
        return [(ProviderName(provider), breaker.get_state()) for provider, breaker in breakers]

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
        # Signal capability gate (PJ-9/P4): the default HealthBasedPolicy
        # ignores latency_p50 and price_hint, so skip the median (lock + copy +
        # statistics.median per provider) and the hint snapshot unless the
        # configured policy declares it consumes them. Unknown injected
        # policies default to True and keep receiving full signals.
        wants_latency = getattr(self._policy, "uses_latency_signal", True)
        wants_price = getattr(self._policy, "uses_price_signal", True)
        if wants_price:
            # Snapshot the price hints once under the lock so every candidate in
            # this selection sees a consistent view (the setter may replace the
            # dict concurrently on another thread).
            with self._signal_lock:
                price_hints = dict(self._last_price_hints)
        else:
            price_hints = {}
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
                    latency_p50=(
                        self.observed_p50(runtime.adapter.name) if wants_latency else None
                    ),
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
        agent_run: _RunContextSnapshot | None = None,
        provider_region: str | None = None,
        modality: Modality = "text",
        media_usage: MediaUsage | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API.

        ``call_id`` is the per-call reconciliation join key. ``possibly_succeeded``
        is the post-send-ambiguous abort flag — left None on every non-abort event.
        ``provider_region`` is the served endpoint's cloud region (Bedrock pricing
        is per model AND region); None for providers without regional pricing.
        ``modality`` is the call modality — ``"text"`` for every chat event; the
        media lifecycle passes the surface's modality (e.g. ``"embedding"``).
        ``media_usage`` carries a non-text surface's non-token quantities; None
        for chat/text events (None-skipped on the wire).
        """
        if not call_id:
            raise RuntimeError("call_id is required for metadata reconciliation")
        agent_run_id, agent_run_name, tags, parent_agent_run_id = (
            _capture_run_context() if agent_run is None else agent_run
        )
        return MetadataEvent(
            model=model,
            provider=ProviderName(provider),
            modality=modality,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_details=token_details,
            media_usage=media_usage,
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
            parent_agent_run_id=parent_agent_run_id,
            agent_run_name=agent_run_name,
            call_id=call_id,
            provider_region=provider_region,
            tags=tags,
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
        agent_run: _RunContextSnapshot | None = None,
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
