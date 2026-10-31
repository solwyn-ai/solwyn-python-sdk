"""Shared sans-I/O logic for Solwyn clients.

Contains _SolwynBase with config, budget logic, metadata formatting,
and candidate selection. No I/O -- sync and async clients inherit
from this and add their own HTTP layer.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
import re
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NoReturn, TypedDict, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from solwyn import _otel
from solwyn._constants import FAILOVER_ERROR_CLASS_MAX_LENGTH, ORDINARY_TOKEN_COUNT_MAX
from solwyn._lifecycle import register_fork_reset
from solwyn._otel import _TraceSnapshot, _validated_trace
from solwyn._privacy import (
    estimate_content_length,
    google_output_bound_view,
    normalize_google_generate_content_kwargs,
    prepare_legacy_google_metering_kwargs,
    with_google_output_cap,
)
from solwyn._routing import (
    CostPolicy,
    HealthBasedPolicy,
    ProviderCandidate,
    RoutingRequest,
    SelectionPolicy,
)
from solwyn._run import _capture_run_context, _fallback_run_span, _RunContextSnapshot
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
    _surface_path_is_reportable,
    _validate_surface_path,
    context_is_declared,
    resolve_surface_rule,
)
from solwyn._token_details import TokenDetails
from solwyn._types import (
    CallStatus,
    CircuitState,
    DenySource,
    FailoverReason,
    MediaUsage,
    MetadataEvent,
    Modality,
    ProviderName,
    VelocityFlag,
)
from solwyn._velocity import VelocityConfig, VelocityMonitor
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerState
from solwyn.config import SolwynConfig
from solwyn.exceptions import (
    ConfigurationError,
    OutputBoundReason,
    OutputBoundUnenforceableError,
    UnsupportedSurfaceError,
    UntrackedSpendSurfaceError,
    UntranslatableRequestError,
)
from solwyn.providers._errors import Disposition, classify_exception

if TYPE_CHECKING:
    from solwyn._registry import ProviderRuntime

logger = logging.getLogger(__name__)

_CallMethod = TypeVar("_CallMethod", bound=Callable[..., Any])


def _with_call_span(method: _CallMethod) -> _CallMethod:
    """Own the entire call lifetime, including preflight and BaseException.

    Streams explicitly transfer ownership before returning. Context attachment
    remains a separate, tightly scoped operation at each I/O seam.
    """
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_call(self: _SolwynBase, *args: Any, **kwargs: Any) -> Any:
            if not self._solwyn_config.otel_spans:
                return await method(self, *args, **kwargs)
            with _otel._start_call_span(parent=_fallback_run_span()) as call_span:
                kwargs["_call_span"] = call_span
                return await method(self, *args, **kwargs)

        return cast("_CallMethod", async_call)

    @functools.wraps(method)
    def sync_call(self: _SolwynBase, *args: Any, **kwargs: Any) -> Any:
        if not self._solwyn_config.otel_spans:
            return method(self, *args, **kwargs)
        with _otel._start_call_span(parent=_fallback_run_span()) as call_span:
            kwargs["_call_span"] = call_span
            return method(self, *args, **kwargs)

    return cast("_CallMethod", sync_call)


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

    The directive writer mutates self._solwyn_config's tuning fields under
    _solwyn_breaker_lock (_apply_failover_tuning_directive); an unlocked mid-call
    re-read of self._solwyn_config can observe a torn mix of old and new tuning
    (PJ-8/R12). The dispatch path therefore consumes tuning ONLY through this
    immutable snapshot, captured once per call under that same lock.
    """

    failover_total_timeout: float
    failover_idempotency: Literal["safe", "never", "always"]
    same_provider_retries: int
    failover_hop_read_timeout: float


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Policy-ordered runtimes plus attribution for a price-driven displacement."""

    runtimes: list[ProviderRuntime]
    cost_routed: bool


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


def _wire_token_quantity(value: int) -> int:
    """Clamp one token quantity into the wire model's validated range.

    Enforcement and lease math stay UNCLAMPED — only the reported artifact is
    bounded. A quantity outside ``[0, ORDINARY_TOKEN_COUNT_MAX]`` raises
    ValidationError at event construction, and every construction site is
    either a denial receipt whose loss is swallowed or a settlement build that
    runs AFTER the provider was paid. A clamped count is always better
    accounting than a destroyed receipt or a crashed paid call.
    """
    return min(max(value, 0), ORDINARY_TOKEN_COUNT_MAX)


# The two halves of FAILOVER_ERROR_CLASS_PATTERN, inverted: everything before
# the first ASCII letter, and every character the pattern's tail does not admit.
# Explicit ASCII ranges, so a non-ASCII letter (which str.isalpha would accept
# and the API would reject) is handled as the foreign character it is.
_ERROR_CLASS_LEADING_NON_LETTERS = re.compile(r"^[^A-Za-z]+")
_ERROR_CLASS_FOREIGN_CHARACTER = re.compile(r"[^A-Za-z0-9_.]")


def _wire_error_class(name: str | None) -> str | None:
    """Map an exception CLASS NAME into the wire model's validated shape.

    Same contract as ``_wire_token_quantity``: only the reported artifact is
    normalized. ``failover_error_class`` is pattern-validated by the API, whose
    ingest route rejects the WHOLE batch when one event fails, and by the SDK's
    own model, which would otherwise raise inside an error handler and mask the
    provider's exception. Real classes fall outside the shape — grpc's
    ``_InactiveRpcError``, any private or generated class — so the name is
    normalized rather than trusted:

    * leading characters up to the first ASCII letter are stripped
      (``_InactiveRpcError`` -> ``InactiveRpcError``);
    * each remaining character outside ``[A-Za-z0-9_.]`` becomes ``_``;
    * the result is cut to ``FAILOVER_ERROR_CLASS_MAX_LENGTH``;
    * ``None`` passes through, and a name with no ASCII letter — or a non-string
      — is ``None``, which the None-skipping serializer drops from the wire.

    Total and deterministic: never raises. It must only ever be handed a class
    name; it is not a scrubber and gives no licence to pass exception text.
    """
    if not isinstance(name, str):
        return None
    stripped = _ERROR_CLASS_LEADING_NON_LETTERS.sub("", name, count=1)
    bounded = stripped[:FAILOVER_ERROR_CLASS_MAX_LENGTH]
    return _ERROR_CLASS_FOREIGN_CHARACTER.sub("_", bounded) or None


# CostPolicy is inert when the request carried NO hint map at all
# (``price_hints is None`` — a legacy/lease/outage response that never priced
# this chain): it degrades to health-based ordering. Warn ONCE per process when
# that fallback is in effect so the no-op is visible without logging on every
# routed call. An explicit empty map (``{}`` — the server priced the chain and
# found nothing priceable) is a deliberate server answer and degrades silently.
# Lock-guarded because the sync client is multi-threaded.
_cost_policy_inactive_warned = False
_cost_policy_warn_lock = threading.Lock()


def _warn_cost_policy_inactive_once() -> None:
    """Emit the CostPolicy-inert warning at most once per process."""
    global _cost_policy_inactive_warned
    with _cost_policy_warn_lock:
        if _cost_policy_inactive_warned:
            return
        _cost_policy_inactive_warned = True
    logger.warning(
        "CostPolicy selected but this budget check carried no price hints; using health-based order"
    )


# Per-process observations contain only structural capability data. The registry
# doubles as the warning latch: every hit increments its counter, while only a
# newly inserted warn-posture key emits the contextual warning.
class _UntrackedSurfaceObservation(TypedDict):
    occurrences: int
    first_seen_at: datetime
    last_seen_at: datetime
    rule_kind: Literal["unmetered_spend", "unknown"]
    capability_scope: str | None
    posture: Literal["warn", "allow"]
    # Local-only latch state; reporters serialize only the wire fields above.
    warning_emitted: bool


_UntrackedSurfaceKey = tuple[str, str, str, str]
_UntrackedObservationNotifier = Callable[..., None]

_UNTRACKED_SURFACE_LIMIT = 512
_untracked_surface_observations: dict[_UntrackedSurfaceKey, _UntrackedSurfaceObservation] = {}
_warn_limit_reached = False
_spend_surface_warn_lock = threading.Lock()


def _reset_warn_locks_after_fork_in_child() -> None:
    global _spend_surface_warn_lock, _cost_policy_warn_lock
    _spend_surface_warn_lock = threading.Lock()
    _cost_policy_warn_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_warn_locks_after_fork_in_child)


def _reset_unmetered_spend_warnings() -> None:
    """Clear per-process untracked observations and warning state for tests."""
    global _warn_limit_reached
    with _spend_surface_warn_lock:
        _untracked_surface_observations.clear()
        _warn_limit_reached = False


def _record_untracked_surface_observation(
    *,
    context: SurfaceContext,
    surface: str,
    rule_kind: Literal["unmetered_spend", "unknown"],
    capability_scope: str | None,
    posture: Literal["warn", "allow"],
    notifier: _UntrackedObservationNotifier | None = None,
) -> Literal["warn", "silent", "full"]:
    """Record one content-free observation and return its registry status."""

    _validate_surface_path(surface)
    reportable = _surface_path_is_reportable(surface)
    observation_key = (
        context.provider,
        context.client_shape,
        context.mode,
        surface,
    )
    with _spend_surface_warn_lock:
        seen_at = datetime.now(UTC)
        observation = _untracked_surface_observations.get(observation_key)
        if observation is not None:
            should_warn = posture == "warn" and not observation["warning_emitted"]
            observation["occurrences"] += 1
            observation["first_seen_at"] = min(observation["first_seen_at"], seen_at)
            observation["last_seen_at"] = max(observation["last_seen_at"], seen_at)
            observation["rule_kind"] = rule_kind
            observation["capability_scope"] = capability_scope
            observation["posture"] = posture
            if should_warn:
                observation["warning_emitted"] = True
                registry_status: Literal["warn", "silent", "full"] = "warn"
            else:
                registry_status = "silent"
        elif len(_untracked_surface_observations) >= _UNTRACKED_SURFACE_LIMIT:
            registry_status = "full"
        else:
            _untracked_surface_observations[observation_key] = {
                "occurrences": 1,
                "first_seen_at": seen_at,
                "last_seen_at": seen_at,
                "rule_kind": rule_kind,
                "capability_scope": capability_scope,
                "posture": posture,
                "warning_emitted": posture == "warn",
            }
            registry_status = "warn" if posture == "warn" else "silent"
    # Never invoke reporter code under the process-global warning lock. The
    # notifier performs only origin-owned in-memory bookkeeping + wakeup; its
    # reporter thread/task owns every network operation. A path the wire cannot
    # carry stays local: counted and warned here, never handed to a reporter.
    if notifier is not None and reportable:
        # Advisory bookkeeping must never affect the provider call. The origin
        # reporter has its own bounded ledger, so global warning-registry
        # saturation is not a reason to suppress this callback.
        with contextlib.suppress(Exception):
            notifier(
                context=context,
                surface=surface,
                rule_kind=rule_kind,
                capability_scope=capability_scope,
                posture=posture,
                seen_at=seen_at,
            )
    return registry_status


def _warn_contextual_surface_once(
    *,
    context: SurfaceContext,
    surface: str,
    rule_kind: Literal["unmetered_spend", "unknown"],
    capability_scope: str | None,
    drifted_from_rule_id: str | None = None,
    notifier: _UntrackedObservationNotifier | None = None,
) -> None:
    """Record every hit and warn once per provider/client-shape/mode/surface."""

    global _warn_limit_reached
    registry_status = _record_untracked_surface_observation(
        context=context,
        surface=surface,
        rule_kind=rule_kind,
        capability_scope=capability_scope,
        posture="warn",
        notifier=notifier,
    )
    wire_ineligible = not _surface_path_is_reportable(surface)
    if registry_status == "silent":
        return
    warn_limit_reached_now = False
    if registry_status == "full":
        with _spend_surface_warn_lock:
            if _warn_limit_reached:
                return
            _warn_limit_reached = True
            warn_limit_reached_now = True
    if warn_limit_reached_now:
        logger.warning(
            "Untracked-surface warning limit (%d) reached; further distinct "
            "surfaces will not be individually reported this process.",
            _UNTRACKED_SURFACE_LIMIT,
        )
        return
    if wire_ineligible:
        logger.warning(
            "Provider '%s' client shape '%s' exposes an untracked public surface "
            "outside advisory reporting limits; no budget check and no cost event "
            "will be emitted, and no advisory report will be sent.",
            context.provider,
            context.client_shape,
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
        "n",
        "extra_body",
        "generation_config",
    }
)
# The only extra_body fields that change how much output a call can produce.
# The OpenAI and Anthropic SDKs merge extra_body over named arguments.
_EXTRA_BODY_BOUND_KEYS = frozenset(
    {"max_tokens", "max_completion_tokens", "n", "enable_thinking", "thinking_budget"}
)
# Under hard_deny a body extension may carry ONLY keys the bound accounts for
# (or that cannot change output length or the model). Anything else, `model`
# included, is refused: an allowlist, because blocklists missed aliases.
_OPENAI_EXTRA_BODY_ALLOWED = frozenset({"max_tokens", "max_completion_tokens", "n"})
_QWEN_EXTRA_BODY_ALLOWED = _OPENAI_EXTRA_BODY_ALLOWED | {"enable_thinking", "thinking_budget"}
_ANTHROPIC_EXTRA_BODY_ALLOWED = frozenset({"max_tokens"})
# Anthropic-style thinking: the API requires budget_tokens < maxTokens.
_BEDROCK_MODEL_FIELDS_ALLOWED = frozenset({"thinking"})
_GOOGLE_CAP_KEYS = ("max_output_tokens", "maxOutputTokens")
_GOOGLE_COUNT_KEYS = ("candidate_count", "candidateCount")
# First Azure OpenAI api-version that accepts max_completion_tokens.
_AZURE_MAX_COMPLETION_TOKENS_SINCE = "2024-09-01"
_OUTPUT_CAP_KEYS_BY_DIALECT = {
    ProviderName.OPENAI.value: frozenset({"max_tokens", "max_completion_tokens"}),
    ProviderName.ANTHROPIC.value: frozenset({"max_tokens"}),
    ProviderName.GOOGLE.value: frozenset({"config", "generation_config", "max_output_tokens"}),
    ProviderName.BEDROCK.value: frozenset({"inferenceConfig"}),
}


def _positive_output_cap(value: object) -> int | None:
    """Return a usable token cap without coercing caller-owned objects."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _nested_value(value: object, *keys: str) -> object:
    """Read the first present scalar among ``keys`` from a dict or SDK object.

    Google config objects may also carry a system instruction, and Bedrock
    request dictionaries carry messages beside ``inferenceConfig``. Reading
    only the named scalar keeps lease sizing outside content-privileged code.
    """
    for key in keys:
        nested = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        if nested is not None:
            return nested
    return None


def _nested_output_cap(value: object, *keys: str) -> int | None:
    """Read one cap from a dict or SDK config object without copying config."""
    return _positive_output_cap(_nested_value(value, *keys))


def _responses_output_bound(kwargs: dict[str, object], default_bound: int) -> int:
    """Return the Responses output bound used for conservative lease sizing.

    Responses uses ``max_output_tokens``, not the chat token aliases. Calls
    without a usable cap use the configured conservative fallback.
    """
    return _positive_output_cap(kwargs.get("max_output_tokens")) or default_bound


def _dialect_output_cap(dialect: str, kwargs: dict[str, object]) -> int | None:
    """Return the effective native cap in one dialect-shaped kwargs mapping."""
    if dialect == ProviderName.GOOGLE.value:
        nested = _nested_output_cap(kwargs.get("config"), *_GOOGLE_CAP_KEYS)
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
    copied = {key: value for key, value in params.items() if key in _OUTPUT_CAP_KEYS}
    extra = copied.get("extra_body")
    if isinstance(extra, dict):
        copied["extra_body"] = {
            key: value for key, value in extra.items() if key in _EXTRA_BODY_BOUND_KEYS
        }
    return copied


class _UnboundableOutput(Exception):
    """Internal: a hop's possible output cannot be held to a bound."""

    def __init__(self, reason: OutputBoundReason) -> None:
        super().__init__(reason)
        self.reason: OutputBoundReason = reason


_OUTPUT_BOUND_HINTS: dict[OutputBoundReason, str] = {
    "malformed_cap_container": (
        "Pass the request's output settings (Google config, Bedrock inferenceConfig, "
        "or a request body extension) as an object so Solwyn can read the output cap."
    ),
    "malformed_candidate_count": (
        "Set the candidate count (n or candidateCount) to a positive integer, or leave it unset."
    ),
    "reasoning_without_budget": (
        "Reasoning is enabled with no reasoning budget, and this provider's output cap "
        "does not include reasoning. Set a positive thinking budget, turn reasoning off, "
        "or use a non-reasoning model."
    ),
    "unsupported_override": (
        "Remove keys Solwyn cannot account for from request body extensions (body "
        "overrides, provider options, extraBody, or additionalModelRequestFields), "
        "and set output limits with the request's own cap parameters."
    ),
    "azure_api_version": (
        "Set max_tokens or max_completion_tokens on the call, or use an Azure OpenAI "
        "api-version that supports max_completion_tokens."
    ),
    "bound_exceeded": (
        "Set the output cap and candidate count on the call itself so the request "
        "fits the output bound Solwyn priced."
    ),
}


def _merged_body(
    kwargs: dict[str, object],
    *,
    strict: bool,
    allowed: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """The body the OpenAI and Anthropic SDKs send: extra_body keys win.

    Strict mode also enforces the hard_deny allowlist: an extra_body key
    outside ``allowed`` (which these SDKs send verbatim) refuses the call.
    """
    extra = kwargs.get("extra_body")
    if extra is None:
        return kwargs
    if not isinstance(extra, dict):
        if strict:
            raise _UnboundableOutput("malformed_cap_container")
        return kwargs
    if strict and not allowed.issuperset(extra):
        raise _UnboundableOutput("unsupported_override")
    return {**kwargs, **extra}


def _candidate_count(value: object, *, strict: bool) -> int:
    """How many candidates one request generates; absent means one."""
    if value is None:
        return 1
    count = _positive_output_cap(value)
    if count is None:
        if strict:
            raise _UnboundableOutput("malformed_candidate_count")
        return 1
    return count


def _reasoning_outside_cap(
    provider: str, model: str, body: dict[str, object], *, strict: bool
) -> int:
    """Reasoning tokens a provider generates beyond its output cap.

    Qwen's ``max_tokens`` excludes thinking, and thinking is on by default:
    unless ``enable_thinking`` is False the call can also spend its whole
    ``thinking_budget``, which must be set. xAI's cap covers visible output
    only and has no reasoning budget, so a reasoning model cannot be bounded.
    """
    if provider == ProviderName.XAI.value:
        effective_model = body.get("model")
        if not isinstance(effective_model, str):
            effective_model = model
        if strict and "non-reasoning" not in effective_model:
            raise _UnboundableOutput("reasoning_without_budget")
        return 0
    if provider != ProviderName.QWEN.value:
        return 0
    if body.get("enable_thinking") is False:
        return 0
    budget = _positive_output_cap(body.get("thinking_budget"))
    if budget is None:
        if strict:
            raise _UnboundableOutput("reasoning_without_budget")
        return 0
    return budget


def _is_config_container(value: object) -> bool:
    """True for a mapping or an SDK config object; False for scalars and arrays."""
    if isinstance(value, dict):
        return True
    if isinstance(value, (str, bytes, list, tuple, int, float, bool)):
        return False
    return isinstance(getattr(value, "__dict__", None), dict)


_GOOGLE_REQUEST_CAP_KEYS = ("config", "generation_config", "max_output_tokens")


def _google_bound_native(*layers: dict[str, object]) -> dict[str, object]:
    """Google cap fields merged across layers exactly as dispatch merges them.

    Lenient: a shape dispatch would refuse sizes at the default, because that
    call never reaches a provider.
    """
    google_layers = [
        {key: value for key, value in layer.items() if key in _GOOGLE_REQUEST_CAP_KEYS}
        for layer in layers
    ]
    try:
        return google_output_bound_view(*google_layers)
    except UntranslatableRequestError:
        return {}


def _google_body_extension(http_options: object) -> bool:
    """True when Google http_options carries a non-empty request body extension.

    google-genai merges ``extra_body`` into the serialized request after
    normalizing its keys, so no key in it can be read safely: under hard_deny
    any content refuses, whatever its spelling.
    """
    extra = _nested_value(http_options, "extra_body", "extraBody")
    # google-genai validates extra_body as a dict or None, so a non-empty
    # mapping is the only shape that can reach the wire.
    return isinstance(extra, Mapping) and len(extra) > 0


def _hop_output_capacity(
    dialect: str,
    provider: str,
    model: str,
    kwargs: dict[str, object],
    *,
    default_bound: int,
    strict: bool,
) -> int:
    """Most output tokens one dialect-shaped hop request can produce.

    ``(cap + reasoning outside the cap) x candidate count``. Lenient sizing
    (``strict=False``) prices the budget check: a missing cap counts as
    ``default_bound`` and a malformed field falls back to one candidate and no
    extra reasoning, because the strict pre-send check refuses those later.
    Strict mode raises ``_UnboundableOutput`` instead, and treats a missing cap
    as unbounded.
    """
    cap: int | None = None
    count = 1
    reasoning = 0
    if dialect == ProviderName.OPENAI.value:
        allowed = (
            _QWEN_EXTRA_BODY_ALLOWED
            if provider == ProviderName.QWEN.value
            else _OPENAI_EXTRA_BODY_ALLOWED
        )
        body = _merged_body(kwargs, strict=strict, allowed=allowed)
        caps = [
            value
            for value in (
                _positive_output_cap(body.get("max_completion_tokens")),
                _positive_output_cap(body.get("max_tokens")),
            )
            if value is not None
        ]
        cap = max(caps) if caps else None
        count = _candidate_count(body.get("n"), strict=strict)
        reasoning = _reasoning_outside_cap(provider, model, body, strict=strict)
    elif dialect == ProviderName.ANTHROPIC.value:
        body = _merged_body(kwargs, strict=strict, allowed=_ANTHROPIC_EXTRA_BODY_ALLOWED)
        cap = _positive_output_cap(body.get("max_tokens"))
    elif dialect == ProviderName.GOOGLE.value:
        config = kwargs.get("config")
        if config is not None and not _is_config_container(config):
            if strict:
                raise _UnboundableOutput("malformed_cap_container")
            config = None
        if strict and _google_body_extension(_nested_value(config, "http_options", "httpOptions")):
            raise _UnboundableOutput("unsupported_override")
        cap = _dialect_output_cap(dialect, {**kwargs, "config": config})
        count = _candidate_count(_nested_value(config, *_GOOGLE_COUNT_KEYS), strict=strict)
    elif dialect == ProviderName.BEDROCK.value:
        inference = kwargs.get("inferenceConfig")
        if inference is not None and not isinstance(inference, dict):
            if strict:
                raise _UnboundableOutput("malformed_cap_container")
            inference = None
        fields = kwargs.get("additionalModelRequestFields")
        if strict and fields is not None:
            if not isinstance(fields, dict):
                raise _UnboundableOutput("malformed_cap_container")
            if not _BEDROCK_MODEL_FIELDS_ALLOWED.issuperset(fields):
                raise _UnboundableOutput("unsupported_override")
        cap = _nested_output_cap(inference, "maxTokens")
    if cap is None:
        if strict:
            raise _UnboundableOutput("bound_exceeded")
        cap = default_bound
    return (cap + reasoning) * count


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
    if provider == ProviderName.AZURE_OPENAI.value:
        # An Azure model is a deployment name the caller chose, so it says
        # nothing about the model family. Never weaken a caller's
        # max_completion_tokens (which counts reasoning) to max_tokens.
        return params
    if "max_completion_tokens" not in params:
        return params
    normalized = dict(params)
    value = normalized.pop("max_completion_tokens")
    normalized["max_tokens"] = value
    return normalized


_OPENAI_CAP_KEYS = ("max_tokens", "max_completion_tokens")


def _merge_openai_cap_layers(*layers: dict[str, object]) -> dict[str, object]:
    """Merge normalized OpenAI-dialect layers so only one cap key survives.

    Each layer is already normalized to its target's key, but an Azure layer
    keeps the caller's own key. When a higher layer sets either cap key, it
    replaces every lower layer's cap, so a default max_tokens can never ride
    beside a per-call max_completion_tokens.
    """
    merged: dict[str, object] = {}
    for layer in layers:
        if any(key in layer for key in _OPENAI_CAP_KEYS):
            for key in _OPENAI_CAP_KEYS:
                merged.pop(key, None)
        merged.update(layer)
    return merged


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
    unbounded hop contributes the configured conservative fallback. A hop's
    bound is its cap plus any reasoning its provider generates outside the
    cap, times its candidate count (``_hop_output_capacity``).
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
                native = _merge_openai_cap_layers(
                    _normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_global_defaults,
                    ),
                    _normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_entry_defaults,
                    ),
                    _normalized_openai_output_cap_layer(
                        runtime.adapter.name,
                        target_model,
                        provider_kwargs,
                    ),
                )
            elif runtime.adapter.dialect == ProviderName.GOOGLE.value:
                native = _google_bound_native(
                    provider_global_defaults, provider_entry_defaults, provider_kwargs
                )
            else:
                native = {
                    **provider_global_defaults,
                    **provider_entry_defaults,
                    **provider_kwargs,
                }
            capacity = _hop_output_capacity(
                runtime.adapter.dialect,
                runtime.adapter.name,
                target_model,
                native,
                default_bound=default_bound,
                strict=False,
            )
        elif primary.adapter.dialect == runtime.adapter.dialect:
            target_name = runtime.adapter.name
            native = _merge_openai_cap_layers(
                _normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_global_defaults,
                ),
                _normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_entry_defaults,
                ),
                _normalized_openai_output_cap_layer(
                    target_name,
                    runtime.entry.model,
                    provider_kwargs,
                ),
            )
            if runtime.adapter.dialect == ProviderName.GOOGLE.value:
                native = _google_bound_native(
                    provider_global_defaults, provider_entry_defaults, provider_kwargs
                )
            # extra_body is endpoint-scoped: a cross-provider hop drops the
            # caller's and sends only the target entry's own (see
            # _build_hop_kwargs).
            native.pop("extra_body", None)
            if "extra_body" in provider_entry_defaults:
                native["extra_body"] = provider_entry_defaults["extra_body"]
            capacity = _hop_output_capacity(
                runtime.adapter.dialect,
                target_name,
                runtime.entry.model,
                native,
                default_bound=default_bound,
                strict=False,
            )
        else:
            source_defaults = _source_output_cap_defaults(
                primary.adapter.dialect,
                provider_entry_defaults,
            )
            if primary.adapter.dialect == ProviderName.OPENAI.value:
                source_model = cast(str, kwargs["model"])
                source_native = _merge_openai_cap_layers(
                    _normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        provider_global_defaults,
                    ),
                    _normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        source_defaults,
                    ),
                    _normalized_openai_output_cap_layer(
                        primary.adapter.name,
                        source_model,
                        provider_kwargs,
                    ),
                )
            elif primary.adapter.dialect == ProviderName.GOOGLE.value:
                source_native = _google_bound_native(
                    provider_global_defaults, source_defaults, provider_kwargs
                )
            else:
                source_native = {
                    **provider_global_defaults,
                    **source_defaults,
                    **provider_kwargs,
                }
            capacity = _hop_output_capacity(
                primary.adapter.dialect,
                primary.adapter.name,
                cast(str, kwargs["model"]),
                source_native,
                default_bound=default_bound,
                strict=False,
            )

        bounds.append(capacity)

    return max(bounds, default=default_bound)


def _azure_accepts_max_completion_tokens(client: object) -> bool:
    """True when an Azure OpenAI client's api-version accepts max_completion_tokens.

    Dated versions from 2024-09-01 on (preview or GA) accept it; so do the v1
    API's ``v1``/``preview``/``latest`` labels and a client with no api-version
    at all (a plain OpenAI client on an Azure v1 base URL).
    """
    version = getattr(client, "_api_version", None)
    if version is None:
        return True
    if not isinstance(version, str):
        return False
    if version in ("v1", "preview", "latest"):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(-preview)?", version) is None:
        return False
    return version[:10] >= _AZURE_MAX_COMPLETION_TOKENS_SINCE


def _google_view_strict(kwargs: dict[str, object]) -> dict[str, Any]:
    """The Google bound view of one final request; a refused shape refuses the call."""
    try:
        return google_output_bound_view(kwargs)
    except UntranslatableRequestError:
        raise _UnboundableOutput("malformed_cap_container") from None


def _google_normalized_strict(kwargs: dict[str, object]) -> dict[str, Any]:
    """One modern Google request with aliases folded into config, as the adapter sends it."""
    try:
        return normalize_google_generate_content_kwargs(kwargs, target_shape="google_genai")
    except UntranslatableRequestError:
        raise _UnboundableOutput("malformed_cap_container") from None


def _legacy_google_view(runtime: ProviderRuntime, kwargs: dict[str, object]) -> dict[str, Any]:
    """Project a legacy Google request with its GenerativeModel constructor defaults."""
    try:
        return prepare_legacy_google_metering_kwargs(runtime.sdk_client, kwargs)
    except UntranslatableRequestError:
        raise _UnboundableOutput("malformed_cap_container") from None


def _with_default_output_cap(
    *,
    runtime: ProviderRuntime,
    model: str,
    kwargs: dict[str, object],
    surface: str,
    default_bound: int,
) -> dict[str, object]:
    """Give an uncapped hop the output cap its reservation already assumed.

    ``_effective_output_bound`` charges an uncapped hop ``default_bound``
    output tokens per candidate, on the lease and the per-call check alike.
    Called only for a call admitted under hard_deny (via
    ``_hard_deny_hop_request``): a final provider request with no positive cap
    gets ``default_bound`` in its dialect's native cap field. A positive caller
    or default cap is never touched; a non-positive or malformed one counts as
    absent, exactly as the bound resolver reads it. The OpenAI and Anthropic
    cap is read from the body the SDK sends, with extra_body merged over named
    arguments. Runs after translation, so a cross-dialect hop that refuses an
    uncapped source still refuses. Returns ``kwargs`` itself when nothing
    changes, else a shallow copy. Raises ``_UnboundableOutput`` when the cap
    cannot be added (an Azure api-version without max_completion_tokens).
    """
    dialect = runtime.adapter.dialect
    if surface == "responses":
        if _positive_output_cap(kwargs.get("max_output_tokens")) is not None:
            return kwargs
        return {**kwargs, "max_output_tokens": default_bound}
    if dialect == ProviderName.GOOGLE.value:
        shape = _client_shape(runtime.sdk_client, dialect)
        effective: dict[str, Any]
        if shape == "google_generativeai":
            # The legacy GenerativeModel may carry its own constructor cap.
            effective = _legacy_google_view(runtime, kwargs)
        else:
            # Fold generation_config and other aliases into config first, as
            # the adapter does before sending, so an aliased explicit cap is
            # seen and never replaced.
            kwargs = _google_normalized_strict(kwargs)
            effective = _google_view_strict(kwargs)
        if _dialect_output_cap(dialect, effective) is not None:
            return kwargs
        return cast(
            "dict[str, object]",
            with_google_output_cap(
                kwargs,
                default_bound,
                target_shape=(
                    "google_generativeai" if shape == "google_generativeai" else "google_genai"
                ),
            ),
        )
    if dialect in (ProviderName.OPENAI.value, ProviderName.ANTHROPIC.value):
        if _dialect_output_cap(dialect, _merged_body(kwargs, strict=False)) is not None:
            return kwargs
    elif _dialect_output_cap(dialect, kwargs) is not None:
        return kwargs
    if dialect == ProviderName.BEDROCK.value:
        inference = kwargs.get("inferenceConfig")
        if inference is None:
            return {**kwargs, "inferenceConfig": {"maxTokens": default_bound}}
        if not isinstance(inference, dict):
            # Malformed container: the strict pre-send check refuses it.
            return kwargs
        return {**kwargs, "inferenceConfig": {**inference, "maxTokens": default_bound}}
    if dialect == ProviderName.OPENAI.value:
        uncapped = {
            key: value
            for key, value in kwargs.items()
            if key not in ("max_tokens", "max_completion_tokens")
        }
        if runtime.adapter.name == ProviderName.AZURE_OPENAI.value:
            # The deployment name says nothing about the model family, so
            # always use the key every current model accepts.
            if not _azure_accepts_max_completion_tokens(runtime.sdk_client):
                raise _UnboundableOutput("azure_api_version")
            return {**uncapped, "max_completion_tokens": default_bound}
        return {
            **uncapped,
            **_normalized_openai_output_cap_layer(
                runtime.adapter.name, model, {"max_tokens": default_bound}
            ),
        }
    if dialect == ProviderName.ANTHROPIC.value:
        return {**kwargs, "max_tokens": default_bound}
    return kwargs


def _final_output_capacity(
    *,
    runtime: ProviderRuntime,
    model: str,
    kwargs: dict[str, object],
    surface: str,
) -> int:
    """Most output the final hop request can produce, or raise ``_UnboundableOutput``."""
    if surface == "responses":
        # Nothing in a Responses body extension is folded into the bound.
        _merged_body(kwargs, strict=True)
        cap = _positive_output_cap(kwargs.get("max_output_tokens"))
        if cap is None:
            raise _UnboundableOutput("bound_exceeded")
        return cap
    dialect = runtime.adapter.dialect
    view: dict[str, Any] = kwargs
    if dialect == ProviderName.GOOGLE.value:
        shape = _client_shape(runtime.sdk_client, dialect)
        if shape == "google_generativeai":
            view = _legacy_google_view(runtime, kwargs)
        else:
            view = _google_view_strict(kwargs)
            # Client-level http_options.extra_body is merged into every request
            # at serialization and would replace the generation config.
            options = getattr(
                getattr(runtime.sdk_client, "_api_client", None), "_http_options", None
            )
            if _google_body_extension(options):
                raise _UnboundableOutput("unsupported_override")
    return _hop_output_capacity(
        dialect, runtime.adapter.name, model, view, default_bound=0, strict=True
    )


def _hard_deny_hop_request(
    *,
    runtime: ProviderRuntime,
    model: str,
    kwargs: dict[str, object],
    surface: str,
    default_bound: int,
    reserved_bound: int,
) -> dict[str, object]:
    """Hold one hop of a hard_deny call to the output bound its check priced.

    Caps an uncapped request at ``default_bound``, then reads the possible
    output of the request the provider would actually receive. When that
    cannot be bounded, or exceeds ``reserved_bound``, the call is refused
    locally with ``OutputBoundUnenforceableError`` before any provider I/O.
    """
    reason: OutputBoundReason
    try:
        capped = _with_default_output_cap(
            runtime=runtime,
            model=model,
            kwargs=kwargs,
            surface=surface,
            default_bound=default_bound,
        )
        capacity = _final_output_capacity(
            runtime=runtime, model=model, kwargs=capped, surface=surface
        )
    except _UnboundableOutput as exc:
        reason = exc.reason
    else:
        if capacity <= reserved_bound:
            return capped
        reason = "bound_exceeded"
    raise OutputBoundUnenforceableError(
        provider=runtime.adapter.name, reason=reason, hint=_OUTPUT_BOUND_HINTS[reason]
    )


@dataclass(frozen=True)
class MediaSurfaceSpec:
    """Per-surface wiring for the ``_media_call`` lifecycle.

    A non-chat surface (decisions, embeddings, images, audio, video) is fully
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
    - ``estimate_input_length``: pre-flight request character count. Defaults to
      the shared content recognizer; surfaces with other input shapes supply a
      length-only privacy helper.
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
    estimate_input_length: Callable[[dict[str, Any]], int] = estimate_content_length

    def usage_unmeasured(self, token_details: TokenDetails | None) -> bool:
        """Text estimates cannot prove output spend; retain the lease's bound."""
        return self.modality == "text" and (token_details is None or token_details.is_estimated)

    def dispatched_spend_unknown(self, exc: BaseException, *, dispatched: bool) -> bool:
        """Text leases retain authority for ambiguous dispatch errors and interrupts."""
        if self.modality != "text" or not dispatched:
            return False
        return (
            not isinstance(exc, Exception)
            or classify_exception(exc) is Disposition.POST_SEND_AMBIGUOUS
        )


def _validate_decisions_kwargs(kwargs: Mapping[str, object]) -> None:
    """Keep OpenAI extra-body precedence from bypassing Decisions metering."""
    extra_body = kwargs.get("extra_body")
    if (
        isinstance(extra_body, Mapping)
        and {"model", "input", "questions", "stream"} & extra_body.keys()
    ):
        raise ConfigurationError(
            "Pass Decisions model, input, and questions as top-level arguments; "
            "metering fields and stream are not supported in extra_body.",
            field="extra_body",
        )


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


def _client_shape(client: object, _dialect: str) -> str:
    """Return the provider-independent structural client-shape identity.

    Shape comes from the SDK object's module/class, never the selected adapter.
    This distinction is load-bearing for explicit provider pins: an Anthropic
    client pinned as OpenAI must form an undeclared OpenAI/Anthropic context
    instead of being mislabeled ``openai_sdk`` from the adapter dialect.
    """

    module = getattr(type(client), "__module__", "")
    class_name = getattr(type(client), "__name__", "")
    if (module == "together" or module.startswith("together.")) and class_name in {
        "Together",
        "AsyncTogether",
    }:
        return "native_together"
    if module == "openai" or module.startswith("openai."):
        return "openai_sdk"
    if module == "anthropic" or module.startswith("anthropic."):
        return "anthropic_sdk"
    if "google.generativeai" in module:
        return "google_generativeai"
    if module == "google.genai" or module.startswith("google.genai."):
        return "google_genai"
    if "aiobotocore" in module:
        return "bedrock_aioboto3"
    if "botocore" in module:
        return "bedrock_boto3"
    return "undeclared_sdk"


def _client_mode(client: object, client_shape: str) -> Literal["sync", "async"] | None:
    """Return a known SDK client's inherent mode, or ``None`` for duck types."""

    class_name = getattr(type(client), "__name__", "")
    if client_shape == "openai_sdk":
        api_mode = _api_client_base_mode(client, sdk_module="openai")
        if api_mode is not None:
            return api_mode
        if class_name in {"AsyncOpenAI", "AsyncAzureOpenAI"}:
            return "async"
        if class_name in {"OpenAI", "AzureOpenAI"}:
            return "sync"
    if client_shape == "native_together":
        return "async" if class_name == "AsyncTogether" else "sync"
    if client_shape == "anthropic_sdk":
        api_mode = _api_client_base_mode(client, sdk_module="anthropic")
        if api_mode is not None:
            return api_mode
        if class_name == "AsyncAnthropic":
            return "async"
        if class_name == "Anthropic":
            return "sync"
    if client_shape == "google_generativeai":
        return "sync"
    if client_shape == "google_genai":
        if class_name == "AsyncClient":
            return "async"
        if class_name == "Client":
            return "sync"
    if client_shape == "bedrock_aioboto3":
        return "async"
    if client_shape == "bedrock_boto3":
        return "sync"
    return None


def _api_client_base_mode(
    client: object,
    *,
    sdk_module: str,
) -> Literal["sync", "async"] | None:
    """Classify an official Stainless client from its SDK-owned base class."""

    base_module = f"{sdk_module}._base_client"
    for base in getattr(type(client), "__mro__", ()):
        if getattr(base, "__module__", "") != base_module:
            continue
        if getattr(base, "__name__", "") == "AsyncAPIClient":
            return "async"
        if getattr(base, "__name__", "") == "SyncAPIClient":
            return "sync"
    return None


def _is_bedrock_runtime_client(client: object) -> bool:
    """Return whether a botocore-shaped client targets Bedrock Runtime."""

    meta = getattr(client, "meta", None)
    service_model = getattr(meta, "service_model", None)
    return getattr(service_model, "service_name", None) == "bedrock-runtime"


def _validate_surface_context(
    context: SurfaceContext,
    *,
    pinned_client: object | None = None,
) -> None:
    """Reject a provider client/mode pairing absent from the reviewed rules."""

    if pinned_client is not None:
        if context.provider == "bedrock" and not _is_bedrock_runtime_client(pinned_client):
            raise ConfigurationError(
                "unsupported provider client pairing: provider 'bedrock' requires "
                "a botocore/aiobotocore bedrock-runtime client",
                field="client",
            )
        client_mode = _client_mode(pinned_client, context.client_shape)
        if client_mode is not None and client_mode != context.mode:
            mode_label = "asynchronous" if client_mode == "async" else "synchronous"
            wrapper = "AsyncSolwyn" if client_mode == "async" else "Solwyn"
            raise ConfigurationError(
                "unsupported provider client/mode pairing: "
                f"{context.provider}/{context.client_shape}/{context.mode}. "
                f"The pinned client is {mode_label}; use {wrapper}.",
                field="client",
            )

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
        for runtime, context in zip(runtimes, runtime_contexts, strict=True):
            _validate_surface_context(
                context,
                pinned_client=runtime.sdk_client if runtime.provider_pinned else None,
            )

        self._solwyn_config = config
        self._solwyn_runtimes = runtimes
        primary = runtimes[0]
        self._solwyn_surface_context = runtime_contexts[0]
        self._solwyn_guard_lock = threading.Lock()
        self._solwyn_guarded_resources: dict[str, _GuardedResource] = {}
        self._solwyn_untracked_observation_notifier: _UntrackedObservationNotifier | None = None
        self._validate_acknowledgments(primary.sdk_client)
        self._solwyn_requested_failover_tuning = {
            name: getattr(config, name) for name in _FAILOVER_TUNING_FIELDS
        }
        self._solwyn_failover_tuning_suppression_logged = False
        self._solwyn_sdk_instance_id = str(uuid.uuid4())
        self._solwyn_velocity = VelocityMonitor(
            VelocityConfig(
                mode=config.velocity_mode,
                repeat_count=config.velocity_repeat_count,
                repeat_window_s=config.velocity_repeat_window_s,
                growth_streak=config.velocity_growth_streak,
                growth_factor=config.velocity_growth_factor,
                accel_floor_per_min=config.velocity_accel_floor_per_min,
                accel_factor=config.velocity_accel_factor,
            )
        )
        # Injectable routing policy: defaults to the health-only policy.
        # Swapping in LatencyPolicy/CostPolicy reorders candidates with ZERO
        # changes to dispatch / translation / budget.
        self._solwyn_policy: SelectionPolicy = selection_policy or HealthBasedPolicy()

        # Per-provider rolling window of recent SUCCESS latencies (ms) for the
        # LatencyPolicy signal. Lock-guarded because the sync client is
        # multi-threaded (the async client is event-loop-serialized; the lock is
        # then uncontended). Pure signal store — no I/O.
        self._solwyn_signal_lock = threading.Lock()
        self._solwyn_latency_windows: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_LATENCY_WINDOW)
        )

        # One circuit breaker per DISTINCT provider across ALL runtimes.
        # Additional providers get lazily-created breakers via
        # _get_circuit_breaker (same jitter).
        self._solwyn_circuit_breakers: dict[str, CircuitBreaker] = {}
        self._solwyn_breaker_lock = threading.Lock()
        for runtime in runtimes:
            provider = runtime.entry.provider.value
            if provider not in self._solwyn_circuit_breakers:
                self._solwyn_circuit_breakers[provider] = self._new_circuit_breaker()

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
        self._solwyn_control_plane_breaker = CircuitBreaker(
            failure_threshold=config.control_plane_failure_threshold,
            recovery_timeout=config.control_plane_recovery_timeout,
            success_threshold=1,
            name="control-plane",
        )

        # Fork repair: a user thread can hold _solwyn_breaker_lock (lazy provider-
        # breaker creation) or _solwyn_signal_lock (latency window append) at fork
        # time; the child would inherit them held by a thread it doesn't have.
        register_fork_reset(self)
        if config.otel_spans:
            _otel._enable_run_spans()

    def _validate_acknowledgments(self, raw_client: object) -> None:
        for token in sorted(self._solwyn_config.acknowledge_untracked):
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
                    context=self._solwyn_surface_context,
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
                context=self._solwyn_surface_context,
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
            value, self._solwyn_surface_context.client_shape
        )

    def _guard_resource(self, value: object, path: str) -> _GuardedResource:
        with self._solwyn_guard_lock:
            cached = self._solwyn_guarded_resources.get(path)
            if cached is not None:
                return cached
            guarded = _GuardedResource(self, value, path)
            self._solwyn_guarded_resources[path] = guarded
            return guarded

    def _has_acknowledged_descendant(self, path: str) -> bool:
        prefix = f"{path}."
        return any(token.startswith(prefix) for token in self._solwyn_config.acknowledge_untracked)

    def _is_exact_raw_response_escape(self, path: str, rule: SurfaceRule) -> bool:
        return (
            rule.capability_scope is CapabilityScope.RAW_RESPONSE
            and rule.token == path
            and path in self._solwyn_config.acknowledge_untracked
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
            token in self._solwyn_config.acknowledge_untracked
            or self._has_acknowledged_descendant(path)
        ):
            return
        scope = (
            rule.capability_scope.value
            if rule is not None and rule.capability_scope is not None
            else None
        )
        kind = rule.kind.value if rule is not None else SurfaceKind.UNKNOWN.value
        posture = self._solwyn_config.on_unmetered
        if posture == "allow":
            _record_untracked_surface_observation(
                context=self._solwyn_surface_context,
                surface=path,
                rule_kind=cast(Literal["unmetered_spend", "unknown"], kind),
                capability_scope=scope,
                posture="allow",
                notifier=self._solwyn_untracked_observation_notifier,
            )
            return
        if posture == "warn":
            _warn_contextual_surface_once(
                context=self._solwyn_surface_context,
                surface=path,
                rule_kind=cast(Literal["unmetered_spend", "unknown"], kind),
                capability_scope=scope,
                drifted_from_rule_id=(drifted_from.rule_id if drifted_from is not None else None),
                notifier=self._solwyn_untracked_observation_notifier,
            )
            return
        raise UntrackedSpendSurfaceError(
            surface=path,
            token=token,
            provider=self._solwyn_surface_context.provider,
            client_shape=self._solwyn_surface_context.client_shape,
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
            context=self._solwyn_surface_context,
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
                provider=self._solwyn_surface_context.provider,
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
        with self._solwyn_guard_lock:
            cached = self._solwyn_guarded_resources.get(path)
        if cached is not None:
            return cached

        rule = resolve_surface_rule(
            context=self._solwyn_surface_context,
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
                    provider=self._solwyn_surface_context.provider,
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
                provider=self._solwyn_surface_context.provider,
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
        """Replace locks and clear process-local velocity state in a child.

        Latency windows, breakers, and the control-plane breaker are plain
        state the child legitimately inherits (the breakers repair
        their own internal locks via their own registration). Run velocity is
        process-local and starts empty with a fresh lock.
        """
        self._solwyn_signal_lock = threading.Lock()
        self._solwyn_breaker_lock = threading.Lock()
        self._solwyn_guard_lock = threading.Lock()
        self._solwyn_velocity._reset_after_fork_in_child()

    def _warn_velocity(self, run_id: str, flags: tuple[str, ...]) -> None:
        """Warn for newly flagged run rules, with monitor-owned rate limiting."""
        now = time.monotonic()
        for rule in flags:
            if self._solwyn_velocity.should_warn(run_id, rule, now):
                logger.warning("velocity.flagged: rule=%s run=%s", rule, run_id)

    def _new_circuit_breaker(self) -> CircuitBreaker:
        """Create a circuit breaker from the configured tuning + jitter."""
        return CircuitBreaker(
            failure_threshold=self._solwyn_config.circuit_breaker_failure_threshold,
            recovery_timeout=self._solwyn_config.circuit_breaker_recovery_timeout,
            success_threshold=self._solwyn_config.circuit_breaker_success_threshold,
            recovery_timeout_jitter=self._solwyn_config.circuit_breaker_recovery_timeout_jitter,
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
            with self._solwyn_breaker_lock:
                return self._tuning_snapshot_locked()

        if allowed:
            effective = dict(self._solwyn_requested_failover_tuning)
        else:
            effective = {
                name: SolwynConfig.model_fields[name].default for name in _FAILOVER_TUNING_FIELDS
            }

        should_log_suppression = False
        with self._solwyn_breaker_lock:
            tuning_changed = any(
                getattr(self._solwyn_config, name) != effective[name]
                for name in _FAILOVER_TUNING_FIELDS
            )
            if tuning_changed:
                for name, value in effective.items():
                    setattr(self._solwyn_config, name, value)
                for breaker in self._solwyn_circuit_breakers.values():
                    breaker.replace_tuning(
                        failure_threshold=self._solwyn_config.circuit_breaker_failure_threshold,
                        recovery_timeout=self._solwyn_config.circuit_breaker_recovery_timeout,
                        success_threshold=self._solwyn_config.circuit_breaker_success_threshold,
                        recovery_timeout_jitter=(
                            self._solwyn_config.circuit_breaker_recovery_timeout_jitter
                        ),
                    )

            suppresses_requested_tuning = any(
                effective[name] != self._solwyn_requested_failover_tuning[name]
                for name in _FAILOVER_TUNING_FIELDS
            )
            if (
                not allowed
                and suppresses_requested_tuning
                and not self._solwyn_failover_tuning_suppression_logged
            ):
                self._solwyn_failover_tuning_suppression_logged = True
                should_log_suppression = True
            snapshot = self._tuning_snapshot_locked()

        if should_log_suppression:
            logger.warning(
                "Custom failover tuning is unavailable for this plan; SDK defaults applied"
            )
        return snapshot

    def _tuning_snapshot_locked(self) -> FailoverTuning:
        """Read one coherent tuning snapshot. Caller must hold _solwyn_breaker_lock."""
        return FailoverTuning(
            failover_total_timeout=self._solwyn_config.failover_total_timeout,
            failover_idempotency=self._solwyn_config.failover_idempotency,
            same_provider_retries=self._solwyn_config.same_provider_retries,
            failover_hop_read_timeout=self._solwyn_config.failover_hop_read_timeout,
        )

    def _get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get the circuit breaker for a provider.

        Lazily creates a circuit breaker if one doesn't exist for this provider.
        """
        with self._solwyn_breaker_lock:
            if provider not in self._solwyn_circuit_breakers:
                self._solwyn_circuit_breakers[provider] = self._new_circuit_breaker()
            return self._solwyn_circuit_breakers[provider]

    def _get_breaker_snapshots(self) -> list[tuple[ProviderName, CircuitBreakerState]]:
        """Return one frozen current-state snapshot per distinct provider.

        Copy the provider/breaker mapping under its lock, then release that lock
        before each breaker acquires its own state lock. This keeps the two lock
        domains independent while allowing providers to be added concurrently.
        """
        with self._solwyn_breaker_lock:
            breakers = list(self._solwyn_circuit_breakers.items())
        return [(ProviderName(provider), breaker.get_state()) for provider, breaker in breakers]

    def record_latency(self, provider: str, ms: float) -> None:
        """Record one observed SUCCESS latency (ms) for a provider (LatencyPolicy).

        Appends to the provider's bounded rolling window. Lock-guarded for the
        multi-threaded sync client (uncontended on the async client). Pure signal
        store — no I/O, no breaker mutation.
        """
        with self._solwyn_signal_lock:
            self._solwyn_latency_windows[provider].append(ms)

    def observed_p50(self, provider: str) -> float | None:
        """Observed p50 (median) latency (ms) for a provider, or None.

        Returns None until at least ``_LATENCY_MIN_SAMPLES`` samples have
        accrued, so an under-sampled provider never jumps the LatencyPolicy
        queue. Lock-guarded; snapshots the window under the lock and computes the
        median outside it.
        """
        with self._solwyn_signal_lock:
            window = self._solwyn_latency_windows.get(provider)
            samples = list(window) if window is not None else []
        if len(samples) < _LATENCY_MIN_SAMPLES:
            return None
        return statistics.median(samples)

    def _select_candidates(
        self, req: RoutingRequest, *, price_hints: Mapping[str, float] | None = None
    ) -> list[ProviderRuntime]:
        """Return policy-ordered runtimes while preserving the legacy list seam."""
        return self._select_candidates_detailed(req, price_hints=price_hints).runtimes

    def _select_candidates_detailed(
        self, req: RoutingRequest, *, price_hints: Mapping[str, float] | None = None
    ) -> CandidateSelection:
        """Order runtimes into attempt order via the pure SelectionPolicy.

        ``price_hints`` are this request's server hints (str provider-name
        keyed). ``None`` or ``{}`` means every candidate carries
        ``price_hint=None``.

        Builds one ProviderCandidate per runtime using NON-MUTATING breaker
        reads only (``state`` / ``recovery_eligible``) — never ``admit()``
        (which consumes a probe). Probe consumption happens exactly once, on the
        single candidate actually attempted, in the dispatch loop.

        ``CandidateSelection.cost_routed`` is True iff a price-signal policy put
        a CLOSED, hinted, DIFFERENT-provider candidate first while the CLOSED
        primary was either unhinted or hinted STRICTLY HIGHER. It is attribution
        only — it never changes the order the policy returned.
        """
        # Signal capability gate (PJ-9/P4): the default HealthBasedPolicy
        # ignores latency_p50 and price_hint, so skip the median unless the
        # configured policy declares it consumes it. Unknown injected policies
        # default to True and keep receiving full signals.
        wants_latency = getattr(self._solwyn_policy, "uses_latency_signal", True)
        wants_price = getattr(self._solwyn_policy, "uses_price_signal", True)
        candidates: list[ProviderCandidate] = []
        for runtime in self._solwyn_runtimes:
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
                    price_hint=(
                        price_hints.get(runtime.adapter.name)
                        if wants_price and price_hints
                        else None
                    ),
                )
            )
        ordered = self._solwyn_policy.order(candidates, req)
        primary = self._solwyn_runtimes[0]
        first = ordered[0] if ordered else None
        primary_cand = next(
            (candidate for candidate in ordered if candidate.runtime is primary),
            None,
        )
        # cost_routed iff a price-signal policy put a CLOSED, hinted,
        # different-provider candidate first while the CLOSED primary was either
        # unhinted or hinted STRICTLY HIGHER. The final conjunct is a plain
        # comparison of the server's own relative hints (the SDK never does
        # arithmetic on them): it stops a custom injected policy — which gets
        # hints by default, since ``uses_price_signal`` is opt-out — from
        # labelling a pricier hop "cost routed". CostPolicy's stable ascending
        # sort can never trip it: if its first candidate is not the primary, its
        # hint is already strictly lower or the primary is unhinted.
        cost_routed = (
            wants_price
            and first is not None
            and first.runtime is not primary
            and first.runtime.entry.provider != primary.entry.provider
            and primary_cand is not None
            and primary_cand.breaker_state is CircuitState.CLOSED
            and first.breaker_state is CircuitState.CLOSED
            and first.price_hint is not None
            and (primary_cand.price_hint is None or first.price_hint < primary_cand.price_hint)
        )
        if isinstance(self._solwyn_policy, CostPolicy) and price_hints is None:
            # The response carried NO hint map (null), so CostPolicy degraded to
            # health-based order — surface the no-op once. Deliberately `is None`
            # and NOT `any(...)`/`not price_hints`: an explicit `{}` means the
            # server priced this chain and found nothing priceable, which is an
            # answer, not a degradation, and must stay silent.
            _warn_cost_policy_inactive_once()
        # Defensive: a custom (possibly misbehaving) injected policy must not be
        # able to inject a runtime that was never in the configured chain into the
        # dispatch walk. Keep ONLY candidates whose runtime is one of our own
        # runtimes (identity check), preserving the policy's order for that valid
        # subset. Drops any foreign/unknown runtime the policy may have appended.
        chain = set(map(id, self._solwyn_runtimes))
        return CandidateSelection(
            runtimes=[candidate.runtime for candidate in ordered if id(candidate.runtime) in chain],
            cost_routed=cost_routed,
        )

    def _extra_client_options(
        self, runtime: ProviderRuntime, *, run_id: str | None
    ) -> dict[str, Any]:
        """Copy-extend this candidate's existing public Anthropic telemetry option.

        Static inspection excludes synthesized attributes and arbitrary descriptors.
        A builtin property uses its public getter (the Anthropic SDK's API); a
        failing getter simply leaves dispatch unchanged. Nested options stay opaque.
        """
        if (
            not self._solwyn_config.anthropic_conversation_id
            or run_id is None
            or runtime.adapter.dialect != "anthropic"
        ):
            return {}
        missing = object()
        try:
            option = inspect.getattr_static(runtime.sdk_client, "open_telemetry", missing)
            if option is missing:
                return {}
            if type(option) is property:
                if option.fget is None:
                    return {}
                option = option.fget(runtime.sdk_client)
            elif inspect.getattr_static(type(option), "__get__", missing) is not missing:
                return {}
            if not isinstance(option, Mapping) or "conversation_id" in option:
                return {}
            return {"open_telemetry": {**option, "conversation_id": run_id}}
        except Exception:
            return {}

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
        trace: _TraceSnapshot | None = None,
        provider_region: str | None = None,
        modality: Modality = "text",
        media_usage: MediaUsage | None = None,
        deny_source: DenySource | None = None,
        deny_reason: str | None = None,
        denied_by_period: str | None = None,
        estimated_output_bound: int | None = None,
        velocity_flags: Collection[str] | None = None,
        receipt_aggregate_count: int | None = None,
        lease_id: str | None = None,
    ) -> MetadataEvent:
        """Build a MetadataEvent for reporting to the cloud API.

        ``call_id`` is the per-call reconciliation join key. ``possibly_succeeded``
        is the post-send-ambiguous abort flag — left None on every non-abort event.
        ``provider_region`` is the served endpoint's cloud region (Bedrock pricing
        is per model AND region); None for providers without regional pricing.
        ``modality`` is the call modality — ``"text"`` for every chat event; the
        media lifecycle passes the surface's modality (e.g. ``"embedding"``).
        ``media_usage`` carries a non-text surface's non-token quantities; None
        for chat/text events (None-skipped on the wire). ``lease_id`` is the
        funding lease of the call's admission, the value on ``budget.lease_id``;
        None for every unfunded or reservation-funded event.
        ``failover_error_class`` is an exception class NAME; this is the single
        choke point that normalizes it (``_wire_error_class``), so no caller
        can build an event the API's pattern would reject.
        """
        if not call_id:
            raise RuntimeError("call_id is required for metadata reconciliation")
        agent_run_id, agent_run_name, tags, parent_agent_run_id = (
            _capture_run_context() if agent_run is None else agent_run
        )
        trace = _validated_trace(trace)
        return MetadataEvent(
            model=model,
            provider=ProviderName(provider),
            modality=modality,
            input_tokens=_wire_token_quantity(input_tokens),
            output_tokens=_wire_token_quantity(output_tokens),
            token_details=token_details,
            media_usage=media_usage,
            latency_ms=latency_ms,
            status=status,
            is_model_fallback=is_model_fallback,
            is_provider_fallback=is_provider_fallback,
            requested_provider=requested_provider,
            requested_model=requested_model,
            failover_reason=failover_reason,
            failover_error_class=_wire_error_class(failover_error_class),
            attempt_index=attempt_index,
            possibly_succeeded=possibly_succeeded,
            service_tier=service_tier,
            sdk_instance_id=sdk_instance_id or self._solwyn_sdk_instance_id,
            timestamp=timestamp or datetime.now(UTC),
            agent_run_id=agent_run_id,
            parent_agent_run_id=parent_agent_run_id,
            agent_run_name=agent_run_name,
            trace_id=trace.trace_id if trace is not None else None,
            span_id=trace.span_id if trace is not None else None,
            trace_sampled=trace.trace_sampled if trace is not None else None,
            call_id=call_id,
            provider_region=provider_region,
            tags=tags,
            deny_source=deny_source,
            deny_reason=deny_reason,
            denied_by_period=denied_by_period,
            estimated_output_bound=(
                None
                if estimated_output_bound is None
                else _wire_token_quantity(estimated_output_bound)
            ),
            velocity_flags=(
                cast("list[VelocityFlag]", list(velocity_flags)) if velocity_flags else None
            ),
            receipt_aggregate_count=receipt_aggregate_count,
            lease_id=lease_id,
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
        trace: _TraceSnapshot | None = None,
        provider_region: str | None = None,
        lease_id: str | None = None,
    ) -> MetadataEvent:
        """Build an error-status MetadataEvent with zeroed token counts.

        Convenience wrapper for the dispatch-failure paths where
        token_details is unavailable and status is always ERROR. ``call_id``
        threads the reconciliation join key; ``possibly_succeeded`` is True only
        on a correctly-not-failed-over post-send-ambiguous abort.
        ``provider_region`` is the FAILED hop's endpoint region — on a
        possibly-succeeded abort the Cloud API needs it to reconcile a
        possibly-landed charge per (model, region); None-skipped otherwise.
        ``lease_id`` is the funding lease of the failed attempt's admission so
        the error event of a lease-admitted call names its lease like its
        served sibling would; None-skipped otherwise.
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
            trace=trace,
            provider_region=provider_region,
            lease_id=lease_id,
        )
