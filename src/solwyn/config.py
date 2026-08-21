"""SolwynConfig -- validated SDK configuration.

Configuration via constructor kwargs or environment variables with the
``SOLWYN_`` prefix (e.g. ``SOLWYN_API_KEY``, ``SOLWYN_API_URL``).
"""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from solwyn._lease import DEFAULT_OUTPUT_BOUND
from solwyn._run import _copy_tags
from solwyn._types import BudgetMode, ProviderEntry
from solwyn._validation import validate_project_key_format
from solwyn._velocity import VELOCITY_HISTORY_LIMIT
from solwyn.exceptions import ConfigurationError

# Environment variable prefix for automatic loading.
_ENV_PREFIX = "SOLWYN_"


class _EnvTags(str):
    """Tag string originating from ``SOLWYN_TAGS``."""


class _EnvAcknowledgments(str):
    """Exact-token string originating from ``SOLWYN_ACKNOWLEDGE_UNTRACKED``."""


class SolwynConfig(BaseModel):
    """Validated configuration for the Solwyn SDK.

    Values can be supplied directly or loaded from environment variables
    (``SOLWYN_API_KEY``, ``SOLWYN_API_URL``, etc.) via a ``@model_validator``.
    """

    # Required fields
    api_key: str

    # Optional fields with defaults
    api_url: str = "https://api.solwyn.ai"
    fail_open: bool = True
    budget_mode: BudgetMode = BudgetMode.ALERT_ONLY

    # Provider failover chain (replaces primary_provider + fallback_model).
    # providers[0] is the primary; the rest are fallbacks in attempt order.
    providers: list[ProviderEntry] = Field(default_factory=list)
    # Global fill-absent defaults (per-entry default_params wins).
    default_params: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] | None = None

    # Compatibility-default handling for provider capabilities that bypass
    # Solwyn interception. Acknowledgments are exact terminal tokens only;
    # graph-aware applicability validation happens in _SolwynBase once the
    # caller's concrete provider client is available.
    on_unmetered: Literal["warn", "raise", "allow"] = "warn"
    acknowledge_untracked: frozenset[str] = Field(default_factory=frozenset)

    # Failover knobs
    failover_total_timeout: FiniteFloat = 30.0
    # Per-hop READ/WRITE bound (seconds), decoupled from the chain deadline
    # (PJ-8/R7). failover_total_timeout is the FAILOVER WINDOW - it bounds the
    # budget pre-flight, per-hop connect slices, Retry-After sleeps, and
    # between-hop advancement - while this field alone bounds how long one
    # dispatched hop may spend reading a legitimate (possibly slow) response.
    # 600.0 matches the openai/anthropic SDK's READ/WRITE default, so the
    # wrapped read/write bound never fires earlier than the unwrapped SDK's
    # would; connect/pool instead track the shrinking failover window. A read
    # timeout is POST_SEND_AMBIGUOUS (re-raised, never failed over, under the
    # default idempotency), so a small value converts slow legitimate
    # generations into ambiguous spend - lower it deliberately.
    failover_hop_read_timeout: FiniteFloat = Field(default=600.0, gt=0)
    failover_idempotency: Literal["safe", "never", "always"] = "safe"
    # Max same-provider retries on a 429 whose Retry-After the provider asked us
    # to honor, BEFORE failing over cross-provider. On such a 429 the dispatch
    # loop sleeps the (deadline-bounded) Retry-After and re-attempts the SAME
    # provider up to this many times; the retried 429 records no breaker verdict
    # until it resolves (success closes/neutralizes the probe, exhaustion records
    # one failure then fails over). A missing/unparseable Retry-After, or one that
    # would not fit the remaining chain deadline, skips the retry and fails over.
    # 0 (default) == immediate failover, today's behavior.
    same_provider_retries: int = Field(default=0, ge=0)
    circuit_breaker_recovery_timeout_jitter: float = 0.2

    # Circuit breaker tuning
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_success_threshold: int = 2

    # Budget check cache
    budget_check_cache_ttl: int = 5

    # Budget pre-flight: per-request timeout for the /budgets/check POST.
    # Deliberately short — the check gates the caller's hot path; the breaker
    # below caps repeated discovery of a control-plane outage.
    budget_check_timeout: float = 1.0

    # Budget leases (PJ-2): run-scoped, token-billed calls draw down a
    # server-granted token lease in memory instead of paying a blocking
    # /budgets/check per call. Kill switch — False routes every call back to
    # the per-call check path. lease_output_bound_default bounds a call's
    # reservation when it carries no max_tokens-family cap; it defaults to the
    # ledger's own constant so the two can never drift.
    lease_enabled: bool = True
    lease_output_bound_default: int = Field(default=DEFAULT_OUTPUT_BOUND, gt=0)

    # Run-scoped, content-free velocity signals. PR-1 is warn-only at the
    # client boundary; ``deny`` is admitted now for the later enforcement stage.
    velocity_mode: Literal["off", "warn", "deny"] = "warn"
    # History-dependent thresholds cannot exceed the retained detailed window;
    # this keeps every accepted configuration achievable.
    velocity_repeat_count: int = Field(default=5, ge=2, le=VELOCITY_HISTORY_LIMIT)
    velocity_repeat_window_s: float = Field(default=60.0, gt=0)
    velocity_growth_streak: int = Field(default=8, ge=3, le=VELOCITY_HISTORY_LIMIT)
    velocity_growth_factor: float = Field(default=3.0, gt=1.0)
    velocity_accel_floor_per_min: int = Field(
        default=30,
        ge=1,
        le=VELOCITY_HISTORY_LIMIT,
    )
    velocity_accel_factor: float = Field(default=3.0, gt=1.0)

    # Control-plane breaker: after this many consecutive check/confirm
    # failures against Solwyn's own API, skip the network call and apply the
    # configured posture (fail_open / local enforcement) instantly for
    # control_plane_recovery_timeout seconds.
    control_plane_failure_threshold: int = 3
    control_plane_recovery_timeout: float = 30.0

    # Reporter tuning. A zero-capacity queue has no defined drop-oldest
    # semantics (the reporter constructors reject it too) — at least one slot.
    reporter_batch_size: int = 50
    reporter_flush_interval: float = 5.0
    reporter_max_queue_size: int = Field(default=10_000, ge=1)
    reporter_max_in_flight: int = 3
    breaker_reporting_enabled: bool = True
    # Advisory reporting defaults on because silence is the failure mode;
    # disable only when optional control-plane egress must be zero.
    report_untracked_surfaces: bool = True
    # Breaker reports POST only when a provider's snapshot changed since the
    # last successful send, plus a periodic full-refresh heartbeat so the
    # dashboard's advisory view never goes stale silently.
    breaker_report_heartbeat: float = Field(default=60.0, gt=0)

    # Reporter at-least-once delivery: these bound the retry/backoff the reporter
    # applies to confirms, settlements, and metadata batches so acknowledged
    # provider spend is not silently lost on a transient control-plane blip, and
    # cap the wall-clock the shutdown/exit flush chain may spend.
    # Validated: a non-positive backoff base would put next_attempt_at in the
    # past (hot retry loop burning the whole attempt budget in one flush
    # cycle), and a negative deadline is already-expired (instant drops).
    reporter_max_send_attempts: int = Field(default=5, ge=1)
    reporter_retry_backoff_base: float = Field(default=1.0, gt=0)
    reporter_retry_backoff_cap: float = Field(default=60.0, gt=0)
    reporter_shutdown_deadline: float = Field(default=5.0, ge=0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Populate missing fields from ``SOLWYN_*`` environment variables."""
        field_env_map = {
            "api_key": "API_KEY",
            "api_url": "API_URL",
            "tags": "TAGS",
            "on_unmetered": "ON_UNMETERED",
            "acknowledge_untracked": "ACKNOWLEDGE_UNTRACKED",
            "fail_open": "FAIL_OPEN",
            "budget_mode": "BUDGET_MODE",
            "circuit_breaker_failure_threshold": "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            "circuit_breaker_recovery_timeout": "CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
            "circuit_breaker_success_threshold": "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
            "budget_check_cache_ttl": "BUDGET_CHECK_CACHE_TTL",
            "budget_check_timeout": "BUDGET_CHECK_TIMEOUT",
            "lease_enabled": "LEASE_ENABLED",
            "lease_output_bound_default": "LEASE_OUTPUT_BOUND_DEFAULT",
            "velocity_mode": "VELOCITY_MODE",
            "velocity_repeat_count": "VELOCITY_REPEAT_COUNT",
            "velocity_repeat_window_s": "VELOCITY_REPEAT_WINDOW_S",
            "velocity_growth_streak": "VELOCITY_GROWTH_STREAK",
            "velocity_growth_factor": "VELOCITY_GROWTH_FACTOR",
            "velocity_accel_floor_per_min": "VELOCITY_ACCEL_FLOOR_PER_MIN",
            "velocity_accel_factor": "VELOCITY_ACCEL_FACTOR",
            "control_plane_failure_threshold": "CONTROL_PLANE_FAILURE_THRESHOLD",
            "control_plane_recovery_timeout": "CONTROL_PLANE_RECOVERY_TIMEOUT",
            "reporter_batch_size": "REPORTER_BATCH_SIZE",
            "reporter_flush_interval": "REPORTER_FLUSH_INTERVAL",
            "reporter_max_queue_size": "REPORTER_MAX_QUEUE_SIZE",
            "reporter_max_in_flight": "REPORTER_MAX_IN_FLIGHT",
            "breaker_reporting_enabled": "BREAKER_REPORTING_ENABLED",
            "report_untracked_surfaces": "REPORT_UNTRACKED_SURFACES",
            "breaker_report_heartbeat": "BREAKER_REPORT_HEARTBEAT",
            "reporter_max_send_attempts": "REPORTER_MAX_SEND_ATTEMPTS",
            "reporter_retry_backoff_base": "REPORTER_RETRY_BACKOFF_BASE",
            "reporter_retry_backoff_cap": "REPORTER_RETRY_BACKOFF_CAP",
            "reporter_shutdown_deadline": "REPORTER_SHUTDOWN_DEADLINE",
        }

        for field, env_suffix in field_env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(f"{_ENV_PREFIX}{env_suffix}")
                if env_val is not None:
                    # Coerce boolean-looking strings
                    if field == "tags":
                        values[field] = _EnvTags(env_val)
                    elif field == "acknowledge_untracked":
                        values[field] = _EnvAcknowledgments(env_val)
                    elif field in {
                        "fail_open",
                        "breaker_reporting_enabled",
                        "lease_enabled",
                        "report_untracked_surfaces",
                    }:
                        values[field] = env_val.lower() in ("true", "1", "yes")
                    else:
                        values[field] = env_val

        return values

    @field_validator("failover_total_timeout", "failover_hop_read_timeout", mode="before")
    @classmethod
    def _reject_boolean_timeout_bounds(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("timeout bounds must be numbers, not booleans")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: object | None) -> dict[str, str] | None:
        if isinstance(value, _EnvTags):
            try:
                value = dict(entry.split("=", 1) for entry in value.split(","))
            except ValueError as exc:
                raise ValueError("SOLWYN_TAGS entries must use key=value") from exc
        try:
            return _copy_tags(value, parameter="tags")
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("acknowledge_untracked", mode="before")
    @classmethod
    def _validate_acknowledgment_tokens(cls, value: object) -> frozenset[str]:
        if isinstance(value, _EnvAcknowledgments):
            if not value.strip():
                return frozenset()
            env_tokens = tuple(part.strip() for part in value.split(","))
            if any(not token for token in env_tokens):
                raise ValueError("SOLWYN_ACKNOWLEDGE_UNTRACKED must not contain empty elements")
            return frozenset(env_tokens)
        if isinstance(value, (str, bytes)):
            raise ValueError("acknowledge_untracked must be a collection of exact tokens")
        if not isinstance(value, Collection):
            raise ValueError("acknowledge_untracked must be a collection of exact tokens") from None
        tokens: set[str] = set()
        for token in value:
            if not isinstance(token, str) or not token:
                raise ValueError("acknowledge_untracked tokens must be non-empty strings")
            tokens.add(token)
        return frozenset(tokens)

    @model_validator(mode="after")
    def _validate_credentials(self) -> SolwynConfig:
        """Validate api_key format after construction."""
        try:
            validate_project_key_format(self.api_key)
        except ValueError as exc:
            raise ConfigurationError(str(exc), field="api_key") from exc

        return self

    @model_validator(mode="after")
    def _check_chain(self) -> SolwynConfig:
        """Require at least one provider entry in the failover chain."""
        if not self.providers:
            raise ConfigurationError("at least one provider entry required", field="providers")
        return self
