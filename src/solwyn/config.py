"""SolwynConfig -- validated SDK configuration.

Configuration via constructor kwargs or environment variables with the
``SOLWYN_`` prefix (e.g. ``SOLWYN_API_KEY``, ``SOLWYN_API_URL``).
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solwyn._types import BudgetMode, ProviderEntry
from solwyn._validation import validate_project_key_format
from solwyn.exceptions import ConfigurationError

# Environment variable prefix for automatic loading.
_ENV_PREFIX = "SOLWYN_"


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

    # Failover knobs
    failover_total_timeout: float = 30.0
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

    # Control-plane breaker: after this many consecutive check/confirm
    # failures against Solwyn's own API, skip the network call and apply the
    # configured posture (fail_open / local enforcement) instantly for
    # control_plane_recovery_timeout seconds.
    control_plane_failure_threshold: int = 3
    control_plane_recovery_timeout: float = 30.0

    # Reporter tuning
    reporter_batch_size: int = 50
    reporter_flush_interval: float = 5.0
    reporter_max_queue_size: int = 10_000
    reporter_max_in_flight: int = 3
    breaker_reporting_enabled: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Populate missing fields from ``SOLWYN_*`` environment variables."""
        field_env_map = {
            "api_key": "API_KEY",
            "api_url": "API_URL",
            "fail_open": "FAIL_OPEN",
            "budget_mode": "BUDGET_MODE",
            "circuit_breaker_failure_threshold": "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            "circuit_breaker_recovery_timeout": "CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
            "circuit_breaker_success_threshold": "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
            "budget_check_cache_ttl": "BUDGET_CHECK_CACHE_TTL",
            "budget_check_timeout": "BUDGET_CHECK_TIMEOUT",
            "control_plane_failure_threshold": "CONTROL_PLANE_FAILURE_THRESHOLD",
            "control_plane_recovery_timeout": "CONTROL_PLANE_RECOVERY_TIMEOUT",
            "reporter_batch_size": "REPORTER_BATCH_SIZE",
            "reporter_flush_interval": "REPORTER_FLUSH_INTERVAL",
            "reporter_max_queue_size": "REPORTER_MAX_QUEUE_SIZE",
            "reporter_max_in_flight": "REPORTER_MAX_IN_FLIGHT",
            "breaker_reporting_enabled": "BREAKER_REPORTING_ENABLED",
        }

        for field, env_suffix in field_env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(f"{_ENV_PREFIX}{env_suffix}")
                if env_val is not None:
                    # Coerce boolean-looking strings
                    if field in {"fail_open", "breaker_reporting_enabled"}:
                        values[field] = env_val.lower() in ("true", "1", "yes")
                    else:
                        values[field] = env_val

        return values

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
