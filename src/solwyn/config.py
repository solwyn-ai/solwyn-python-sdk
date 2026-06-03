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
    # Global fill-absent defaults (per-entry default_params wins; see §5.2).
    default_params: dict[str, Any] = Field(default_factory=dict)

    # Failover knobs (§4.4 / §6.3 / §6.5)
    failover_total_timeout: float = 30.0
    failover_idempotency: Literal["safe", "never", "always"] = "safe"
    same_provider_retries: int = 0
    circuit_breaker_recovery_timeout_jitter: float = 0.2

    # Circuit breaker tuning
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_success_threshold: int = 2

    # Budget check cache
    budget_check_cache_ttl: int = 5

    # Reporter tuning
    reporter_batch_size: int = 50
    reporter_flush_interval: float = 5.0
    reporter_max_queue_size: int = 10_000
    reporter_max_in_flight: int = 3

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
            "reporter_batch_size": "REPORTER_BATCH_SIZE",
            "reporter_flush_interval": "REPORTER_FLUSH_INTERVAL",
            "reporter_max_queue_size": "REPORTER_MAX_QUEUE_SIZE",
            "reporter_max_in_flight": "REPORTER_MAX_IN_FLIGHT",
        }

        for field, env_suffix in field_env_map.items():
            if field not in values or values[field] is None:
                env_val = os.environ.get(f"{_ENV_PREFIX}{env_suffix}")
                if env_val is not None:
                    # Coerce boolean-looking strings
                    if field == "fail_open":
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
