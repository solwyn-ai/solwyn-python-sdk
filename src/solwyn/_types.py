"""Vendored enums and wire-format models for SDK <-> API contracts.

Pydantic models for API request/response contracts.
Excludes API-internal types: ProjectConfig, ProviderHealth,
NotificationEventType, Environment, BudgetPeriod.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# TokenDetails lives in a separate module to avoid a circular import:
# _types -> TokenDetails -> (if merged here) _types.
from solwyn._constants import (
    AGENT_RUN_ID_MAX_LENGTH,
    AGENT_RUN_NAME_MAX_LENGTH,
    SERVICE_TIER_MAX_LENGTH,
)
from solwyn._token_details import TokenDetails

# ── Enums ────────────────────────────────────────────────────────────────


class BudgetMode(StrEnum):
    """How the SDK reacts when a budget limit is reached."""

    ALERT_ONLY = "alert_only"
    HARD_DENY = "hard_deny"


class CircuitState(StrEnum):
    """Circuit breaker states for provider health tracking."""

    CLOSED = "closed"  # Normal operation — requests flow through
    OPEN = "open"  # Failing — reject requests, try fallback
    HALF_OPEN = "half_open"  # Testing recovery — allow probe requests


class ProviderName(StrEnum):
    """Supported LLM provider identifiers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class CallStatus(StrEnum):
    """Outcome status for LLM call metadata events."""

    SUCCESS = "success"
    ERROR = "error"
    BUDGET_DENIED = "budget_denied"


class FailoverReason(StrEnum):
    """Why the router advanced past the requested primary target."""

    CIRCUIT_OPEN = "circuit_open"  # primary breaker was OPEN — never attempted
    PRIMARY_ERROR = "primary_error"  # primary attempt raised before success
    MODEL_FALLBACK = "model_fallback"  # same-provider model swap


# ── Config models ───────────────────────────────────────────────────────


class ProviderEntry(BaseModel):
    """One link in the configured provider/model failover chain.

    Decision D: deliberately carries NO ``api_key`` / ``base_url``. Solwyn
    never accepts, stores, or logs a provider credential — credentials live
    only in the caller's client objects. ``extra="forbid"`` makes any attempt
    to smuggle one in a hard error.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = Field(..., description="Target LLM provider")
    model: str = Field(..., max_length=100, description="Model name for this provider")
    default_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Fill-absent default request params for this entry",
    )


# ── Wire-format models ──────────────────────────────────────────────────


class MetadataEvent(BaseModel):
    """Telemetry event sent from SDK to API after each LLM call.

    Contains token/latency metadata only — never prompts, responses, or
    SDK-computed costs.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize telemetry events without null-valued optional fields."""
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("MetadataEvent serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    model: str = Field(..., max_length=100, description="LLM model name (e.g. gpt-4o)")
    provider: ProviderName = Field(..., description="LLM provider")
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    token_details: TokenDetails | None = Field(
        None, description="Full token breakdown from provider adapter"
    )
    latency_ms: float = Field(..., description="End-to-end call latency in ms")
    status: CallStatus = Field(..., description="Call outcome")
    is_model_fallback: bool = Field(..., description="same-provider model swap ONLY")
    is_provider_fallback: bool = Field(default=False, description="served provider != requested")
    requested_provider: ProviderName | None = Field(default=None)
    requested_model: str | None = Field(default=None, max_length=100)
    failover_reason: FailoverReason | None = Field(default=None)
    failover_error_class: str | None = Field(
        default=None,
        max_length=64,
        description="type(exc).__name__ ONLY — never str(exc)",
    )
    attempt_index: int = Field(default=0, ge=0, description="0=primary, 1=first fallback")
    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "uuid per intercepted call; join key for cache-hit spend reconciliation "
            "(§8.4). Always present on the wire — it is NOT content."
        ),
    )
    possibly_succeeded: bool | None = Field(
        default=None,
        description=(
            "post-send-ambiguous abort flag for reconciliation (§8.4); True only on a "
            "correctly-not-failed-over post-send-ambiguous abort. None default keeps "
            "every non-abort event clean on the wire."
        ),
    )
    service_tier: str | None = Field(
        default=None,
        max_length=SERVICE_TIER_MAX_LENGTH,
        description="Provider service tier from the response, when available.",
    )
    sdk_instance_id: str = Field(..., description="Unique SDK instance identifier")
    timestamp: datetime = Field(..., description="When the LLM call completed (UTC)")
    agent_run_id: str | None = Field(
        default=None,
        max_length=AGENT_RUN_ID_MAX_LENGTH,
        description=(
            "Stable id for the active solwyn.run() scope. None when no scope is "
            "active — the API synthesizes _auto-{sdk_instance_id}-{YYYY-MM-DD} "
            "server-side from the event timestamp."
        ),
    )
    agent_run_name: str | None = Field(
        default=None,
        max_length=AGENT_RUN_NAME_MAX_LENGTH,
        description="Human-readable label passed to solwyn.run(name).",
    )


class BudgetCheckRequest(BaseModel):
    """Pre-flight budget check sent before an LLM call."""

    model_config = ConfigDict(extra="forbid")

    estimated_input_tokens: int = Field(
        ..., ge=0, description="Estimated input token count for the pending call"
    )
    model: str = Field(..., max_length=100, description="LLM model name")
    provider: ProviderName = Field(..., description="Target provider")
    fallback_providers: list[ProviderName] = Field(
        default_factory=list,
        description="Configured failover providers, in attempt order (chain hint)",
    )
    fallback_models: list[Annotated[str, Field(max_length=100)]] = Field(
        default_factory=list,
        max_length=8,
        description="Failover models aligned element-for-element with fallback_providers",
    )

    @model_validator(mode="after")
    def _check_chain_hint_alignment(self) -> BudgetCheckRequest:
        """fallback_providers and fallback_models must align element-for-element."""
        if len(self.fallback_models) != len(self.fallback_providers):
            raise ValueError("fallback_models and fallback_providers must have equal length")
        return self


class BudgetCheckResponse(BaseModel):
    """API response to a budget check request."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = Field(..., description="Whether the call is within budget")
    remaining_budget: float = Field(..., description="Remaining budget in USD for current period")
    reservation_id: str | None = Field(
        None, description="Budget reservation ID (for cost reconciliation)"
    )
    mode: BudgetMode = Field(..., description="Current budget enforcement mode")
    budget_limit: float = Field(..., description="Total budget limit for current period in USD")
    current_usage: float = Field(..., description="Current spend in USD for this period")
    denied_by_period: str | None = Field(
        ..., description="Which budget period triggered denial (e.g. 'daily')"
    )
    project_id: str = Field(..., description="Project identifier resolved from the API key")
    price_hints: dict[ProviderName, float] | None = Field(
        default=None,
        description=(
            "Server-provided RELATIVE price signal per provider for cost routing; "
            "SDK never computes price"
        ),
    )


class BudgetConfirmRequest(BaseModel):
    """Post-call budget confirmation sent after an LLM call completes."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str = Field(
        ..., description="Budget reservation ID returned by BudgetCheckResponse"
    )
    model: str = Field(..., max_length=100, description="LLM model name used for the call")
    provider: ProviderName = Field(
        ..., description="Provider that actually served the call (required; §8.1)"
    )
    is_provider_fallback: bool = Field(
        default=False, description="served provider != requested provider"
    )
    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=("uuid per intercepted call; dedups confirm vs metadata reconciliation (§8.4)"),
    )
    token_details: TokenDetails = Field(
        ..., description="Actual token breakdown from the provider adapter"
    )
