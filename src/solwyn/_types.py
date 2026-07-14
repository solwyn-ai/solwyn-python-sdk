"""Vendored enums and wire-format models for SDK <-> API contracts.

Pydantic models for API request/response contracts.
Excludes API-internal types: ProjectConfig, ProviderHealth,
NotificationEventType, Environment, BudgetPeriod.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    StringConstraints,
    model_serializer,
    model_validator,
)

# TokenDetails lives in a separate module to avoid a circular import:
# _types -> TokenDetails -> (if merged here) _types.
from solwyn._constants import (
    AGENT_RUN_ID_MAX_LENGTH,
    AGENT_RUN_NAME_MAX_LENGTH,
    MODEL_NAME_MAX_LENGTH,
    PROVIDER_REGION_MAX_LENGTH,
    SERVICE_TIER_MAX_LENGTH,
    TAG_KEY_MAX_LENGTH,
    TAG_VALUE_MAX_LENGTH,
    TAGS_MAX_KEYS,
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
    """Supported LLM provider identifiers.

    The first four are native API dialects (Bedrock's Converse API is its
    own dialect). The rest are OpenAI-compatible
    providers: they speak the Chat Completions dialect but are distinct
    providers for attribution, pricing, budget enforcement, and circuit
    breaking. ``OPENAI_COMPATIBLE`` is the generic catch-all for any
    unrecognized OpenAI-compatible endpoint (custom proxies, new vendors).

    Wire contract: the Cloud API must accept every value here on
    MetadataEvent/BudgetCheckRequest/BudgetConfirmRequest before an SDK
    carrying it is released (API deploys first).
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    # OpenAI-compatible providers (Chat Completions dialect)
    XAI = "xai"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    QWEN = "qwen"
    ZAI = "zai"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    AZURE_OPENAI = "azure_openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LMSTUDIO = "lmstudio"
    OPENAI_COMPATIBLE = "openai_compatible"


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


# The CONTRACTUAL tier values, pinned lock-step with the Cloud API's
# ServiceTier literal — its confirm model rejects anything else. The
# observational MetadataEvent.service_tier stays a bounded str (any provider
# echo rides telemetry); only the confirm, which settles the budget
# enforcement counter at a tier-repriced rate, is value-strict.
ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "standard", "optimized"]


# Call modality discriminator. This is the ONLY modality field on the wire; its
# default "text" lets an SDK that predates the field omit it safely (the API
# deploys the field FIRST). The SERVER's pricing card unit — not this label alone —
# selects the billing basis; core bills text tokens only until per-modality
# cards land. Vendored lock-step with core shared/models.py's Modality.
Modality = Literal["text", "image", "audio", "video", "embedding"]

TagKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=TAG_KEY_MAX_LENGTH, strict=True),
]
TagValue = Annotated[
    str,
    StringConstraints(max_length=TAG_VALUE_MAX_LENGTH, strict=True),
]


# ── Config models ───────────────────────────────────────────────────────


class ProviderEntry(BaseModel):
    """One link in the configured provider/model failover chain.

    Deliberately carries NO ``api_key`` / ``base_url``. Solwyn
    never accepts, stores, or logs a provider credential — credentials live
    only in the caller's client objects. ``extra="forbid"`` makes any attempt
    to smuggle one in a hard error.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = Field(..., description="Target LLM provider")
    model: str = Field(
        ..., max_length=MODEL_NAME_MAX_LENGTH, description="Model name for this provider"
    )
    default_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Fill-absent default request params for this entry",
    )


# ── Wire-format models ──────────────────────────────────────────────────


class BreakerStateReport(BaseModel):
    """One SDK instance's current provider circuit-breaker snapshot."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = Field(..., description="Provider identifier")
    state: CircuitState = Field(..., description="Current SDK circuit-breaker state")
    failure_count: int = Field(..., ge=0, description="Current breaker failure count")
    success_count: int = Field(..., ge=0, description="Current breaker success count")
    reported_at: datetime = Field(..., description="Wall-clock time the SDK took the snapshot")
    sdk_instance_id: str = Field(
        ...,
        max_length=100,
        description="Bounded SDK instance identifier used to isolate Redis snapshots",
    )


class MediaUsage(BaseModel):
    """Non-token billable quantities + variant selectors for a media call.

    ``TokenDetails`` is int-only by design
    (a normalized TOKEN breakdown); the non-token quantities a per-unit priced
    surface bills on — image counts, media seconds, character counts — live
    here instead, so the token channel stays doubly int-locked and media
    quantities are never shoehorned into it.

    Every quantity is ``None`` when the SDK cannot observe it — never a
    zero-as-default. A known-unit pricing card with a ``None`` quantity routes
    to the server's ``is_unpriced``/$0 lane (never $0-priced-as-real), so an
    unobservable quantity is never silently settled as a real $0 price. The
    server's pricing card unit — not any single field here — selects which
    quantity is the billing basis.

    Vendored lock-step with core ``shared/solwyn_shared/models.py``'s
    ``MediaUsage`` (SDK is self-contained — mirrored inline, no
    ``solwyn_shared`` import).
    """

    model_config = ConfigDict(extra="forbid")

    image_count: int | None = Field(
        default=None, ge=0, description="Images generated/edited (per_image unit)"
    )
    generation_count: int | None = Field(
        default=None,
        ge=0,
        description="Discrete generations for per_video / per_generation / per_song units",
    )
    video_seconds: float | None = Field(
        default=None, ge=0, description="Video duration in seconds (per_second / per_minute units)"
    )
    audio_seconds: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Audio duration in seconds (fractional; e.g. whisper verbose_json usage.seconds)"
        ),
    )
    input_characters: int | None = Field(
        default=None,
        ge=0,
        description=(
            "TTS input character count (per_*_chars units; length measured in the firewall)"
        ),
    )
    resolution: str | None = Field(
        default=None,
        max_length=32,
        description="Variant selector matched against the pricing card's grid (e.g. '1024x1024')",
    )
    quality: str | None = Field(
        default=None,
        max_length=32,
        description="Variant selector matched against the pricing card's grid (e.g. 'hd', 'low')",
    )
    is_estimated: bool = Field(
        default=False,
        description=(
            "True when these quantities are SDK-side estimates rather than "
            "provider-reported. False for exact request-derived quantities (an "
            "image count from n= is a true known quantity, not an approximation)."
        ),
    )


class MetadataEvent(BaseModel):
    """Telemetry event sent from SDK to API after each LLM call.

    Carries automatic token/latency metadata plus optional explicit
    customer-supplied tags, which are outside the zero-content guarantee. It
    never carries prompts, responses, or SDK-computed costs.
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

    model: str = Field(
        ..., max_length=MODEL_NAME_MAX_LENGTH, description="LLM model name (e.g. gpt-4o)"
    )
    provider: ProviderName = Field(..., description="LLM provider")
    modality: Modality = Field(
        default="text",
        description=(
            "Call modality. Defaults to 'text' so pre-modality SDKs omit it "
            "safely (API-first). The pricing card's unit — not this label alone "
            "— selects the billing basis; core bills text tokens only until "
            "per-modality cards land."
        ),
    )
    input_tokens: int = Field(..., ge=0, description="Input token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    token_details: TokenDetails | None = Field(
        None, description="Full token breakdown from provider adapter"
    )
    media_usage: MediaUsage | None = Field(
        default=None,
        description=(
            "Non-token billable quantities (image counts, media seconds, "
            "character counts) + variant selectors for non-text modalities. "
            "None for chat/text calls — the None-skipping serializer keeps their "
            "wire bytes unchanged."
        ),
    )
    latency_ms: float = Field(..., description="End-to-end call latency in ms")
    status: CallStatus = Field(..., description="Call outcome")
    is_model_fallback: bool = Field(..., description="same-provider model swap ONLY")
    is_provider_fallback: bool = Field(default=False, description="served provider != requested")
    requested_provider: ProviderName | None = Field(default=None)
    requested_model: str | None = Field(default=None, max_length=MODEL_NAME_MAX_LENGTH)
    failover_reason: FailoverReason | None = Field(default=None)
    failover_error_class: str | None = Field(
        default=None,
        max_length=64,
        description="type(exc).__name__ ONLY — never str(exc)",
    )
    attempt_index: int = Field(default=0, ge=0, description="0=primary, 1=first fallback")
    call_id: str = Field(
        ...,
        description=(
            "uuid per intercepted call; join key for cache-hit spend reconciliation. "
            "Always present on the wire — it is NOT content."
        ),
    )
    possibly_succeeded: bool | None = Field(
        default=None,
        description=(
            "post-send-ambiguous abort flag for reconciliation; True only on a "
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
    provider_region: str | None = Field(
        default=None,
        max_length=PROVIDER_REGION_MAX_LENGTH,
        description=(
            "Cloud region of the endpoint that served the call (e.g. 'us-east-1' "
            "for Bedrock, whose pricing is keyed per model AND region). None for "
            "providers without regional pricing — the None-skipping serializer "
            "keeps their wire bytes unchanged."
        ),
    )
    tags: dict[TagKey, TagValue] | None = Field(
        default=None,
        max_length=TAGS_MAX_KEYS,
        description=(
            "Explicit customer-supplied metadata for grouping and export. "
            "Never derived from prompts or responses."
        ),
    )


class BudgetCheckRequest(BaseModel):
    """Pre-flight budget check sent before an LLM call."""

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize checks without null-valued optional fields.

        ``estimated_media`` and ``agent_run_id`` are optional fields; skipping
        them when None keeps unscoped text/chat check wire bytes byte-identical
        to the legacy wire. The always-present fields (modality and the
        fallback-chain lists) never go None, so they always serialize.
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("BudgetCheckRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    estimated_input_tokens: int = Field(
        ..., ge=0, description="Estimated input token count for the pending call"
    )
    model: str = Field(..., max_length=MODEL_NAME_MAX_LENGTH, description="LLM model name")
    provider: ProviderName = Field(..., description="Target provider")
    modality: Modality = Field(
        default="text",
        description=(
            "Call modality of the pending request. Defaults to 'text' so "
            "pre-modality SDKs omit it safely (API-first). The pricing card's "
            "unit selects the billing basis; core bills text tokens only until "
            "per-modality cards land."
        ),
    )
    estimated_media: MediaUsage | None = Field(
        default=None,
        description=(
            "Pre-flight non-token quantities (image counts, media seconds, "
            "character counts) + variant selectors for a non-text pending call, "
            "so the server prices a precise check-time cost. None for text/chat "
            "checks — the None-skipping serializer keeps their wire bytes "
            "unchanged."
        ),
    )
    fallback_providers: list[ProviderName] = Field(
        default_factory=list,
        description="Configured failover providers, in attempt order (chain hint)",
    )
    fallback_models: list[Annotated[str, Field(max_length=MODEL_NAME_MAX_LENGTH)]] = Field(
        default_factory=list,
        max_length=8,
        description="Failover models aligned element-for-element with fallback_providers",
    )
    agent_run_id: str | None = Field(
        default=None,
        max_length=AGENT_RUN_ID_MAX_LENGTH,
        description="Stable id for the active solwyn.run() scope, when present.",
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

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize confirms without null-valued optional fields.

        provider_region is the only optional field; skipping it when None keeps
        the bearer-key providers' confirm wire bytes unchanged (the Cloud-API
        model forbids unknown keys).
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("BudgetConfirmRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    reservation_id: str = Field(
        ..., description="Budget reservation ID returned by BudgetCheckResponse"
    )
    model: str = Field(
        ..., max_length=MODEL_NAME_MAX_LENGTH, description="LLM model name used for the call"
    )
    provider: ProviderName = Field(
        ..., description="Provider that actually served the call (required)"
    )
    modality: Modality = Field(
        default="text",
        description=(
            "Call modality of the served call. Defaults to 'text' so pre-modality "
            "SDKs omit it safely (API-first). The pricing card's unit settles the "
            "budget counter on the right basis; core bills text tokens only until "
            "per-modality cards land."
        ),
    )
    is_provider_fallback: bool = Field(
        default=False, description="served provider != requested provider"
    )
    call_id: str = Field(
        ...,
        description=("uuid per intercepted call; dedups confirm vs metadata reconciliation"),
    )
    token_details: TokenDetails = Field(
        ..., description="Actual token breakdown from the provider adapter"
    )
    media_usage: MediaUsage | None = Field(
        default=None,
        description=(
            "Actual non-token billable quantities (image counts, media seconds, "
            "character counts) + variant selectors for a non-text served call, so "
            "the server settles the enforcement counter on the right basis. None "
            "for chat/text confirms — the None-skipping serializer keeps their "
            "wire bytes unchanged."
        ),
    )
    provider_region: str | None = Field(
        default=None,
        max_length=PROVIDER_REGION_MAX_LENGTH,
        description=(
            "Cloud region of the serving endpoint, for per-region pricing "
            "(Bedrock). None for providers without regional pricing."
        ),
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description=(
            "Service tier echoed by the provider response. Bedrock confirms "
            "settle the budget enforcement counter at the tier-repriced rate "
            "(flex 0.5x / priority 1.75x / optimized 1.25x) so hard-deny "
            "tracks real spend; None settles at Standard rates. Ignored for "
            "providers without per-tier pricing."
        ),
    )
