"""Vendored enums and wire-format models for SDK <-> API contracts.

Pydantic models for API request/response contracts.
Excludes API-internal types: ProjectConfig, ProviderHealth,
NotificationEventType, Environment, BudgetPeriod.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    StringConstraints,
    field_validator,
    model_serializer,
    model_validator,
)

# TokenDetails lives in a separate module to avoid a circular import:
# _types -> TokenDetails -> (if merged here) _types.
from solwyn._constants import (
    AGENT_RUN_ID_MAX_LENGTH,
    AGENT_RUN_NAME_MAX_LENGTH,
    CALL_ID_MAX_LENGTH,
    CALL_ID_PATTERN,
    HOLDER_ID_MAX_LENGTH,
    LEASE_ID_MAX_LENGTH,
    MODEL_NAME_MAX_LENGTH,
    ORDINARY_TOKEN_COUNT_MAX,
    PROVIDER_REGION_MAX_LENGTH,
    SERVICE_TIER_MAX_LENGTH,
    TAG_KEY_MAX_LENGTH,
    TAG_VALUE_MAX_LENGTH,
    TAGS_MAX_KEYS,
)
from solwyn._token_details import TokenDetails

# ── Enums ────────────────────────────────────────────────────────────────

DenySource: TypeAlias = Literal[
    "server",
    "sticky_replay",
    "local_enforcement",
    "lease_exhausted",
    "local_velocity",
    "run_terminated",
    "aggregate_replay",
]
VelocityFlag: TypeAlias = Literal[
    "repeat_size",
    "monotonic_growth",
    "rate_acceleration",
]


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
    COST_ROUTED = "cost_routed"  # routing policy put a healthy non-primary provider first


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

UntrackedClientShape = Literal[
    "openai_sdk",
    "native_together",
    "anthropic_sdk",
    "google_generativeai",
    "google_genai",
    "bedrock_boto3",
    "bedrock_aioboto3",
]
UntrackedRuleKind = Literal["unmetered_spend", "unknown"]
UntrackedPosture = Literal["warn", "allow"]
UntrackedScope = Literal[
    "operation",
    "client",
    "resource",
    "raw_response",
    "arbitrary_endpoint",
]
SURFACE_PATH_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,7}$"

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


class UntrackedSurfaceReport(BaseModel):
    """Content-free structural observation for a surface the SDK does not meter."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    client_shape: UntrackedClientShape
    mode: Literal["sync", "async"]
    surface: str = Field(..., max_length=128, pattern=SURFACE_PATH_PATTERN)
    rule_kind: UntrackedRuleKind
    capability_scope: UntrackedScope | None = None
    posture: UntrackedPosture
    occurrences: int = Field(..., ge=1, le=1_000_000_000)
    first_seen_at: datetime
    last_seen_at: datetime
    sdk_instance_id: str = Field(..., max_length=100)
    report_id: str = Field(..., max_length=36, pattern=CALL_ID_PATTERN)

    @field_validator("first_seen_at", "last_seen_at")
    @classmethod
    def normalize_seen_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value

    @model_validator(mode="after")
    def _check_seen_order(self) -> UntrackedSurfaceReport:
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at must be greater than or equal to first_seen_at")
        return self


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
        default=None,
        ge=0,
        le=100_000_000,
        description="Images generated/edited (per_image unit)",
    )
    generation_count: int | None = Field(
        default=None,
        ge=0,
        le=100_000_000,
        description="Discrete generations for per_video / per_generation / per_song units",
    )
    video_seconds: float | None = Field(
        default=None,
        ge=0,
        le=100_000_000,
        description="Video duration in seconds (per_second / per_minute units)",
    )
    audio_seconds: float | None = Field(
        default=None,
        ge=0,
        le=100_000_000,
        description=(
            "Audio duration in seconds (fractional; e.g. whisper verbose_json usage.seconds)"
        ),
    )
    input_characters: int | None = Field(
        default=None,
        ge=0,
        le=100_000_000,
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
        ..., max_length=MODEL_NAME_MAX_LENGTH, description="LLM model name (e.g. gpt-5.5)"
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
    input_tokens: int = Field(
        ..., ge=0, le=ORDINARY_TOKEN_COUNT_MAX, description="Input token count"
    )
    output_tokens: int = Field(
        ..., ge=0, le=ORDINARY_TOKEN_COUNT_MAX, description="Output token count"
    )
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
        max_length=CALL_ID_MAX_LENGTH,
        pattern=CALL_ID_PATTERN,
        description=(
            "uuid per intercepted call; join key for cache-hit spend reconciliation. "
            "Always present on the wire — it is NOT content. Canonical lowercase "
            "UUID, matching the API's own pin: this id and its confirm's are the "
            "SAME id, so both halves of the join key answer to one shape."
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
    parent_agent_run_id: str | None = Field(
        default=None,
        max_length=AGENT_RUN_ID_MAX_LENGTH,
        description=(
            "Immediate enclosing solwyn.run() scope id for orchestrator-to-child "
            "hierarchies. None for top-level and unscoped events."
        ),
    )
    agent_run_name: str | None = Field(
        default=None,
        max_length=AGENT_RUN_NAME_MAX_LENGTH,
        description="Human-readable label passed to solwyn.run(name).",
    )
    lease_id: str | None = Field(
        default=None,
        max_length=LEASE_ID_MAX_LENGTH,
        description=(
            "Budget lease whose claim funded this call's admission, from "
            "LeaseGrantResponse; the same id the call's confirm carries. None "
            "when the call was funded by a per-call reservation, admitted from "
            "the allow cache, admitted fail-open or uncounted, or denied. Server-"
            "minted opaque id (lse_...), never content; bounded by "
            "LEASE_ID_MAX_LENGTH exactly as BudgetConfirmRequest.lease_id (D7)."
        ),
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
    deny_source: DenySource | None = Field(
        default=None,
        description="Structural source of a denied-call receipt.",
    )
    deny_reason: str | None = Field(
        default=None,
        max_length=64,
        description="Bounded structural denial reason; never derived from content.",
    )
    denied_by_period: str | None = Field(
        default=None,
        max_length=32,
        description="Budget period responsible for a denial, when available.",
    )
    estimated_output_bound: int | None = Field(
        default=None,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Exact output-token bound used by the corresponding pre-flight.",
    )
    velocity_flags: list[VelocityFlag] | None = Field(
        default=None,
        max_length=8,
        description="Content-free velocity rule names observed for this call.",
    )
    receipt_aggregate_count: int | None = Field(
        default=None,
        ge=1,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Number of receipts represented by an aggregate replay event.",
    )
    receipt_pricing_input_tokens: int | None = Field(
        default=None,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description=(
            "Original per-call input-token count used only to select the pricing "
            "card for a homogeneous aggregate receipt. Requires receipt_aggregate_count."
        ),
    )

    @model_validator(mode="after")
    def _pricing_basis_requires_aggregate(self) -> MetadataEvent:
        """Keep the pricing-basis hint scoped to homogeneous aggregate replays."""
        if self.receipt_pricing_input_tokens is not None and self.receipt_aggregate_count is None:
            raise ValueError("receipt_pricing_input_tokens requires receipt_aggregate_count")
        return self


class FailoverDirective(BaseModel):
    """Versioned server policy for SDK-managed provider failover."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1"]
    failover_tuning_allowed: bool


class RunControlDirective(BaseModel):
    """Versioned server instruction that stops one exact agent run."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["1"]
    action: Literal["terminate"]
    agent_run_id: str = Field(..., max_length=AGENT_RUN_ID_MAX_LENGTH)
    reason: str = Field(..., max_length=64)


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

        Optional fields are skipped when None. Runtime request construction
        always opts in to the v1 failover directive; direct model construction
        may omit the version for compatibility tests and legacy callers.
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
    tags: dict[TagKey, TagValue] | None = Field(
        default=None,
        max_length=TAGS_MAX_KEYS,
        description=(
            "Exact explicit tag snapshot captured for the pending call, used for "
            "tag-scoped budget admission."
        ),
    )
    failover_directive_version: Literal["1"] | None = Field(
        default=None,
        description="Explicit opt-in to the version 1 server failover directive.",
    )
    run_directive_version: Literal["1"] | None = Field(
        default=None,
        description="Explicit opt-in to version 1 server run-control directives.",
    )
    price_hints_version: Literal["1"] | None = Field(
        default=None,
        description=(
            "Explicit opt-in to server price hints on the response. Omitted requests keep "
            "price_hints null (no statement)."
        ),
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
        None,
        description=(
            "Which budget period triggered denial (e.g. 'daily', 'agent_run'). "
            "INTENTIONALLY defaulted, not required-nullable: the API serializes "
            "directive-v1 responses (the SDK's only wire) with exclude_none, so "
            "allow responses omit the key entirely. Deny-side drift is covered "
            "live by tests/integration/test_live_contract.py."
        ),
    )
    project_id: str = Field(..., description="Project identifier resolved from the API key")
    price_hints: dict[ProviderName, float] | None = Field(
        default=None,
        description=(
            "Server-provided RELATIVE price signal per provider for cost routing; "
            "SDK never computes price"
        ),
    )
    failover_directive: FailoverDirective | None = Field(
        default=None,
        description="Versioned server policy for SDK-managed failover.",
    )
    run_control: RunControlDirective | None = Field(
        default=None,
        description="Versioned server instruction for the exact requested run.",
    )


class LeaseGrantRequest(BaseModel):
    """Ask the API for a token-denominated budget lease for a run.

    The lease replaces the per-call ``/budgets/check`` for run-scoped,
    token-billed traffic: the SDK draws the grant down in memory and renews
    ahead of need. ``fail_open`` declares the client's configured unreachable
    posture, echoed back as ``posture.on_unreachable`` so the grant is the
    single self-describing artifact the outage ladder reads.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize grant requests without null-valued optional fields."""
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("LeaseGrantRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    agent_run_id: str = Field(
        ...,
        max_length=AGENT_RUN_ID_MAX_LENGTH,
        description="Stable id for the solwyn.run() scope the lease is drawn for",
    )
    holder_id: str = Field(
        ...,
        max_length=HOLDER_ID_MAX_LENGTH,
        description="SDK instance id — the lease holder identity",
    )
    model: str = Field(
        ..., max_length=MODEL_NAME_MAX_LENGTH, description="Primary model for the run"
    )
    provider: ProviderName = Field(..., description="Primary provider for the run")
    fallback_providers: list[ProviderName] = Field(
        default_factory=list,
        description="Configured failover providers, in attempt order (declared set)",
    )
    fallback_models: list[Annotated[str, Field(max_length=MODEL_NAME_MAX_LENGTH)]] = Field(
        default_factory=list,
        max_length=8,
        description="Failover models aligned element-for-element with fallback_providers",
    )
    fail_open: bool = Field(
        default=True,
        description="Client's configured unreachable posture (echoed as posture.on_unreachable)",
    )
    estimated_input_tokens: int = Field(
        default=0, ge=0, description="Triggering call's input estimate (demand hint)"
    )
    run_directive_version: Literal["1"] | None = Field(
        default=None,
        description="Explicit opt-in to version 1 server run-control directives.",
    )

    @model_validator(mode="after")
    def _check_chain_hint_alignment(self) -> LeaseGrantRequest:
        """fallback_providers and fallback_models must align element-for-element."""
        if len(self.fallback_models) != len(self.fallback_providers):
            raise ValueError("fallback_models and fallback_providers must have equal length")
        return self


class LeaseRenewRequest(BaseModel):
    """Renew a lease, acknowledging the generation the holder operates under.

    Doubles as the periodic report: ``spent_tokens`` is the trued-up drawdown
    since the last successful report, ``uncounted_*`` the outage fail-open
    tally. ``model``/``provider``/chain optionally re-declare the model set —
    the server UNIONS them into the lease's declared set.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize renewals without null-valued optional fields.

        The optional re-declaration fields are skipped when None so a renewal
        that re-declares nothing keeps its wire bytes minimal (the Cloud-API
        model forbids unknown keys).
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("LeaseRenewRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    lease_id: str = Field(..., max_length=LEASE_ID_MAX_LENGTH, description="Lease being renewed")
    holder_id: str = Field(
        ..., max_length=HOLDER_ID_MAX_LENGTH, description="SDK instance id — the lease holder"
    )
    generation: int = Field(
        ...,
        ge=0,
        description="ECHO of the grant the holder operates under — also acknowledges it",
    )
    spent_tokens: int = Field(
        default=0, ge=0, description="Trued-up drawdown since the last successful report"
    )
    reserved_tokens: int = Field(
        default=0, ge=0, description="Currently in-flight reservations (demand hint)"
    )
    uncounted_calls: int = Field(
        default=0, ge=0, description="Outage fail-open call tally since the last report"
    )
    uncounted_tokens: int = Field(
        default=0, ge=0, description="Outage fail-open token tally since the last report"
    )
    model: str | None = Field(
        default=None,
        max_length=MODEL_NAME_MAX_LENGTH,
        description="Optional re-declaration; the server unions it into the declared set",
    )
    provider: ProviderName | None = Field(
        default=None, description="Optional provider re-declaration"
    )
    fallback_providers: list[ProviderName] = Field(
        default_factory=list, description="Optional failover provider re-declaration"
    )
    fallback_models: list[Annotated[str, Field(max_length=MODEL_NAME_MAX_LENGTH)]] = Field(
        default_factory=list,
        max_length=8,
        description="Failover models aligned element-for-element with fallback_providers",
    )
    run_directive_version: Literal["1"] | None = Field(
        default=None,
        description="Explicit opt-in to version 1 server run-control directives.",
    )

    @model_validator(mode="after")
    def _check_chain_hint_alignment(self) -> LeaseRenewRequest:
        """fallback_providers and fallback_models must align element-for-element."""
        if len(self.fallback_models) != len(self.fallback_providers):
            raise ValueError("fallback_models and fallback_providers must have equal length")
        return self


class LeaseSurrenderRequest(BaseModel):
    """Release a lease cleanly (DHCPRELEASE-style) with a final true-up."""

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_none(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Serialize surrenders without null-valued optional fields."""
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("LeaseSurrenderRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    lease_id: str = Field(
        ..., max_length=LEASE_ID_MAX_LENGTH, description="Lease being surrendered"
    )
    holder_id: str = Field(
        ..., max_length=HOLDER_ID_MAX_LENGTH, description="SDK instance id — the lease holder"
    )
    generation: int = Field(..., ge=0, description="Generation the holder operates under")
    spent_tokens: int = Field(default=0, ge=0, description="Final true-up report")


class LeasePosture(BaseModel):
    """The customer-chosen verdicts the outage ladder reads off a grant."""

    model_config = ConfigDict(extra="forbid")

    mode: BudgetMode = Field(..., description="The CUSTOMER's configured project budget mode")
    on_unreachable: Literal["fail_open", "local_enforce"] = Field(
        ..., description="Posture when Solwyn is unreachable (echo of the client's fail_open)"
    )


class LeaseGrantResponse(BaseModel):
    """API response to a lease grant OR renewal.

    The lease fields are present iff the run is eligible AND allowed; the
    display snapshot is always present. Core serializes directive-v1 style
    (``exclude_none``), so the conditionally-omitted fields carry explicit
    ``None`` defaults per repo convention — their server-side drift is caught
    by tests/integration/test_live_contract.py, not by required-field
    validation.
    """

    model_config = ConfigDict(extra="forbid")

    eligible: bool = Field(
        ..., description="False → the run is lease-ineligible; the SDK uses the legacy path"
    )
    ineligible_reason: str | None = Field(
        default=None,
        description=(
            "Why the run is lease-ineligible (e.g. 'unit_priced_model', "
            "'zero_rate_model', 'scoped_rules_present'). Omitted when eligible."
        ),
    )
    allowed: bool = Field(
        ..., description="False → authoritative HARD DENY verdict (feeds sticky-deny)"
    )
    denied_by_period: str | None = Field(
        default=None,
        description=(
            "Which budget period triggered denial (e.g. 'daily', 'agent_run'). "
            "Deny-only; omitted on allow responses."
        ),
    )
    lease_id: str | None = Field(
        default=None,
        max_length=LEASE_ID_MAX_LENGTH,
        description=(
            "Granted lease id. Bounded lock-step with the renew/surrender "
            "REQUEST bound: a drifted id must fail HERE, where the enforcer "
            "treats a malformed grant as ineligible and drops to the per-call "
            "path, not later when the renewal/surrender request is built on a "
            "customer call."
        ),
    )
    generation: int | None = Field(
        default=None, description="Grant generation — starts at 1, +1 per renewal (fencing)"
    )
    granted_tokens: int | None = Field(
        default=None,
        description=(
            "Tokens granted for local drawdown. May be 0 (e.g. alert_only past "
            "cap): admission then falls to the legacy per-call path."
        ),
    )
    refresh_interval_s: float | None = Field(
        default=None, description="Seconds until a renewal is due (server jittered)"
    )
    lease_length_s: float | None = Field(
        default=None, description="Seconds of granted authority from receipt (monotonic)"
    )
    headroom_share_tokens: int | None = Field(
        default=None, description="This holder's apportioned share of run headroom"
    )
    posture: LeasePosture | None = Field(
        default=None, description="Customer's mode + unreachable posture for the outage ladder"
    )
    final_grant: bool | None = Field(
        default=None, description="True → wind-down; no further renewal will be granted"
    )
    project_id: str = Field(..., description="Project identifier resolved from the API key")
    mode: BudgetMode = Field(..., description="Current budget enforcement mode")
    budget_limit: float = Field(..., description="Total budget limit for current period in USD")
    current_usage: float = Field(..., description="Current spend in USD for this period")
    remaining_budget: float = Field(..., description="Remaining budget in USD for current period")
    run_control: RunControlDirective | None = Field(
        default=None,
        description="Versioned server instruction for the exact leased run.",
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

        Every optional field is skipped when None. That keeps the bearer-key
        providers' confirm wire bytes unchanged (provider_region/service_tier/
        media_usage) AND makes the settlement key exclusive on the wire: a
        reservation-settled confirm carries no ``lease_id`` key and a
        lease-settled confirm carries no ``reservation_id`` key (the Cloud-API
        model forbids unknown keys).
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("BudgetConfirmRequest serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        return {key: value for key, value in serialized.items() if value is not None}

    reservation_id: str | None = Field(
        default=None,
        description=(
            "Budget reservation ID returned by BudgetCheckResponse. Exactly one "
            "of reservation_id / lease_id must be set."
        ),
    )
    lease_id: str | None = Field(
        default=None,
        description=(
            "Lease the call drew down (PJ-2 lease settlement). Exactly one of "
            "reservation_id / lease_id must be set."
        ),
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
        max_length=CALL_ID_MAX_LENGTH,
        pattern=CALL_ID_PATTERN,
        description=(
            "uuid per intercepted call; dedups confirm vs metadata reconciliation. "
            "Canonical lowercase UUID, matching the API's own pin — a confirm is "
            "the settlement of real spend, so an id the server would 422 must fail "
            "here, at the seam that built it."
        ),
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

    @model_validator(mode="after")
    def _check_exactly_one_settlement_key(self) -> BudgetConfirmRequest:
        """A confirm settles EITHER a reservation OR a lease — never both/neither."""
        if (self.reservation_id is None) == (self.lease_id is None):
            raise ValueError("exactly one of reservation_id / lease_id must be set")
        return self
