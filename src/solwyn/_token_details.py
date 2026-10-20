"""TokenDetails — normalized token usage breakdown.

Normalized token usage breakdown for one LLM call.
"""

from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from solwyn._constants import ORDINARY_TOKEN_COUNT_MAX

# Compaction subset -> the parent bucket that already includes it. Omitted from
# the wire when 0, so every payload without compaction stays byte-identical.
COMPACTION_SUBSET_PARENTS: dict[str, str] = {
    "compaction_input_tokens": "input_tokens",
    "compaction_cached_input_tokens": "cached_input_tokens",
    "compaction_cache_creation_5m_tokens": "cache_creation_5m_tokens",
    "compaction_cache_creation_1h_tokens": "cache_creation_1h_tokens",
    "compaction_output_tokens": "output_tokens",
}


class TokenDetails(BaseModel):
    """Normalized token usage breakdown for one LLM call.

    Provider adapters populate whichever fields their API exposes; the rest
    stay at 0.  The API uses this struct to compute exact costs rather than
    trusting SDK-side estimates.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_optional_defaults(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Omit ``is_estimated`` when False and each ``compaction_*`` when 0.

        Optional wire additions are OMITTED when absent, never emitted as
        defaults — so provider-reported (non-estimated) payloads stay
        byte-identical to the pre-``is_estimated`` wire, and calls without
        server-side compaction stay byte-identical to the pre-compaction wire.
        The Cloud-API change is isolated to the calls that need it.
        ``is_estimated=True`` (estimation fallback fired) and non-zero
        compaction counts still serialize.
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("TokenDetails serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        if not self.is_estimated:
            serialized.pop("is_estimated", None)
        for name in COMPACTION_SUBSET_PARENTS:
            if getattr(self, name) == 0:
                serialized.pop(name, None)
        return serialized

    input_tokens: int = Field(default=0, ge=0, description="Total input tokens (normalized)")
    output_tokens: int = Field(default=0, ge=0, description="Total output tokens (normalized)")
    cached_input_tokens: int = Field(
        default=0, ge=0, description="Input tokens served from prompt cache"
    )
    cache_creation_5m_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Tokens written to prompt cache with 5-minute TTL (priced at 1.25x base input rate)"
        ),
    )
    cache_creation_1h_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens written to prompt cache with 1-hour TTL (priced at 2x base input rate)",
    )
    reasoning_tokens: int = Field(
        default=0, ge=0, description="Tokens used for chain-of-thought / thinking"
    )
    audio_input_tokens: int = Field(default=0, ge=0, description="Audio input tokens (OpenAI)")
    audio_output_tokens: int = Field(default=0, ge=0, description="Audio output tokens (OpenAI)")
    accepted_prediction_tokens: int = Field(
        default=0, ge=0, description="Predicted output tokens accepted (OpenAI)"
    )
    rejected_prediction_tokens: int = Field(
        default=0, ge=0, description="Predicted output tokens rejected (OpenAI)"
    )
    tool_use_input_tokens: int = Field(
        default=0, ge=0, description="Tokens used for tool/function definitions (Google)"
    )
    image_input_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Image tokens on the INPUT side, a SUBSET of input_tokens (image ⊂ "
            "input, mirroring reasoning ⊂ output). Documented, NOT enforced. "
            "Token-billed image models (e.g. gpt-image-2) price these at their "
            "own image-input rate; the server derives text input = input − image."
        ),
    )
    image_output_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Image tokens on the OUTPUT side, a SUBSET of output_tokens "
            "(image_output ⊂ output). Documented, NOT enforced. Token-billed "
            "image models price these at their own image-output rate."
        ),
    )
    compaction_input_tokens: int = Field(
        default=0,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description=(
            "Input tokens of Anthropic server-side compaction iterations, "
            "normalized like input_tokens (base + cache reads + cache writes). "
            "A SUBSET of input_tokens: the adapter adds compaction usage into "
            "the parent buckets because Anthropic reports it only in "
            "usage.iterations, outside the top-level usage fields."
        ),
    )
    compaction_cached_input_tokens: int = Field(
        default=0,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Compaction cache reads — a SUBSET of cached_input_tokens",
    )
    compaction_cache_creation_5m_tokens: int = Field(
        default=0,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Compaction 5-minute cache writes — a SUBSET of cache_creation_5m_tokens",
    )
    compaction_cache_creation_1h_tokens: int = Field(
        default=0,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Compaction 1-hour cache writes — a SUBSET of cache_creation_1h_tokens",
    )
    compaction_output_tokens: int = Field(
        default=0,
        ge=0,
        le=ORDINARY_TOKEN_COUNT_MAX,
        description="Compaction output tokens — a SUBSET of output_tokens",
    )
    is_estimated: bool = Field(
        default=False,
        description=(
            "True when the provider returned no usage data and these counts are "
            "SDK-side length-based estimates (explicit degradation marker — the "
            "API can surface estimated costs distinctly). False for "
            "provider-reported counts."
        ),
    )

    @model_validator(mode="after")
    def _compaction_subsets_fit_parents(self) -> "TokenDetails":
        # Mirrors Core's shared model: each compaction bucket is included in
        # its parent, and compaction cache buckets are included in
        # compaction_input_tokens.
        for name, parent in COMPACTION_SUBSET_PARENTS.items():
            if getattr(self, name) > getattr(self, parent):
                raise ValueError(f"{parent} must include {name}")
        compaction_cache_total = (
            self.compaction_cached_input_tokens
            + self.compaction_cache_creation_5m_tokens
            + self.compaction_cache_creation_1h_tokens
        )
        if compaction_cache_total > self.compaction_input_tokens:
            raise ValueError(
                "compaction_input_tokens must include compaction_cached_input_tokens, "
                "compaction_cache_creation_5m_tokens, and compaction_cache_creation_1h_tokens"
            )
        # The non-compaction remainder must satisfy the whole-call cache
        # invariant on its own: its uncached input cannot go negative.
        cache_total = (
            self.cached_input_tokens + self.cache_creation_5m_tokens + self.cache_creation_1h_tokens
        )
        if (self.compaction_input_tokens - compaction_cache_total) > (
            self.input_tokens - cache_total
        ):
            raise ValueError("uncached input_tokens must include the uncached compaction input")
        return self

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens.  Excluded from serialization."""
        return self.input_tokens + self.output_tokens
