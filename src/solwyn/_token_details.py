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
)


class TokenDetails(BaseModel):
    """Normalized token usage breakdown for one LLM call.

    Provider adapters populate whichever fields their API exposes; the rest
    stay at 0.  The API uses this struct to compute exact costs rather than
    trusting SDK-side estimates.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _serialize_without_default_is_estimated(
        self,
        handler: SerializerFunctionWrapHandler,
        _info: SerializationInfo,
    ) -> dict[str, Any]:
        """Omit ``is_estimated`` from the wire when False (the default).

        Optional wire additions are OMITTED when absent, never emitted as
        defaults — so provider-reported (non-estimated) payloads stay
        byte-identical to the pre-``is_estimated`` wire, and the Cloud-API
        change is isolated to the estimation-fallback calls that need it.
        ``is_estimated=True`` (estimation fallback fired) still serializes.
        """
        data = handler(self)
        if not isinstance(data, dict):
            raise RuntimeError("TokenDetails serializer expected dict output")
        serialized = cast(dict[str, Any], data)
        if not self.is_estimated:
            serialized.pop("is_estimated", None)
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
    is_estimated: bool = Field(
        default=False,
        description=(
            "True when the provider returned no usage data and these counts are "
            "SDK-side length-based estimates (explicit degradation marker — the "
            "API can surface estimated costs distinctly). False for "
            "provider-reported counts."
        ),
    )

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens.  Excluded from serialization."""
        return self.input_tokens + self.output_tokens
