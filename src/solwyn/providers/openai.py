"""OpenAI provider adapter — token extraction only, no pricing.

Handles two OpenAI API response shapes:
- Chat Completions API: usage.prompt_tokens / usage.completion_tokens
- Responses API:        usage.input_tokens  / usage.output_tokens

Detail sub-objects (prompt_tokens_details, completion_tokens_details,
input_tokens_details, output_tokens_details) may be None on older responses
or when not requested — all missing fields default to 0.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from solwyn._constants import SERVICE_TIER_MAX_LENGTH
from solwyn._token_details import TokenDetails

logger = logging.getLogger(__name__)


def _extract_openai_usage(response: Any) -> TokenDetails:
    """Extract token usage from a Chat Completions or Responses API response.

    Module-level so both OpenAIAdapter and OpenAIStreamAccumulator can call it
    without instantiating an adapter. Returns TokenDetails() with all zeros
    when usage is unavailable. Never raises.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenDetails()

    # Detect which API shape we have
    if hasattr(usage, "prompt_tokens"):
        return _extract_chat_completions(usage)
    if hasattr(usage, "input_tokens"):
        return _extract_responses_api(usage)

    return TokenDetails()


def _extract_chat_completions(usage: Any) -> TokenDetails:
    """Extract from Chat Completions API usage object."""
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    return TokenDetails(
        input_tokens=getattr(usage, "prompt_tokens", None) or 0,
        output_tokens=getattr(usage, "completion_tokens", None) or 0,
        cached_input_tokens=getattr(prompt_details, "cached_tokens", None) or 0,
        audio_input_tokens=getattr(prompt_details, "audio_tokens", None) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", None) or 0,
        audio_output_tokens=getattr(completion_details, "audio_tokens", None) or 0,
        accepted_prediction_tokens=getattr(completion_details, "accepted_prediction_tokens", None)
        or 0,
        rejected_prediction_tokens=getattr(completion_details, "rejected_prediction_tokens", None)
        or 0,
    )


def _extract_responses_api(usage: Any) -> TokenDetails:
    """Extract from Responses API usage object.

    Mirrors _extract_chat_completions field-for-field — the Responses API
    exposes the same sub-objects under different names (input_tokens_details
    instead of prompt_tokens_details, output_tokens_details instead of
    completion_tokens_details). All 8 token categories that Chat Completions
    surfaces are surfaced here too.
    """
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)

    return TokenDetails(
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=getattr(usage, "output_tokens", None) or 0,
        cached_input_tokens=getattr(input_details, "cached_tokens", None) or 0,
        audio_input_tokens=getattr(input_details, "audio_tokens", None) or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None) or 0,
        audio_output_tokens=getattr(output_details, "audio_tokens", None) or 0,
        accepted_prediction_tokens=getattr(output_details, "accepted_prediction_tokens", None) or 0,
        rejected_prediction_tokens=getattr(output_details, "rejected_prediction_tokens", None) or 0,
    )


def _extract_service_tier(response: Any) -> str | None:
    """Return a bounded service_tier value, or None when absent/non-string."""
    tier = getattr(response, "service_tier", None)
    if not isinstance(tier, str):
        return None
    if len(tier) > SERVICE_TIER_MAX_LENGTH:
        logger.warning(
            "OpenAI service_tier exceeds %d characters; truncating",
            SERVICE_TIER_MAX_LENGTH,
        )
        return tier[:SERVICE_TIER_MAX_LENGTH]
    return tier


class OpenAIAdapter:
    """Extracts normalized TokenDetails from OpenAI API responses."""

    @property
    def name(self) -> str:
        return "openai"

    def detect_client(self, client: Any) -> bool:
        """Return True if the client's module path contains 'openai'."""
        return "openai" in getattr(type(client), "__module__", "")

    def detect_model(self, model: str) -> bool:
        """Return True for gpt-*, o3*, o4* model prefixes."""
        return model.startswith("gpt-") or model.startswith("o3") or model.startswith("o4")

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from a Chat Completions or Responses API response."""
        return _extract_openai_usage(response)

    def extract_service_tier(self, response: Any) -> str | None:
        """Return service_tier from an OpenAI response, or None if absent.

        OpenAI tiers (default/flex/priority/batch) price at different rates;
        the API stores this on cost_events for future per-tier repricing.
        """
        return _extract_service_tier(response)

    def extract_region(self, client: Any) -> str | None:
        """OpenAI pricing is not regional."""
        return None

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject stream_options so usage appears in the final chunk."""
        kwargs = dict(kwargs)
        stream_options = dict(kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        kwargs["stream_options"] = stream_options
        return kwargs

    def create_stream_accumulator(self) -> OpenAIStreamAccumulator:
        return OpenAIStreamAccumulator()

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Chat Completions hop: streaming rides the ``stream=True`` kwarg.

        timeout/max_retries are ignored — the dispatcher already applied them
        via the SDK's ``with_options``.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.chat.completions.create, kwargs

    def unwrap_stream_source(self, response: Any) -> Any:
        """The streaming call returns the iterable itself."""
        return response

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """OpenAI-dialect callers iterate the stream object directly."""
        return wrapper


class OpenAIStreamAccumulator:
    """Accumulates usage from OpenAI streaming chunks.

    OpenAI includes usage only in the final chunk when the caller sets
    stream_options={"include_usage": True}. We save that chunk and
    delegate to the same extraction logic as non-streaming responses.
    """

    def __init__(self) -> None:
        self._usage_chunk: Any | None = None

    def observe(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage_chunk = chunk

    def finalize(self) -> TokenDetails:
        if self._usage_chunk is None:
            return TokenDetails()
        return _extract_openai_usage(self._usage_chunk)

    def get_service_tier(self) -> str | None:
        """Return service_tier from the saved final-chunk, or None if absent."""
        if self._usage_chunk is None:
            return None
        return _extract_service_tier(self._usage_chunk)
