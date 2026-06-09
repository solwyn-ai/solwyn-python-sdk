"""Google/Gemini provider adapter — token extraction only, no pricing.

Google usage normalization:

    output_tokens (normalized) = candidates_token_count + thoughts_token_count

Google reports thinking tokens (thoughts_token_count) SEPARATELY from the
model's candidate output (candidates_token_count). The candidates count does
NOT include thoughts. Our normalized output_tokens sums both.

reasoning_tokens = thoughts_token_count (the raw thinking count preserved).

Optional fields (thoughts_token_count, cached_content_token_count,
tool_use_prompt_token_count) may be absent on simpler responses — all missing
fields default to 0 via getattr guards.

The usage data lives on response.usage_metadata (not response.usage).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from solwyn._token_details import TokenDetails


def _extract_google_usage(usage_metadata: Any) -> TokenDetails:
    """Extract token usage from a Google usage_metadata object.

    Module-level so both GoogleAdapter and GoogleStreamAccumulator can call it
    without instantiating an adapter. Accepts usage_metadata directly (not the
    full response). Returns TokenDetails() with all zeros when usage_metadata
    is None. Never raises.

    Normalization:
    - input_tokens = prompt_token_count
    - output_tokens = candidates_token_count + thoughts_token_count
      (candidates does NOT include thoughts — must sum both)
    - reasoning_tokens = thoughts_token_count
    - cached_input_tokens = cached_content_token_count
    - tool_use_input_tokens = tool_use_prompt_token_count
    """
    if usage_metadata is None:
        return TokenDetails()

    input_tokens = getattr(usage_metadata, "prompt_token_count", None) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", None) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", None) or 0
    cached = getattr(usage_metadata, "cached_content_token_count", None) or 0
    tool_use = getattr(usage_metadata, "tool_use_prompt_token_count", None) or 0

    return TokenDetails(
        input_tokens=input_tokens,
        output_tokens=candidates + thoughts,
        reasoning_tokens=thoughts,
        cached_input_tokens=cached,
        tool_use_input_tokens=tool_use,
    )


def _mapping_from_config(value: object) -> dict[str, Any]:
    """Return a shallow dict view of a provider config object."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dict(dumped)
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    return {}


def _with_google_http_bound(
    kwargs: dict[str, Any], *, timeout: float, max_retries: int
) -> dict[str, Any]:
    """Inject google-genai per-request HTTP options without importing the extra.

    google-genai accepts ``config={"http_options": ...}`` for generate_content
    calls, and its timeout is milliseconds. We preserve caller config/http_options
    but override the timeout and retry attempts because the chain deadline is a
    mandatory Solwyn bound.
    """
    bounded = dict(kwargs)
    config = _mapping_from_config(bounded.get("config"))
    http_options = _mapping_from_config(config.get("http_options"))
    retry_options = _mapping_from_config(http_options.get("retry_options"))

    timeout_ms = max(1, int(timeout * 1000))
    retry_options["attempts"] = max(1, max_retries + 1)
    http_options["timeout"] = timeout_ms
    http_options["retry_options"] = retry_options
    config["http_options"] = http_options
    bounded["config"] = config
    return bounded


class GoogleAdapter:
    """Extracts normalized TokenDetails from Google Gemini API responses."""

    @property
    def name(self) -> str:
        return "google"

    @property
    def dialect(self) -> Literal["google"]:
        return "google"

    def detect_client(self, client: Any) -> bool:
        """Return True if client module path contains 'google.genai' or 'google.generativeai'."""
        module = getattr(type(client), "__module__", "")
        return "google.genai" in module or "google.generativeai" in module

    def detect_model(self, model: str) -> bool:
        """Return True for gemini-* model prefix."""
        return model.startswith("gemini-")

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from a Google GenerateContentResponse."""
        usage_metadata = getattr(response, "usage_metadata", None)
        return _extract_google_usage(usage_metadata)

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """Google always reports usage_metadata — no estimated fallback."""
        return None

    def extract_service_tier(self, response: Any) -> str | None:
        """Google responses do not expose a service tier."""
        return None

    def extract_region(self, client: Any) -> str | None:
        """Gemini API pricing is not regional."""
        return None

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """Google streams include usage_metadata by default — no changes needed."""
        return dict(kwargs)

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> GoogleStreamAccumulator:
        """Google chunks carry usage_metadata by default — the input estimate
        is not needed."""
        return GoogleStreamAccumulator()

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """generate_content hop: streaming selects the dedicated method.

        google-genai has no ``with_options``, so the mandatory per-hop bound is
        injected as per-request ``config.http_options`` here. Its methods take
        no ``stream`` kwarg — strip it; streaming intent picks the method.
        """
        kwargs = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
        kwargs.pop("stream", None)
        if is_streaming:
            return client.models.generate_content_stream, kwargs
        return client.models.generate_content, kwargs

    def unwrap_stream_source(self, response: Any) -> Any:
        """The streaming call returns the iterable itself."""
        return response

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """Google-dialect callers iterate the stream object directly."""
        return wrapper


class GoogleStreamAccumulator:
    """Accumulates usage from Google streaming chunks.

    Google includes usage_metadata on most chunks. Later chunks have
    more complete data, so we keep the last observed usage_metadata object
    (not the full chunk — avoids retaining candidate text, safety ratings, etc.)
    and extract from it directly.
    """

    def __init__(self) -> None:
        self._last_usage_metadata: Any | None = None

    def observe(self, chunk: Any) -> None:
        usage_metadata = getattr(chunk, "usage_metadata", None)
        if usage_metadata is not None:
            self._last_usage_metadata = usage_metadata

    def finalize(self) -> TokenDetails:
        return _extract_google_usage(self._last_usage_metadata)

    def get_service_tier(self) -> str | None:
        """Google streams do not expose a service tier."""
        return None
