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

from typing import Any

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


class GoogleAdapter:
    """Extracts normalized TokenDetails from Google Gemini API responses."""

    @property
    def name(self) -> str:
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

    def extract_service_tier(self, response: Any) -> str | None:
        """Google responses do not expose a service tier."""
        return None

    def extract_region(self, client: Any) -> str | None:
        """Gemini API pricing is not regional."""
        return None

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Google streams include usage_metadata by default — no changes needed."""
        return dict(kwargs)

    def create_stream_accumulator(self) -> GoogleStreamAccumulator:
        return GoogleStreamAccumulator()


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
