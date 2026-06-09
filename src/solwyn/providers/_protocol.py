"""Provider adapter protocol — extraction only, no pricing.

The SDK is a context engine: adapters extract token details from provider
responses. Cost calculation lives in the Cloud API's PricingService.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator


@runtime_checkable
class ProviderAdapter(Protocol):
    """Interface for provider-specific token extraction.

    Each provider adapter is responsible for:
    - Identifying whether a given SDK client or model string belongs to it
    - Extracting normalized TokenDetails from a raw provider response

    No pricing logic belongs here. Adapters are pure extraction.
    """

    @property
    def name(self) -> str:
        """Provider name (e.g. 'openai', 'anthropic', 'google')."""
        ...

    def detect_client(self, client: Any) -> bool:
        """Return True if this adapter handles the given SDK client instance."""
        ...

    def detect_model(self, model: str) -> bool:
        """Return True if this adapter handles the given model name."""
        ...

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract normalized token usage from a provider response object.

        Must return TokenDetails() with all zeros when usage is unavailable.
        Must never raise — return zeros for any unexpected response shape.
        """
        ...

    def extract_service_tier(self, response: Any) -> str | None:
        """Extract provider service tier from a response, or None when unavailable."""
        ...

    def extract_region(self, client: Any) -> str | None:
        """Return the cloud region the client targets, or None.

        Part of the cost-attribution contract: Bedrock pricing is keyed per
        model AND region, so the served region rides metadata/confirm events.
        Providers without regional pricing return None.
        """
        ...

    def prepare_streaming(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare call kwargs for streaming if needed.

        For example, OpenAI needs stream_options={"include_usage": True}
        to get usage data in the final streaming chunk.

        Returns a (possibly modified) copy of kwargs. Must not mutate the input.
        """
        ...

    def create_stream_accumulator(self) -> StreamUsageAccumulator:
        """Create a fresh accumulator for a new streaming response."""
        ...
