"""Provider adapter protocol — extraction only, no pricing.

The SDK is a context engine: adapters extract token details from provider
responses. Cost calculation lives in the Cloud API's PricingService.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator

# The four API shapes the SDK knows how to dispatch and translate.
# A provider's *dialect* is the wire shape it speaks; its *name* is the
# attribution identity (budgets, pricing, circuit breaking). All
# OpenAI-compatible providers share the "openai" dialect under distinct names;
# Bedrock's Converse API is its own dialect.
Dialect = Literal["openai", "anthropic", "google", "bedrock"]

# Non-chat billable SURFACES the SDK can dispatch through the media lifecycle.
# "chat" is not listed: it flows through ``ProviderAdapter.prepare_call`` (the
# always-present chat seam). These are the surfaces later batches wire onto
# ``prepare_media_call`` (P1.7 embeddings, P2 images, P3 audio, P4 video).
MediaSurface = Literal["embeddings", "images", "audio", "video"]


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
        """Provider name (e.g. 'openai', 'groq', 'openrouter').

        Always a valid ``ProviderName`` value — this is the attribution
        identity used for budgets, metadata, and circuit breaking.
        """
        ...

    @property
    def dialect(self) -> Dialect:
        """API dialect this provider speaks ('openai', 'anthropic', 'google').

        Drives dispatch method selection and cross-provider translation.
        Distinct from ``name``: every OpenAI-compatible provider returns
        'openai' here while keeping its own attribution name.
        """
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

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """Estimated TokenDetails when the response carries NO usage block, else None.

        Adapters for providers that always report usage return None
        unconditionally. OpenAI-compatible adapters return a length-based
        estimate marked ``is_estimated=True`` when ``response.usage`` is absent —
        the explicit-degradation fallback that keeps budgets enforcing instead
        of silently recording zero spend. Must never raise.
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

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """Prepare call kwargs for streaming if needed.

        For example, OpenAI needs stream_options={"include_usage": True}
        to get usage data in the final streaming chunk.

        ``cross_provider`` is True when this adapter serves a FAILOVER hop for
        a request authored against a different provider — adapters may then
        sanitize options that were meant for the original target (e.g. strip
        stream_options before a provider that rejects it). On the caller's own
        configured target their explicit options pass through untouched.

        Returns a (possibly modified) copy of kwargs. Must not mutate the input.
        """
        ...

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> StreamUsageAccumulator:
        """Create a fresh accumulator for a new streaming response.

        ``estimated_input_tokens`` is the pre-call length-based input estimate.
        Accumulators for providers that always report streaming usage ignore
        it; OpenAI-compatible accumulators keep it for the missing-usage
        estimated fallback (marked ``is_estimated=True``).
        """
        ...

    def prepare_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Select the provider call method and shape kwargs for one CHAT hop.

        This is the CHAT surface seam — hardcoded to the provider's chat
        completion method. Non-chat media surfaces (embeddings, images, audio,
        video) dispatch through ``prepare_media_call`` on ``MediaSurfaceAdapter``
        instead; keeping them separate is what lets this method stay the stable,
        always-present chat entry point.

        Sans-I/O: returns the bound SDK method (sync OR async client — same
        attribute path; the caller invokes/awaits it) plus a shaped COPY of
        kwargs. Owns every provider-specific dispatch quirk: streaming intent
        (``stream=True`` kwarg vs a dedicated method), per-request HTTP bounds
        for SDKs without ``with_options`` (timeout/max_retries; others ignore
        them), and any model-key rename. Must not mutate the input kwargs and
        must never read prompt content — key-level shaping only.
        """
        ...

    def unwrap_stream_source(self, response: Any) -> Any:
        """Return the iterable to wrap from a streaming-call response.

        Identity for providers whose streaming call returns the iterable
        itself. Bedrock's ConverseStream nests it (``response["stream"]``) —
        the INNER event stream (whose close() releases the connection) is
        what the stream wrapper must own.
        """
        ...

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """Reshape a wrapped stream into THIS adapter's caller dialect.

        Called on the PRIMARY (caller-dialect) adapter with the wrapper and
        the SERVED hop's raw response. Identity (return the wrapper) for
        dialects whose callers iterate the stream object directly; Bedrock
        callers iterate ``result["stream"]``, so its adapter rebuilds the
        boto3 contract dict around the wrapper.
        """
        ...


class MediaSurfaceAdapter(Protocol):
    """Optional extension: adapters that serve non-chat media SURFACES.

    ``ProviderAdapter.prepare_call`` dispatches the CHAT surface only. An
    adapter that also serves media surfaces (embeddings, images, audio, video)
    implements ``prepare_media_call`` — the per-surface dispatch seam the media
    lifecycle (``_media_call``) consumes.

    Deliberately kept OUT of the base ``ProviderAdapter`` protocol (and NOT
    ``@runtime_checkable``): chat-only adapters, and adapters whose media
    surfaces land in later batches, must still satisfy ``ProviderAdapter``
    without being forced to grow this method before they serve any surface.
    The lifecycle discovers it duck-typed and raises ``UnsupportedSurfaceError``
    when it is absent or the requested surface has no branch yet.
    """

    def prepare_media_call(
        self,
        surface: str,
        client: Any,
        kwargs: dict[str, Any],
        *,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Select the SDK method and shape kwargs for one NON-chat media hop.

        The per-surface analogue of ``prepare_call``: given a ``surface`` key
        (a ``MediaSurface`` value), returns the bound SDK method (sync OR async
        client — same attribute path; the caller invokes/awaits it) plus a
        shaped COPY of kwargs. Owns the same provider-specific quirks as
        ``prepare_call`` (per-request HTTP bounds for SDKs without
        ``with_options``, any key rename) but for the media method path (e.g.
        ``client.embeddings.create``). Media surfaces do not stream in v1, so
        there is no ``is_streaming`` axis here.

        Raises ``UnsupportedSurfaceError`` for a surface this adapter does not
        serve. Must not mutate the input kwargs and must never read prompt or
        media content — key-level shaping only.
        """
        ...
