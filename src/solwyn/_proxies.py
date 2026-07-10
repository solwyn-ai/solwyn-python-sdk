"""Provider-specific proxy classes for LLM API interception.

These thin delegation wrappers let ``Solwyn.chat.completions.create()``
(and the Anthropic/Google equivalents) route through ``_intercepted_call``
while passing everything else through to the underlying client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from solwyn._base import MediaSurfaceSpec, _warn_unmetered_spend_surface_once
from solwyn._privacy import (
    estimate_content_length,
    estimate_embedding_input_tokens,
    estimate_tokens_from_length,
)
from solwyn._token_details import TokenDetails

if TYPE_CHECKING:
    from solwyn.client import AsyncSolwyn, Solwyn


def _bedrock_internal_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rename boto3's ``modelId`` to the pipeline's uniform ``model`` key.

    The whole interception pipeline (estimation, budget check, candidate walk,
    translation) keys on ``kwargs["model"]``; dispatch renames it back to
    ``modelId`` for the actual boto3 call. Raises TypeError (the same class
    boto3 raises for a missing required kwarg) when ``modelId`` is absent.
    """
    if "modelId" not in kwargs:
        raise TypeError("converse() requires the 'modelId' keyword argument")
    renamed = dict(kwargs)
    renamed["model"] = renamed.pop("modelId")
    return renamed


# ---------------------------------------------------------------------------
# Embeddings surface (openai dialect: native OpenAI + all compat profiles)
# ---------------------------------------------------------------------------


def _extract_embedding_usage(response: Any) -> TokenDetails | None:
    """Pull the billable input quantity from an embeddings response's usage block.

    Embeddings emit NO output tokens, so ``usage.prompt_tokens`` is the entire
    billable basis (output stays a TRUE zero, priced at rate 0.0 server-side).
    Native OpenAI always reports it; a compat endpoint that omits, zeroes, or
    garbles it yields None so the request-side estimator takes over rather than
    settling a silent $0. Never raises — the media lifecycle then falls back to
    ``measure_request``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return None
    return TokenDetails(input_tokens=prompt_tokens)


def _measure_embedding_request(kwargs: dict[str, Any], provider: str) -> TokenDetails | None:
    """Request-derived input-token estimate for a usage-less embeddings response.

    Length-only measurement via the privacy-firewall recognizer, marked
    ``is_estimated=True`` (explicit degradation, mirroring the compat chat
    missing-usage fallback). Returns None when nothing measurable is present so
    the billable quantity stays None — never a zero-as-default.
    """
    estimate = estimate_embedding_input_tokens(kwargs, provider)
    if estimate <= 0:
        return None
    return TokenDetails(input_tokens=estimate, is_estimated=True)


def _embeddings_spec(solwyn: Solwyn | AsyncSolwyn) -> MediaSurfaceSpec:
    """Build the embeddings ``MediaSurfaceSpec`` for one client.

    ``surface="embeddings"`` is the adapter dispatch key; ``modality="embedding"``
    is the server billing modality. One spec covers native OpenAI plus all 14
    compat profiles — they share the openai dialect and the
    ``usage.prompt_tokens`` response shape. The request-side estimator binds the
    primary provider name so the char->token ratio matches it.
    """
    provider = solwyn._adapter.name
    return MediaSurfaceSpec(
        surface="embeddings",
        modality="embedding",
        extract_usage=_extract_embedding_usage,
        measure_request=lambda kwargs: _measure_embedding_request(kwargs, provider),
    )


# ---------------------------------------------------------------------------
# Embeddings surface (google dialect: client.models.embed_content)
# ---------------------------------------------------------------------------


def _extract_google_embedding_usage(response: Any) -> TokenDetails | None:
    """Pull the billable input quantity from a Google embeddings response.

    Google bills gemini-embedding models on input tokens; when a response
    carries usage it rides on ``usage_metadata.prompt_token_count`` — the same
    snake_case attribute the chat adapter reads (``_extract_google_usage``), NOT
    the wire form ``usageMetadata.promptTokenCount``, because the google-genai
    SDK exposes snake_case Python attributes. Embeddings emit no output tokens,
    so that count is the whole billable basis. A response that omits, zeroes, or
    garbles it yields None so the request-side estimator takes over rather than
    settling a silent $0. Never raises — the media lifecycle then falls back to
    ``measure_request``. (Today's ``EmbedContentResponse`` exposes no
    ``usage_metadata`` at all, so this returns None in practice and the estimator
    drives billing; the getattr path stays forward-compatible if google adds it.)
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return None
    return TokenDetails(input_tokens=prompt_tokens)


def _measure_google_embedding_request(kwargs: dict[str, Any]) -> TokenDetails | None:
    """Request-derived input-token estimate for a usage-less Google embeddings response.

    Google's ``embed_content`` request text rides on ``contents=`` (a str or a
    list of str/parts), NOT the openai ``input=`` key, so measurement reuses the
    privacy-firewall ``estimate_content_length`` recognizer (which already
    understands google-shaped ``contents=``) and ratio-converts with google's
    char/token ratio, marked ``is_estimated=True``. Length-only: the input text
    is never retained, logged, or concatenated. Returns None when nothing
    measurable is present so the billable quantity stays None — never a
    zero-as-default.
    """
    char_count = estimate_content_length(kwargs)
    if char_count <= 0:
        return None
    return TokenDetails(
        input_tokens=estimate_tokens_from_length(char_count, "google"), is_estimated=True
    )


def _google_embeddings_spec() -> MediaSurfaceSpec:
    """Build the Google embeddings ``MediaSurfaceSpec``.

    ``surface="embeddings"`` is the adapter dispatch key (``GoogleAdapter``
    routes it to ``client.models.embed_content``); ``modality="embedding"`` is
    the server billing modality. Unlike the openai spec this needs no bound
    provider argument — google's provider name is fixed, so the request
    estimator hardcodes the google char/token ratio.
    """
    return MediaSurfaceSpec(
        surface="embeddings",
        modality="embedding",
        extract_usage=_extract_google_embedding_usage,
        measure_request=_measure_google_embedding_request,
    )


# ---------------------------------------------------------------------------
# Sync proxies
# ---------------------------------------------------------------------------


class _SyncChatCompletionsProxy:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() with budget/circuit/reporting."""
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return getattr(self._solwyn._client.chat.completions, name)


class _SyncChatProxy:
    """Proxy for client.chat that provides .completions.create()."""

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self.completions = _SyncChatCompletionsProxy(solwyn)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-completions attributes (OpenAI dialect only).

        This proxy is only useful for OpenAI-dialect clients (OpenAI itself
        plus every OpenAI-compatible provider). Any attribute that is not
        ``completions`` (set in __init__) falls through here.
        """
        if self._solwyn._dialect == "openai":
            return getattr(self._solwyn._client.chat, name)
        raise AttributeError(
            f"'chat.{name}' is not supported. "
            f"The Solwyn chat proxy is OpenAI-dialect-specific; Anthropic uses "
            f"'messages' and Google uses 'models'."
        )


class _SyncEmbeddingsProxy:
    """Proxy for client.embeddings that routes create() through the media lifecycle.

    ``client.embeddings.create()`` (OpenAI's embeddings API, shared by every
    OpenAI-compatible provider) flows through ``_media_call`` instead of the raw
    client, so embeddings spend is budget-checked, confirmed, and reported. Every
    other ``embeddings`` attribute passes through to the underlying client. The
    per-client spec is built once at construction (provider is fixed then).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _embeddings_spec(solwyn)

    def create(self, **kwargs: Any) -> Any:
        """Intercept embeddings.create() with budget/confirm/reporting."""
        return self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's embeddings."""
        return getattr(self._solwyn._client.embeddings, name)


class _SyncMessagesProxy:
    """Proxy for client.messages that intercepts create().

    Enables ``client.messages.create()`` (Anthropic's documented API)
    to go through _intercepted_call instead of __getattr__ pass-through.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.messages, name)


class _SyncModelsProxy:
    """Proxy for client.models that intercepts generate_content(), generate_content_stream(),
    and embed_content().

    Enables ``client.models.generate_content()`` (Google's documented API)
    to go through _intercepted_call. The generate_content_stream() method
    passes _force_stream=True so _intercepted_call dispatches to the correct
    underlying SDK method. ``embed_content()`` routes through the media
    lifecycle (``_media_call``) so embeddings spend is budget-checked,
    confirmed, and reported.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._embeddings_spec = _google_embeddings_spec()

    def generate_content(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(**kwargs)

    def generate_content_stream(self, **kwargs: Any) -> Any:
        return self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    def embed_content(self, **kwargs: Any) -> Any:
        """Intercept models.embed_content() through the media lifecycle.

        An EXPLICIT method (not __getattr__) so embeddings spend is
        budget-checked, confirmed, and reported instead of passing through
        untracked. Because it is defined on the class, it never reaches
        __getattr__ — generate_images/generate_videos keep riding that
        warn-once pass-through until their interception phase ships.
        """
        return self._solwyn._media_call(self._embeddings_spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Google's non-chat media surfaces (generate_images/generate_videos) are
        # methods on client.models, so they arrive here rather than on
        # Solwyn.__getattr__ — warn-once pass-through per the P1.10 posture (P1.8
        # delegates this warn to P1.10). embed_content never reaches here: the
        # explicit method above intercepts it (P1.8).
        attribute = getattr(self._solwyn._client.models, name)
        _warn_unmetered_spend_surface_once(
            adapter=self._solwyn._adapter, dialect=self._solwyn._dialect, surface=name
        )
        return attribute


# ---------------------------------------------------------------------------
# Async proxies
# ---------------------------------------------------------------------------


class _AsyncChatCompletionsProxy:
    """Async proxy for client.chat.completions that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() with budget/circuit/reporting."""
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return getattr(self._solwyn._client.chat.completions, name)


class _AsyncChatProxy:
    """Async proxy for client.chat that provides .completions.create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self.completions = _AsyncChatCompletionsProxy(solwyn)

    def __getattr__(self, name: str) -> Any:
        if self._solwyn._dialect == "openai":
            return getattr(self._solwyn._client.chat, name)
        raise AttributeError(
            f"'chat.{name}' is not supported. "
            f"The Solwyn chat proxy is OpenAI-dialect-specific; Anthropic uses "
            f"'messages' and Google uses 'models'."
        )


class _AsyncEmbeddingsProxy:
    """Async proxy for client.embeddings that routes create() through the media lifecycle.

    Mirror of ``_SyncEmbeddingsProxy``: ``client.embeddings.create()`` flows
    through the async ``_media_call``; every other attribute passes through to
    the underlying client's embeddings.
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _embeddings_spec(solwyn)

    async def create(self, **kwargs: Any) -> Any:
        """Intercept embeddings.create() with budget/confirm/reporting."""
        return await self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's embeddings."""
        return getattr(self._solwyn._client.embeddings, name)


class _AsyncMessagesProxy:
    """Async proxy for client.messages that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solwyn._client.messages, name)


class _AsyncModelsProxy:
    """Async proxy for client.models.

    Intercepts generate_content(), generate_content_stream(), and embed_content().
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._embeddings_spec = _google_embeddings_spec()

    async def generate_content(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(**kwargs)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        return await self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    async def embed_content(self, **kwargs: Any) -> Any:
        """Intercept models.embed_content() through the async media lifecycle.

        Mirror of ``_SyncModelsProxy.embed_content``: an EXPLICIT method (not
        __getattr__) so embeddings spend is budget-checked, confirmed, and
        reported. Being defined on the class, it never reaches __getattr__ —
        generate_images/generate_videos keep their warn-once pass-through.
        """
        return await self._solwyn._media_call(self._embeddings_spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # See _SyncModelsProxy.__getattr__: Google media surfaces (generate_images/
        # generate_videos) warn-once pass-through per the P1.10 posture.
        # embed_content never reaches here — the explicit method above intercepts it.
        attribute = getattr(self._solwyn._client.models, name)
        _warn_unmetered_spend_surface_once(
            adapter=self._solwyn._adapter, dialect=self._solwyn._dialect, surface=name
        )
        return attribute
