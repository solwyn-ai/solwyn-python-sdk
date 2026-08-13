"""Provider-specific proxy classes for LLM API interception.

These thin delegation wrappers let ``Solwyn.chat.completions.create()``
(and the Anthropic/Google equivalents) route through ``_intercepted_call``
while routing every provider pass-through through the shared surface policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from solwyn._base import MediaSurfaceSpec
from solwyn._privacy import (
    estimate_content_length,
    estimate_embedding_input_tokens,
    estimate_tokens_from_length,
    measure_google_image_media,
    measure_image_media,
    measure_openai_video_media,
    measure_speech_media,
    measure_video_media,
)
from solwyn._surfaces import SurfaceCondition, SurfaceSource
from solwyn._token_details import TokenDetails
from solwyn._types import MediaUsage
from solwyn.providers.openai import (
    _AUDIO_OP_KEY,
    _IMAGE_OP_KEY,
    _extract_image_usage,
    _extract_transcription_usage,
    _is_untracked_tts_model,
    _measure_transcription_media,
)

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
# Images surface (openai dialect: native gpt-image token-billed + compat per-image)
# ---------------------------------------------------------------------------


def _measure_image_media(kwargs: dict[str, Any], _response: Any) -> MediaUsage:
    """Settled request-derived image ``MediaUsage`` (config values only).

    The response is intentionally IGNORED: image billing quantities are
    request-DETERMINED (n images at a given size/quality). gpt-image echoes
    size/quality at the top level of its response, but the request params are
    preferred and measured in the firewall — response fields are never read for
    billing (privacy). Because the estimate is EXACT for images, this settled
    measurement and the pre-flight estimate use the same firewall builder.
    """
    return measure_image_media(kwargs)


def _images_spec() -> MediaSurfaceSpec:
    """Build the images ``MediaSurfaceSpec`` (openai dialect: native + all compat).

    ``surface="images"`` is the adapter dispatch key; ``modality="image"`` is the
    server billing modality. ONE spec covers native OpenAI (token-billed gpt-image,
    whose usage carries image buckets) AND every compat profile (per-image,
    usage-less):

    - ``extract_usage`` reads gpt-image token usage (input/output incl. image
      buckets), None when the endpoint reports none (compat FLUX returns
      ``usage: null``; dall-e reports no usage).
    - ``measure_request`` returns None: images have NO request-derived TOKEN
      estimate (image tokens exist only in the gpt-image response usage). Image
      quantities are non-token and ride ``measure_media`` instead.
    - ``measure_media`` / ``estimate_media`` build the request-derived
      ``MediaUsage`` (n->image_count, size->resolution, quality->quality) so BOTH
      bases ride the call when observable; the server's card unit picks (native
      gpt-image → token card; compat/dall-e → per-image card).
    """
    return MediaSurfaceSpec(
        surface="images",
        modality="image",
        extract_usage=_extract_image_usage,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_image_media,
        estimate_media=measure_image_media,
    )


# ---------------------------------------------------------------------------
# Images surface (google dialect: client.models.generate_images — imagen)
# ---------------------------------------------------------------------------


def _measure_google_image_media(kwargs: dict[str, Any], _response: Any) -> MediaUsage:
    """Settled request-derived image ``MediaUsage`` for a Google generate_images call.

    The response is intentionally IGNORED: imagen exposes NO usage, so the
    billable quantity is request-DETERMINED (``config.number_of_images``). Because
    the estimate is EXACT for images, this settled measurement and the pre-flight
    estimate share the same firewall builder.
    """
    return measure_google_image_media(kwargs)


def _google_images_spec() -> MediaSurfaceSpec:
    """Build the Google images ``MediaSurfaceSpec`` (imagen: request-derived per-image).

    ``surface="images"`` is the adapter dispatch key (``GoogleAdapter`` routes it
    to ``client.models.generate_images``); ``modality="image"`` is the server
    billing modality. imagen models are token-usage-less and priced on flat
    per-image cards, so:

    - ``extract_usage`` returns None: imagen responses carry no token usage. (The
      TOKEN-billed google image model, gemini-3-pro-image, rides the CHAT surface
      ``generate_content`` instead — its image tokens flow via the usageMetadata
      modality buckets, NOT this seam.)
    - ``measure_request`` returns None: no request-derived TOKEN estimate exists.
    - ``measure_media`` / ``estimate_media`` build the request-derived
      ``MediaUsage`` (``config.number_of_images`` -> ``image_count``) so the
      per-image quantity is the sole billable basis — an EXACT count, never a
      silent $0.
    """
    return MediaSurfaceSpec(
        surface="images",
        modality="image",
        extract_usage=lambda _response: None,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_google_image_media,
        estimate_media=measure_google_image_media,
    )


# ---------------------------------------------------------------------------
# Video surface (google dialect: client.models.generate_videos — veo)
# ---------------------------------------------------------------------------


def _measure_video_media(kwargs: dict[str, Any], _response: Any) -> MediaUsage:
    """Settled request-derived video ``MediaUsage`` for a Google generate_videos call.

    The response is intentionally IGNORED: generate_videos returns a long-running
    operation with no usage, so the billable quantity is request-DETERMINED
    (``config.duration_seconds`` at the ``config.resolution`` variant). Video always
    settles at INITIATION marked ``is_estimated=True`` — the same firewall builder
    serves both the pre-flight estimate and this settled measurement.
    """
    return measure_video_media(kwargs)


def _google_videos_spec() -> MediaSurfaceSpec:
    """Build the Google video ``MediaSurfaceSpec`` (veo: request-derived per-second).

    ``surface="video"`` is the adapter dispatch key (``GoogleAdapter`` routes it
    to ``client.models.generate_videos``); ``modality="video"`` is the server
    billing modality. Video generation is asynchronous — the call returns a
    long-running operation carrying no usage — so quantities are request-derived
    and billing settles at INITIATION:

    - ``extract_usage`` returns None: the operation object carries no token usage.
    - ``measure_request`` returns None: no request-derived TOKEN estimate exists.
    - ``measure_media`` / ``estimate_media`` build the request-derived
      ``MediaUsage`` (``config.duration_seconds`` -> ``video_seconds`` at the
      ``config.resolution`` variant), ALWAYS ``is_estimated=True``. The pre-flight
      estimate is precise (seconds x the resolution variant's per-second rate
      server-side, so an oversized request is denied before the provider is
      called); the settle reuses the same builder because the returned operation
      offers nothing better. The over-count is deliberate and conservative — the
      provider does not charge for failed/blocked generations. An absent duration
      leaves ``video_seconds`` None so the call is tracked unpriced, never a
      guessed duration and never a silent $0.
    """
    return MediaSurfaceSpec(
        surface="video",
        modality="video",
        extract_usage=lambda _response: None,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_video_media,
        estimate_media=measure_video_media,
    )


# ---------------------------------------------------------------------------
# Audio transcriptions surface (openai dialect: native + all compat incl. Groq)
# ---------------------------------------------------------------------------


def _transcriptions_spec() -> MediaSurfaceSpec:
    """Build the audio-transcriptions ``MediaSurfaceSpec`` (openai dialect).

    ``surface="audio"`` is the adapter dispatch key (routes to
    ``client.audio.transcriptions.create``); ``modality="audio"`` is the server
    billing modality. ONE spec covers both billable shapes the transcription
    models report — the shared extractor discriminates on ``usage.type``:

    - ``extract_usage`` returns the TOKEN basis for the token-billed models
      (gpt-4o-transcribe / gpt-4o-mini-transcribe), whose ``audio_tokens`` ride
      ``audio_input_tokens``; None for the duration-billed model (whisper-1) and
      for a non-JSON response with no usage.
    - ``measure_request`` returns None: audio duration is not derivable from the
      request (the SDK never touches the file bytes).
    - ``measure_media`` returns the DURATION basis (``usage.seconds`` ->
      ``audio_seconds``) for whisper-1; None otherwise.

    A non-JSON response_format (text/srt/vtt) carries no usage → both bases None →
    the call is tracked UNPRICED, with a one-time hint (emitted by the extractor)
    to use a JSON response_format for priced tracking.
    """
    return MediaSurfaceSpec(
        surface="audio",
        modality="audio",
        extract_usage=_extract_transcription_usage,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_transcription_media,
    )


# ---------------------------------------------------------------------------
# Audio speech (TTS) surface (openai dialect: native + all compat)
# ---------------------------------------------------------------------------


def _measure_speech_media(kwargs: dict[str, Any], _response: Any) -> MediaUsage | None:
    """Settled request-derived TTS ``MediaUsage`` (input character count only).

    The response is intentionally IGNORED: a TTS response is raw audio bytes with
    NO usage metadata, so the SOLE billable basis is the request's ``input`` text
    LENGTH, measured in the firewall (``input_characters``). Because the character
    count is EXACT, this settled measurement and the pre-flight estimate share the
    same firewall builder. Returns None when ``input`` is non-str/absent so an
    unobservable quantity stays None rather than a zero-as-default.
    """
    return measure_speech_media(kwargs)


def _speech_spec() -> MediaSurfaceSpec:
    """Build the audio-speech (TTS) ``MediaSurfaceSpec`` (openai dialect).

    ``surface="audio"`` is the adapter dispatch key (the op marker routes it to
    ``client.audio.speech.create``); ``modality="audio"`` is the server billing
    modality. TTS responses carry ZERO usage metadata, so:

    - ``extract_usage`` returns None: there is no response TOKEN basis to read.
    - ``measure_request`` returns None: TTS has no request-derived TOKEN estimate
      (audio-output tokens exist only in a response this surface never receives).
    - ``measure_media`` / ``estimate_media`` build the request-derived
      ``MediaUsage`` (``input`` length -> ``input_characters``) so the exact
      character count is the sole billable basis on BOTH the pre-flight check and
      the settled confirm — a char-priced card prices chars/1e6 x rate server-side.

    Token-billed TTS models (gpt-4o-mini-tts) publish no usage at all; those calls
    are carved out UPSTREAM (in the speech proxy) and never reach this spec.
    """
    return MediaSurfaceSpec(
        surface="audio",
        modality="audio",
        extract_usage=lambda _response: None,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_speech_media,
        estimate_media=measure_speech_media,
    )


# ---------------------------------------------------------------------------
# Video surface (openai dialect: client.videos.create — Sora)
# ---------------------------------------------------------------------------


def _measure_openai_video_media(kwargs: dict[str, Any], _response: Any) -> MediaUsage:
    """Settled request-derived video ``MediaUsage`` for an OpenAI videos.create call.

    The response is intentionally IGNORED: videos.create returns an async video
    job with no usage, so the billable quantity is request-DETERMINED (top-level
    ``seconds`` at the ``size``-derived resolution label). Video always settles at
    INITIATION marked ``is_estimated=True`` — the same firewall builder serves both
    the pre-flight estimate and this settled measurement.
    """
    return measure_openai_video_media(kwargs)


def _openai_videos_spec() -> MediaSurfaceSpec:
    """Build the OpenAI video ``MediaSurfaceSpec`` (Sora: request-derived per-second).

    ``surface="video"`` is the adapter dispatch key (``OpenAIAdapter`` routes it to
    ``client.videos.create``); ``modality="video"`` is the server billing modality.
    Video generation is asynchronous — the call returns a video job carrying no
    usage — so quantities are request-derived and billing settles at INITIATION:

    - ``extract_usage`` returns None: the job object carries no token usage.
    - ``measure_request`` returns None: no request-derived TOKEN estimate exists.
    - ``measure_media`` / ``estimate_media`` build the request-derived
      ``MediaUsage`` (top-level ``seconds`` → ``video_seconds`` at the
      ``size``-derived resolution label), ALWAYS ``is_estimated=True``. The
      pre-flight estimate is precise (seconds × the resolution variant's
      per-second rate server-side, so an oversized request is denied before the
      provider is called); the settle reuses the same builder because the returned
      job offers nothing better. The over-count is deliberate and conservative.
      ``seconds`` / ``size`` default to OpenAI's documented values when absent, so
      a bare call is priced, never a silent $0.
    """
    return MediaSurfaceSpec(
        surface="video",
        modality="video",
        extract_usage=lambda _response: None,
        measure_request=lambda _kwargs: None,
        measure_media=_measure_openai_video_media,
        estimate_media=measure_openai_video_media,
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
        self._solwyn._enforce_explicit_surface(
            "chat.completions.create", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.chat.completions,
            name=name,
            path=f"chat.completions.{name}",
            source=SurfaceSource.RAW,
        )


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
            return self._solwyn._resolve_public_attribute(
                self._solwyn._client.chat,
                name=name,
                path=f"chat.{name}",
                source=SurfaceSource.RAW,
            )
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
    other ``embeddings`` attribute is resolved by the shared surface policy. The
    per-client spec is built once at construction (provider is fixed then).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _embeddings_spec(solwyn)

    def create(self, **kwargs: Any) -> Any:
        """Intercept embeddings.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("embeddings.create", source=SurfaceSource.WRAPPER)
        return self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's embeddings."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.embeddings,
            name=name,
            path=f"embeddings.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncImagesProxy:
    """Proxy for client.images that routes generate()/edit() through the media lifecycle.

    ``client.images.generate()`` and ``client.images.edit()`` (OpenAI's images
    API, shared by every OpenAI-compatible provider) flow through ``_media_call``
    so image spend is budget-checked, confirmed, and reported. Both carry a
    billable basis: native gpt-image reports token usage (with image buckets),
    and every dialect carries the request-derived per-image quantity — BOTH are
    sent when observable, and the server's card unit picks. Every other
    ``images`` attribute (``create_variation``, …) is resolved by the shared
    surface policy. On a non-openai client the media seam raises
    ``UnsupportedSurfaceError`` (that adapter serves no images seam). The
    per-client spec is built once (the spec is provider-agnostic; the adapter
    dispatch differs).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _images_spec()

    def generate(self, **kwargs: Any) -> Any:
        """Intercept images.generate() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("images.generate", source=SurfaceSource.WRAPPER)
        return self._solwyn._media_call(self._spec, **kwargs)

    def edit(self, **kwargs: Any) -> Any:
        """Intercept images.edit() through the same lifecycle.

        Both images ops share the ONE ``images`` surface; the private op marker
        (stripped in ``prepare_media_call``) routes this hop to
        ``client.images.edit`` rather than ``.generate``.
        """
        self._solwyn._enforce_explicit_surface("images.edit", source=SurfaceSource.WRAPPER)
        return self._solwyn._media_call(self._spec, **{**kwargs, _IMAGE_OP_KEY: "edit"})

    def __getattr__(self, name: str) -> Any:
        """Pass through non-generate/edit attributes to the client's images."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.images,
            name=name,
            path=f"images.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncAudioTranscriptionsProxy:
    """Proxy for client.audio.transcriptions that routes create() through the media lifecycle.

    ``client.audio.transcriptions.create()`` (OpenAI's transcription API, shared by
    every OpenAI-compatible provider incl. Groq whisper) flows through
    ``_media_call`` so transcription spend is budget-checked, confirmed, and
    reported. The billable basis is per-model: token usage for the
    gpt-4o-transcribe family (``audio_input_tokens``), duration usage for whisper-1
    (``audio_seconds``); a non-JSON response_format yields no usage and is tracked
    UNPRICED. Every other ``transcriptions`` attribute is resolved by the shared
    surface policy. The per-client spec is built once (it is provider-agnostic;
    the adapter dispatch differs).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _transcriptions_spec()

    def create(self, **kwargs: Any) -> Any:
        """Intercept audio.transcriptions.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface(
            "audio.transcriptions.create", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's audio.transcriptions."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio.transcriptions,
            name=name,
            path=f"audio.transcriptions.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncAudioSpeechProxy:
    """Proxy for client.audio.speech that routes create() through the media lifecycle.

    ``client.audio.speech.create()`` (OpenAI's TTS API, shared by every
    OpenAI-compatible provider) flows through ``_media_call`` so speech spend is
    budget-checked, confirmed, and reported. TTS responses carry NO usage
    metadata, so the sole billable basis is the request's ``input`` character
    count, measured in the firewall (``input_characters``) and priced server-side.

    Token-billed TTS models (``gpt-4o-mini-tts`` and its dated snapshots) publish
    no usage of any kind, so their audio-output tokens are unobservable. The
    shared conditional policy decides whether those calls warn, raise, or pass
    through to the raw client. Every other ``speech`` attribute is resolved by
    the shared surface policy. On a non-openai client the media seam raises
    ``UnsupportedSurfaceError`` (that adapter serves no audio seam). The
    per-client spec is built once (it is provider-agnostic; the adapter dispatch
    differs).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _speech_spec()

    def create(self, **kwargs: Any) -> Any:
        """Intercept audio.speech.create(); carve out untracked token-billed TTS models.

        A token-billed TTS model has no observable usage, so it cannot be priced;
        the shared conditional policy runs before its raw dispatch. Every other
        model rides the media lifecycle with the ``audio`` op marker selecting
        speech.
        """
        self._solwyn._enforce_explicit_surface("audio.speech.create", source=SurfaceSource.WRAPPER)
        if _is_untracked_tts_model(kwargs.get("model")):
            self._solwyn._enforce_explicit_surface(
                "audio.speech.create",
                source=SurfaceSource.SYNTHETIC_POLICY,
                condition=SurfaceCondition.OPENAI_UNTRACKED_TTS_MODEL,
            )
            return self._solwyn._client.audio.speech.create(**kwargs)
        return self._solwyn._media_call(self._spec, **{**kwargs, _AUDIO_OP_KEY: "speech"})

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's audio.speech."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio.speech,
            name=name,
            path=f"audio.speech.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncAudioProxy:
    """Proxy for client.audio with guarded pass-throughs and tracked media seams.

    ``transcriptions`` and ``speech`` are the intercepted audio sub-surfaces (their
    ``create`` routes through the media lifecycle). ``translations`` and every
    other ``audio`` attribute use the shared surface policy. On a non-openai
    client the media seams raise ``UnsupportedSurfaceError`` (that adapter serves
    no audio seam).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self.transcriptions = _SyncAudioTranscriptionsProxy(solwyn)
        self.speech = _SyncAudioSpeechProxy(solwyn)

    @property
    def translations(self) -> Any:
        """Resolve the client's untracked audio translations resource."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio,
            name="translations",
            path="audio.translations",
            source=SurfaceSource.RAW,
        )

    def __getattr__(self, name: str) -> Any:
        """Pass through other audio attributes (e.g. with_raw_response) to the client."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio,
            name=name,
            path=f"audio.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncVideosProxy:
    """Proxy for client.videos that routes create() through the media lifecycle.

    ``client.videos.create()`` (OpenAI's Sora video API) flows through
    ``_media_call`` so video spend is budget-checked, confirmed, and reported.
    Video generation is asynchronous — the call returns a video job carrying no
    usage — so the sole billable basis is the request-derived per-second
    ``MediaUsage`` (top-level ``seconds`` at the ``size``-derived resolution
    label), settled at INITIATION with ``is_estimated=True``. The returned job
    object is passed back untouched: callers poll it themselves; the SDK never
    wraps or polls it. Every other ``videos`` attribute (``retrieve``,
    ``download_content``, …) is resolved by the shared surface policy. Sora is
    OpenAI-only, so on a non-openai client (including OpenAI-compatible profiles)
    ``.create()`` fails loud with ``UnsupportedSurfaceError`` (that adapter serves
    no video seam). The per-client spec is built once (it is provider-agnostic;
    the adapter dispatch differs).
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._spec = _openai_videos_spec()

    def create(self, **kwargs: Any) -> Any:
        """Intercept videos.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("videos.create", source=SurfaceSource.WRAPPER)
        return self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's videos."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.videos,
            name=name,
            path=f"videos.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncMessagesProxy:
    """Proxy for client.messages that intercepts create().

    Enables ``client.messages.create()`` (Anthropic's documented API)
    to go through _intercepted_call instead of __getattr__ pass-through.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn

    def create(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface("messages.create", source=SurfaceSource.WRAPPER)
        return self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.messages,
            name=name,
            path=f"messages.{name}",
            source=SurfaceSource.RAW,
        )


class _SyncModelsProxy:
    """Proxy for client.models that intercepts generate_content(), generate_content_stream(),
    embed_content(), generate_images(), and generate_videos().

    Enables ``client.models.generate_content()`` (Google's documented API)
    to go through _intercepted_call. The generate_content_stream() method
    passes _force_stream=True so _intercepted_call dispatches to the correct
    underlying SDK method. ``embed_content()``, ``generate_images()``, and
    ``generate_videos()`` route through the media lifecycle (``_media_call``) so
    their spend is budget-checked, confirmed, and reported.
    """

    def __init__(self, solwyn: Solwyn) -> None:
        self._solwyn = solwyn
        self._embeddings_spec = _google_embeddings_spec()
        self._images_spec = _google_images_spec()
        self._videos_spec = _google_videos_spec()

    def generate_content(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface(
            "models.generate_content", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._intercepted_call(**kwargs)

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface(
            "models.generate_content_stream", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    def embed_content(self, **kwargs: Any) -> Any:
        """Intercept models.embed_content() through the media lifecycle.

        An EXPLICIT method (not __getattr__) so embeddings spend is
        budget-checked, confirmed, and reported instead of passing through
        untracked. Because it is defined on the class, it never reaches
        __getattr__.
        """
        self._solwyn._enforce_explicit_surface("models.embed_content", source=SurfaceSource.WRAPPER)
        return self._solwyn._media_call(self._embeddings_spec, **kwargs)

    def generate_images(self, **kwargs: Any) -> Any:
        """Intercept models.generate_images() (imagen) through the media lifecycle.

        An EXPLICIT method (not __getattr__) so image spend is budget-checked,
        confirmed, and reported. imagen exposes no token usage; the billable
        basis is the request-derived per-image ``MediaUsage``
        (``config.number_of_images``). Being defined on the class, it never
        reaches __getattr__.
        """
        self._solwyn._enforce_explicit_surface(
            "models.generate_images", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._media_call(self._images_spec, **kwargs)

    def generate_videos(self, **kwargs: Any) -> Any:
        """Intercept models.generate_videos() (veo) through the media lifecycle.

        An EXPLICIT method (not __getattr__) so video spend is budget-checked,
        confirmed, and reported. Video generation is asynchronous — the call
        returns a long-running operation carrying no usage — so the billable
        basis is the request-derived per-second ``MediaUsage``
        (``config.duration_seconds`` at ``config.resolution``), settled at
        INITIATION with ``is_estimated=True``. The returned operation object is
        passed back untouched: callers poll it themselves; the SDK never wraps or
        polls it.
        """
        self._solwyn._enforce_explicit_surface(
            "models.generate_videos", source=SurfaceSource.WRAPPER
        )
        return self._solwyn._media_call(self._videos_spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Unrecognized client.models methods arrive here rather than on
        # Solwyn.__getattr__. The shared resolver applies exact-path policy;
        # explicit tracked methods above never reach this seam.
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.models,
            name=name,
            path=f"models.{name}",
            source=SurfaceSource.RAW,
        )


# ---------------------------------------------------------------------------
# Async proxies
# ---------------------------------------------------------------------------


class _AsyncChatCompletionsProxy:
    """Async proxy for client.chat.completions that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() with budget/circuit/reporting."""
        self._solwyn._enforce_explicit_surface(
            "chat.completions.create", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to OpenAI's chat.completions."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.chat.completions,
            name=name,
            path=f"chat.completions.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncChatProxy:
    """Async proxy for client.chat that provides .completions.create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self.completions = _AsyncChatCompletionsProxy(solwyn)

    def __getattr__(self, name: str) -> Any:
        if self._solwyn._dialect == "openai":
            return self._solwyn._resolve_public_attribute(
                self._solwyn._client.chat,
                name=name,
                path=f"chat.{name}",
                source=SurfaceSource.RAW,
            )
        raise AttributeError(
            f"'chat.{name}' is not supported. "
            f"The Solwyn chat proxy is OpenAI-dialect-specific; Anthropic uses "
            f"'messages' and Google uses 'models'."
        )


class _AsyncEmbeddingsProxy:
    """Async proxy for client.embeddings that routes create() through the media lifecycle.

    Mirror of ``_SyncEmbeddingsProxy``: ``client.embeddings.create()`` flows
    through the async ``_media_call``; every other attribute uses the shared
    surface policy.
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _embeddings_spec(solwyn)

    async def create(self, **kwargs: Any) -> Any:
        """Intercept embeddings.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("embeddings.create", source=SurfaceSource.WRAPPER)
        return await self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's embeddings."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.embeddings,
            name=name,
            path=f"embeddings.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncImagesProxy:
    """Async proxy for client.images that routes generate()/edit() through the lifecycle.

    Mirror of ``_SyncImagesProxy``: ``client.images.generate()`` and
    ``client.images.edit()`` flow through the async ``_media_call``; every other
    attribute uses the shared surface policy.
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _images_spec()

    async def generate(self, **kwargs: Any) -> Any:
        """Intercept images.generate() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("images.generate", source=SurfaceSource.WRAPPER)
        return await self._solwyn._media_call(self._spec, **kwargs)

    async def edit(self, **kwargs: Any) -> Any:
        """Intercept images.edit() through the same async lifecycle (op marker routes it)."""
        self._solwyn._enforce_explicit_surface("images.edit", source=SurfaceSource.WRAPPER)
        return await self._solwyn._media_call(self._spec, **{**kwargs, _IMAGE_OP_KEY: "edit"})

    def __getattr__(self, name: str) -> Any:
        """Pass through non-generate/edit attributes to the client's images."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.images,
            name=name,
            path=f"images.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncAudioTranscriptionsProxy:
    """Async proxy for client.audio.transcriptions that routes create() through the lifecycle.

    Mirror of ``_SyncAudioTranscriptionsProxy``: ``client.audio.transcriptions
    .create()`` flows through the async ``_media_call``; every other attribute
    uses the shared surface policy.
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _transcriptions_spec()

    async def create(self, **kwargs: Any) -> Any:
        """Intercept audio.transcriptions.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface(
            "audio.transcriptions.create", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's audio.transcriptions."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio.transcriptions,
            name=name,
            path=f"audio.transcriptions.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncAudioSpeechProxy:
    """Async proxy for client.audio.speech that routes create() through the lifecycle.

    Mirror of ``_SyncAudioSpeechProxy``: ``client.audio.speech.create()`` flows
    through the async ``_media_call`` (billed on the request's ``input`` character
    count; TTS responses carry no usage). The shared conditional policy decides
    whether the untracked token-billed model (``gpt-4o-mini-tts``) warns, raises,
    or awaits a direct pass-through; every other attribute uses the same resolver.
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _speech_spec()

    async def create(self, **kwargs: Any) -> Any:
        """Intercept audio.speech.create(); carve out untracked token-billed TTS models."""
        self._solwyn._enforce_explicit_surface("audio.speech.create", source=SurfaceSource.WRAPPER)
        if _is_untracked_tts_model(kwargs.get("model")):
            self._solwyn._enforce_explicit_surface(
                "audio.speech.create",
                source=SurfaceSource.SYNTHETIC_POLICY,
                condition=SurfaceCondition.OPENAI_UNTRACKED_TTS_MODEL,
            )
            return await self._solwyn._client.audio.speech.create(**kwargs)
        return await self._solwyn._media_call(self._spec, **{**kwargs, _AUDIO_OP_KEY: "speech"})

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's audio.speech."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio.speech,
            name=name,
            path=f"audio.speech.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncAudioProxy:
    """Async proxy for client.audio with guarded pass-throughs and tracked media seams.

    Mirror of ``_SyncAudioProxy``: attribute access is synchronous, so the
    pass-through policy and intercepted ``transcriptions`` / ``speech``
    sub-proxies behave exactly as on the sync proxy; only their ``create`` differs
    (it awaits the async ``_media_call``).
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self.transcriptions = _AsyncAudioTranscriptionsProxy(solwyn)
        self.speech = _AsyncAudioSpeechProxy(solwyn)

    @property
    def translations(self) -> Any:
        """Resolve the client's untracked audio translations resource."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio,
            name="translations",
            path="audio.translations",
            source=SurfaceSource.RAW,
        )

    def __getattr__(self, name: str) -> Any:
        """Pass through other audio attributes to the client."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.audio,
            name=name,
            path=f"audio.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncVideosProxy:
    """Async proxy for client.videos that routes create() through the lifecycle.

    Mirror of ``_SyncVideosProxy``: ``client.videos.create()`` (OpenAI's Sora
    video API) flows through the async ``_media_call``, settled on the
    request-derived per-second ``MediaUsage`` (top-level ``seconds`` at the
    ``size``-derived resolution label) with ``is_estimated=True`` (the async video
    job carries no usage). The returned job is passed back untouched — callers
    poll it themselves. Every other ``videos`` attribute uses the shared surface
    policy; on a non-openai client ``.create()`` fails loud with
    ``UnsupportedSurfaceError`` (Sora is OpenAI-only).
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._spec = _openai_videos_spec()

    async def create(self, **kwargs: Any) -> Any:
        """Intercept videos.create() with budget/confirm/reporting."""
        self._solwyn._enforce_explicit_surface("videos.create", source=SurfaceSource.WRAPPER)
        return await self._solwyn._media_call(self._spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through non-create attributes to the client's videos."""
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.videos,
            name=name,
            path=f"videos.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncMessagesProxy:
    """Async proxy for client.messages that intercepts create()."""

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn

    async def create(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface("messages.create", source=SurfaceSource.WRAPPER)
        return await self._solwyn._intercepted_call(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.messages,
            name=name,
            path=f"messages.{name}",
            source=SurfaceSource.RAW,
        )


class _AsyncModelsProxy:
    """Async proxy for client.models.

    Intercepts generate_content(), generate_content_stream(), embed_content(),
    generate_images(), and generate_videos().
    """

    def __init__(self, solwyn: AsyncSolwyn) -> None:
        self._solwyn = solwyn
        self._embeddings_spec = _google_embeddings_spec()
        self._images_spec = _google_images_spec()
        self._videos_spec = _google_videos_spec()

    async def generate_content(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface(
            "models.generate_content", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._intercepted_call(**kwargs)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self._solwyn._enforce_explicit_surface(
            "models.generate_content_stream", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._intercepted_call(_force_stream=True, **kwargs)

    async def embed_content(self, **kwargs: Any) -> Any:
        """Intercept models.embed_content() through the async media lifecycle.

        Mirror of ``_SyncModelsProxy.embed_content``: an EXPLICIT method (not
        __getattr__) so embeddings spend is budget-checked, confirmed, and
        reported. Being defined on the class, it never reaches __getattr__.
        """
        self._solwyn._enforce_explicit_surface("models.embed_content", source=SurfaceSource.WRAPPER)
        return await self._solwyn._media_call(self._embeddings_spec, **kwargs)

    async def generate_images(self, **kwargs: Any) -> Any:
        """Intercept models.generate_images() (imagen) through the async media lifecycle.

        Mirror of ``_SyncModelsProxy.generate_images``: an EXPLICIT method (not
        __getattr__) so image spend is budget-checked, confirmed, and reported on
        the request-derived per-image ``MediaUsage`` basis.
        """
        self._solwyn._enforce_explicit_surface(
            "models.generate_images", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._media_call(self._images_spec, **kwargs)

    async def generate_videos(self, **kwargs: Any) -> Any:
        """Intercept models.generate_videos() (veo) through the async media lifecycle.

        Mirror of ``_SyncModelsProxy.generate_videos``: an EXPLICIT method (not
        __getattr__) so video spend is budget-checked, confirmed, and reported on
        the request-derived per-second ``MediaUsage`` basis, settled at INITIATION
        with ``is_estimated=True``. The returned long-running operation is passed
        back untouched — callers poll it themselves.
        """
        self._solwyn._enforce_explicit_surface(
            "models.generate_videos", source=SurfaceSource.WRAPPER
        )
        return await self._solwyn._media_call(self._videos_spec, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # See _SyncModelsProxy.__getattr__: the shared resolver applies exact-path
        # policy, and explicit tracked methods never reach this seam.
        return self._solwyn._resolve_public_attribute(
            self._solwyn._client.models,
            name=name,
            path=f"models.{name}",
            source=SurfaceSource.RAW,
        )
