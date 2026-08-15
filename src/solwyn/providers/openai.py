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
import threading
from collections.abc import Callable
from typing import Any, Literal

from solwyn._constants import SERVICE_TIER_MAX_LENGTH
from solwyn._token_details import TokenDetails
from solwyn._types import MediaUsage
from solwyn.exceptions import UnsupportedSurfaceError

logger = logging.getLogger(__name__)


def _usage_value(value: object) -> int:
    """Coerce one usage field to a non-negative int; garbage degrades to 0.

    TokenDetails fields are ``Field(ge=0)``, so a negative or non-int value
    from a misbehaving endpoint would raise ValidationError out of extraction
    paths contracted to never raise. Treating garbage as 0 lets the existing
    zero-usage handling (and the compat estimation fallback) take over.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


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
    cached_tokens = getattr(
        prompt_details if prompt_details is not None else usage,
        "cached_tokens",
        None,
    )

    return TokenDetails(
        input_tokens=_usage_value(getattr(usage, "prompt_tokens", None)),
        output_tokens=_usage_value(getattr(usage, "completion_tokens", None)),
        cached_input_tokens=_usage_value(cached_tokens),
        # OpenAI's provider-default/minimum 30-minute cache writes use the
        # dataset's existing cache_write_5m bucket to preserve the wire contract.
        cache_creation_5m_tokens=_usage_value(getattr(prompt_details, "cache_write_tokens", None)),
        audio_input_tokens=_usage_value(getattr(prompt_details, "audio_tokens", None)),
        reasoning_tokens=_usage_value(getattr(completion_details, "reasoning_tokens", None)),
        audio_output_tokens=_usage_value(getattr(completion_details, "audio_tokens", None)),
        accepted_prediction_tokens=_usage_value(
            getattr(completion_details, "accepted_prediction_tokens", None)
        ),
        rejected_prediction_tokens=_usage_value(
            getattr(completion_details, "rejected_prediction_tokens", None)
        ),
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
        input_tokens=_usage_value(getattr(usage, "input_tokens", None)),
        output_tokens=_usage_value(getattr(usage, "output_tokens", None)),
        cached_input_tokens=_usage_value(getattr(input_details, "cached_tokens", None)),
        cache_creation_5m_tokens=_usage_value(getattr(input_details, "cache_write_tokens", None)),
        audio_input_tokens=_usage_value(getattr(input_details, "audio_tokens", None)),
        reasoning_tokens=_usage_value(getattr(output_details, "reasoning_tokens", None)),
        audio_output_tokens=_usage_value(getattr(output_details, "audio_tokens", None)),
        accepted_prediction_tokens=_usage_value(
            getattr(output_details, "accepted_prediction_tokens", None)
        ),
        rejected_prediction_tokens=_usage_value(
            getattr(output_details, "rejected_prediction_tokens", None)
        ),
    )


def _extract_image_usage(response: Any) -> TokenDetails | None:
    """Extract token usage from a gpt-image images.generate / images.edit response.

    Module-level so the images ``MediaSurfaceSpec`` (in ``_proxies``) can reuse
    it for native OpenAI and any compat endpoint that shares the shape. Token-
    billed image models (gpt-image-2) report the Responses-style usage shape
    with per-modality buckets (probe-verified 2026-07-10):

        usage.input_tokens / usage.output_tokens
        usage.input_tokens_details  = {image_tokens, text_tokens}
        usage.output_tokens_details = {image_tokens, text_tokens}

    The image buckets map onto ``image_input_tokens`` / ``image_output_tokens``
    (image ⊂ input, image_output ⊂ output). There is NO cached image-input
    bucket — the probe confirmed none exists, so this reads none. Returns None
    when the response carries no usable usage (a compat images endpoint returns
    ``usage: null``, and dall-e models report no usage) so the request-derived
    ``MediaUsage`` becomes the sole billable basis rather than a silent $0.
    Never raises.

    (The Responses-API image-generation TOOL is out of scope here: its usage
    lands on the chat surface's top-level ``response.tool_usage.image_gen`` and
    must NOT be intercepted through this images proxy.)
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _usage_value(getattr(usage, "input_tokens", None))
    output_tokens = _usage_value(getattr(usage, "output_tokens", None))
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return TokenDetails(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        image_input_tokens=_usage_value(getattr(input_details, "image_tokens", None)),
        image_output_tokens=_usage_value(getattr(output_details, "image_tokens", None)),
    )


# Private marker the images proxy sets so images.edit vs images.generate can
# ride the ONE "images" media surface through the shared lifecycle (mirrors the
# chat ``_force_stream`` marker). The surface key is just "images", so the
# operation rides kwargs and is STRIPPED here before the SDK call.
_IMAGE_OP_KEY = "_solwyn_image_op"


def _prepare_image_media_call(
    client: Any, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Select client.images.generate / .edit and shape a COPY of kwargs.

    Shared by the OpenAI adapter and every OpenAI-compatible adapter (same
    ``client.images`` surface). Pops the private ``_IMAGE_OP_KEY`` marker
    (default ``"generate"``) — never sends it to the SDK — and NEVER reads the
    ``prompt`` / ``image`` / ``mask`` content, only re-shapes keys.
    """
    shaped = dict(kwargs)
    operation = shaped.pop(_IMAGE_OP_KEY, "generate")
    method = client.images.edit if operation == "edit" else client.images.generate
    return method, shaped


# Per-process warn-once latch for a transcription that returns no billable basis
# (a non-JSON response_format yields a plain string with no usage). Lock-guarded
# for the multi-threaded sync client; reset in tests via
# ``_reset_transcription_unpriced_warning``.
_transcription_unpriced_warned = False
_transcription_unpriced_warn_lock = threading.Lock()


def _warn_transcription_unpriced_once() -> None:
    """Hint once that a JSON response_format restores priced transcription tracking.

    Fires when an ``audio.transcriptions`` response carries no usable usage — the
    non-JSON response formats (``text`` / ``srt`` / ``vtt``) return a plain string
    with no usage block, so the call is tracked but UNPRICED. Content-free: the
    message names no request or response data.
    """
    global _transcription_unpriced_warned
    with _transcription_unpriced_warn_lock:
        if _transcription_unpriced_warned:
            return
        _transcription_unpriced_warned = True
    # Logging handlers may run arbitrary code; keep them outside the lock.
    logger.warning(
        "Audio transcription returned no usage data; the call is tracked but "
        "UNPRICED. A non-JSON response_format (text/srt/vtt) carries no billable "
        "basis — pass response_format='json' or 'verbose_json' for priced tracking."
    )


def _reset_transcription_unpriced_warning() -> None:
    """Clear the per-process transcription-unpriced warn latch. Test-support hook only."""
    global _transcription_unpriced_warned
    with _transcription_unpriced_warn_lock:
        _transcription_unpriced_warned = False


def _transcription_token_details(usage: Any) -> TokenDetails | None:
    """TokenDetails from a token-usage transcription block (gpt-4o-transcribe family).

    The token-billed models report ``usage.input_tokens`` / ``usage.output_tokens``
    with an ``input_token_details`` sub-object carrying ``text_tokens`` and
    ``audio_tokens``. ``audio_tokens`` maps onto ``audio_input_tokens`` (audio ⊂
    input, mirroring image ⊂ input); the text side is ``input_tokens −
    audio_input_tokens``, derived server-side. Returns None when neither side
    carries a positive count so the media lifecycle never settles a real $0.
    """
    input_tokens = _usage_value(getattr(usage, "input_tokens", None))
    output_tokens = _usage_value(getattr(usage, "output_tokens", None))
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    input_details = getattr(usage, "input_token_details", None)
    return TokenDetails(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        audio_input_tokens=_usage_value(getattr(input_details, "audio_tokens", None)),
    )


def _transcription_duration_media(usage: Any) -> MediaUsage | None:
    """MediaUsage from a duration-usage transcription block (whisper-1).

    whisper-1 reports ``usage.seconds`` — an INTEGER whole-second count that is the
    provider's billed basis, present on BOTH the default ``json`` and
    ``verbose_json`` formats. It maps onto ``MediaUsage.audio_seconds``; the
    fractional top-level ``duration`` field is deliberately NOT read (it is not the
    billed basis). Returns None when ``seconds`` is absent or non-positive so an
    unobservable quantity stays None rather than a zero-as-default.
    """
    seconds = getattr(usage, "seconds", None)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    return MediaUsage(audio_seconds=float(seconds))


def _transcription_usage_basis(response: Any) -> tuple[TokenDetails | None, MediaUsage | None]:
    """Discriminate one transcription response's usage into (token, media) bases.

    ONE extractor for both billable shapes: it switches on ``usage.type``. A
    ``"tokens"`` block yields the TokenDetails basis (gpt-4o-transcribe family); a
    ``"duration"`` block yields the MediaUsage basis (whisper-1). A response with
    NO usage block (a non-JSON response_format returns a plain string) yields
    ``(None, None)`` and hints once that a JSON response_format restores priced
    tracking. An unrecognized ``type`` also yields ``(None, None)`` — unpriced, but
    without the JSON hint (that message would misdescribe it). Duck-typed and never
    raises; both wrapper hooks share this so the discrimination lives in one place.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        _warn_transcription_unpriced_once()
        return None, None
    usage_type = getattr(usage, "type", None)
    if usage_type == "tokens":
        return _transcription_token_details(usage), None
    if usage_type == "duration":
        return None, _transcription_duration_media(usage)
    return None, None


def _extract_transcription_usage(response: Any) -> TokenDetails | None:
    """Token basis of a transcription response, or None. See ``_transcription_usage_basis``.

    Module-level so the audio ``MediaSurfaceSpec`` (in ``_proxies``) reuses it for
    native OpenAI and every compat endpoint (incl. Groq) that shares the shape.
    """
    return _transcription_usage_basis(response)[0]


def _measure_transcription_media(_kwargs: dict[str, Any], response: Any) -> MediaUsage | None:
    """Duration basis of a transcription response, or None. See ``_transcription_usage_basis``.

    The request is not read: audio duration is only observable in the response's
    duration-usage block, never derivable from the request (which carries the file
    bytes this SDK never touches).
    """
    return _transcription_usage_basis(response)[1]


# Private marker the audio proxies set so speech.create vs transcriptions.create
# can ride the ONE "audio" media surface through the shared lifecycle (mirrors the
# images ``_IMAGE_OP_KEY``). The surface key is just "audio", so the operation
# rides kwargs and is STRIPPED here before the SDK call. Default is transcriptions.
_AUDIO_OP_KEY = "_solwyn_audio_op"


def _prepare_audio_media_call(
    client: Any, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Select ``client.audio.transcriptions.create`` / ``.speech.create`` and shape a COPY.

    Shared by the OpenAI adapter and every OpenAI-compatible adapter (same
    ``client.audio`` surface, incl. Groq whisper). Pops the private
    ``_AUDIO_OP_KEY`` marker (default ``"transcriptions"``) — never sends it to the
    SDK — and routes ``"speech"`` to ``client.audio.speech.create``. NEVER reads
    the ``file`` bytes / ``input`` text / ``prompt``, only re-shapes keys into a
    defensive copy.
    """
    shaped = dict(kwargs)
    operation = shaped.pop(_AUDIO_OP_KEY, "transcriptions")
    if operation == "speech":
        return client.audio.speech.create, shaped
    return client.audio.transcriptions.create, shaped


# Token-billed TTS models publish NO usage metadata whatsoever — their
# audio-output tokens are unobservable, so these calls cannot be priced and are
# passed through UNTRACKED (a WARNED pass-through, never a silent $0) by product
# decision. A prefix match keeps the carve-out narrow (today exactly
# ``gpt-4o-mini-tts`` and its dated snapshots) and trivially removable if such a
# model ever begins reporting usage. Sanctioned exception to the
# no-hardcoded-model-names norm; keyed on the model string so it applies on the
# native OpenAI adapter and every OpenAI-compatible adapter alike.
_UNTRACKED_TTS_MODEL_PREFIXES = ("gpt-4o-mini-tts",)


def _is_untracked_tts_model(model: Any) -> bool:
    """True when a TTS model is token-billed with no observable usage (untracked).

    Prefix match on the model string (covers dated snapshots such as
    ``gpt-4o-mini-tts-2026-01-01``). A non-str model is never matched — the media
    lifecycle then handles it normally.
    """
    if not isinstance(model, str):
        return False
    return model.startswith(_UNTRACKED_TTS_MODEL_PREFIXES)


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

    @property
    def dialect(self) -> Literal["openai"]:
        return "openai"

    def detect_client(self, client: Any) -> bool:
        """Return True if the client's module path contains 'openai'.

        Registered AFTER the OpenAI-compatible adapters, which claim
        openai-module clients whose base_url points at another vendor —
        so reaching this adapter means the client targets OpenAI itself.
        """
        return "openai" in getattr(type(client), "__module__", "")

    def detect_model(self, model: str) -> bool:
        """Return True for gpt-*, o3*, o4* model prefixes."""
        return model.startswith("gpt-") or model.startswith("o3") or model.startswith("o4")

    def extract_usage(self, response: Any) -> TokenDetails:
        """Extract token usage from a Chat Completions or Responses API response."""
        return _extract_openai_usage(response)

    def estimate_missing_usage(
        self, response: Any, *, estimated_input_tokens: int
    ) -> TokenDetails | None:
        """OpenAI itself always reports usage — no estimated fallback."""
        return None

    def extract_service_tier(self, response: Any) -> str | None:
        """Return service_tier from an OpenAI response, or None if absent.

        OpenAI tiers (default/flex/priority/batch) price at different rates;
        the API stores this on cost_events for future per-tier repricing.
        """
        return _extract_service_tier(response)

    def extract_region(self, client: Any) -> str | None:
        """OpenAI pricing is not regional."""
        return None

    def prepare_streaming(
        self, kwargs: dict[str, Any], *, cross_provider: bool = False
    ) -> dict[str, Any]:
        """Inject stream_options so usage appears in the final chunk."""
        kwargs = dict(kwargs)
        stream_options = dict(kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        kwargs["stream_options"] = stream_options
        return kwargs

    def create_stream_accumulator(
        self, *, estimated_input_tokens: int = 0
    ) -> OpenAIStreamAccumulator:
        """OpenAI streams always carry usage (include_usage is injected) —
        the input estimate is not needed."""
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

    def prepare_responses_call(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        is_streaming: bool,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Duck-typed Responses dispatch seam, like the media surface seams.

        This foundation targets native OpenAI only. Streaming sets ``stream``
        but deliberately omits ``stream_options`` because Responses rejects it;
        usage instead arrives on the terminal ``response.completed`` event.
        """
        kwargs = dict(kwargs)
        if is_streaming:
            kwargs["stream"] = True
        return client.responses.create, kwargs

    def prepare_media_call(
        self,
        surface: str,
        client: Any,
        kwargs: dict[str, Any],
        *,
        timeout: float,
        max_retries: int,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Per-surface dispatch seam for non-chat media surfaces.

        Embeddings route to ``client.embeddings.create``; images route to
        ``client.images.generate`` / ``.edit``; audio routes to
        ``client.audio.transcriptions.create`` (or ``.speech.create`` when the op
        marker selects it); video routes to ``client.videos.create`` (Sora) — the
        same ``(method, shaped_kwargs)`` shape ``prepare_call`` returns for chat,
        with a defensive COPY of kwargs (never mutate/alias the caller's dict). An
        unrecognized surface fails loud with ``UnsupportedSurfaceError``.
        timeout/max_retries are ignored for SDKs with ``with_options`` (the
        dispatcher already applied them); a branch for an SDK without it applies
        them itself, like ``prepare_call``.
        """
        if surface == "embeddings":
            return client.embeddings.create, dict(kwargs)
        if surface == "images":
            return _prepare_image_media_call(client, kwargs)
        if surface == "audio":
            return _prepare_audio_media_call(client, kwargs)
        if surface == "video":
            return client.videos.create, dict(kwargs)
        raise UnsupportedSurfaceError(surface=surface, provider=self.name)

    def unwrap_stream_source(self, response: Any) -> Any:
        """The streaming call returns the iterable itself."""
        return response

    def wrap_stream_result(self, wrapper: Any, served_response: Any) -> Any:
        """OpenAI-dialect callers iterate the stream object directly."""
        return wrapper


class OpenAIStreamAccumulator:
    """Accumulates usage from OpenAI streaming chunks.

    Chat Completions includes usage in a final chunk when the caller sets
    stream_options={"include_usage": True}; Responses embeds its usage-bearing
    response on the terminal ``response.completed`` event. We save either object
    and delegate to the same extraction logic as non-streaming responses.
    """

    def __init__(self) -> None:
        self._usage_chunk: Any | None = None

    def observe(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage_chunk = chunk
            return

        response = getattr(chunk, "response", None)
        if response is not None and getattr(response, "usage", None) is not None:
            self._usage_chunk = response

    def finalize(self) -> TokenDetails:
        if self._usage_chunk is None:
            return TokenDetails()
        return _extract_openai_usage(self._usage_chunk)

    def get_service_tier(self) -> str | None:
        """Return service_tier from the saved final-chunk, or None if absent."""
        if self._usage_chunk is None:
            return None
        return _extract_service_tier(self._usage_chunk)
