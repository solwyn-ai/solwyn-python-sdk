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

from solwyn._privacy import normalize_google_generate_content_kwargs
from solwyn._token_details import TokenDetails
from solwyn.exceptions import UnsupportedSurfaceError


def _modality_label(value: Any) -> str | None:
    """Normalize a ModalityTokenCount ``modality`` field to an UPPERCASE string.

    google-genai's ``MediaModality`` is a ``str`` enum whose value is the label
    ("TEXT", "IMAGE", "AUDIO", ...); a plain string is handled too. Duck-typed
    (``.value`` then ``.name``) so no provider SDK import is needed. Returns None
    for an unreadable shape so the caller simply skips that bucket. Never raises.
    """
    if isinstance(value, str):
        label: Any = value
    else:
        label = getattr(value, "value", None)
        if not isinstance(label, str):
            label = getattr(value, "name", None)
    if not isinstance(label, str):
        return None
    return label.upper()


def _modality_token_sum(details: Any, modality: str) -> int:
    """Sum ``token_count`` across a per-modality details list for one label.

    ``prompt_tokens_details`` / ``candidates_tokens_details`` are lists of
    ModalityTokenCount ``{modality, token_count}`` (snake_case SDK attrs — never
    the camelCase wire names). Absent (None) or non-list details, or entries with
    a garbage count, contribute 0 — so a response WITHOUT per-modality details
    leaves every mapped field at 0 exactly as before. Never raises.
    """
    if not isinstance(details, (list, tuple)):
        return 0
    total = 0
    for entry in details:
        if _modality_label(getattr(entry, "modality", None)) != modality:
            continue
        count = getattr(entry, "token_count", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            continue
        total += count
    return total


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

    Per-modality buckets (when present) map from the ModalityTokenCount lists
    ``prompt_tokens_details`` (input side) and ``candidates_tokens_details``
    (output side):
    - IMAGE -> image_input_tokens / image_output_tokens (a SUBSET of input /
      output; this is what makes a gemini-3-pro-image ``generate_content`` call
      carry image-output tokens so the server prices them at the image rate).
    - AUDIO -> audio_input_tokens / audio_output_tokens.
    TEXT buckets need no field — they are the remainder. Absent details leave
    these fields at 0.
    """
    if usage_metadata is None:
        return TokenDetails()

    input_tokens = getattr(usage_metadata, "prompt_token_count", None) or 0
    candidates = getattr(usage_metadata, "candidates_token_count", None) or 0
    thoughts = getattr(usage_metadata, "thoughts_token_count", None) or 0
    cached = getattr(usage_metadata, "cached_content_token_count", None) or 0
    tool_use = getattr(usage_metadata, "tool_use_prompt_token_count", None) or 0

    prompt_details = getattr(usage_metadata, "prompt_tokens_details", None)
    candidates_details = getattr(usage_metadata, "candidates_tokens_details", None)

    return TokenDetails(
        input_tokens=input_tokens,
        output_tokens=candidates + thoughts,
        reasoning_tokens=thoughts,
        cached_input_tokens=cached,
        tool_use_input_tokens=tool_use,
        image_input_tokens=_modality_token_sum(prompt_details, "IMAGE"),
        image_output_tokens=_modality_token_sum(candidates_details, "IMAGE"),
        audio_input_tokens=_modality_token_sum(prompt_details, "AUDIO"),
        audio_output_tokens=_modality_token_sum(candidates_details, "AUDIO"),
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
    but override the timeout and retry attempts because the hop read bound is a
    mandatory Solwyn bound (google-genai cannot split connect from read; the
    failover window still gates advancement between hops).
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


def _is_legacy_google_client(client: object) -> bool:
    """Return whether *client* is a deprecated google.generativeai model."""
    return "google.generativeai" in getattr(type(client), "__module__", "")


def _with_legacy_google_request_bound(kwargs: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """Apply copied legacy GAPIC timeout/retry bounds.

    ``google.generativeai`` forwards this mapping directly to its GAPIC method.
    ``retry=None`` disables that method's default retry decorator, preventing
    retry stacking; unlike google-genai, the legacy package exposes no import-free
    attempt-count request shape.
    """
    bounded = dict(kwargs)
    request_options = _mapping_from_config(bounded.get("request_options"))
    request_options["timeout"] = timeout
    request_options["retry"] = None
    bounded["request_options"] = request_options
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
        """Select and shape a Google generation hop for either SDK family.

        google-genai uses dedicated streaming/non-streaming methods and carries
        the mandatory bound in ``config.http_options``. The legacy
        google.generativeai client uses one root method with ``stream=`` and
        forwards copied ``request_options`` to GAPIC. In both cases the returned
        kwargs are defensive copies.
        """
        if _is_legacy_google_client(client):
            legacy_kwargs = normalize_google_generate_content_kwargs(
                kwargs,
                target_shape="google_generativeai",
            )
            legacy_kwargs = _with_legacy_google_request_bound(legacy_kwargs, timeout=timeout)
            legacy_kwargs.pop("model", None)
            legacy_kwargs["stream"] = is_streaming
            return client.generate_content, legacy_kwargs

        kwargs = normalize_google_generate_content_kwargs(
            kwargs,
            target_shape="google_genai",
        )
        kwargs = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
        kwargs.pop("stream", None)
        if is_streaming:
            return client.models.generate_content_stream, kwargs
        return client.models.generate_content, kwargs

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

        Embeddings route to ``client.models.embed_content``; images route to
        ``client.models.generate_images`` (imagen); video routes to
        ``client.models.generate_videos`` (veo). Unlike the openai seam — where
        the dispatcher already applied the per-hop bound via ``with_options`` —
        google-genai has no ``with_options``, so (exactly as ``prepare_call``
        does for chat) the mandatory bound is injected here as per-request
        ``config.http_options`` via ``_with_google_http_bound``, which also
        returns a defensive COPY (never mutates the caller's kwargs). Note the
        bound rides ``config.http_options`` alongside the caller's own ``config``
        keys (e.g. ``number_of_images``, ``duration_seconds``) — those are
        preserved. Audio is not wired and fails loud with the structural,
        content-free ``UnsupportedSurfaceError``.
        """
        if surface == "embeddings":
            bounded = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
            return client.models.embed_content, bounded
        if surface == "images":
            bounded = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
            return client.models.generate_images, bounded
        if surface == "video":
            bounded = _with_google_http_bound(kwargs, timeout=timeout, max_retries=max_retries)
            return client.models.generate_videos, bounded
        raise UnsupportedSurfaceError(surface=surface, provider=self.name)

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
