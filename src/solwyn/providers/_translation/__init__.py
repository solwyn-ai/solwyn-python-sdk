"""Cross-provider request/response translation (design spec §5).

================================ PRIVACY-CRITICAL ==============================
This is one of EXACTLY TWO content-privileged modules (the other is
``solwyn/_privacy.py``). It is the only place in the SDK that reshapes customer
prompt CONTENT. The content-touching contract (spec §7) is non-negotiable:

  * In-memory transform ONLY. Content is never stored on a long-lived object;
    it lives only inside the call frame and the returned kwargs/response that go
    STRAIGHT to the destination provider SDK.
  * No I/O. This module imports neither the stdlib log module nor any HTTP
    client, and holds no log handle of any kind. It must never reach
    ``_reporter``, ``_budget``, or any client pointed at ``config.api_url``.
  * Fail loudly with ``UntranslatableRequestError(source, target, feature)``
    carrying STRUCTURAL labels only — NEVER the offending value or any prompt
    content. Re-raise across the boundary with ``... from None`` so a provider
    exception's text can never leak via ``__cause__``/``__context__``.
  * Pure. Same input -> same output; no global state.

If you change this file you must keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================

Public surface (keyed by provider name ``openai`` | ``anthropic`` | ``google``):

  to_canonical(provider, kwargs)        -> CanonicalRequest
  from_canonical(provider, canonical, *, model) -> dict  (native target kwargs)
  normalize_response(*, served, requested, response) -> object (duck-typed)

The canonical form is intentionally tiny (spec §5.1): ~10 request fields, ~5
response fields. Anything outside the subset RAISES on the first cross-provider
hop, BEFORE any network call. Usage extraction is NOT done here — the existing
extraction adapters own that.
"""

from __future__ import annotations

from typing import Any

from solwyn.providers._translation._anthropic import (
    _anthropic_response_to_canonical,
    _anthropic_stream_chunk_to_deltas,
    _anthropic_to_canonical,
    _canonical_response_to_anthropic,
    _canonical_to_anthropic,
    _stream_delta_to_anthropic,
)
from solwyn.providers._translation._common import (
    normalize_finish_reason as normalize_finish_reason,
)
from solwyn.providers._translation._google import (
    _canonical_response_to_google,
    _canonical_to_google,
    _google_response_to_canonical,
    _google_stream_chunk_to_deltas,
    _google_to_canonical,
    _stream_delta_to_google,
)
from solwyn.providers._translation._guardrails import (
    _guard,
    _validate_provider,
)
from solwyn.providers._translation._guardrails import (
    fail_cross_provider_tool_stream as fail_cross_provider_tool_stream,
)
from solwyn.providers._translation._models import (
    CanonicalMessage as CanonicalMessage,
)
from solwyn.providers._translation._models import (
    CanonicalRequest as CanonicalRequest,
)
from solwyn.providers._translation._models import (
    CanonicalResponse as CanonicalResponse,
)
from solwyn.providers._translation._models import (
    CanonicalTool as CanonicalTool,
)
from solwyn.providers._translation._models import (
    TextPart as TextPart,
)
from solwyn.providers._translation._models import (
    ToolChoice as ToolChoice,
)
from solwyn.providers._translation._models import (
    ToolResultPart as ToolResultPart,
)
from solwyn.providers._translation._models import (
    ToolUsePart as ToolUsePart,
)
from solwyn.providers._translation._models import (
    _StreamDelta,
)
from solwyn.providers._translation._openai import (
    _canonical_response_to_openai,
    _canonical_to_openai,
    _openai_response_to_canonical,
    _openai_stream_chunk_to_deltas,
    _openai_to_canonical,
    _stream_delta_to_openai,
)

__all__ = [
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalTool",
    "ToolChoice",
    "fail_cross_provider_tool_stream",
    "from_canonical",
    "normalize_response",
    "normalize_finish_reason",
    "to_canonical",
    "translate_stream_chunk",
]

# --------------------------------------------------------------------------- #
# to_canonical — parse a native request dict into the canonical form.          #
# --------------------------------------------------------------------------- #
def to_canonical(provider: str, kwargs: dict[str, Any]) -> CanonicalRequest:
    """Parse a native ``provider`` request dict into the canonical request.

    RAISES ``UntranslatableRequestError`` for anything outside the §5 subset.
    """
    _validate_provider(provider)
    # Guard ALL caller-content access: any non-Untranslatable error (a malformed
    # value would otherwise surface as a raw pydantic ValidationError / ValueError
    # / KeyError carrying the offending value) is converted to a STRUCTURAL
    # ``malformed_request`` with ``from None`` (fix [B]).
    with _guard(provider, "*"):
        if provider == "openai":
            return _openai_to_canonical(kwargs)
        if provider == "anthropic":
            return _anthropic_to_canonical(kwargs)
        return _google_to_canonical(kwargs)


# --------------------------------------------------------------------------- #
# from_canonical — render the canonical request into target native kwargs.     #
# --------------------------------------------------------------------------- #
def from_canonical(provider: str, canonical: CanonicalRequest, *, model: str) -> dict[str, Any]:
    """Render *canonical* into ``provider`` native kwargs (model included)."""
    _validate_provider(provider)
    # Guard caller-content access on the render side too (fix [B]).
    with _guard("*", provider):
        if provider == "openai":
            return _canonical_to_openai(canonical, model)
        if provider == "anthropic":
            return _canonical_to_anthropic(canonical, model)
        return _canonical_to_google(canonical, model)


# --------------------------------------------------------------------------- #
# normalize_response — served-provider response -> requested-provider shape.    #
# --------------------------------------------------------------------------- #
def normalize_response(*, served: str, requested: str, response: object) -> object:
    """Reshape a *served*-provider response into a *requested*-shaped object.

    Returns a DUCK-TYPED ``SimpleNamespace`` tree exposing the native access
    path the caller wrote (e.g. ``resp.choices[0].message.content`` for an
    OpenAI-dialect caller). A same-provider call is identity (no reshape).
    Only invoked on a real cross-provider hop.
    """
    _validate_provider(served)
    _validate_provider(requested)
    if served == requested:
        return response

    # Guard response-content access too: a malformed served value (wrong-typed
    # field) must convert to a structural error, not leak via a raw exception
    # (fix [B]).
    with _guard(served, requested):
        canonical = _response_to_canonical(served, response)
        if requested == "openai":
            return _canonical_response_to_openai(canonical)
        if requested == "anthropic":
            return _canonical_response_to_anthropic(canonical)
        return _canonical_response_to_google(canonical)


def _response_to_canonical(served: str, response: object) -> CanonicalResponse:
    if served == "openai":
        return _openai_response_to_canonical(response)
    if served == "anthropic":
        return _anthropic_response_to_canonical(response)
    return _google_response_to_canonical(response)


# --------------------------------------------------------------------------- #
# §6.6 Per-chunk stream translation — map ONE served-stream event into ZERO     #
# OR MORE chunks in the requested provider's native streaming dialect.          #
# --------------------------------------------------------------------------- #
# A streamed _StreamDelta is the tiny canonical event the per-provider parsers
# emit and the per-provider renderers consume. Exactly one of (text, finish) is
# set per delta; a served structural event yields NO _StreamDelta at all (an
# empty list out of translate_stream_chunk). This mirrors normalize_response's
# served->canonical->requested pipeline at the chunk granularity. Tool-call and
# non-text media on a cross-provider hop are OUT of the v1 streaming subset and
# RAISE structurally (no chunk content ever reaches the error). The
# _CROSS_PROVIDER_*_STREAM labels are defined near the top of the module so the
# pre-dispatch fail-loud helper can share them.


def translate_stream_chunk(*, served: str, requested: str, chunk: object) -> list[object]:
    """Map ONE raw *served*-stream chunk into ZERO OR MORE *requested* chunks.

    Returns a list because one foreign streaming event can map to zero caller
    chunks (a structural event: message_start / content_block_start / stop) or
    one caller chunk (a text delta or a terminal finish). TEXT streaming only:
    a tool-call or non-text media part on a cross-provider hop is out of the v1
    subset and RAISES ``UntranslatableRequestError`` with a STRUCTURAL label —
    chunk content is NEVER read into the error.
    """
    _validate_provider(served)
    _validate_provider(requested)
    # Guard chunk-content access: a malformed served chunk must convert to a
    # structural error, never leak via a raw exception (fix [B], parity with
    # normalize_response).
    with _guard(served, requested):
        deltas = _stream_chunk_to_deltas(served, requested, chunk)
        return [_stream_delta_to_requested(requested, delta) for delta in deltas]


def _stream_chunk_to_deltas(served: str, requested: str, chunk: object) -> list[_StreamDelta]:
    if served == "openai":
        return _openai_stream_chunk_to_deltas(served, requested, chunk)
    if served == "anthropic":
        return _anthropic_stream_chunk_to_deltas(served, requested, chunk)
    return _google_stream_chunk_to_deltas(served, requested, chunk)


# --------------- canonical _StreamDelta -> requested native chunk ----------- #
def _stream_delta_to_requested(requested: str, delta: _StreamDelta) -> object:
    if requested == "openai":
        return _stream_delta_to_openai(delta)
    if requested == "anthropic":
        return _stream_delta_to_anthropic(delta)
    return _stream_delta_to_google(delta)

