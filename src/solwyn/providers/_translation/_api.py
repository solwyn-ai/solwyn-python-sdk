"""Public dispatchers routing cross-provider translation across dialects.

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. This module
routes customer prompt CONTENT through an in-memory transform ONLY, wrapping
every content access in the shared ``_guard`` so a malformed value converts to a
STRUCTURAL error with ``... from None`` — never the offending value. No I/O: this
module imports neither the stdlib log module nor any HTTP client, and holds no
log handle of any kind. Pure: same input -> same output; no global state. Keep
the firewall tests (``tests/unit/test_privacy_firewall.py`` /
``test_translation.py``) green.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from ._anthropic import (
    _anthropic_response_to_canonical,
    _anthropic_stream_chunk_to_deltas,
    _anthropic_to_canonical,
    _canonical_response_to_anthropic,
    _canonical_to_anthropic,
    _stream_delta_to_anthropic,
)
from ._google import (
    _canonical_response_to_google,
    _canonical_to_google,
    _google_response_to_canonical,
    _google_stream_chunk_to_deltas,
    _google_to_canonical,
    _stream_delta_to_google,
)
from ._guardrails import (
    _guard,
    _validate_provider,
)
from ._models import (
    CanonicalRequest,
    CanonicalResponse,
    _StreamDelta,
)
from ._openai import (
    _canonical_response_to_openai,
    _canonical_to_openai,
    _openai_response_to_canonical,
    _openai_stream_chunk_to_deltas,
    _openai_to_canonical,
    _stream_delta_to_openai,
)


# --------------------------------------------------------------------------- #
# to_canonical — parse a native request dict into the canonical form.          #
# --------------------------------------------------------------------------- #
def to_canonical(provider: str, kwargs: dict[str, Any]) -> CanonicalRequest:
    """Parse a native ``provider`` request dict into the canonical request.

    RAISES ``UntranslatableRequestError`` for anything outside the subset.
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
# Per-chunk stream translation — map ONE served-stream event into ZERO          #
# OR MORE chunks in the requested provider's native streaming dialect.          #
# --------------------------------------------------------------------------- #
# See ``_StreamDelta`` in ``_models`` for the canonical per-chunk event contract
# (text-OR-finish; a served structural event emits no delta; tool-call / non-text
# media on a cross-provider hop RAISE structurally — chunk content never reaches
# the error).


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
