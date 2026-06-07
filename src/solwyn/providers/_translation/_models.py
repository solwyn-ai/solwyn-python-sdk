"""Canonical Pydantic models for cross-provider translation.

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. These models
carry customer prompt CONTENT through an in-memory transform ONLY. No I/O: this
module imports neither the stdlib log module nor any HTTP client, and holds no
log handle of any kind. Same input -> same output; no global state. Keep the
firewall tests (``tests/unit/test_privacy_firewall.py`` /
``test_translation.py``) green.
===============================================================================
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Canonical models (Pydantic v2, extra="forbid")                               #
# --------------------------------------------------------------------------- #
class _Canonical(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextPart(_Canonical):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Canonical):
    type: Literal["image"] = "image"
    # Exactly one of (media_type+data) for base64, or url for a public HTTPS image.
    media_type: str | None = None
    data: str | None = None  # base64 payload
    url: str | None = None


class ToolUsePart(_Canonical):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultPart(_Canonical):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str


ContentPart = Annotated[
    TextPart | ImagePart | ToolUsePart | ToolResultPart,
    Field(discriminator="type"),
]


class CanonicalMessage(_Canonical):
    role: Literal["user", "assistant"]
    content: list[ContentPart]


class CanonicalTool(_Canonical):
    name: str
    description: str | None = None
    parameters: dict[str, Any]


class ToolChoice(_Canonical):
    mode: Literal["auto", "required", "none", "force"]
    name: str | None = None  # set only when mode == "force"


class CanonicalRequest(_Canonical):
    system: str | None = None
    messages: list[CanonicalMessage]
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    tools: list[CanonicalTool] | None = None
    tool_choice: ToolChoice | None = None
    # Structural request-shaping flag (NOT content). Default True == provider
    # default. Carried so the target-aware ``from_canonical`` can RAISE when a
    # ``parallel_tool_calls=False`` request targets Google, which has no
    # first-class equivalent. This is the only non-content addition.
    parallel_tool_calls: bool = True


class CanonicalToolCall(_Canonical):
    id: str
    name: str
    arguments: dict[str, Any]


class CanonicalResponse(_Canonical):
    text: str | None = None
    tool_calls: list[CanonicalToolCall] | None = None
    finish_reason: Literal["stop", "length", "tool_use", "content_filter"] | None = None
    model: str | None = None


# A streamed _StreamDelta is the tiny canonical event the per-provider parsers
# emit and the per-provider renderers consume. Exactly one of (text, finish) is
# set per delta; a served structural event yields NO _StreamDelta at all (an
# empty list out of translate_stream_chunk). This mirrors normalize_response's
# served->canonical->requested pipeline at the chunk granularity. Tool-call and
# non-text media on a cross-provider hop are OUT of the v1 streaming subset and
# RAISE structurally (no chunk content ever reaches the error). The
# _CROSS_PROVIDER_*_STREAM labels live in ``_guardrails`` so the pre-dispatch
# fail-loud helper can share them.


class _StreamDelta(_Canonical):
    # A text increment OR a terminal finish-reason; never both. ``text`` carries
    # streamed CONTENT and lives only inside this in-memory frame.
    text: str | None = None
    finish_reason: Literal["stop", "length", "tool_use", "content_filter"] | None = None
