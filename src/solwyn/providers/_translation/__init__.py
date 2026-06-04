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

import json
from types import SimpleNamespace
from typing import Any, Literal, NoReturn

from solwyn.exceptions import UntranslatableRequestError
from solwyn.providers._translation._models import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTool,
    CanonicalToolCall,
    ContentPart,
    ImagePart,
    TextPart,
    ToolChoice,
    ToolResultPart,
    ToolUsePart,
    _StreamDelta,
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

_PROVIDERS = ("openai", "anthropic", "google")

# Cross-provider streaming structural labels (§6.6). A tool-call or non-text media
# delta is out of the v1 streaming subset and RAISES with one of these; chunk
# content NEVER reaches the error. Defined here so the pre-dispatch fail-loud
# helper and the per-chunk translator share one label.
_CROSS_PROVIDER_TOOL_STREAM = "cross_provider_tool_stream"
_CROSS_PROVIDER_MULTIMODAL_STREAM = "cross_provider_multimodal_stream"


# --------------------------------------------------------------------------- #
# Fail-loud helper                                                             #
# --------------------------------------------------------------------------- #
def _raise(source: str, target: str, feature: str) -> NoReturn:
    """Raise a STRUCTURAL UntranslatableRequestError, suppressing any context.

    ``from None`` sets ``__suppress_context__``/drops ``__cause__`` so no
    provider-exception text or prompt content leaks through the chained-exception
    display. When this is raised from inside a ``_Guard`` that is handling a
    content-bearing exception, ``raise ... from None`` would still re-bind
    ``__context__`` to that exception object (retaining the offending value on the
    raised error even though it is hidden from display); the guard NULLS
    ``__context__`` after the fact for defense-in-depth (fix [E]).
    """
    raise UntranslatableRequestError(source=source, target=target, feature=feature) from None


def _validate_provider(provider: str) -> None:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Known: {list(_PROVIDERS)}")


def fail_cross_provider_tool_stream(*, source: str, target: str) -> NoReturn:
    """Abort a cross-provider TOOL-using STREAMING hop (P3; §6.6, §4.1 Decision A).

    P3 ships per-chunk text-stream normalization, so a PLAIN-TEXT cross-provider
    streaming hop now proceeds. A TOOL-using streamed response, however, cannot be
    normalized cross-provider — tool-call deltas are out of the v1 streaming
    subset. RAISE a structural ``cross_provider_tool_stream`` error BEFORE dispatch
    so the chain aborts cleanly (no foreign stream served). STRUCTURAL labels only.
    """
    _raise(source, target, _CROSS_PROVIDER_TOOL_STREAM)


# --------------------------------------------------------------------------- #
# Malformed-content guard (fix [B]).                                           #
# --------------------------------------------------------------------------- #
# A malformed caller value (non-string text, non-dict tool input, wrong-typed
# field) would otherwise surface as a raw pydantic ValidationError / ValueError
# / KeyError whose text embeds the offending value. ``_guard`` runs the risky
# construction and converts ANY non-Untranslatable exception into a STRUCTURAL
# UntranslatableRequestError with ``from None`` (suppressing the context so the
# offending value cannot leak via __cause__/__context__/__notes__).
_MALFORMED_REQUEST = "malformed_request"


class _Guard:
    """Context manager that converts content-bearing errors to structural ones.

    Usage::

        with _guard(source, target, feature):
            risky_pydantic_construction()
    """

    __slots__ = ("_source", "_target", "_feature")

    def __init__(self, source: str, target: str, feature: str) -> None:
        self._source = source
        self._target = target
        self._feature = feature

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> Literal[False]:
        # NEVER suppresses (always returns False or re-raises a structural error),
        # so mypy knows the guarded ``with`` body must return or propagate.
        if exc is None:
            return False
        # Already-structural errors pass through unchanged (they carry no value).
        if isinstance(exc, UntranslatableRequestError):
            return False
        # Anything else (ValidationError/ValueError/KeyError/TypeError) may carry
        # the offending value in its text — convert to a structural label and
        # DROP the chain so nothing leaks.
        try:
            _raise(self._source, self._target, self._feature)
        except UntranslatableRequestError as structural:
            # ``raise ... from None`` inside _raise re-binds __context__ to the
            # content-bearing ``exc`` we are handling here (whose text may embed
            # the offending chunk/value). SEVER it so the offending value is
            # retained NOWHERE on the raised object — not merely hidden from the
            # traceback display (fix [E]). __cause__ is already None.
            structural.__context__ = None
            raise


def _guard(source: str, target: str, feature: str = _MALFORMED_REQUEST) -> _Guard:
    return _Guard(source, target, feature)


# --------------------------------------------------------------------------- #
# Unsupported scalar/structural keys, checked per inbound dialect (§5.5).      #
# The label is the structural key name; the value is NEVER read into the error.#
# --------------------------------------------------------------------------- #
# Keys that are unsupported regardless of value (presence alone -> RAISE).
_FORBIDDEN_TOP_LEVEL: dict[str, str] = {
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "top_k": "top_k",
    "response_format": "response_format",
    "response_schema": "response_schema",
    "logprobs": "logprobs",
    "top_logprobs": "top_logprobs",
    "logit_bias": "logit_bias",
    "service_tier": "service_tier",
    "reasoning_effort": "reasoning_effort",
    "reasoning": "reasoning",
    "thinking": "thinking",
    "thinking_config": "thinking_config",
    "cache_control": "cache_control",
    "cached_content": "cached_content",
    "response_mime_type": "response_mime_type",
}

# Recognized canonical request keys per inbound dialect. Anything outside the
# union of {recognized scalars/structures, these} at the relevant scope is an
# unrecognized native kwarg and RAISES "unsupported_kwarg.<key>" (fail-closed,
# fix [A]). The key NAME is an API field identifier (not prompt content), so it
# is an acceptable STRUCTURAL label; the VALUE is never read into the error.
_RECOGNIZED_OPENAI_TOP_LEVEL = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
)
_RECOGNIZED_ANTHROPIC_TOP_LEVEL = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "stop_sequences",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
)
_RECOGNIZED_GOOGLE_TOP_LEVEL = frozenset(
    {"model", "contents", "config", "max_output_tokens", "stream"}
)
_RECOGNIZED_GOOGLE_CONFIG = frozenset(
    {
        "system_instruction",
        "max_output_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
        "tools",
        "tool_config",
        "parallel_tool_calls",
        "candidate_count",
    }
)


# Caller-controlled content-part discriminators (a block ``type``) map to FIXED
# structural labels (fix [C]). An unrecognized discriminator must NEVER be
# interpolated into the feature label (it could carry caller data); it falls
# back to a CONSTANT. ``multimodal.*`` is the consistent label for non-text,
# non-image media (document/audio/video/PDF), used across all parsers (fix [L]).
_MULTIMODAL_BLOCK_LABELS: dict[str, str] = {
    "document": "multimodal.document",
    "file": "multimodal.document",
    "input_file": "multimodal.document",
    "audio": "multimodal.audio",
    "input_audio": "multimodal.audio",
    "video": "multimodal.video",
}
_CONTENT_PART_UNKNOWN = "content_part.unknown"
_MULTIMODAL_UNKNOWN = "multimodal.unknown"
_OPENAI_UNSUPPORTED_TOOL = "openai.unsupported_tool"


def _content_part_label(btype: object) -> str:
    """Fixed structural label for an unrecognized content-part discriminator.

    Multimodal media types map to ``multimodal.*``; anything else falls back to
    a CONSTANT. The raw discriminator is NEVER returned (it may carry caller
    data — fix [C]).
    """
    if isinstance(btype, str) and btype in _MULTIMODAL_BLOCK_LABELS:
        return _MULTIMODAL_BLOCK_LABELS[btype]
    return _CONTENT_PART_UNKNOWN


# Anthropic proprietary tool ``type`` prefixes (server tools, not custom funcs).
_ANTHROPIC_PROPRIETARY_TOOL_PREFIXES = (
    "computer",
    "bash",
    "text_editor",
    "web_search",
    "code_execution",
)
# Google built-in tool keys (a tool entry is a dict with one of these keys).
_GOOGLE_BUILTIN_TOOL_KEYS = (
    "google_search",
    "google_search_retrieval",
    "code_execution",
    "url_context",
)


def _check_forbidden_keys(provider: str, kwargs: dict[str, Any]) -> None:
    """RAISE on any §5.5 inference-control / structured-output / caching key,
    then FAIL CLOSED on any remaining unrecognized native kwarg (fix [A]).

    Scans top-level kwargs AND (for Google) the nested ``config`` dict, since
    Gemini hangs generation controls off ``config.*``. Specific labels (the
    forbidden-key set, ``n>1``, ``candidate_count>1``) take precedence over the
    generic ``unsupported_kwarg.<key>`` fallback.
    """
    scopes: list[dict[str, Any]] = [kwargs]
    config = kwargs.get("config")
    if isinstance(config, dict):
        scopes.append(config)
    if provider == "anthropic":
        mcp = kwargs.get("mcp_servers")
        if mcp:
            _raise(provider, "*", "anthropic.mcp_servers")
    # 1. Dedicated labels first (these win over the generic fallback).
    for scope in scopes:
        for key, label in _FORBIDDEN_TOP_LEVEL.items():
            if key in scope and scope[key] is not None:
                _raise(provider, "*", label)
        if "n" in scope and isinstance(scope["n"], int) and scope["n"] > 1:
            _raise(provider, "*", "n>1")
        # Gemini's n>1 equivalent (fix [J]).
        cc = scope.get("candidate_count")
        if isinstance(cc, int) and cc > 1:
            _raise(provider, "*", "candidate_count>1")

    # 2. Fail closed: any remaining unrecognized key is dropped silently today;
    #    instead RAISE so anything outside the canonical subset is caught on the
    #    first cross-provider hop. Only the key NAME (an API field identifier)
    #    reaches the label — never the value.
    if provider == "openai":
        for key in kwargs:
            if key not in _RECOGNIZED_OPENAI_TOP_LEVEL:
                _raise(provider, "*", f"unsupported_kwarg.{key}")
    elif provider == "anthropic":
        for key in kwargs:
            if key not in _RECOGNIZED_ANTHROPIC_TOP_LEVEL:
                _raise(provider, "*", f"unsupported_kwarg.{key}")
    else:  # google
        for key in kwargs:
            if key not in _RECOGNIZED_GOOGLE_TOP_LEVEL:
                _raise(provider, "*", f"unsupported_kwarg.{key}")
        if isinstance(config, dict):
            for key in config:
                if key not in _RECOGNIZED_GOOGLE_CONFIG:
                    _raise(provider, "*", f"unsupported_kwarg.{key}")


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


def _common_scalars(provider: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull temperature/top_p/stop/stream out of a flat kwargs scope."""
    out: dict[str, Any] = {}
    temperature = kwargs.get("temperature")
    if temperature is not None:
        if float(temperature) > 1.0:
            _raise(provider, "*", "temperature>1.0")
        out["temperature"] = float(temperature)
    top_p = kwargs.get("top_p")
    if top_p is not None:
        out["top_p"] = float(top_p)
    out["stop"] = _normalize_stop(provider, kwargs)
    out["stream"] = bool(kwargs.get("stream", False))
    return out


def _normalize_stop(provider: str, scope: dict[str, Any]) -> list[str] | None:
    raw = scope.get("stop")
    if raw is None and isinstance(scope.get("stop_sequences"), list):
        raw = scope.get("stop_sequences")
    if raw is None:
        return None
    stops = [raw] if isinstance(raw, str) else list(raw)
    if len(stops) > 4:
        _raise(provider, "*", "stop>4")
    return stops


# ------------------------------- OpenAI -> canonical ------------------------ #
def _openai_to_canonical(kwargs: dict[str, Any]) -> CanonicalRequest:
    # Responses API shape uses input=/instructions=, not messages= (§5.5).
    if "input" in kwargs or "instructions" in kwargs:
        _raise("openai", "*", "responses_api")
    _check_forbidden_keys("openai", kwargs)

    raw_messages = kwargs.get("messages") or []
    system_chunks: list[str] = []
    messages: list[CanonicalMessage] = []
    # Pre-scan tool_call name map for OpenAI tool result envelopes.
    call_name_by_id: dict[str, str] = {}
    pending_tool_call_ids: set[str] = set()

    for msg in raw_messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            # Multiple system/developer messages CONCATENATE (join with "\n\n")
            # rather than keeping only the last (silent content loss; fix [H1]).
            system_chunks.append(_openai_system_text(msg))
            continue
        if role == "assistant":
            cm, pending = _openai_assistant_to_canonical(msg, call_name_by_id)
            pending_tool_call_ids |= pending
            messages.append(cm)
            continue
        if role == "tool":
            messages.append(_openai_tool_result_to_canonical(msg, pending_tool_call_ids))
            continue
        if role == "user":
            messages.append(_openai_user_to_canonical(msg))
            continue
        _raise("openai", "*", "unknown_message_role")

    if pending_tool_call_ids:
        _raise("openai", "*", "dangling_tool_call")

    system = "\n\n".join(system_chunks) if system_chunks else None
    return CanonicalRequest(
        system=system,
        messages=messages,
        max_tokens=_openai_max_tokens(kwargs),
        tools=_openai_tools_to_canonical(kwargs.get("tools")),
        tool_choice=_openai_tool_choice_to_canonical(kwargs.get("tool_choice")),
        parallel_tool_calls=kwargs.get("parallel_tool_calls", True) is not False,
        **_common_scalars("openai", kwargs),
    )


def _openai_max_tokens(kwargs: dict[str, Any]) -> int:
    value = kwargs.get("max_completion_tokens")
    if value is None:
        value = kwargs.get("max_tokens")
    if value is None:
        _raise("openai", "*", "missing_max_tokens")
    return int(value)


def _openai_system_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    # A list-content system message (block list) is not a plain string (§5.2).
    _raise("openai", "*", "system_block_list")


def _openai_user_to_canonical(msg: dict[str, Any]) -> CanonicalMessage:
    return CanonicalMessage(role="user", content=_openai_parts(msg.get("content")))


def _openai_parts(content: Any) -> list[ContentPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(text=content)]
    parts: list[ContentPart] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append(TextPart(text=block.get("text", "")))
        elif btype == "image_url":
            parts.append(_openai_image_to_canonical(block.get("image_url", {})))
        else:
            # FIXED structural label — the caller-controlled discriminator is
            # NEVER interpolated into the feature (fix [C]).
            _raise("openai", "*", _content_part_label(btype))
    return parts


def _secure_image_url(provider: str, url: str) -> ImagePart:
    """Accept only ``https://`` image URLs; an ``http://`` (insecure) or other
    non-https URL RAISES ``image.insecure_url`` (fix [H2])."""
    if url.startswith("https://"):
        return ImagePart(url=url)
    _raise(provider, "*", "image.insecure_url")


def _openai_image_to_canonical(image_url: dict[str, Any]) -> ImagePart:
    url = image_url.get("url", "")
    if url.startswith("data:"):
        media_type, data = _parse_data_uri(url)
        return ImagePart(media_type=media_type, data=data)
    if url.startswith("http"):
        return _secure_image_url("openai", url)
    _raise("openai", "*", "image.opaque_handle")


def _openai_assistant_to_canonical(
    msg: dict[str, Any], call_name_by_id: dict[str, str]
) -> tuple[CanonicalMessage, set[str]]:
    parts: list[ContentPart] = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        parts.append(TextPart(text=text))
    elif isinstance(text, list):
        parts.extend(_openai_parts(text))
    tool_calls = msg.get("tool_calls") or []
    names = [c.get("function", {}).get("name") for c in tool_calls]
    if len(names) != len(set(names)) and len(tool_calls) > 1:
        _raise("openai", "*", "parallel_same_name_tool_calls")
    pending: set[str] = set()
    for call in tool_calls:
        fn = call.get("function", {})
        call_id = call["id"]
        name = fn.get("name", "")
        call_name_by_id[call_id] = name
        pending.add(call_id)
        parts.append(
            ToolUsePart(
                id=call_id,
                name=name,
                input=_loads_args(fn.get("arguments")),
            )
        )
    return CanonicalMessage(role="assistant", content=parts), pending


def _openai_tool_result_to_canonical(msg: dict[str, Any], pending: set[str]) -> CanonicalMessage:
    call_id = msg.get("tool_call_id", "")
    if call_id not in pending:
        _raise("openai", "*", "orphan_tool_result")
    pending.discard(call_id)
    return CanonicalMessage(
        role="user",
        content=[ToolResultPart(tool_use_id=call_id, content=_as_text(msg.get("content")))],
    )


# Known OpenAI non-function (proprietary) tool ``type`` identifiers. These are
# fixed API field identifiers (not caller content), so they may surface in the
# structural label. Anything OUTSIDE this set is treated as caller-controlled and
# falls back to a CONSTANT so an exotic type can never echo back (fix [C]).
_OPENAI_KNOWN_TOOL_TYPES = frozenset(
    {
        "file_search",
        "web_search",
        "web_search_preview",
        "computer_use_preview",
        "code_interpreter",
        "image_generation",
        "mcp",
    }
)


def _openai_tools_to_canonical(tools: Any) -> list[CanonicalTool] | None:
    if not tools:
        return None
    out: list[CanonicalTool] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype != "function":
            if isinstance(ttype, str) and ttype in _OPENAI_KNOWN_TOOL_TYPES:
                _raise("openai", "*", f"openai.{ttype}")
            _raise("openai", "*", _OPENAI_UNSUPPORTED_TOOL)
        fn = tool.get("function", {})
        out.append(
            CanonicalTool(
                name=fn.get("name", ""),
                description=fn.get("description"),
                parameters=fn.get("parameters", {}),
            )
        )
    return out


def _openai_tool_choice_to_canonical(choice: Any) -> ToolChoice | None:
    if choice is None:
        return None
    if choice == "auto":
        return ToolChoice(mode="auto")
    if choice == "required":
        return ToolChoice(mode="required")
    if choice == "none":
        return ToolChoice(mode="none")
    if isinstance(choice, dict) and choice.get("type") == "function":
        return ToolChoice(mode="force", name=choice.get("function", {}).get("name"))
    _raise("openai", "*", "tool_choice.unknown")


# ----------------------------- Anthropic -> canonical ----------------------- #
def _anthropic_to_canonical(kwargs: dict[str, Any]) -> CanonicalRequest:
    _check_forbidden_keys("anthropic", kwargs)
    system = _anthropic_system(kwargs.get("system"))
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is None:
        _raise("anthropic", "*", "missing_max_tokens")

    messages: list[CanonicalMessage] = []
    pending: set[str] = set()
    for msg in kwargs.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            cm, p = _anthropic_assistant_to_canonical(content)
            pending |= p
            messages.append(cm)
        elif role == "user":
            cm = _anthropic_user_to_canonical(content, pending)
            messages.append(cm)
        else:
            _raise("anthropic", "*", "unknown_message_role")

    if pending:
        _raise("anthropic", "*", "dangling_tool_call")

    return CanonicalRequest(
        system=system,
        messages=messages,
        max_tokens=int(max_tokens),
        tools=_anthropic_tools_to_canonical(kwargs.get("tools")),
        tool_choice=_anthropic_tool_choice_to_canonical(kwargs.get("tool_choice")),
        parallel_tool_calls=kwargs.get("parallel_tool_calls", True) is not False,
        **_common_scalars("anthropic", kwargs),
    )


def _anthropic_system(system: Any) -> str | None:
    if system is None or isinstance(system, str):
        return system
    # A block-list system (used for cache_control) is not a plain string (§5.2).
    _raise("anthropic", "*", "system_block_list")


def _anthropic_parts(content: Any) -> list[ContentPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(text=content)]
    parts: list[ContentPart] = []
    for block in content:
        if "cache_control" in block:
            _raise("anthropic", "*", "cache_control")
        btype = block.get("type")
        if btype == "text":
            parts.append(TextPart(text=block.get("text", "")))
        elif btype == "image":
            parts.append(_anthropic_image_to_canonical(block.get("source", {})))
        else:
            # Fixed-label mapping (multimodal.* for document/audio/video,
            # constant otherwise); the discriminator never echoes (fix [C]/[L]).
            _raise("anthropic", "*", _content_part_label(btype))
    return parts


def _anthropic_image_to_canonical(source: dict[str, Any]) -> ImagePart:
    stype = source.get("type")
    if stype == "base64":
        return ImagePart(media_type=source.get("media_type"), data=source.get("data"))
    if stype == "url":
        return _secure_image_url("anthropic", source.get("url", ""))
    # file_id / Files-API handles are provider-specific (§5.4) -> RAISE.
    _raise("anthropic", "*", "image.opaque_handle")


def _anthropic_assistant_to_canonical(content: Any) -> tuple[CanonicalMessage, set[str]]:
    parts: list[ContentPart] = []
    pending: set[str] = set()
    tool_use_names: list[str] = []
    if isinstance(content, str):
        parts.append(TextPart(text=content))
    else:
        for block in content or []:
            if "cache_control" in block:
                _raise("anthropic", "*", "cache_control")
            btype = block.get("type")
            if btype == "text":
                parts.append(TextPart(text=block.get("text", "")))
            elif btype == "tool_use":
                tool_use_names.append(block.get("name", ""))
                pending.add(block["id"])
                parts.append(
                    ToolUsePart(
                        id=block["id"],
                        name=block.get("name", ""),
                        input=dict(block.get("input") or {}),
                    )
                )
            else:
                _raise("anthropic", "*", _content_part_label(btype))
    if len(tool_use_names) != len(set(tool_use_names)) and len(tool_use_names) > 1:
        _raise("anthropic", "*", "parallel_same_name_tool_calls")
    return CanonicalMessage(role="assistant", content=parts), pending


def _anthropic_user_to_canonical(content: Any, pending: set[str]) -> CanonicalMessage:
    # A user turn may carry tool_result blocks (resolving prior tool_use).
    parts: list[ContentPart] = []
    if isinstance(content, str):
        parts.append(TextPart(text=content))
    else:
        for block in content or []:
            if "cache_control" in block:
                _raise("anthropic", "*", "cache_control")
            btype = block.get("type")
            if btype == "tool_result":
                tuid = block.get("tool_use_id", "")
                if tuid not in pending:
                    _raise("anthropic", "*", "orphan_tool_result")
                pending.discard(tuid)
                parts.append(
                    ToolResultPart(tool_use_id=tuid, content=_as_text(block.get("content")))
                )
            elif btype == "text":
                parts.append(TextPart(text=block.get("text", "")))
            elif btype == "image":
                parts.append(_anthropic_image_to_canonical(block.get("source", {})))
            else:
                _raise("anthropic", "*", _content_part_label(btype))
    return CanonicalMessage(role="user", content=parts)


def _anthropic_tools_to_canonical(tools: Any) -> list[CanonicalTool] | None:
    if not tools:
        return None
    out: list[CanonicalTool] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype is not None and ttype != "custom":
            # Proprietary server tools (computer_use/bash/web_search/...) -> RAISE.
            for prefix in _ANTHROPIC_PROPRIETARY_TOOL_PREFIXES:
                if ttype.startswith(prefix):
                    _raise("anthropic", "*", f"anthropic.{prefix}")
            _raise("anthropic", "*", f"anthropic.{ttype}")
        out.append(
            CanonicalTool(
                name=tool.get("name", ""),
                description=tool.get("description"),
                parameters=tool.get("input_schema", {}),
            )
        )
    return out


def _anthropic_tool_choice_to_canonical(choice: Any) -> ToolChoice | None:
    if choice is None:
        return None
    ctype = choice.get("type")
    if ctype == "auto":
        return ToolChoice(mode="auto")
    if ctype == "any":
        return ToolChoice(mode="required")
    if ctype == "none":
        return ToolChoice(mode="none")
    if ctype == "tool":
        return ToolChoice(mode="force", name=choice.get("name"))
    _raise("anthropic", "*", "tool_choice.unknown")


# ------------------------------- Google -> canonical ------------------------ #
def _google_to_canonical(kwargs: dict[str, Any]) -> CanonicalRequest:
    _check_forbidden_keys("google", kwargs)
    config = kwargs.get("config") or {}
    if not isinstance(config, dict):
        config = {}

    system = config.get("system_instruction")
    if system is not None and not isinstance(system, str):
        _raise("google", "*", "system_block_list")
    max_tokens = config.get("max_output_tokens") or kwargs.get("max_output_tokens")
    if max_tokens is None:
        _raise("google", "*", "missing_max_tokens")

    messages: list[CanonicalMessage] = []
    pending: set[str] = set()
    for turn in kwargs.get("contents") or []:
        role = turn.get("role", "user")
        parts_in = turn.get("parts") or []
        if role == "model":
            cm, p = _google_model_to_canonical(parts_in)
            pending |= p
            messages.append(cm)
        elif role in ("user", "tool", "function"):
            cm = _google_user_to_canonical(parts_in, pending, role)
            messages.append(cm)
        else:
            _raise("google", "*", "unknown_message_role")

    if pending:
        _raise("google", "*", "dangling_tool_call")

    # config-level scalars live under config.* for Gemini.
    scalar_scope = {
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "stop": config.get("stop_sequences"),
        "stream": kwargs.get("stream", False),
    }
    return CanonicalRequest(
        system=system,
        messages=messages,
        max_tokens=int(max_tokens),
        tools=_google_tools_to_canonical(config.get("tools")),
        tool_choice=_google_tool_choice_to_canonical(config.get("tool_config")),
        parallel_tool_calls=config.get("parallel_tool_calls", True) is not False,
        **_common_scalars("google", scalar_scope),
    )


def _google_model_to_canonical(parts_in: list[dict[str, Any]]) -> tuple[CanonicalMessage, set[str]]:
    parts: list[ContentPart] = []
    pending: set[str] = set()
    fc_names: list[str] = []
    for part in parts_in:
        if "text" in part:
            parts.append(TextPart(text=part.get("text", "")))
        elif "function_call" in part:
            fc = part["function_call"]
            name = fc.get("name", "")
            # Gemini now returns call ids; when absent, mint one deterministically
            # from the name so the matching function_response resolves it.
            call_id = fc.get("id") or f"call_{name}"
            fc_names.append(name)
            pending.add(call_id)
            parts.append(ToolUsePart(id=call_id, name=name, input=dict(fc.get("args") or {})))
        elif "inline_data" in part or "file_data" in part:
            _raise("google", "*", "image.opaque_handle")
        else:
            _raise("google", "*", "content_part.unknown")
    if len(fc_names) != len(set(fc_names)) and len(fc_names) > 1:
        _raise("google", "*", "parallel_same_name_tool_calls")
    return CanonicalMessage(role="assistant", content=parts), pending


def _google_user_to_canonical(
    parts_in: list[dict[str, Any]], pending: set[str], role: str
) -> CanonicalMessage:
    parts: list[ContentPart] = []
    for part in parts_in:
        if "text" in part:
            parts.append(TextPart(text=part.get("text", "")))
        elif "function_response" in part:
            fr = part["function_response"]
            tuid = fr.get("id") or f"call_{fr.get('name', '')}"
            if tuid not in pending:
                _raise("google", "*", "orphan_tool_result")
            pending.discard(tuid)
            parts.append(
                ToolResultPart(tool_use_id=tuid, content=_as_text(_unwrap_google_response(fr)))
            )
        elif "inline_data" in part:
            parts.append(_google_inline_image_to_canonical(part["inline_data"]))
        elif "file_data" in part:
            # gs:// / Files-API handles are provider-specific (§5.4) -> RAISE.
            _raise("google", "*", "image.opaque_handle")
        else:
            _raise("google", "*", "content_part.unknown")
    return CanonicalMessage(role="user", content=parts)


# Top-level MIME categories that map to a FIXED multimodal label. Anything else
# (the mime is caller-controlled) falls back to a CONSTANT so it never echoes.
_MIME_CATEGORY_LABELS: dict[str, str] = {
    "audio": "multimodal.audio",
    "video": "multimodal.video",
    "application": "multimodal.document",
    "text": "multimodal.document",
}


def _unwrap_google_response(fr: dict[str, Any]) -> Any:
    """Unwrap a Google ``function_response.response`` back to bare content.

    ``from_canonical(google)`` wraps a tool_result content as
    ``response={"result": <content>}``. Reading ``fr["response"]`` and json-dumping
    the whole dict would corrupt a hop into-and-out-of Google, so unwrap the
    single-key ``{"result": ...}`` envelope symmetrically (fix [G]).
    """
    response = fr.get("response")
    if isinstance(response, dict) and list(response.keys()) == ["result"]:
        return response["result"]
    return response


def _google_inline_image_to_canonical(inline: dict[str, Any]) -> ImagePart:
    mime = inline.get("mime_type", "")
    if not mime.startswith("image/"):
        # FIXED structural label — the caller-controlled mime never echoes (fix [C]).
        category = mime.split("/", 1)[0] if isinstance(mime, str) else ""
        _raise("google", "*", _MIME_CATEGORY_LABELS.get(category, _MULTIMODAL_UNKNOWN))
    return ImagePart(media_type=mime, data=inline.get("data"))


def _google_tools_to_canonical(tools: Any) -> list[CanonicalTool] | None:
    if not tools:
        return None
    out: list[CanonicalTool] = []
    for tool in tools:
        for builtin in _GOOGLE_BUILTIN_TOOL_KEYS:
            if builtin in tool:
                _raise("google", "*", f"google.{builtin}")
        for decl in tool.get("function_declarations", []):
            out.append(
                CanonicalTool(
                    name=decl.get("name", ""),
                    description=decl.get("description"),
                    parameters=decl.get("parameters_json_schema") or decl.get("parameters", {}),
                )
            )
    return out or None


def _google_tool_choice_to_canonical(tool_config: Any) -> ToolChoice | None:
    if not tool_config:
        return None
    fcc = tool_config.get("function_calling_config", {})
    mode = fcc.get("mode")
    if mode == "AUTO":
        return ToolChoice(mode="auto")
    if mode == "NONE":
        return ToolChoice(mode="none")
    if mode == "ANY":
        allowed = fcc.get("allowed_function_names")
        if allowed:
            return ToolChoice(mode="force", name=allowed[0])
        return ToolChoice(mode="required")
    _raise("google", "*", "tool_choice.unknown")


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


# ------------------------------ canonical -> OpenAI ------------------------- #
def _canonical_to_openai(canonical: CanonicalRequest, model: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if canonical.system is not None:
        messages.append({"role": "system", "content": canonical.system})
    for msg in canonical.messages:
        messages.extend(_canonical_msg_to_openai(msg))

    out: dict[str, Any] = {
        "model": model,
        # MUST emit max_completion_tokens, never max_tokens (§5.2).
        "max_completion_tokens": canonical.max_tokens,
        "messages": messages,
    }
    _apply_common_to_openai(out, canonical)
    # parallel_tool_calls=False is a portable OpenAI flag (do not silently drop;
    # fix [F]). Only emitted when explicitly disabled (default True is provider
    # default and stays implicit).
    if canonical.parallel_tool_calls is False:
        out["parallel_tool_calls"] = False
    if canonical.tools is not None:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in canonical.tools
        ]
    if canonical.tool_choice is not None:
        out["tool_choice"] = _canonical_tool_choice_to_openai(canonical.tool_choice)
    return out


def _apply_common_to_openai(out: dict[str, Any], canonical: CanonicalRequest) -> None:
    if canonical.temperature is not None:
        out["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        out["top_p"] = canonical.top_p
    if canonical.stop is not None:
        out["stop"] = canonical.stop
    if canonical.stream:
        out["stream"] = True


def _canonical_msg_to_openai(msg: CanonicalMessage) -> list[dict[str, Any]]:
    if msg.role == "assistant":
        return [_canonical_assistant_to_openai(msg)]
    # User turn: tool_result parts become separate role:tool messages.
    tool_results = [p for p in msg.content if isinstance(p, ToolResultPart)]
    if tool_results:
        # OpenAI cannot represent a tool_result turn that also carries text/image
        # content — rendering only the role:tool messages would silently DROP the
        # other parts. Fail loud at the render boundary (Anthropic CAN represent
        # mixed turns, so this is render-side, not canonicalization-side).
        if any(not isinstance(p, ToolResultPart) for p in msg.content):
            _raise("*", "openai", "tool_result.mixed_content")
        return [
            {"role": "tool", "tool_call_id": p.tool_use_id, "content": p.content}
            for p in tool_results
        ]
    return [{"role": "user", "content": _canonical_parts_to_openai(msg.content)}]


def _canonical_assistant_to_openai(msg: CanonicalMessage) -> dict[str, Any]:
    text_parts = [p for p in msg.content if isinstance(p, TextPart)]
    tool_uses = [p for p in msg.content if isinstance(p, ToolUsePart)]
    out: dict[str, Any] = {"role": "assistant"}
    out["content"] = "".join(p.text for p in text_parts) if text_parts else None
    if tool_uses:
        out["tool_calls"] = [
            {
                "id": p.id,
                "type": "function",
                "function": {"name": p.name, "arguments": json.dumps(p.input)},
            }
            for p in tool_uses
        ]
    return out


def _canonical_parts_to_openai(parts: list[ContentPart]) -> Any:
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            out.append({"type": "image_url", "image_url": {"url": _image_to_url(part)}})
    return out


def _canonical_tool_choice_to_openai(choice: ToolChoice) -> Any:
    if choice.mode in ("auto", "required", "none"):
        return choice.mode
    return {"type": "function", "function": {"name": choice.name}}


# ----------------------------- canonical -> Anthropic ----------------------- #
def _canonical_to_anthropic(canonical: CanonicalRequest, model: str) -> dict[str, Any]:
    # Drop turns that render to an empty content list — Anthropic 400s on
    # content=[] (an empty assistant turn has no text and no tool_calls; fix [I]).
    messages = [
        rendered
        for m in canonical.messages
        if (rendered := _canonical_msg_to_anthropic(m))["content"]
    ]
    out: dict[str, Any] = {
        "model": model,
        "max_tokens": canonical.max_tokens,  # required (§5.2)
        "messages": messages,
    }
    if canonical.system is not None:
        out["system"] = canonical.system
    if canonical.temperature is not None:
        out["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        out["top_p"] = canonical.top_p
    if canonical.stop is not None:
        out["stop_sequences"] = canonical.stop
    if canonical.stream:
        out["stream"] = True
    if canonical.tools is not None:
        out["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in canonical.tools
        ]
    if canonical.tool_choice is not None or canonical.parallel_tool_calls is False:
        out["tool_choice"] = _canonical_tool_choice_to_anthropic(
            canonical.tool_choice, parallel_tool_calls=canonical.parallel_tool_calls
        )
    return out


def _canonical_msg_to_anthropic(msg: CanonicalMessage) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for part in msg.content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image", "source": _image_to_anthropic_source(part)})
        elif isinstance(part, ToolUsePart):
            blocks.append(
                {"type": "tool_use", "id": part.id, "name": part.name, "input": part.input}
            )
        elif isinstance(part, ToolResultPart):
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": part.tool_use_id,
                    "content": part.content,
                }
            )
    return {"role": msg.role, "content": blocks}


def _image_to_anthropic_source(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "url", "url": part.url}
    return {"type": "base64", "media_type": part.media_type, "data": part.data}


def _canonical_tool_choice_to_anthropic(
    choice: ToolChoice | None, *, parallel_tool_calls: bool = True
) -> dict[str, Any]:
    # parallel_tool_calls=False maps to Anthropic's disable_parallel_tool_use
    # (fix [F]); valid only on auto/any/tool. Default to auto when no explicit
    # tool_choice was set but parallel-disable was requested.
    if choice is None:
        result: dict[str, Any] = {"type": "auto"}
    elif choice.mode == "auto":
        result = {"type": "auto"}
    elif choice.mode == "required":
        result = {"type": "any"}
    elif choice.mode == "none":
        result = {"type": "none"}
    else:
        result = {"type": "tool", "name": choice.name}
    if parallel_tool_calls is False and result["type"] != "none":
        result["disable_parallel_tool_use"] = True
    return result


# ------------------------------ canonical -> Google ------------------------- #
def _canonical_to_google(canonical: CanonicalRequest, model: str) -> dict[str, Any]:
    # Google has no first-class equivalent of disabling parallel tool calls (§5.5).
    if canonical.parallel_tool_calls is False:
        _raise("*", "google", "parallel_tool_calls=False")
    # Google function_response carries the function NAME, which the canonical
    # tool_result does not. Recover it from the matching tool_use across turns.
    name_by_id = _tool_use_name_map(canonical.messages)
    contents = [_canonical_msg_to_google(m, name_by_id) for m in canonical.messages]
    config: dict[str, Any] = {"max_output_tokens": canonical.max_tokens}
    if canonical.system is not None:
        config["system_instruction"] = canonical.system
    if canonical.temperature is not None:
        config["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        config["top_p"] = canonical.top_p
    if canonical.stop is not None:
        config["stop_sequences"] = canonical.stop
    if canonical.tools is not None:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters_json_schema": t.parameters,
                    }
                    for t in canonical.tools
                ]
            }
        ]
    if canonical.tool_choice is not None:
        config["tool_config"] = _canonical_tool_choice_to_google(canonical.tool_choice)

    out: dict[str, Any] = {"model": model, "contents": contents, "config": config}
    if canonical.stream:
        out["stream"] = True
    return out


def _tool_use_name_map(messages: list[CanonicalMessage]) -> dict[str, str]:
    """Map each tool_use id -> its function name across the whole history."""
    out: dict[str, str] = {}
    for msg in messages:
        for part in msg.content:
            if isinstance(part, ToolUsePart):
                out[part.id] = part.name
    return out


def _canonical_msg_to_google(msg: CanonicalMessage, name_by_id: dict[str, str]) -> dict[str, Any]:
    # tool_result parts make this a `tool` turn; otherwise user/model.
    parts: list[dict[str, Any]] = []
    if any(isinstance(p, ToolResultPart) for p in msg.content):
        # Google cannot represent a tool_result turn mixed with text/image parts —
        # emitting only function_response would silently DROP the other parts. Fail
        # loud at the render boundary (Anthropic CAN represent mixed turns).
        if any(not isinstance(p, ToolResultPart) for p in msg.content):
            _raise("*", "google", "tool_result.mixed_content")
        for p in msg.content:
            if isinstance(p, ToolResultPart):
                parts.append(
                    {
                        "function_response": {
                            "id": p.tool_use_id,
                            "name": name_by_id.get(p.tool_use_id, p.tool_use_id),
                            "response": {"result": p.content},
                        }
                    }
                )
        return {"role": "tool", "parts": parts}

    role = "model" if msg.role == "assistant" else "user"
    for part in msg.content:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
        elif isinstance(part, ImagePart):
            parts.append(_image_to_google_part(part))
        elif isinstance(part, ToolUsePart):
            parts.append({"function_call": {"id": part.id, "name": part.name, "args": part.input}})
    return {"role": role, "parts": parts}


def _image_to_google_part(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        # file_data.file_uri is the Files-API/gs:// handle channel, not a plain
        # public URL; emitting one there is likely invalid. Be conservative and
        # FAIL LOUD rather than ship a possibly-broken mapping (fix [K]). Inline
        # base64 images toward Google still work.
        _raise("*", "google", "image.url_unsupported_google")
    return {"inline_data": {"mime_type": part.media_type, "data": part.data}}


def _canonical_tool_choice_to_google(choice: ToolChoice) -> dict[str, Any]:
    if choice.mode == "auto":
        return {"function_calling_config": {"mode": "AUTO"}}
    if choice.mode == "required":
        return {"function_calling_config": {"mode": "ANY"}}
    if choice.mode == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    return {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [choice.name],
        }
    }


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


def _openai_response_to_canonical(response: object) -> CanonicalResponse:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return CanonicalResponse(model=getattr(response, "model", None))
    choice = choices[0]
    message = getattr(choice, "message", None)
    text = getattr(message, "content", None)
    raw_calls = getattr(message, "tool_calls", None) or []
    tool_calls = [
        CanonicalToolCall(
            id=getattr(c, "id", ""),
            name=getattr(getattr(c, "function", None), "name", ""),
            arguments=_loads_args(getattr(getattr(c, "function", None), "arguments", None)),
        )
        for c in raw_calls
    ] or None
    return CanonicalResponse(
        text=text,
        tool_calls=tool_calls,
        finish_reason=normalize_finish_reason("openai", getattr(choice, "finish_reason", None)),
        model=getattr(response, "model", None),
    )


def _anthropic_response_to_canonical(response: object) -> CanonicalResponse:
    blocks = getattr(response, "content", None) or []
    text_chunks: list[str] = []
    tool_calls: list[CanonicalToolCall] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_chunks.append(getattr(block, "text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                CanonicalToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=dict(getattr(block, "input", None) or {}),
                )
            )
    return CanonicalResponse(
        text="".join(text_chunks) or None,
        tool_calls=tool_calls or None,
        finish_reason=normalize_finish_reason("anthropic", getattr(response, "stop_reason", None)),
        model=getattr(response, "model", None),
    )


def _google_response_to_canonical(response: object) -> CanonicalResponse:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return CanonicalResponse(model=getattr(response, "model", None))
    candidate = candidates[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    text_chunks: list[str] = []
    tool_calls: list[CanonicalToolCall] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text is not None:
            text_chunks.append(part_text)
        fc = getattr(part, "function_call", None)
        if fc is not None:
            tool_calls.append(
                CanonicalToolCall(
                    id=getattr(fc, "id", "") or "",
                    name=getattr(fc, "name", ""),
                    arguments=dict(getattr(fc, "args", None) or {}),
                )
            )
    finish_reason = normalize_finish_reason("google", getattr(candidate, "finish_reason", None))
    # §5.4: Google encodes a tool call as STOP + function_call (no dedicated tool
    # finish reason). When tool calls are present and the mapped reason is the
    # plain "stop", upgrade to "tool_use" so the caller sees the tool finish
    # reason on normalization (fix [E]).
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_use"
    return CanonicalResponse(
        text="".join(text_chunks) or None,
        tool_calls=tool_calls or None,
        finish_reason=finish_reason,
        model=getattr(response, "model", None),
    )


def _canonical_response_to_openai(canonical: CanonicalResponse) -> object:
    tool_calls = None
    if canonical.tool_calls:
        tool_calls = [
            SimpleNamespace(
                id=tc.id,
                type="function",
                function=SimpleNamespace(name=tc.name, arguments=json.dumps(tc.arguments)),
            )
            for tc in canonical.tool_calls
        ]
    message = SimpleNamespace(role="assistant", content=canonical.text, tool_calls=tool_calls)
    finish = _denormalize_finish_reason("openai", canonical.finish_reason)
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice], model=canonical.model)


def _canonical_response_to_anthropic(canonical: CanonicalResponse) -> object:
    blocks: list[SimpleNamespace] = []
    if canonical.text:
        blocks.append(SimpleNamespace(type="text", text=canonical.text))
    for tc in canonical.tool_calls or []:
        blocks.append(SimpleNamespace(type="tool_use", id=tc.id, name=tc.name, input=tc.arguments))
    return SimpleNamespace(
        content=blocks,
        stop_reason=_denormalize_finish_reason("anthropic", canonical.finish_reason),
        model=canonical.model,
    )


class _GoogleResponse:
    """Duck-typed google-genai response shape (fix [C]).

    A Google drop-in caller does not only read ``candidates[0].content.parts[*]``
    — the idiomatic google-genai accessors are ``response.text`` (the
    concatenation of every text part) and ``response.function_calls`` (the
    function-call objects). The OpenAI (``.choices[0].message.content``) and
    Anthropic (``.content[0].text``) idioms already survive cross-provider
    failover; this gives the Google-shaped normalized object the same fidelity so
    a Google drop-in user's ``response.text`` / ``response.function_calls`` keep
    working after failover. Pure in-memory transform; no provider import.
    """

    def __init__(self, candidates: list[SimpleNamespace], model: str | None) -> None:
        self.candidates = candidates
        self.model = model

    def _parts(self) -> list[SimpleNamespace]:
        if not self.candidates:
            return []
        content = getattr(self.candidates[0], "content", None)
        return list(getattr(content, "parts", None) or [])

    @property
    def text(self) -> str | None:
        """Concatenation of all text parts; ``None`` when there is no text."""
        chunks = [p.text for p in self._parts() if getattr(p, "text", None) is not None]
        return "".join(chunks) if chunks else None

    @property
    def function_calls(self) -> list[SimpleNamespace]:
        """The function-call parts (idiomatic google-genai), in order."""
        return [fc for p in self._parts() if (fc := getattr(p, "function_call", None)) is not None]


def _canonical_response_to_google(canonical: CanonicalResponse) -> object:
    parts: list[SimpleNamespace] = []
    if canonical.text:
        parts.append(SimpleNamespace(text=canonical.text, function_call=None))
    for tc in canonical.tool_calls or []:
        parts.append(
            SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(id=tc.id, name=tc.name, args=tc.arguments),
            )
        )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts, role="model"),
        finish_reason=_denormalize_finish_reason("google", canonical.finish_reason),
    )
    return _GoogleResponse(candidates=[candidate], model=canonical.model)


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


# ------------------------------ served: OpenAI ------------------------------ #
def _openai_stream_chunk_to_deltas(
    served: str, requested: str, chunk: object
) -> list[_StreamDelta]:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return []
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    # A tool-call delta is out of the v1 streaming subset -> RAISE (no content).
    if getattr(delta, "tool_calls", None):
        _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
    out: list[_StreamDelta] = []
    text = getattr(delta, "content", None)
    if text:
        out.append(_StreamDelta(text=text))
    finish_raw = getattr(choice, "finish_reason", None)
    if finish_raw is not None:
        out.append(_StreamDelta(finish_reason=normalize_finish_reason(served, finish_raw)))
    return out


# ---------------------------- served: Anthropic ----------------------------- #
def _anthropic_stream_chunk_to_deltas(
    served: str, requested: str, chunk: object
) -> list[_StreamDelta]:
    etype = getattr(chunk, "type", None)
    if etype == "content_block_delta":
        delta = getattr(chunk, "delta", None)
        dtype = getattr(delta, "type", None)
        if dtype == "text_delta":
            return [_StreamDelta(text=getattr(delta, "text", "") or "")]
        # input_json_delta (tool-call args) / any non-text block delta is out of
        # the v1 streaming subset -> RAISE structurally (never read the value).
        _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
    if etype == "message_delta":
        stop_raw = getattr(getattr(chunk, "delta", None), "stop_reason", None)
        if stop_raw is None:
            return []
        return [_StreamDelta(finish_reason=normalize_finish_reason(served, stop_raw))]
    # content_block_start may open a tool_use block — that whole block is out of
    # the v1 subset, so fail loud at the START rather than mid-stream.
    if etype == "content_block_start":
        block = getattr(chunk, "content_block", None)
        if getattr(block, "type", None) not in (None, "text"):
            _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
        return []
    # message_start / content_block_stop / message_stop -> no caller chunk.
    return []


# ------------------------------ served: Google ------------------------------ #
def _google_stream_chunk_to_deltas(
    served: str, requested: str, chunk: object
) -> list[_StreamDelta]:
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return []
    candidate = candidates[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    out: list[_StreamDelta] = []
    for part in parts:
        # A function_call part (tool-call stream) is out of the v1 subset.
        if getattr(part, "function_call", None) is not None:
            _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
        # inline_data / file_data (image/non-text media) is out of the v1 subset.
        if (
            getattr(part, "inline_data", None) is not None
            or getattr(part, "file_data", None) is not None
        ):
            _raise(served, requested, _CROSS_PROVIDER_MULTIMODAL_STREAM)
        text = getattr(part, "text", None)
        if text:
            out.append(_StreamDelta(text=text))
    finish_raw = getattr(candidate, "finish_reason", None)
    if finish_raw is not None:
        out.append(_StreamDelta(finish_reason=normalize_finish_reason(served, finish_raw)))
    return out


# --------------- canonical _StreamDelta -> requested native chunk ----------- #
def _stream_delta_to_requested(requested: str, delta: _StreamDelta) -> object:
    if requested == "openai":
        return _stream_delta_to_openai(delta)
    if requested == "anthropic":
        return _stream_delta_to_anthropic(delta)
    return _stream_delta_to_google(delta)


def _stream_delta_to_openai(delta: _StreamDelta) -> object:
    finish = _denormalize_finish_reason("openai", delta.finish_reason)
    inner = SimpleNamespace(
        index=0,
        delta=SimpleNamespace(role=None, content=delta.text),
        finish_reason=finish,
    )
    return SimpleNamespace(choices=[inner], model=None)


def _stream_delta_to_anthropic(delta: _StreamDelta) -> object:
    if delta.finish_reason is not None:
        stop = _denormalize_finish_reason("anthropic", delta.finish_reason)
        return SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason=stop),
        )
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=delta.text),
    )


def _stream_delta_to_google(delta: _StreamDelta) -> object:
    finish = _denormalize_finish_reason("google", delta.finish_reason)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=delta.text)]),
        finish_reason=finish,
    )
    return SimpleNamespace(candidates=[candidate])


# --------------------------------------------------------------------------- #
# §5.4 Finish-reason normalization (and the inverse for response shaping).      #
# --------------------------------------------------------------------------- #
_FINISH_TO_CANONICAL: dict[str, dict[str, str]] = {
    "openai": {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "pause_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_use",
        "refusal": "content_filter",
    },
    "google": {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "BLOCKLIST": "content_filter",
    },
}

_CANONICAL_TO_FINISH: dict[str, dict[str, str]] = {
    "openai": {
        "stop": "stop",
        "length": "length",
        "tool_use": "tool_calls",
        "content_filter": "content_filter",
    },
    "anthropic": {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_use": "tool_use",
        "content_filter": "refusal",
    },
    "google": {
        "stop": "STOP",
        "length": "MAX_TOKENS",
        "tool_use": "STOP",
        "content_filter": "SAFETY",
    },
}


def normalize_finish_reason(
    served: str, raw: str | None
) -> Literal["stop", "length", "tool_use", "content_filter"] | None:
    """Map a provider-native finish/stop reason onto the canonical 4-value set."""
    if raw is None:
        return None
    mapped = _FINISH_TO_CANONICAL.get(served, {}).get(raw)
    return mapped  # type: ignore[return-value]


def _denormalize_finish_reason(requested: str, canonical: str | None) -> str | None:
    if canonical is None:
        return None
    return _CANONICAL_TO_FINISH.get(requested, {}).get(canonical)


# --------------------------------------------------------------------------- #
# Small in-memory helpers (pure; never log, never store).                      #
# --------------------------------------------------------------------------- #
def _parse_data_uri(uri: str) -> tuple[str, str]:
    """Split ``data:<media_type>;base64,<data>`` -> (media_type, data) VERBATIM.

    Anthropic strictly validates ``media_type`` vs bytes (§5.4), so the media
    type is preserved exactly as written.
    """
    header, _, data = uri.partition(",")
    meta = header[len("data:") :]
    media_type = meta.split(";", 1)[0]
    return media_type, data


def _image_to_url(part: ImagePart) -> str:
    if part.url is not None:
        return part.url
    return f"data:{part.media_type};base64,{part.data}"


def _loads_args(raw: Any) -> dict[str, Any]:
    """Decode a tool-call ``arguments`` value to a JSON object.

    OpenAI encodes arguments as a JSON STRING; Anthropic/Google as an object.
    Never logs the value; a malformed payload yields an empty dict structurally.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _as_text(content: Any) -> str:
    """Coerce a tool-result payload to a plain string (in-memory only)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                chunks.append(str(block["text"]))
            elif isinstance(block, str):
                chunks.append(block)
        return "".join(chunks)
    return json.dumps(content)
