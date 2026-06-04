"""Structural errors, guards, and provider validation for translation (spec §5.5).

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. Everything
here fails loudly with STRUCTURAL labels only — NEVER the offending value or any
prompt content. Re-raises across the boundary with ``... from None`` so a
provider exception's text can never leak via ``__cause__``/``__context__``. No
I/O: this module imports neither the stdlib log module nor any HTTP client, and
holds no log handle of any kind. Keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Literal, NoReturn

from solwyn.exceptions import UntranslatableRequestError

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
