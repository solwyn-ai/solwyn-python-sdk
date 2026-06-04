"""Anthropic request/response/stream dialect for cross-provider translation.

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. This module
reshapes customer prompt CONTENT through an in-memory transform ONLY; it never
stores it on a long-lived object. No I/O: this module imports neither the stdlib
log module nor any HTTP client, and holds no log handle of any kind. Pure: same
input -> same output; no global state. Keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ._common import (
    _as_text,
    _common_scalars,
    _denormalize_finish_reason,
    _secure_image_url,
    normalize_finish_reason,
)
from ._guardrails import (
    _ANTHROPIC_PROPRIETARY_TOOL_PREFIXES,
    _CROSS_PROVIDER_TOOL_STREAM,
    _check_forbidden_keys,
    _content_part_label,
    _raise,
)
from ._models import (
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
