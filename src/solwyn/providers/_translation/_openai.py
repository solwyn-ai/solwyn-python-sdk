"""OpenAI request/response/stream dialect for cross-provider translation.

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

import json
from types import SimpleNamespace
from typing import Any

from ._common import (
    _as_text,
    _common_scalars,
    _denormalize_finish_reason,
    _image_to_url,
    _loads_args,
    _parse_data_uri,
    _secure_image_url,
    normalize_finish_reason,
)
from ._guardrails import (
    _CROSS_PROVIDER_TOOL_STREAM,
    _OPENAI_UNSUPPORTED_TOOL,
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


def _stream_delta_to_openai(delta: _StreamDelta) -> object:
    finish = _denormalize_finish_reason("openai", delta.finish_reason)
    inner = SimpleNamespace(
        index=0,
        delta=SimpleNamespace(role=None, content=delta.text),
        finish_reason=finish,
    )
    return SimpleNamespace(choices=[inner], model=None)
