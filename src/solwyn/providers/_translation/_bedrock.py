"""Bedrock Converse request/response/stream dialect for cross-provider translation.

================================ PRIVACY-CRITICAL ==============================
Part of the content-privileged ``providers/_translation`` package. This module
reshapes customer prompt CONTENT through an in-memory transform ONLY; it never
stores it on a long-lived object. No I/O and no log handle of any kind. Pure:
same input -> same output; no global state. Keep the firewall tests
(``tests/unit/test_privacy_firewall.py`` / ``test_translation.py``) green.
===============================================================================

Dialect notes. Converse kwargs reach this module carrying the uniform internal
``model`` key (the proxy renames boto3's ``modelId`` at the interception
boundary; dispatch renames it back). Generation controls nest under
``inferenceConfig``; tools nest under ``toolConfig`` with ``toolSpec`` /
``inputSchema.json`` wrappers; ``system`` is a LIST of content blocks; content
blocks are single-key UNION dicts (``{"text": ...}``, ``{"toolUse": ...}``) with
no ``type`` discriminator; images carry RAW bytes (base64 in the canonical
form); responses and stream events are plain DICTS, so rendering INTO this
dialect produces dicts (not SimpleNamespace trees).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from ._common import (
    _common_scalars,
    _denormalize_finish_reason,
    normalize_finish_reason,
)
from ._guardrails import (
    _CROSS_PROVIDER_MULTIMODAL_STREAM,
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


# ----------------------------- Bedrock -> canonical ------------------------- #
def _bedrock_to_canonical(kwargs: dict[str, Any]) -> CanonicalRequest:
    _check_forbidden_keys("bedrock", kwargs)
    system = _bedrock_system(kwargs.get("system"))
    inference_config = kwargs.get("inferenceConfig") or {}
    max_tokens = inference_config.get("maxTokens")
    if max_tokens is None:
        # inferenceConfig.maxTokens is optional on Bedrock (model default),
        # but the canonical subset requires an explicit bound — the SDK will
        # not invent one for a cross-provider hop.
        _raise("bedrock", "*", "missing_max_tokens")

    messages: list[CanonicalMessage] = []
    pending: set[str] = set()
    for msg in kwargs.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            cm, p = _bedrock_assistant_to_canonical(content)
            pending |= p
            messages.append(cm)
        elif role == "user":
            messages.append(_bedrock_user_to_canonical(content, pending))
        else:
            _raise("bedrock", "*", "unknown_message_role")

    if pending:
        _raise("bedrock", "*", "dangling_tool_call")

    tool_config = kwargs.get("toolConfig") or {}
    return CanonicalRequest(
        system=system,
        messages=messages,
        max_tokens=int(max_tokens),
        tools=_bedrock_tools_to_canonical(tool_config.get("tools")),
        tool_choice=_bedrock_tool_choice_to_canonical(tool_config.get("toolChoice")),
        **_common_scalars(
            "bedrock",
            {
                "temperature": inference_config.get("temperature"),
                "top_p": inference_config.get("topP"),
                "stop": inference_config.get("stopSequences"),
                "stream": kwargs.get("stream", False),
            },
        ),
    )


def _bedrock_system(system: Any) -> str | None:
    """Join the SystemContentBlock list's text blocks into one canonical string.

    cachePoint / guardContent system blocks have no cross-provider equivalent
    and RAISE structurally.
    """
    if system is None:
        return None
    chunks: list[str] = []
    for block in system:
        _check_bedrock_block_modifiers(block)
        if "text" in block:
            chunks.append(block.get("text") or "")
        else:
            _raise("bedrock", "*", _bedrock_content_label(block))
    return "\n".join(chunks) if chunks else None


def _check_bedrock_block_modifiers(block: Mapping[str, Any]) -> None:
    """RAISE on block kinds that have no cross-provider meaning anywhere."""
    if "cachePoint" in block:
        # Bedrock's prompt-cache checkpoint == Anthropic's cache_control;
        # share the label so dashboards see one structural feature.
        _raise("bedrock", "*", "cache_control")
    if "guardContent" in block:
        _raise("bedrock", "*", "bedrock.guard_content")
    if "reasoningContent" in block:
        _raise("bedrock", "*", "reasoning")


def _bedrock_content_label(block: Mapping[str, Any]) -> str:
    """Fixed structural label for an unsupported Bedrock content block.

    Bedrock blocks are single-key unions, so the discriminator is the lone
    key. Only a RECOGNIZED multimodal key maps to a ``multimodal.*`` label;
    anything else falls back to the constant — the key is never interpolated
    beyond the fixed map (it could be attacker-controlled).
    """
    for key in block:
        label = _content_part_label(key)
        if label != "content_part.unknown":
            return label
    return "content_part.unknown"


def _bedrock_image_to_canonical(image: Mapping[str, Any]) -> ImagePart:
    source = image.get("source") or {}
    if "bytes" in source:
        encoded = base64.b64encode(source["bytes"]).decode("ascii")
        return ImagePart(media_type=f"image/{image.get('format') or ''}", data=encoded)
    # s3Location (and any future source member) is a provider-bound handle.
    _raise("bedrock", "*", "image.opaque_handle")


def _bedrock_assistant_to_canonical(content: Any) -> tuple[CanonicalMessage, set[str]]:
    parts: list[ContentPart] = []
    pending: set[str] = set()
    tool_use_names: list[str] = []
    for block in content or []:
        _check_bedrock_block_modifiers(block)
        if "text" in block:
            parts.append(TextPart(text=block.get("text") or ""))
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_use_names.append(tool_use.get("name", ""))
            pending.add(tool_use["toolUseId"])
            parts.append(
                ToolUsePart(
                    id=tool_use["toolUseId"],
                    name=tool_use.get("name", ""),
                    input=dict(tool_use.get("input") or {}),
                )
            )
        elif "image" in block:
            parts.append(_bedrock_image_to_canonical(block["image"]))
        else:
            _raise("bedrock", "*", _bedrock_content_label(block))
    if len(tool_use_names) != len(set(tool_use_names)) and len(tool_use_names) > 1:
        _raise("bedrock", "*", "parallel_same_name_tool_calls")
    return CanonicalMessage(role="assistant", content=parts), pending


def _bedrock_user_to_canonical(content: Any, pending: set[str]) -> CanonicalMessage:
    parts: list[ContentPart] = []
    for block in content or []:
        _check_bedrock_block_modifiers(block)
        if "toolResult" in block:
            tool_result = block["toolResult"]
            tuid = tool_result.get("toolUseId", "")
            if tuid not in pending:
                _raise("bedrock", "*", "orphan_tool_result")
            pending.discard(tuid)
            parts.append(
                ToolResultPart(
                    tool_use_id=tuid,
                    content=_bedrock_tool_result_text(tool_result.get("content")),
                )
            )
        elif "text" in block:
            parts.append(TextPart(text=block.get("text") or ""))
        elif "image" in block:
            parts.append(_bedrock_image_to_canonical(block["image"]))
        else:
            _raise("bedrock", "*", _bedrock_content_label(block))
    return CanonicalMessage(role="user", content=parts)


def _bedrock_tool_result_text(content: Any) -> str:
    """Flatten a toolResult content block list ({text} | {json}) to a string."""
    if content is None:
        return ""
    chunks: list[str] = []
    for block in content:
        if isinstance(block, Mapping):
            if isinstance(block.get("text"), str):
                chunks.append(block["text"])
            elif "json" in block:
                chunks.append(json.dumps(block["json"]))
    return "".join(chunks)


def _bedrock_tools_to_canonical(tools: Any) -> list[CanonicalTool] | None:
    if not tools:
        return None
    out: list[CanonicalTool] = []
    for tool in tools:
        if "cachePoint" in tool:
            _raise("bedrock", "*", "cache_control")
        tool_spec = tool.get("toolSpec")
        if tool_spec is None:
            _raise("bedrock", "*", "bedrock.unsupported_tool")
        input_schema = tool_spec.get("inputSchema") or {}
        out.append(
            CanonicalTool(
                name=tool_spec.get("name", ""),
                description=tool_spec.get("description"),
                parameters=dict(input_schema.get("json") or {}),
            )
        )
    return out


def _bedrock_tool_choice_to_canonical(choice: Any) -> ToolChoice | None:
    if choice is None:
        return None
    if "auto" in choice:
        return ToolChoice(mode="auto")
    if "any" in choice:
        return ToolChoice(mode="required")
    if "tool" in choice:
        return ToolChoice(mode="force", name=(choice.get("tool") or {}).get("name"))
    _raise("bedrock", "*", "tool_choice.unknown")


# ----------------------------- canonical -> Bedrock ------------------------- #
def _canonical_to_bedrock(canonical: CanonicalRequest, model: str) -> dict[str, Any]:
    if canonical.parallel_tool_calls is False:
        # Bedrock has no disable-parallel-tool-use equivalent.
        _raise("*", "bedrock", "parallel_tool_calls")

    # Drop turns that render to an empty content list — Converse rejects
    # content=[] (mirrors the Anthropic renderer).
    messages = [
        rendered
        for m in canonical.messages
        if (rendered := _canonical_msg_to_bedrock(m))["content"]
    ]
    inference_config: dict[str, Any] = {"maxTokens": canonical.max_tokens}
    if canonical.temperature is not None:
        inference_config["temperature"] = canonical.temperature
    if canonical.top_p is not None:
        inference_config["topP"] = canonical.top_p
    if canonical.stop is not None:
        inference_config["stopSequences"] = canonical.stop

    out: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "inferenceConfig": inference_config,
    }
    if canonical.system is not None:
        out["system"] = [{"text": canonical.system}]
    if canonical.stream:
        # Dispatch strips this and selects converse vs converse_stream.
        out["stream"] = True
    if canonical.tools is not None:
        tool_config: dict[str, Any] = {
            "tools": [_canonical_tool_to_bedrock(t) for t in canonical.tools]
        }
        if canonical.tool_choice is not None:
            tool_config["toolChoice"] = _canonical_tool_choice_to_bedrock(canonical.tool_choice)
        out["toolConfig"] = tool_config
    return out


def _canonical_tool_to_bedrock(tool: CanonicalTool) -> dict[str, Any]:
    tool_spec: dict[str, Any] = {
        "name": tool.name,
        "inputSchema": {"json": tool.parameters},
    }
    if tool.description is not None:
        tool_spec["description"] = tool.description
    return {"toolSpec": tool_spec}


def _canonical_tool_choice_to_bedrock(choice: ToolChoice) -> dict[str, Any]:
    if choice.mode == "auto":
        return {"auto": {}}
    if choice.mode == "required":
        return {"any": {}}
    if choice.mode == "force":
        return {"tool": {"name": choice.name}}
    # Bedrock ToolChoice is auto | any | tool — there is no "none".
    _raise("*", "bedrock", "tool_choice.none")


def _canonical_msg_to_bedrock(msg: CanonicalMessage) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for part in msg.content:
        if isinstance(part, TextPart):
            blocks.append({"text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(_image_to_bedrock_block(part))
        elif isinstance(part, ToolUsePart):
            blocks.append(
                {"toolUse": {"toolUseId": part.id, "name": part.name, "input": part.input}}
            )
        elif isinstance(part, ToolResultPart):
            blocks.append(
                {
                    "toolResult": {
                        "toolUseId": part.tool_use_id,
                        "content": [{"text": part.content}],
                    }
                }
            )
    return {"role": msg.role, "content": blocks}


def _image_to_bedrock_block(part: ImagePart) -> dict[str, Any]:
    if part.url is not None:
        # Bedrock ImageSource is raw bytes or s3Location — no public-URL member.
        _raise("*", "bedrock", "image.url_unsupported")
    media_type = part.media_type or ""
    image_format = media_type.split("/", 1)[1] if "/" in media_type else media_type
    return {
        "image": {
            "format": image_format,
            "source": {"bytes": base64.b64decode(part.data or "")},
        }
    }


# ----------------------------- response shaping ----------------------------- #
def _bedrock_response_to_canonical(response: object) -> CanonicalResponse:
    output = response.get("output") if isinstance(response, Mapping) else None
    message = output.get("message") if isinstance(output, Mapping) else None
    blocks = message.get("content") if isinstance(message, Mapping) else None

    text_chunks: list[str] = []
    tool_calls: list[CanonicalToolCall] = []
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        if isinstance(block.get("text"), str):
            text_chunks.append(block["text"])
        elif isinstance(block.get("toolUse"), Mapping):
            tool_use = block["toolUse"]
            tool_calls.append(
                CanonicalToolCall(
                    id=tool_use.get("toolUseId", ""),
                    name=tool_use.get("name", ""),
                    arguments=dict(tool_use.get("input") or {}),
                )
            )
    stop_reason = response.get("stopReason") if isinstance(response, Mapping) else None
    return CanonicalResponse(
        text="".join(text_chunks) or None,
        tool_calls=tool_calls or None,
        finish_reason=normalize_finish_reason("bedrock", stop_reason),
        # Converse responses do not echo a model id.
        model=None,
    )


def _canonical_response_to_bedrock(canonical: CanonicalResponse) -> object:
    content: list[dict[str, Any]] = []
    if canonical.text:
        content.append({"text": canonical.text})
    for tc in canonical.tool_calls or []:
        content.append({"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.arguments}})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": _denormalize_finish_reason("bedrock", canonical.finish_reason),
    }


# ----------------------------- served: Bedrock ------------------------------ #
def _bedrock_stream_chunk_to_deltas(
    served: str, requested: str, chunk: object
) -> list[_StreamDelta]:
    if not isinstance(chunk, Mapping):
        return []

    delta_event = chunk.get("contentBlockDelta")
    if isinstance(delta_event, Mapping):
        delta = delta_event.get("delta") or {}
        if "text" in delta:
            return [_StreamDelta(text=delta.get("text") or "")]
        if "toolUse" in delta:
            _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
        # reasoningContent / citation / image deltas are out of the v1 subset.
        _raise(served, requested, _CROSS_PROVIDER_MULTIMODAL_STREAM)

    stop_event = chunk.get("messageStop")
    if isinstance(stop_event, Mapping):
        reason = normalize_finish_reason("bedrock", stop_event.get("stopReason"))
        if reason is None:
            return []
        return [_StreamDelta(finish_reason=reason)]

    start_event = chunk.get("contentBlockStart")
    if isinstance(start_event, Mapping):
        # The start union is toolUse / toolResult / image — all out of the v1
        # text subset; fail loud at the START rather than mid-stream. Plain
        # text blocks emit no contentBlockStart at all.
        if start_event.get("start"):
            _raise(served, requested, _CROSS_PROVIDER_TOOL_STREAM)
        return []

    # messageStart / contentBlockStop / metadata -> no caller chunk.
    return []


def _stream_delta_to_bedrock(delta: _StreamDelta) -> object:
    if delta.finish_reason is not None:
        return {
            "messageStop": {
                "stopReason": _denormalize_finish_reason("bedrock", delta.finish_reason)
            }
        }
    return {"contentBlockDelta": {"delta": {"text": delta.text}, "contentBlockIndex": 0}}
