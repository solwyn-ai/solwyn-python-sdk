"""Google request/response/stream dialect for cross-provider translation.

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
    normalize_finish_reason,
)
from ._guardrails import (
    _CROSS_PROVIDER_MULTIMODAL_STREAM,
    _CROSS_PROVIDER_TOOL_STREAM,
    _GOOGLE_BUILTIN_TOOL_KEYS,
    _MULTIMODAL_UNKNOWN,
    _check_forbidden_keys,
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
            # gs:// / Files-API handles are provider-specific -> RAISE.
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


# ------------------------------ canonical -> Google ------------------------- #
def _canonical_to_google(canonical: CanonicalRequest, model: str) -> dict[str, Any]:
    # Google has no first-class equivalent of disabling parallel tool calls.
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
    # Google encodes a tool call as STOP + function_call (no dedicated tool
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


def _stream_delta_to_google(delta: _StreamDelta) -> object:
    finish = _denormalize_finish_reason("google", delta.finish_reason)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=delta.text)]),
        finish_reason=finish,
    )
    return SimpleNamespace(candidates=[candidate])
