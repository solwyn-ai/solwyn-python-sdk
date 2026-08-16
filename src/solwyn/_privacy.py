"""Private, privacy-sensitive helpers — PRIVACY-CRITICAL.

PRIVACY-CRITICAL
================
This module is the only place in the SDK that touches customer prompt
content directly. Code here must obey three rules:

  1. NEVER pass prompt content to a logger (`logger.*`) — not even in
     a formatted string, not even at DEBUG level. CI enforces this
     with `tests/unit/test_privacy_firewall.py`.
  2. NEVER store prompt content on a long-lived object — compute and
     discard within the current function call.
  3. NEVER include prompt content in exception arguments. If a
     computation fails, log `type(exc).__name__` only.

If you add a new helper here, add a corresponding enforcement test.
"""

from __future__ import annotations

from typing import Any, NoReturn

from solwyn._types import MediaUsage
from solwyn.exceptions import UntranslatableRequestError

_LEGACY_GOOGLE_GENERATION_CONFIG_KEYS = frozenset(
    {
        "candidate_count",
        "stop_sequences",
        "max_output_tokens",
        "temperature",
        "top_p",
    }
)
_GOOGLE_GENERATE_CONTENT_KEYS = frozenset(
    {
        "model",
        "contents",
        "stream",
        "config",
        "generation_config",
        "max_output_tokens",
        "safety_settings",
        "tools",
        "tool_config",
        "system_instruction",
        "request_options",
    }
)
_GOOGLE_SCHEMA_TYPES = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
}


def _raise_google_shape(feature: str) -> NoReturn:
    """Raise a structural Google-shape error with no retained input context."""
    try:
        raise UntranslatableRequestError(
            source="google",
            target="google",
            feature=feature,
        ) from None
    except UntranslatableRequestError as clean:
        clean.__context__ = None
        raise


def _google_config_mapping(value: object, *, feature: str) -> dict[str, Any]:
    """Return an ephemeral mapping view of a Google request config."""
    try:
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
            return {key: item for key, item in attrs.items() if not key.startswith("_")}
    except Exception:
        _raise_google_shape(feature)
    _raise_google_shape(feature)


def _normalize_google_tools(tools: object, *, target_shape: str) -> object:
    """Translate the documented function-tool subset between Google SDKs.

    Tool descriptions and schemas may contain customer-authored material, so the
    reshape stays inside this content-privileged module and is purely ephemeral.
    Built-in/provider-specific tool shapes fail closed instead of being guessed.
    """
    if tools is None:
        return None
    if not isinstance(tools, (list, tuple)):
        _raise_google_shape("unsupported_google_tools_shape")
    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        tool_mapping = _google_config_mapping(tool, feature="unsupported_google_tool")
        if set(tool_mapping) != {"function_declarations"}:
            _raise_google_shape("unsupported_google_tool")
        declarations = tool_mapping["function_declarations"]
        if not isinstance(declarations, (list, tuple)):
            _raise_google_shape("unsupported_google_function_declarations")
        normalized_declarations: list[dict[str, Any]] = []
        for declaration in declarations:
            mapped = _google_config_mapping(
                declaration,
                feature="unsupported_google_function_declaration",
            )
            allowed = {
                "name",
                "description",
                "parameters",
                "parameters_json_schema",
            }
            if not set(mapped).issubset(allowed):
                _raise_google_shape("unsupported_google_function_declaration")
            if target_shape == "google_generativeai":
                if "parameters" not in mapped and "parameters_json_schema" in mapped:
                    mapped["parameters"] = mapped["parameters_json_schema"]
                mapped.pop("parameters_json_schema", None)
                # Legacy requires the keyword even though modern permits it to
                # be absent; an empty description preserves that absence.
                mapped.setdefault("description", "")
            else:
                if "parameters_json_schema" not in mapped and "parameters" in mapped:
                    mapped["parameters_json_schema"] = mapped["parameters"]
                mapped.pop("parameters", None)
            normalized_declarations.append(mapped)
        normalized_tools.append({"function_declarations": normalized_declarations})
    return normalized_tools


def _normalize_google_tool_config(value: object) -> dict[str, Any]:
    """Copy the common function-calling config subset defensively."""
    mapped = _google_config_mapping(
        value,
        feature="unsupported_google_tool_config_shape",
    )
    if set(mapped) != {"function_calling_config"}:
        _raise_google_shape("unsupported_google_tool_config_shape")
    calling = _google_config_mapping(
        mapped["function_calling_config"],
        feature="unsupported_google_tool_config_shape",
    )
    allowed = {"mode", "allowed_function_names"}
    if not set(calling).issubset(allowed):
        _raise_google_shape("unsupported_google_tool_config_shape")
    return {"function_calling_config": calling}


def _copy_legacy_google_tool_config(value: object) -> object:
    """Copy dict layers the legacy SDK mutates while preserving native objects."""
    if not isinstance(value, dict):
        return value
    copied = dict(value)
    calling = copied.get("function_calling_config")
    if isinstance(calling, dict):
        copied["function_calling_config"] = dict(calling)
    return copied


def _legacy_google_schema_dict_to_json(value: object) -> dict[str, Any]:
    """Clean a proto ``Schema.to_dict`` result into the translation subset."""
    schema = _google_config_mapping(value, feature="unsupported_google_constructor_tools")
    type_name = schema.pop("type_", None)
    if not isinstance(type_name, str) or type_name not in _GOOGLE_SCHEMA_TYPES:
        _raise_google_shape("unsupported_google_constructor_tools")
    converted: dict[str, Any] = {"type": _GOOGLE_SCHEMA_TYPES[type_name]}
    for key in ("description", "enum", "required"):
        field = schema.pop(key, None)
        if field:
            converted[key] = field
    properties = schema.pop("properties", None)
    if properties:
        if not isinstance(properties, dict):
            _raise_google_shape("unsupported_google_constructor_tools")
        converted["properties"] = {
            key: _legacy_google_schema_dict_to_json(item) for key, item in properties.items()
        }
    items = schema.pop("items", None)
    if items:
        converted["items"] = _legacy_google_schema_dict_to_json(items)
    if any(field not in (None, "", False, 0, "0", [], {}) for field in schema.values()):
        _raise_google_shape("unsupported_google_constructor_tools")
    return converted


def _legacy_google_request_tools(value: object) -> list[dict[str, Any]]:
    """Clean request-proto function tools into modern Google dictionaries."""
    if not isinstance(value, list):
        _raise_google_shape("unsupported_google_constructor_tools")
    tools: list[dict[str, Any]] = []
    for raw_tool in value:
        tool = _google_config_mapping(
            raw_tool,
            feature="unsupported_google_constructor_tools",
        )
        if set(tool) != {"function_declarations"} or not isinstance(
            tool["function_declarations"], list
        ):
            _raise_google_shape("unsupported_google_constructor_tools")
        declarations: list[dict[str, Any]] = []
        for raw_declaration in tool["function_declarations"]:
            declaration = _google_config_mapping(
                raw_declaration,
                feature="unsupported_google_constructor_tools",
            )
            if not set(declaration).issubset({"name", "description", "parameters"}):
                _raise_google_shape("unsupported_google_constructor_tools")
            name = declaration.get("name")
            description = declaration.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                _raise_google_shape("unsupported_google_constructor_tools")
            declarations.append(
                {
                    "name": name,
                    "description": description,
                    "parameters_json_schema": (
                        _legacy_google_schema_dict_to_json(declaration["parameters"])
                        if declaration.get("parameters") is not None
                        else {}
                    ),
                }
            )
        tools.append({"function_declarations": declarations})
    return tools


def _legacy_google_request_system(value: object) -> str:
    """Extract the supported single-text-part system instruction."""
    system = _google_config_mapping(
        value,
        feature="unsupported_google_constructor_system_instruction",
    )
    parts = system.get("parts")
    if set(system) - {"parts", "role"} or not isinstance(parts, list) or len(parts) != 1:
        _raise_google_shape("unsupported_google_constructor_system_instruction")
    part = _google_config_mapping(
        parts[0],
        feature="unsupported_google_constructor_system_instruction",
    )
    if set(part) != {"text"} or not isinstance(part["text"], str):
        _raise_google_shape("unsupported_google_constructor_system_instruction")
    return part["text"]


def _legacy_google_request_payload(
    client: object,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build and copy the legacy SDK's effective no-I/O request payload."""
    prepare_request = getattr(client, "_prepare_request", None)
    if not callable(prepare_request) or "contents" not in kwargs:
        _raise_google_shape("unsupported_google_constructor_defaults")
    request = prepare_request(
        contents=kwargs["contents"],
        generation_config=kwargs.get("generation_config"),
        safety_settings=kwargs.get("safety_settings"),
        tools=kwargs.get("tools"),
        tool_config=kwargs.get("tool_config"),
    )
    to_dict = getattr(type(request), "to_dict", None)
    if not callable(to_dict):
        _raise_google_shape("unsupported_google_constructor_defaults")
    return _google_config_mapping(
        to_dict(
            request,
            use_integers_for_enums=False,
            preserving_proto_field_name=True,
        ),
        feature="unsupported_google_constructor_defaults",
    )


def _prepare_legacy_google_translation_source_kwargs(
    client: object, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Use legacy ``_prepare_request`` as the constructor/call merge authority."""
    normalized = normalize_google_generate_content_kwargs(
        kwargs,
        target_shape="google_generativeai",
    )
    payload = _legacy_google_request_payload(client, normalized)
    if payload.get("cached_content"):
        _raise_google_shape("unsupported_google_constructor_cached_content")

    generation = _google_config_mapping(
        payload.get("generation_config"),
        feature="unsupported_google_constructor_generation_config",
    )
    config = {
        key: value
        for key, value in generation.items()
        if key in _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
    }
    if any(
        value not in (None, "", False, 0, "0", [], {})
        for key, value in generation.items()
        if key not in _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
    ):
        _raise_google_shape("unsupported_google_constructor_generation_config")
    if payload.get("system_instruction"):
        config["system_instruction"] = _legacy_google_request_system(payload["system_instruction"])
    if payload.get("safety_settings"):
        config["safety_settings"] = payload["safety_settings"]
    if payload.get("tools"):
        config["tools"] = _legacy_google_request_tools(payload["tools"])
    if payload.get("tool_config"):
        config["tool_config"] = payload["tool_config"]

    contents = payload.get("contents")
    if not isinstance(contents, list) or not contents:
        _raise_google_shape("unsupported_google_contents_shape")
    final_turn = contents[-1]
    if isinstance(final_turn, dict) and not final_turn.get("role"):
        final_turn = {**final_turn, "role": "user"}
        contents = [*contents[:-1], final_turn]
    result: dict[str, Any] = {"contents": contents, "config": config}
    if "model" in normalized:
        result["model"] = normalized["model"]
    if normalized.get("stream"):
        result["stream"] = True
    return result


def prepare_legacy_google_translation_source_kwargs(
    client: object, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Prepare one ephemeral legacy request for fallback translation, without I/O."""
    try:
        return _prepare_legacy_google_translation_source_kwargs(client, kwargs)
    except UntranslatableRequestError as structural:
        structural.__context__ = None
        raise
    except Exception:
        _raise_google_shape("unsupported_google_constructor_defaults")


def _merge_legacy_google_metering_layers(
    layers: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Merge legacy request layers without rejecting provider-native options."""
    merged: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    for layer in layers:
        copied = dict(layer)
        modern_config = _google_config_mapping(
            copied.get("config"),
            feature="unsupported_google_estimation_shape",
        )
        native_generation = _google_config_mapping(
            copied.get("generation_config"),
            feature="unsupported_google_estimation_shape",
        )
        generation.update(
            {
                key: value
                for key, value in modern_config.items()
                if key in _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
            }
        )
        generation.update(native_generation)
        if "max_output_tokens" in copied:
            generation["max_output_tokens"] = copied["max_output_tokens"]
        for key in ("model", "contents", "safety_settings", "tools", "tool_config"):
            if key in copied:
                value = copied[key]
            elif key in modern_config:
                value = modern_config[key]
            else:
                continue
            merged[key] = _copy_legacy_google_tool_config(value) if key == "tool_config" else value
    if generation:
        merged["generation_config"] = generation
    return merged


def prepare_legacy_google_metering_kwargs(
    client: object,
    *layers: dict[str, Any],
) -> dict[str, Any]:
    """Project an effective legacy request onto content and output-cap fields.

    Unlike failover translation, metering must accept every request shape the
    legacy SDK itself accepts. Unrelated native options remain inside the SDK's
    no-I/O ``_prepare_request`` merge and are deliberately ignored here.
    """
    try:
        merged = _merge_legacy_google_metering_layers(layers)
        payload = _legacy_google_request_payload(client, merged)
        generation = _google_config_mapping(
            payload.get("generation_config"),
            feature="unsupported_google_estimation_shape",
        )
        config: dict[str, Any] = {}
        if "max_output_tokens" in generation:
            config["max_output_tokens"] = generation["max_output_tokens"]
        if payload.get("system_instruction") is not None:
            config["system_instruction"] = payload["system_instruction"]
        projected: dict[str, Any] = {
            "contents": payload.get("contents"),
            "config": config,
        }
        if "model" in merged:
            projected["model"] = merged["model"]
        return projected
    except UntranslatableRequestError as structural:
        structural.__context__ = None
        raise
    except Exception:
        _raise_google_shape("unsupported_google_estimation_shape")


def _normalize_google_generate_content_kwargs(
    kwargs: dict[str, Any], *, target_shape: str
) -> dict[str, Any]:
    """Normalize one ephemeral Google request layer for the served SDK shape.

    ``google-genai`` accepts ``model``, ``contents``, and a single ``config``;
    deprecated ``google.generativeai`` accepts ``contents`` plus
    ``generation_config`` and top-level tool/safety options. This helper maps the
    documented common generation subset without importing either SDK. It is
    called independently for global defaults, entry defaults, and per-call
    values so the ordinary merge order keeps caller precedence across aliases.

    Customer contents and system/tool material are never logged or retained;
    they move opaquely through defensive, short-lived mappings only.
    """
    if target_shape not in {"google_genai", "google_generativeai"}:
        raise RuntimeError(f"unsupported Google client shape: {target_shape}")

    normalized = dict(kwargs)
    for key in normalized:
        if key not in _GOOGLE_GENERATE_CONTENT_KEYS:
            _raise_google_shape("unsupported_google_kwargs")
    if target_shape == "google_generativeai":
        if "system_instruction" in normalized:
            _raise_google_shape("google.system_instruction_per_call")

        modern_config_value = normalized.pop("config", None)
        modern_config = _google_config_mapping(
            modern_config_value,
            feature="unsupported_google_config_shape",
        )
        unsupported_config = set(modern_config) - (
            _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
            | {"system_instruction", "safety_settings", "tools", "tool_config"}
        )
        if unsupported_config:
            _raise_google_shape("unsupported_google_config")
        if "system_instruction" in modern_config:
            # Legacy GenerativeModel stores this at construction and exposes no
            # per-call parameter. Mutating the shared model would be racy.
            _raise_google_shape("google.system_instruction_per_call")

        mapped_generation = {
            key: value
            for key, value in modern_config.items()
            if key in _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
        }
        if "max_output_tokens" in normalized:
            mapped_generation["max_output_tokens"] = normalized.pop("max_output_tokens")
        native_generation_value = normalized.get("generation_config")
        if native_generation_value is not None and mapped_generation:
            native_generation = _google_config_mapping(
                native_generation_value,
                feature="unsupported_google_generation_config_shape",
            )
            unknown = set(native_generation) - _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
            if unknown:
                _raise_google_shape("unsupported_google_generation_config")
            mapped_generation.update(native_generation)
            normalized["generation_config"] = mapped_generation
        elif native_generation_value is None and mapped_generation:
            normalized["generation_config"] = mapped_generation

        if "safety_settings" not in normalized and "safety_settings" in modern_config:
            normalized["safety_settings"] = modern_config["safety_settings"]
        if "tools" not in normalized and "tools" in modern_config:
            normalized["tools"] = _normalize_google_tools(
                modern_config["tools"], target_shape=target_shape
            )
        if "tool_config" not in normalized and modern_config.get("tool_config") is not None:
            normalized["tool_config"] = _normalize_google_tool_config(modern_config["tool_config"])
        # Already-top-level legacy values remain native passthrough. Only values
        # extracted from modern ``config`` above need cross-shape conversion.
        # Dict tool configs still get a defensive nested copy because the legacy
        # SDK's converter mutates them in place during provider dispatch.
        if "tool_config" in normalized:
            normalized["tool_config"] = _copy_legacy_google_tool_config(normalized["tool_config"])
        return normalized

    request_options = normalized.pop("request_options", None)
    if request_options:
        _raise_google_shape("google.request_options")

    legacy_generation_value = normalized.pop("generation_config", None)
    legacy_generation = _google_config_mapping(
        legacy_generation_value,
        feature="unsupported_google_generation_config_shape",
    )
    unknown_generation = set(legacy_generation) - _LEGACY_GOOGLE_GENERATION_CONFIG_KEYS
    if unknown_generation:
        _raise_google_shape("unsupported_google_generation_config")
    if "max_output_tokens" in normalized:
        legacy_generation["max_output_tokens"] = normalized.pop("max_output_tokens")

    native_config_value = normalized.pop("config", None)
    native_config = _google_config_mapping(
        native_config_value,
        feature="unsupported_google_config_shape",
    )
    mapped_config = dict(legacy_generation)
    for key in ("safety_settings", "tools", "tool_config", "system_instruction"):
        if key not in normalized:
            continue
        value = normalized.pop(key)
        if key == "tools":
            value = _normalize_google_tools(value, target_shape=target_shape)
        elif key == "tool_config" and value is not None:
            value = _normalize_google_tool_config(value)
        mapped_config[key] = value
    mapped_config.update(native_config)
    if mapped_config or native_config_value is not None or legacy_generation_value is not None:
        normalized["config"] = mapped_config
    return normalized


def normalize_google_generate_content_kwargs(
    kwargs: dict[str, Any], *, target_shape: str
) -> dict[str, Any]:
    """Normalize one request layer and suppress content-bearing conversion errors."""
    try:
        return _normalize_google_generate_content_kwargs(kwargs, target_shape=target_shape)
    except UntranslatableRequestError as structural:
        structural.__context__ = None
        raise
    except Exception:
        _raise_google_shape("malformed_google_request")


def merge_google_generate_content_kwargs(
    *layers: dict[str, Any],
    target_shape: str,
) -> dict[str, Any]:
    """Merge precedence-ordered Google layers with field-level config precedence."""
    config_key = "generation_config" if target_shape == "google_generativeai" else "config"
    merged: dict[str, Any] = {}
    merged_config: dict[str, Any] = {}
    saw_config = False
    for layer in layers:
        normalized = normalize_google_generate_content_kwargs(layer, target_shape=target_shape)
        config = normalized.pop(config_key, None)
        if config is not None:
            saw_config = True
            merged_config.update(_google_config_mapping(config, feature="malformed_google_request"))
        merged.update(normalized)
    if saw_config:
        merged[config_key] = merged_config
    return merged


def normalize_google_translation_source_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize the documented Google source subset for canonical translation.

    The canonical Google translator consumes modern structured turns. Legacy's
    ordinary string contents are represented as one user text turn. Already
    structured modern turns pass via defensive container copies; every other
    legacy content type fails with one constant structural label before provider
    I/O, rather than leaking a value through a conversion exception.
    """
    normalized = normalize_google_generate_content_kwargs(
        kwargs,
        target_shape="google_genai",
    )
    try:
        contents = normalized.get("contents")
        if isinstance(contents, str):
            normalized["contents"] = [{"role": "user", "parts": [{"text": contents}]}]
            return normalized
        if not isinstance(contents, list):
            _raise_google_shape("unsupported_google_contents_shape")
        turns: list[dict[str, Any]] = []
        for turn in contents:
            if not isinstance(turn, dict):
                _raise_google_shape("unsupported_google_contents_shape")
            parts = turn.get("parts")
            if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
                _raise_google_shape("unsupported_google_contents_shape")
            turns.append({**turn, "parts": list(parts)})
        normalized["contents"] = turns
        return normalized
    except UntranslatableRequestError as structural:
        structural.__context__ = None
        raise
    except Exception:
        _raise_google_shape("unsupported_google_contents_shape")


def normalize_legacy_google_generate_content_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Normalize legacy ``GenerativeModel.generate_content`` call arguments.

    The deprecated ``google.generativeai`` client accepts exactly one positional
    argument: customer ``contents``. Keep that content-bearing reshape inside
    the privacy firewall, return only a short-lived defensive copy, and retain
    normal Python duplicate-argument behavior.
    """
    if len(args) > 1:
        raise TypeError(
            f"generate_content() takes 1 positional argument but {len(args)} were given"
        )
    normalized = dict(kwargs)
    if not args:
        return normalized
    if "contents" in normalized:
        raise TypeError("generate_content() got multiple values for argument 'contents'")
    normalized["contents"] = args[0]
    return normalized


def estimate_content_length(kwargs: dict[str, Any]) -> int:
    """Return the total character length of prompt content in kwargs.

    Walks messages/system/contents and sums string lengths WITHOUT
    concatenating them into a joined string. The returned integer is
    safe to log — it is not reversible to prompt content.

    Args:
        kwargs: The LLM call kwargs dict. Handles OpenAI/Anthropic/Bedrock
            messages (Bedrock Converse content blocks are ``{"text": ...}``
            dicts, covered by the block walk), the Anthropic string system
            prompt, the Bedrock system block LIST, and Google contents.

    Returns:
        Total character count (0 if no recognizable content keys).
    """
    total = 0

    messages = kwargs.get("messages", [])
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        total += len(text)

    system = kwargs.get("system")
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        # Bedrock Converse: system is a list of SystemContentBlock dicts.
        # Non-text blocks (cachePoint/guardContent) carry no countable text.
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str):
                    total += len(text)

    try:
        config_value = kwargs.get("config")
        if config_value is not None:
            config = _google_config_mapping(
                config_value,
                feature="unsupported_google_estimation_shape",
            )
            total += _google_prompt_text_length(config.get("system_instruction"))
        total += _google_prompt_text_length(kwargs.get("contents"))
    except UntranslatableRequestError as structural:
        structural.__context__ = None
        raise
    except Exception:
        _raise_google_shape("unsupported_google_estimation_shape")

    return total


def _google_prompt_text_length(value: object) -> int:
    """Count text in Google string/content/part shapes without joining it."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_google_prompt_text_length(item) for item in value)
    mapped = _google_config_mapping(
        value,
        feature="unsupported_google_estimation_shape",
    )
    text = mapped.get("text")
    total = len(text) if isinstance(text, str) else 0
    parts = mapped.get("parts")
    if parts is not None:
        if not isinstance(parts, (list, tuple)):
            _raise_google_shape("unsupported_google_estimation_shape")
        total += sum(_google_prompt_text_length(part) for part in parts)
    return total


def estimate_responses_content_length(kwargs: dict[str, Any]) -> int:
    """Return Responses request character length without retaining content.

    This length-only heuristic recognizer must not reject calls. It is
    deliberately separate from ``estimate_content_length`` so media calls do
    not double-count embeddings or text-to-speech ``input``.
    """
    total = 0

    instructions = kwargs.get("instructions")
    if isinstance(instructions, str):
        total += len(instructions)

    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        return total + len(input_value)
    if not isinstance(input_value, list):
        return total

    for item in input_value:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)

        output = item.get("output")
        if isinstance(output, str):
            total += len(output)
        elif isinstance(output, list):
            for part in output:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)

    return total


def estimate_tokens_from_length(char_count: int, provider: str) -> int:
    """Convert a character count to a token estimate using per-provider ratios.

    These are heuristic ratios that match tiktoken's observed behavior.
    They are NOT tiktoken-exact — the exact path is intentionally removed
    because it required materializing the joined prompt text.

    Args:
        char_count: Number of characters in the prompt content.
        provider: One of "openai", "anthropic", "google", "bedrock".

    Returns:
        Estimated token count.
    """
    ratio_by_provider = {
        "openai": 4.0,
        "anthropic": 3.8,
        "google": 4.0,
        # Bedrock hosts many model families with different tokenizers; 4.0 is
        # the deliberate multi-family default (pre-flight estimate only — the
        # settled cost always comes from provider-reported usage).
        "bedrock": 4.0,
    }
    ratio = ratio_by_provider.get(provider, 4.0)
    return max(1, int(char_count / ratio))


def _chars_to_embedding_tokens(char_count: int, provider: str) -> int:
    """Chars -> tokens via the provider ratio, with 0 chars staying 0.

    ``estimate_tokens_from_length`` floors at 1 token for any positive char
    count; that floor is wrong for a purely pre-tokenized batch (0 text chars),
    which would otherwise gain a spurious 1-token estimate. Guard the zero case.
    """
    if char_count <= 0:
        return 0
    return estimate_tokens_from_length(char_count, provider)


def _estimate_embedding_list_tokens(items: list[Any], provider: str) -> int:
    """Token estimate for a list-valued embeddings ``input=`` (lengths only).

    Element type disambiguates the three list shapes without materializing any
    text — nothing is concatenated, stored, or returned beyond the integer:

    - ``str`` items: a text batch; character lengths summed then ratio-converted.
    - ``int`` items: token ids of ONE pre-tokenized sequence — each counts as a
      single token (``len()`` of the list IS the token count; running
      chars/ratio over token ids would be wrong).
    - ``list`` items: a batch of pre-tokenized sequences — each inner list
      contributes its own length.
    """
    char_total = 0
    token_total = 0
    for item in items:
        if isinstance(item, str):
            char_total += len(item)
        elif isinstance(item, bool):
            # bool is an int subclass but is never a token id — ignore it.
            continue
        elif isinstance(item, int):
            # list[int]: one pre-tokenized sequence; each int is one token id.
            token_total += 1
        elif isinstance(item, list):
            # list[list[int]]: this inner sequence's length is its token count.
            token_total += len(item)
    return token_total + _chars_to_embedding_tokens(char_total, provider)


def estimate_embedding_input_tokens(kwargs: dict[str, Any], provider: str) -> int:
    """Estimate input tokens for an embeddings request's ``input=`` (lengths only).

    The request-side fallback used when an embeddings response reports no usable
    usage (``measure_request`` in the embeddings ``MediaSurfaceSpec``). Recognizes
    the four ``input=`` shapes the OpenAI embeddings API accepts, measuring
    LENGTHS only — the input text/ids are never retained, logged, or concatenated:

    - ``str``: one text; character count converted to tokens via the provider ratio.
    - ``list[str]``: a batch of texts; summed character count ratio-converted.
    - ``list[int]``: ONE pre-tokenized sequence; ``len()`` IS the token count.
    - ``list[list[int]]``: a batch of pre-tokenized sequences; summed inner lengths.

    Pre-tokenized ids map to tokens 1:1 — dividing their count by the char/token
    ratio would undercount. Returns 0 when ``input=`` is absent or an unrecognized
    shape, so the caller keeps the billable quantity None rather than settling a
    zero-as-default price.
    """
    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        return _chars_to_embedding_tokens(len(input_value), provider)
    if isinstance(input_value, list):
        return _estimate_embedding_list_tokens(input_value, provider)
    return 0


def _image_count(kwargs: dict[str, Any]) -> int:
    """Requested image count from ``n=``, defaulting to 1 (config value only).

    ``n`` is a COUNT, not prompt content. The OpenAI images API contract
    defaults ``n`` to 1 when omitted, so a missing ``n`` is a TRUE known
    quantity (1 image), never a zero-as-default. A non-int / bool / <1 value is
    garbage and also degrades to the documented default of 1.
    """
    n = kwargs.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return 1
    return n


def _image_selector(value: Any) -> str | None:
    """A request variant selector (``size`` / ``quality``) as a bounded string.

    Both are CONFIG enums (e.g. ``"1024x1024"``, ``"hd"``), never prompt
    content, matched server-side against the pricing card's variant grid. A
    non-str or over-length (the ``MediaUsage`` cap is 32) value degrades to None
    so a garbage selector neither raises out of the builder nor mismatches the
    grid — it simply carries no selector.
    """
    if not isinstance(value, str):
        return None
    if len(value) > 32:
        return None
    return value


def measure_image_media(kwargs: dict[str, Any]) -> MediaUsage:
    """Request-derived ``MediaUsage`` for an images.generate/edit call.

    CONFIG values only — ``n`` -> ``image_count``, ``size`` -> ``resolution``,
    ``quality`` -> ``quality``. The customer's ``prompt=`` (and the ``image=`` /
    ``mask=`` bytes on an edit) are NEVER read, logged, or retained; only these
    non-content request parameters are measured. ``is_estimated`` stays False:
    these are the EXACT request parameters that determine billing, not
    length-based approximations — so the same builder serves both the pre-flight
    ``estimated_media`` (a precise check) and the settled ``media_usage``.
    """
    return MediaUsage(
        image_count=_image_count(kwargs),
        resolution=_image_selector(kwargs.get("size")),
        quality=_image_selector(kwargs.get("quality")),
    )


def measure_speech_media(kwargs: dict[str, Any]) -> MediaUsage | None:
    """Request-derived ``MediaUsage`` for a TTS ``audio.speech.create`` call.

    The ONLY billable basis for text-to-speech is the request's ``input`` text
    LENGTH — the response is raw audio bytes with no usage metadata of any kind.
    Measures ``len(input)`` in the firewall: the input TEXT is never retained,
    logged, or concatenated, only its character count (which is not reversible to
    content). A non-str or absent ``input`` yields None so an unobservable
    quantity stays None rather than a zero-as-default. ``is_estimated`` stays
    False: a character count is EXACT, not a length-based approximation — so the
    same builder serves both the pre-flight ``estimated_media`` (a precise check)
    and the settled ``media_usage`` (chars/1e6 x rate priced server-side).
    """
    value = kwargs.get("input")
    if not isinstance(value, str):
        return None
    return MediaUsage(input_characters=len(value))


def _google_image_count(config: object) -> int:
    """Requested image count from a ``generate_images`` ``config``, defaulting to 1.

    google-genai carries ``number_of_images`` INSIDE the ``config=`` argument
    (unlike the openai images API's top-level ``n``): either a
    ``GenerateImagesConfig`` object (attribute access) or a plain dict — both
    handled duck-typed, no provider SDK import. ``number_of_images`` is a COUNT,
    not prompt content. The imagen API contract defaults it to 1 when omitted, so
    a missing value is a TRUE known quantity (1 image), never a zero-as-default.
    A non-int / bool / <1 value is garbage and also degrades to the documented
    default of 1.
    """
    if isinstance(config, dict):
        value = config.get("number_of_images")
    else:
        value = getattr(config, "number_of_images", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def measure_google_image_media(kwargs: dict[str, Any]) -> MediaUsage:
    """Request-derived ``MediaUsage`` for a Google ``generate_images`` call.

    CONFIG values only — ``config.number_of_images`` -> ``image_count``. The
    customer's ``prompt=`` is NEVER read, logged, or retained; only this
    non-content count is measured. imagen responses expose NO usage, so this
    request-derived count is the SOLE billable basis (imagen cards are flat
    per-image). ``is_estimated`` stays False: this is the EXACT request parameter
    that determines billing, not a length-based approximation — so the same
    builder serves both the pre-flight ``estimated_media`` and the settled
    ``media_usage``.
    """
    return MediaUsage(image_count=_google_image_count(kwargs.get("config")))


def _video_duration_seconds(config: object) -> float | None:
    """Requested video duration in seconds from a ``generate_videos`` ``config``.

    google-genai carries ``duration_seconds`` INSIDE the ``config=`` argument
    (a ``GenerateVideosConfig`` object or a plain dict — both handled duck-typed,
    no provider SDK import). It is a DURATION, not prompt content. The field has
    NO SDK-level default (the SDK sends nothing when it is omitted) and the
    provider's per-model default is not uniformly published, so an absent value
    stays None (the call is tracked UNPRICED) rather than settling a guessed
    duration — a per-second billed surface must never bill a duration the caller
    did not request. A non-numeric / bool / negative value is garbage and also
    degrades to None. ``bool`` is excluded explicitly (it is an ``int`` subclass).
    """
    if isinstance(config, dict):
        value = config.get("duration_seconds")
    else:
        value = getattr(config, "duration_seconds", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _video_resolution(config: object) -> str | None:
    """Video resolution variant selector (e.g. ``"720p"``) from a ``config``.

    A CONFIG enum, never prompt content, matched server-side against the pricing
    card's per-second variant grid. Read duck-typed from the dict or object
    ``config``; a non-str or over-length value degrades to None via the shared
    bounded-selector guard (an absent selector simply carries no variant).
    """
    if isinstance(config, dict):
        value = config.get("resolution")
    else:
        value = getattr(config, "resolution", None)
    return _image_selector(value)


def measure_video_media(kwargs: dict[str, Any]) -> MediaUsage:
    """Request-derived ``MediaUsage`` for a Google ``generate_videos`` call.

    CONFIG values only — ``config.duration_seconds`` -> ``video_seconds``,
    ``config.resolution`` -> ``resolution``. The customer's ``prompt=`` (and any
    seed ``image=`` bytes) are NEVER read, logged, or retained; only these
    non-content request parameters are measured. ``is_estimated`` is ALWAYS True:
    generate_videos returns immediately with a long-running operation that carries
    no usage, so billing settles at INITIATION from the request parameters — a
    deliberate, conservative over-count until completion-observation exists (the
    provider does not charge for failed/blocked generations). The same builder
    serves both the pre-flight ``estimated_media`` (a precise per-second check —
    an oversized request is denied before the provider is called) and the settled
    ``media_usage`` (the operation object carries nothing better). ``video_seconds``
    stays None when duration is absent, so the call is tracked unpriced rather
    than billed on a guessed duration.
    """
    config = kwargs.get("config")
    return MediaUsage(
        video_seconds=_video_duration_seconds(config),
        resolution=_video_resolution(config),
        is_estimated=True,
    )


# OpenAI's videos.create (Sora) documents stable defaults in its API reference:
# ``seconds`` defaults to "4" and ``size`` to "720x1280". Unlike google-genai's
# generate_videos (no published default → absent stays unpriced), these are
# published-and-stable, so an omitted param settles the documented default —
# billing the value the provider itself applies is faithful, not a guess.
_OPENAI_VIDEO_DEFAULT_SECONDS = 4.0
_OPENAI_VIDEO_DEFAULT_SIZE = "720x1280"


def _coerce_video_seconds(value: object) -> float | None:
    """Coerce a videos.create ``seconds`` value to a non-negative float, or None.

    OpenAI carries ``seconds`` as a DIGIT STRING ("4"/"8"/"12" per the API
    reference); a numeric int/float is accepted duck-typed. It is a DURATION, not
    prompt content. A non-digit string, a negative, ``bool`` (an ``int`` subclass,
    excluded explicitly), or any other shape is garbage → None (tracked unpriced),
    never a guessed duration. Never raises.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return float(value) if value.isdigit() else None
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def _openai_video_seconds(kwargs: dict[str, Any]) -> float | None:
    """Requested clip duration from a TOP-LEVEL ``seconds`` param (defaulting to 4).

    OpenAI's videos.create takes ``seconds`` at the TOP LEVEL of the request (not
    inside a ``config=`` object like google-genai). An absent (or ``None``) value
    settles the API's documented default of 4 seconds — billing the value the
    provider applies is faithful. A present-but-garbage value degrades to None
    (tracked unpriced) rather than a guessed duration; see ``_coerce_video_seconds``.
    """
    if kwargs.get("seconds") is None:
        return _OPENAI_VIDEO_DEFAULT_SECONDS
    return _coerce_video_seconds(kwargs.get("seconds"))


def _video_size_label(size: str) -> str | None:
    """Normalize a ``"WIDTHxHEIGHT"`` size string to a resolution LABEL, or None.

    The server's per-second video pricing variants are keyed by resolution LABELS
    ("720p", "1024p", "1080p"), so the request's pixel ``size`` is reduced to
    ``min(width, height) + "p"`` — orientation-independent: "1280x720" and
    "720x1280" both yield "720p", "1792x1024"/"1024x1792" → "1024p",
    "1920x1080"/"1080x1920" → "1080p". Parsing is case-insensitive on the ``x``
    separator. A string that is not exactly two digit parts is UNPARSEABLE and
    yields None so the caller passes the raw string through as the selector (the
    server fails loud on a selector miss rather than mispricing). Never raises.
    """
    parts = size.lower().split("x")
    if len(parts) != 2:
        return None
    width, height = parts
    if not width.isdigit() or not height.isdigit():
        return None
    return f"{min(int(width), int(height))}p"


def _normalize_video_size(value: object) -> str | None:
    """Resolution LABEL from a videos.create ``size`` selector, or None.

    A parseable ``"WIDTHxHEIGHT"`` size normalizes to its ``min(w, h) + "p"``
    label; an unparseable STRING passes through raw (the server fails loud on a
    selector miss rather than mispricing). Both paths run through the shared
    bounded-selector guard (``_image_selector``), so a non-str or over-length
    value degrades to None and never raises out of the builder.
    """
    if not isinstance(value, str):
        return None
    label = _video_size_label(value)
    return _image_selector(label if label is not None else value)


def _openai_video_size(kwargs: dict[str, Any]) -> str | None:
    """Resolution LABEL from a TOP-LEVEL ``size`` param (defaulting to 720x1280).

    OpenAI's videos.create takes ``size`` at the TOP LEVEL of the request. An
    absent (or ``None``) value settles the API's documented default of "720x1280"
    (→ "720p"); a present value normalizes via ``_normalize_video_size``.
    """
    value = kwargs.get("size")
    if value is None:
        value = _OPENAI_VIDEO_DEFAULT_SIZE
    return _normalize_video_size(value)


def measure_openai_video_media(kwargs: dict[str, Any]) -> MediaUsage:
    """Request-derived ``MediaUsage`` for an OpenAI ``videos.create`` (Sora) call.

    TOP-LEVEL request params only — ``seconds`` → ``video_seconds``, ``size``
    (normalized to a resolution label) → ``resolution``. The customer's
    ``prompt=`` (and any ``input_reference`` bytes) are NEVER read, logged, or
    retained; only these non-content request parameters are measured.
    ``is_estimated`` is ALWAYS True: videos.create returns immediately with an
    async video job that carries no usage, so billing settles at INITIATION from
    the request parameters — a deliberate, conservative over-count until
    completion-observation exists. The same builder serves both the pre-flight
    ``estimated_media`` (a precise per-second check — an oversized request is
    denied before the provider is called) and the settled ``media_usage`` (the
    job object carries nothing better). ``seconds`` and ``size`` default to
    OpenAI's documented values when absent (billing the value the provider
    applies), so a bare call is priced, not silently $0.
    """
    return MediaUsage(
        video_seconds=_openai_video_seconds(kwargs),
        resolution=_openai_video_size(kwargs),
        is_estimated=True,
    )


def _part_text_length(part: Any) -> int:
    """Sum string lengths of one delta/message part: content, reasoning text,
    and tool-call function arguments. Length-only; nothing is concatenated,
    stored, or returned beyond an integer."""
    total = 0
    content = getattr(part, "content", None)
    if isinstance(content, str):
        total += len(content)
    # DeepSeek-style reasoning text rides on a separate field.
    reasoning = getattr(part, "reasoning_content", None)
    if isinstance(reasoning, str):
        total += len(reasoning)
    # Tool-calling output lives entirely in function arguments — without this,
    # a tool-only response from a usage-less endpoint would estimate to zero.
    tool_calls = getattr(part, "tool_calls", None) or []
    for tool_call in tool_calls:
        arguments = getattr(getattr(tool_call, "function", None), "arguments", None)
        if isinstance(arguments, str):
            total += len(arguments)
    return total


def estimate_stream_chunk_content_length(chunk: Any) -> int:
    """Return the character length of an OpenAI-dialect stream chunk's delta text.

    Length-only measurement for the missing-streaming-usage fallback: when an
    OpenAI-compatible provider never emits a usage block, accumulated delta
    lengths (text + reasoning + tool-call arguments) estimate output tokens.
    Sums string lengths WITHOUT concatenating or storing the text. The
    returned integer is not reversible to content.

    Returns the total so far for any unexpected chunk shape. Never raises —
    this runs inside the live stream path on arbitrary endpoint output, and a
    malformed chunk must not convert a deliverable stream into a failure.
    """
    total = 0
    try:
        for choice in getattr(chunk, "choices", None) or []:
            total += _part_text_length(getattr(choice, "delta", None))
    except Exception:
        return total
    return total


def estimate_response_content_length(response: Any) -> int:
    """Return the character length of an OpenAI-dialect response's message text.

    Length-only measurement for the missing-usage fallback on NON-streaming
    responses from OpenAI-compatible providers. Counts message text, reasoning
    text, and tool-call arguments. Sums string lengths WITHOUT concatenating
    or storing the text. The returned integer is not reversible to content.

    Returns the total so far for any unexpected response shape. Never raises.
    """
    total = 0
    try:
        for choice in getattr(response, "choices", None) or []:
            total += _part_text_length(getattr(choice, "message", None))
    except Exception:
        return total
    return total
