"""Bedrock Converse dialect translation contract tests.

Mirrors test_translation.py for the Bedrock dialect: round-trip is the
contract AND the drift alarm; every unsupported item must RAISE
``UntranslatableRequestError`` with a STRUCTURAL feature label and NO
offending value anywhere on the error.

Bedrock-dialect specifics under test:
- kwargs carry the uniform internal ``model`` key (the proxy renames boto3's
  ``modelId`` at the interception boundary; dispatch renames it back).
- generation controls nest under ``inferenceConfig`` (maxTokens/temperature/
  topP/stopSequences); tools nest under ``toolConfig`` with ``toolSpec`` /
  ``inputSchema.json`` wrappers.
- ``system`` is a LIST of content blocks, images are raw bytes (base64 in the
  canonical form), and responses/stream events are plain DICTS.
"""

from __future__ import annotations

import base64

import pytest

from solwyn.exceptions import UntranslatableRequestError
from solwyn.providers._translation import (
    CanonicalRequest,
    from_canonical,
    normalize_finish_reason,
    normalize_response,
    to_canonical,
    translate_stream_chunk,
)

SECRET = "SUPER_SECRET_PROMPT_a1b2c3"

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakebytes"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def bedrock_req(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "inferenceConfig": {"maxTokens": 256},
    }
    base.update(over)
    return base


def _assert_no_value_leak(exc: UntranslatableRequestError, *values: str) -> None:
    blob = str(exc) + repr(exc) + repr(exc.args) + repr(exc.feature)
    for value in values:
        assert value not in blob


# --------------------------------------------------------------------------- #
# to_canonical — Bedrock dialect parsing                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestBedrockToCanonical:
    def test_basic_request(self) -> None:
        canonical = to_canonical("bedrock", bedrock_req())
        assert canonical.max_tokens == 256
        assert canonical.messages[0].role == "user"
        assert canonical.messages[0].content[0].text == "hello"  # type: ignore[union-attr]
        assert canonical.system is None

    def test_inference_config_scalars(self) -> None:
        canonical = to_canonical(
            "bedrock",
            bedrock_req(
                inferenceConfig={
                    "maxTokens": 128,
                    "temperature": 0.5,
                    "topP": 0.9,
                    "stopSequences": ["END"],
                }
            ),
        )
        assert canonical.max_tokens == 128
        assert canonical.temperature == 0.5
        assert canonical.top_p == 0.9
        assert canonical.stop == ["END"]

    def test_system_block_list_joined(self) -> None:
        canonical = to_canonical(
            "bedrock",
            bedrock_req(system=[{"text": "You are helpful"}, {"text": "Be brief"}]),
        )
        assert canonical.system == "You are helpful\nBe brief"

    def test_missing_max_tokens_raises(self) -> None:
        req = bedrock_req(inferenceConfig={})
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "missing_max_tokens"

    def test_temperature_above_one_raises(self) -> None:
        # Bedrock temperature is documented 0..1; canonical enforces <= 1.0.
        req = bedrock_req(inferenceConfig={"maxTokens": 10, "temperature": 1.5})
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "temperature>1.0"

    def test_tools_and_tool_choice(self) -> None:
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        canonical = to_canonical(
            "bedrock",
            bedrock_req(
                toolConfig={
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "inputSchema": {"json": schema},
                            }
                        }
                    ],
                    "toolChoice": {"tool": {"name": "get_weather"}},
                }
            ),
        )
        assert canonical.tools is not None
        assert canonical.tools[0].name == "get_weather"
        assert canonical.tools[0].parameters == schema
        assert canonical.tool_choice is not None
        assert canonical.tool_choice.mode == "force"
        assert canonical.tool_choice.name == "get_weather"

    @pytest.mark.parametrize(
        "choice,mode",
        [({"auto": {}}, "auto"), ({"any": {}}, "required")],
    )
    def test_tool_choice_modes(self, choice: dict[str, object], mode: str) -> None:
        canonical = to_canonical(
            "bedrock",
            bedrock_req(
                toolConfig={
                    "tools": [{"toolSpec": {"name": "t", "inputSchema": {"json": {}}}}],
                    "toolChoice": choice,
                }
            ),
        )
        assert canonical.tool_choice is not None
        assert canonical.tool_choice.mode == mode

    def test_tool_use_and_result_round(self) -> None:
        canonical = to_canonical(
            "bedrock",
            bedrock_req(
                messages=[
                    {"role": "user", "content": [{"text": "weather?"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tu_1",
                                    "name": "get_weather",
                                    "input": {"city": "Berlin"},
                                }
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": "tu_1",
                                    "content": [{"json": {"temp": 21}}],
                                    "status": "success",
                                }
                            }
                        ],
                    },
                ]
            ),
        )
        assert canonical.messages[1].content[0].id == "tu_1"  # type: ignore[union-attr]
        assert canonical.messages[2].content[0].tool_use_id == "tu_1"  # type: ignore[union-attr]

    def test_dangling_tool_use_raises(self) -> None:
        req = bedrock_req(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "tu_1", "name": "t", "input": {}}}],
                },
            ]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "dangling_tool_call"

    def test_orphan_tool_result_raises(self) -> None:
        req = bedrock_req(
            messages=[
                {
                    "role": "user",
                    "content": [{"toolResult": {"toolUseId": "tu_x", "content": []}}],
                },
            ]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "orphan_tool_result"

    def test_image_bytes_to_canonical_base64(self) -> None:
        canonical = to_canonical(
            "bedrock",
            bedrock_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": "png", "source": {"bytes": PNG_BYTES}}},
                            {"text": "describe"},
                        ],
                    }
                ]
            ),
        )
        image = canonical.messages[0].content[0]
        assert image.media_type == "image/png"  # type: ignore[union-attr]
        assert image.data == PNG_B64  # type: ignore[union-attr]

    def test_image_s3_location_raises_opaque_handle(self) -> None:
        req = bedrock_req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"s3Location": {"uri": "s3://bucket/key"}},
                            }
                        }
                    ],
                }
            ]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "image.opaque_handle"

    def test_cache_point_block_raises(self) -> None:
        req = bedrock_req(
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "hi"}, {"cachePoint": {"type": "default"}}],
                }
            ]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "cache_control"

    def test_system_cache_point_raises(self) -> None:
        req = bedrock_req(system=[{"text": "hi"}, {"cachePoint": {"type": "default"}}])
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "cache_control"

    def test_guard_content_block_raises(self) -> None:
        req = bedrock_req(system=[{"guardContent": {"text": {"text": SECRET}}}])
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "bedrock.guard_content"
        _assert_no_value_leak(exc_info.value, SECRET)

    def test_reasoning_content_block_raises(self) -> None:
        req = bedrock_req(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"reasoningContent": {"reasoningText": {"text": SECRET}}}],
                }
            ]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "reasoning"
        _assert_no_value_leak(exc_info.value, SECRET)

    def test_document_block_raises_multimodal(self) -> None:
        req = bedrock_req(
            messages=[{"role": "user", "content": [{"document": {"name": "d", "format": "pdf"}}]}]
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "multimodal.document"

    def test_guardrail_config_raises(self) -> None:
        req = bedrock_req(guardrailConfig={"guardrailIdentifier": "g1", "guardrailVersion": "1"})
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "bedrock.guardrail_config"

    @pytest.mark.parametrize(
        "key",
        ["additionalModelRequestFields", "promptVariables", "requestMetadata", "outputConfig"],
    )
    def test_unrecognized_top_level_kwarg_fails_closed(self, key: str) -> None:
        req = bedrock_req(**{key: {"x": 1}})
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == f"unsupported_kwarg.{key}"

    def test_unrecognized_inference_config_key_fails_closed(self) -> None:
        req = bedrock_req(inferenceConfig={"maxTokens": 10, "topK": 40})
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "unsupported_kwarg.topK"

    def test_unknown_message_role_raises(self) -> None:
        req = bedrock_req(messages=[{"role": "tool", "content": [{"text": "x"}]}])
        with pytest.raises(UntranslatableRequestError) as exc_info:
            to_canonical("bedrock", req)
        assert exc_info.value.feature == "unknown_message_role"


# --------------------------------------------------------------------------- #
# from_canonical — rendering INTO the Bedrock dialect                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestCanonicalToBedrock:
    def test_basic_render(self) -> None:
        canonical = to_canonical(
            "anthropic",
            {
                "model": "claude-3-5-sonnet",
                "max_tokens": 99,
                "system": "be helpful",
                "temperature": 0.3,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        kwargs = from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert kwargs["model"] == "amazon.nova-pro-v1:0"
        assert kwargs["system"] == [{"text": "be helpful"}]
        assert kwargs["inferenceConfig"] == {"maxTokens": 99, "temperature": 0.3}
        assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]

    def test_stop_sequences_render(self) -> None:
        canonical = to_canonical(
            "openai",
            {
                "model": "gpt-4o",
                "max_tokens": 10,
                "stop": ["END"],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        kwargs = from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert kwargs["inferenceConfig"]["stopSequences"] == ["END"]  # type: ignore[index]

    def test_tools_render(self) -> None:
        schema = {"type": "object", "properties": {}}
        canonical = to_canonical(
            "openai",
            {
                "model": "gpt-4o",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "description": "d", "parameters": schema},
                    }
                ],
                "tool_choice": "required",
            },
        )
        kwargs = from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        tool_config = kwargs["toolConfig"]
        assert tool_config["tools"][0]["toolSpec"]["name"] == "f"  # type: ignore[index]
        assert tool_config["tools"][0]["toolSpec"]["inputSchema"] == {"json": schema}  # type: ignore[index]
        assert tool_config["toolChoice"] == {"any": {}}  # type: ignore[index]

    def test_tool_choice_none_raises(self) -> None:
        # Bedrock ToolChoice supports auto/any/tool only — there is no "none".
        canonical = to_canonical(
            "openai",
            {
                "model": "gpt-4o",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                "tool_choice": "none",
            },
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert exc_info.value.feature == "tool_choice.none"

    def test_parallel_tool_calls_false_raises(self) -> None:
        # No Bedrock equivalent of disable-parallel-tool-use.
        canonical = to_canonical(
            "openai",
            {
                "model": "gpt-4o",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
                "parallel_tool_calls": False,
            },
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert exc_info.value.feature == "parallel_tool_calls"

    def test_image_base64_renders_raw_bytes(self) -> None:
        canonical = to_canonical(
            "anthropic",
            {
                "model": "claude-3-5-sonnet",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": PNG_B64,
                                },
                            }
                        ],
                    }
                ],
            },
        )
        kwargs = from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        block = kwargs["messages"][0]["content"][0]  # type: ignore[index]
        assert block["image"]["format"] == "png"
        assert block["image"]["source"]["bytes"] == PNG_BYTES

    def test_image_url_raises(self) -> None:
        # Bedrock ImageSource is bytes or s3Location — no public-URL member.
        canonical = to_canonical(
            "anthropic",
            {
                "model": "claude-3-5-sonnet",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://x.test/i.png"},
                            }
                        ],
                    }
                ],
            },
        )
        with pytest.raises(UntranslatableRequestError) as exc_info:
            from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert exc_info.value.feature == "image.url_unsupported"

    def test_tool_round_trip_bedrock_to_bedrock(self) -> None:
        original = bedrock_req(
            messages=[
                {"role": "user", "content": [{"text": "weather?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_1",
                                "name": "get_weather",
                                "input": {"city": "Berlin"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "tu_1", "content": [{"text": "21C"}]}}
                    ],
                },
            ],
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_weather",
                            "description": "d",
                            "inputSchema": {"json": {"type": "object"}},
                        }
                    }
                ]
            },
        )
        canonical = to_canonical("bedrock", original)
        rendered = from_canonical(
            "bedrock", canonical, model="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert rendered["messages"][1]["content"][0]["toolUse"]["toolUseId"] == "tu_1"  # type: ignore[index]
        assert rendered["messages"][2]["content"][0]["toolResult"]["toolUseId"] == "tu_1"  # type: ignore[index]

    def test_empty_assistant_turn_dropped(self) -> None:
        canonical = CanonicalRequest(
            messages=[],
            max_tokens=10,
        )
        rendered = from_canonical("bedrock", canonical, model="amazon.nova-pro-v1:0")
        assert rendered["messages"] == []


# --------------------------------------------------------------------------- #
# Response normalization (both directions)                                     #
# --------------------------------------------------------------------------- #
def _converse_response_dict() -> dict[str, object]:
    return {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "hi there"},
                    {"toolUse": {"toolUseId": "tu_9", "name": "f", "input": {"a": 1}}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
        "metrics": {"latencyMs": 5},
    }


@pytest.mark.unit
class TestBedrockResponseNormalization:
    def test_bedrock_served_to_openai_requested(self) -> None:
        result = normalize_response(
            served="bedrock", requested="openai", response=_converse_response_dict()
        )
        message = result.choices[0].message  # type: ignore[attr-defined]
        assert message.content == "hi there"
        assert message.tool_calls[0].id == "tu_9"
        assert result.choices[0].finish_reason == "tool_calls"  # type: ignore[attr-defined]

    def test_anthropic_served_to_bedrock_requested(self) -> None:
        from types import SimpleNamespace

        served = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            model="claude-3-5-sonnet",
        )
        result = normalize_response(served="anthropic", requested="bedrock", response=served)
        assert result["output"]["message"]["content"][0]["text"] == "ok"  # type: ignore[index]
        assert result["output"]["message"]["role"] == "assistant"  # type: ignore[index]
        assert result["stopReason"] == "end_turn"  # type: ignore[index]

    def test_bedrock_to_bedrock_is_identity(self) -> None:
        response = _converse_response_dict()
        assert normalize_response(served="bedrock", requested="bedrock", response=response) is (
            response
        )

    @pytest.mark.parametrize(
        "stop_reason,canonical",
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_use"),
            ("guardrail_intervened", "content_filter"),
            ("content_filtered", "content_filter"),
            ("model_context_window_exceeded", "length"),
        ],
    )
    def test_finish_reason_normalization(self, stop_reason: str, canonical: str) -> None:
        assert normalize_finish_reason("bedrock", stop_reason) == canonical


# --------------------------------------------------------------------------- #
# Stream chunk translation (both directions)                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestBedrockStreamChunks:
    def test_text_delta_from_bedrock_to_openai(self) -> None:
        chunks = translate_stream_chunk(
            served="bedrock",
            requested="openai",
            chunk={"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hel"  # type: ignore[attr-defined]

    def test_message_stop_maps_to_finish(self) -> None:
        chunks = translate_stream_chunk(
            served="bedrock",
            requested="openai",
            chunk={"messageStop": {"stopReason": "end_turn"}},
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].finish_reason == "stop"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "event",
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}},
        ],
    )
    def test_structural_events_emit_no_chunks(self, event: dict[str, object]) -> None:
        assert translate_stream_chunk(served="bedrock", requested="openai", chunk=event) == []

    def test_tool_use_delta_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as exc_info:
            translate_stream_chunk(
                served="bedrock",
                requested="openai",
                chunk={
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": SECRET}},
                        "contentBlockIndex": 0,
                    }
                },
            )
        assert exc_info.value.feature == "cross_provider_tool_stream"
        _assert_no_value_leak(exc_info.value, SECRET)

    def test_tool_use_block_start_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as exc_info:
            translate_stream_chunk(
                served="bedrock",
                requested="openai",
                chunk={
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": "t", "name": "f"}},
                        "contentBlockIndex": 1,
                    }
                },
            )
        assert exc_info.value.feature == "cross_provider_tool_stream"

    def test_reasoning_delta_raises_multimodal(self) -> None:
        with pytest.raises(UntranslatableRequestError) as exc_info:
            translate_stream_chunk(
                served="bedrock",
                requested="openai",
                chunk={
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": {"text": SECRET}},
                        "contentBlockIndex": 0,
                    }
                },
            )
        assert exc_info.value.feature == "cross_provider_multimodal_stream"
        _assert_no_value_leak(exc_info.value, SECRET)

    def test_text_delta_from_anthropic_to_bedrock(self) -> None:
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hi"),
        )
        chunks = translate_stream_chunk(served="anthropic", requested="bedrock", chunk=chunk)
        assert chunks == [{"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}}]

    def test_finish_from_anthropic_to_bedrock(self) -> None:
        from types import SimpleNamespace

        chunk = SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="max_tokens"),
        )
        chunks = translate_stream_chunk(served="anthropic", requested="bedrock", chunk=chunk)
        assert chunks == [{"messageStop": {"stopReason": "max_tokens"}}]
