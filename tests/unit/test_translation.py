"""Cross-provider translation contract tests (design spec §5).

Round-trip is the contract AND the drift alarm: a fully-resolved request must
survive ``to_canonical -> from_canonical`` with correct id remap and argument
encoding. Every §5.5 unsupported item must RAISE ``UntranslatableRequestError``
with a STRUCTURAL feature label and NO offending value anywhere in the error.
"""

from __future__ import annotations

import pytest

from solwyn.exceptions import UntranslatableRequestError
from solwyn.providers._translation import (
    CanonicalRequest,
    from_canonical,
    normalize_response,
    to_canonical,
)

# A value we plant inside prompt content / argument values and then assert never
# leaks into any UntranslatableRequestError surface.
SECRET = "SUPER_SECRET_PROMPT_a1b2c3"


# --------------------------------------------------------------------------- #
# Helpers — minimal native request dicts per provider dialect.                 #
# --------------------------------------------------------------------------- #
def openai_req(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "gpt-4o",
        "max_completion_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    base.update(over)
    return base


def anthropic_req(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    base.update(over)
    return base


def google_req(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "gemini-1.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "config": {"max_output_tokens": 256},
    }
    base.update(over)
    return base


def _assert_no_value_leak(exc: UntranslatableRequestError, *values: str) -> None:
    """The error must carry STRUCTURAL labels only — never an offending value."""
    blob = str(exc) + repr(exc) + repr(exc.args) + repr(exc.feature)
    for value in values:
        assert value not in blob


# --------------------------------------------------------------------------- #
# Canonical model invariants (§5.1)                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestCanonicalModel:
    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            CanonicalRequest(
                system=None,
                messages=[],
                max_tokens=10,
                temperature=None,
                top_p=None,
                stop=None,
                stream=False,
                tools=None,
                tool_choice=None,
                bogus_field=1,  # type: ignore[call-arg]
            )

    def test_max_tokens_required(self) -> None:
        canonical = to_canonical("anthropic", anthropic_req(max_tokens=512))
        assert canonical.max_tokens == 512


# --------------------------------------------------------------------------- #
# §5.2 Request field mapping                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFieldMapping:
    def test_openai_emits_max_completion_tokens_not_max_tokens(self) -> None:
        canonical = to_canonical("openai", openai_req(max_completion_tokens=128))
        out = from_canonical("openai", canonical, model="gpt-4o")
        assert out["max_completion_tokens"] == 128
        assert "max_tokens" not in out

    def test_openai_accepts_legacy_max_tokens_input(self) -> None:
        # Inbound Chat Completions may still use max_tokens; canonical normalizes.
        canonical = to_canonical("openai", openai_req(max_tokens=77, max_completion_tokens=None))
        assert canonical.max_tokens == 77

    def test_anthropic_hoists_system_to_top_level(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                ]
            ),
        )
        assert canonical.system == "be terse"
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert out["system"] == "be terse"
        assert out["max_tokens"] == 256
        # system must NOT remain as a message
        assert all(m["role"] != "system" for m in out["messages"])

    def test_google_nests_under_config(self) -> None:
        canonical = to_canonical("openai", openai_req(stop=["X"]))
        out = from_canonical("google", canonical, model="gemini-1.5-pro")
        assert out["config"]["max_output_tokens"] == 256
        assert out["config"]["stop_sequences"] == ["X"]

    def test_google_system_instruction(self) -> None:
        canonical = to_canonical("anthropic", anthropic_req(system="be terse"))
        out = from_canonical("google", canonical, model="gemini-1.5-pro")
        assert out["config"]["system_instruction"] == "be terse"

    def test_temperature_in_range_passes(self) -> None:
        canonical = to_canonical("openai", openai_req(temperature=0.5))
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert out["temperature"] == 0.5

    def test_temperature_above_one_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(temperature=1.5))
        assert ei.value.feature == "temperature>1.0"
        _assert_no_value_leak(ei.value, "1.5")

    def test_top_p_straight_passes(self) -> None:
        canonical = to_canonical("openai", openai_req(top_p=0.9))
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert out["top_p"] == 0.9

    def test_stop_over_four_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(stop=["a", "b", "c", "d", "e"]))
        assert ei.value.feature == "stop>4"

    def test_stop_string_normalizes_to_list(self) -> None:
        canonical = to_canonical("openai", openai_req(stop="STOP"))
        assert canonical.stop == ["STOP"]
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert out["stop_sequences"] == ["STOP"]

    def test_anthropic_consecutive_same_role_does_not_raise(self) -> None:
        # §5.2: do NOT repair-or-RAISE on consecutive same-role turns.
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ]
            ),
        )
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert len(out["messages"]) == 2


# --------------------------------------------------------------------------- #
# §5.3 Tools & tool_choice                                                     #
# --------------------------------------------------------------------------- #
TOOL = {
    "name": "get_weather",
    "description": "Look up weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.unit
class TestToolDeclarations:
    def test_openai_wrapper(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
            ),
        )
        out = from_canonical("openai", canonical, model="gpt-4o")
        assert out["tools"][0]["type"] == "function"
        assert out["tools"][0]["function"]["name"] == "get_weather"
        assert out["tools"][0]["function"]["parameters"] == TOOL["parameters"]

    def test_anthropic_wrapper(self) -> None:
        canonical = to_canonical(
            "openai", openai_req(tools=[{"type": "function", "function": TOOL}])
        )
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        decl = out["tools"][0]
        assert decl["name"] == "get_weather"
        assert decl["input_schema"] == TOOL["parameters"]
        assert "type" not in decl  # anthropic decl has no top-level type

    def test_google_wrapper(self) -> None:
        canonical = to_canonical(
            "openai", openai_req(tools=[{"type": "function", "function": TOOL}])
        )
        out = from_canonical("google", canonical, model="gemini-1.5-pro")
        decls = out["config"]["tools"][0]["function_declarations"]
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["parameters_json_schema"] == TOOL["parameters"]

    def test_anthropic_inbound_tool_decl(self) -> None:
        canonical = to_canonical(
            "anthropic",
            anthropic_req(
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "input_schema": TOOL["parameters"],
                    }
                ]
            ),
        )
        out = from_canonical("openai", canonical, model="gpt-4o")
        assert out["tools"][0]["function"]["name"] == "get_weather"

    def test_google_inbound_tool_decl(self) -> None:
        canonical = to_canonical(
            "google",
            google_req(
                config={
                    "max_output_tokens": 256,
                    "tools": [
                        {
                            "function_declarations": [
                                {
                                    "name": "get_weather",
                                    "description": "Look up weather",
                                    "parameters_json_schema": TOOL["parameters"],
                                }
                            ]
                        }
                    ],
                }
            ),
        )
        out = from_canonical("anthropic", canonical, model="claude-3-5-sonnet")
        assert out["tools"][0]["name"] == "get_weather"


@pytest.mark.unit
class TestToolChoice:
    @pytest.mark.parametrize(
        ("native_choice", "expected"),
        [
            ("auto", "auto"),
            ("required", "required"),
            ("none", "none"),
        ],
    )
    def test_openai_simple_choices(self, native_choice: str, expected: str) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                tool_choice=native_choice,
            ),
        )
        out = from_canonical("openai", canonical, model="gpt-4o")
        assert out["tool_choice"] == expected

    def test_openai_force_specific(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
        )
        out = from_canonical("openai", canonical, model="gpt-4o")
        assert out["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_anthropic_choices(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                tool_choice="auto",
            ),
        )
        assert from_canonical("anthropic", canonical, model="m")["tool_choice"] == {"type": "auto"}
        canonical = to_canonical(
            "openai",
            openai_req(tools=[{"type": "function", "function": TOOL}], tool_choice="required"),
        )
        assert from_canonical("anthropic", canonical, model="m")["tool_choice"] == {"type": "any"}
        canonical = to_canonical(
            "openai",
            openai_req(tools=[{"type": "function", "function": TOOL}], tool_choice="none"),
        )
        assert from_canonical("anthropic", canonical, model="m")["tool_choice"] == {"type": "none"}

    def test_anthropic_force_specific(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        assert out["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_google_choices(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(tools=[{"type": "function", "function": TOOL}], tool_choice="auto"),
        )
        cfg = from_canonical("google", canonical, model="m")["config"]["tool_config"][
            "function_calling_config"
        ]
        assert cfg["mode"] == "AUTO"
        canonical = to_canonical(
            "openai",
            openai_req(tools=[{"type": "function", "function": TOOL}], tool_choice="required"),
        )
        cfg = from_canonical("google", canonical, model="m")["config"]["tool_config"][
            "function_calling_config"
        ]
        assert cfg["mode"] == "ANY"
        canonical = to_canonical(
            "openai",
            openai_req(tools=[{"type": "function", "function": TOOL}], tool_choice="none"),
        )
        cfg = from_canonical("google", canonical, model="m")["config"]["tool_config"][
            "function_calling_config"
        ]
        assert cfg["mode"] == "NONE"

    def test_google_force_specific(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
        )
        cfg = from_canonical("google", canonical, model="m")["config"]["tool_config"][
            "function_calling_config"
        ]
        assert cfg["mode"] == "ANY"
        assert cfg["allowed_function_names"] == ["get_weather"]


# --------------------------------------------------------------------------- #
# §5.3 Tool-result round-trip (the most error-prone surface)                  #
# --------------------------------------------------------------------------- #
def _openai_resolved_tool_history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "weather in paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]


def _anthropic_resolved_tool_history() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "weather in paris?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}],
        },
    ]


@pytest.mark.unit
class TestToolResultRoundTrip:
    def test_openai_to_anthropic_args_become_object(self) -> None:
        canonical = to_canonical("openai", openai_req(messages=_openai_resolved_tool_history()))
        out = from_canonical("anthropic", canonical, model="m")
        # assistant tool_use carries a JSON OBJECT input (not a string)
        assistant = out["messages"][1]
        tool_use = assistant["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["input"] == {"city": "paris"}
        assert tool_use["name"] == "get_weather"
        # the tool result is remapped to a tool_result block keyed by tool_use_id
        result_msg = out["messages"][2]
        block = result_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == tool_use["id"]
        assert block["content"] == "sunny"

    def test_anthropic_to_openai_args_become_string(self) -> None:
        import json

        canonical = to_canonical(
            "anthropic", anthropic_req(messages=_anthropic_resolved_tool_history())
        )
        out = from_canonical("openai", canonical, model="gpt-4o")
        assistant = out["messages"][1]
        call = assistant["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        # arguments must be a JSON STRING on OpenAI
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"city": "paris"}
        # tool result becomes role:tool keyed by tool_call_id
        result = out["messages"][2]
        assert result["role"] == "tool"
        assert result["tool_call_id"] == call["id"]
        assert result["content"] == "sunny"

    def test_openai_anthropic_openai_full_roundtrip(self) -> None:
        import json

        original = _openai_resolved_tool_history()
        canonical = to_canonical("openai", openai_req(messages=original))
        anthro = from_canonical("anthropic", canonical, model="m")
        canonical2 = to_canonical("anthropic", anthropic_req(messages=anthro["messages"]))
        back = from_canonical("openai", canonical2, model="gpt-4o")
        call = back["messages"][1]["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"city": "paris"}
        assert back["messages"][2]["content"] == "sunny"
        # ids remap consistently: result points at the (re)issued call id
        assert back["messages"][2]["tool_call_id"] == call["id"]

    def test_openai_to_google_function_response(self) -> None:
        canonical = to_canonical("openai", openai_req(messages=_openai_resolved_tool_history()))
        out = from_canonical("google", canonical, model="m")
        contents = out["contents"]
        # model turn carries a functionCall part
        model_turn = contents[1]
        assert model_turn["role"] == "model"
        fc_part = model_turn["parts"][0]
        assert fc_part["function_call"]["name"] == "get_weather"
        assert fc_part["function_call"]["args"] == {"city": "paris"}
        # tool turn carries a functionResponse part keyed by name
        tool_turn = contents[2]
        assert tool_turn["role"] == "tool"
        fr_part = tool_turn["parts"][0]
        assert fr_part["function_response"]["name"] == "get_weather"

    def test_google_to_openai_roundtrip(self) -> None:
        import json

        google_history = [
            {"role": "user", "parts": [{"text": "weather?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "fc_1",
                            "name": "get_weather",
                            "args": {"city": "paris"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "function_response": {
                            "id": "fc_1",
                            "name": "get_weather",
                            "response": {"weather": "sunny"},
                        }
                    }
                ],
            },
        ]
        canonical = to_canonical("google", google_req(contents=google_history))
        out = from_canonical("openai", canonical, model="gpt-4o")
        call = out["messages"][1]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "paris"}
        assert out["messages"][2]["tool_call_id"] == call["id"]


# --------------------------------------------------------------------------- #
# §5.4 Multimodal                                                              #
# --------------------------------------------------------------------------- #
PNG_B64 = "iVBORw0KGgoAAAANS"


@pytest.mark.unit
class TestMultimodal:
    def test_openai_base64_image_to_anthropic(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                            },
                        ],
                    }
                ]
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        blocks = out["messages"][0]["content"]
        img = next(b for b in blocks if b["type"] == "image")
        assert img["source"]["type"] == "base64"
        # media_type parsed VERBATIM from the data: URI
        assert img["source"]["media_type"] == "image/png"
        assert img["source"]["data"] == PNG_B64

    def test_public_https_url_image_translates(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://ex.com/a.png"},
                            }
                        ],
                    }
                ]
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        img = out["messages"][0]["content"][0]
        assert img["source"]["type"] == "url"
        assert img["source"]["url"] == "https://ex.com/a.png"

    def test_data_uri_media_type_parsed_verbatim(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{PNG_B64}"},
                            }
                        ],
                    }
                ]
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        img = out["messages"][0]["content"][0]
        assert img["source"]["media_type"] == "image/jpeg"

    def test_gs_uri_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "file_data": {
                                        "file_uri": f"gs://bucket/{SECRET}.png",
                                        "mime_type": "image/png",
                                    }
                                }
                            ],
                        }
                    ]
                ),
            )
        assert "image" in ei.value.feature
        _assert_no_value_leak(ei.value, SECRET)

    def test_anthropic_file_id_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "file",
                                        "file_id": f"file_{SECRET}",
                                    },
                                }
                            ],
                        }
                    ]
                ),
            )
        _assert_no_value_leak(ei.value, SECRET)

    def test_audio_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": SECRET, "format": "wav"},
                                }
                            ],
                        }
                    ]
                ),
            )
        _assert_no_value_leak(ei.value, SECRET)


# --------------------------------------------------------------------------- #
# §5.4 Finish reason normalization                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFinishReason:
    @pytest.mark.parametrize(
        ("served", "raw", "expected"),
        [
            ("openai", "stop", "stop"),
            ("openai", "length", "length"),
            ("openai", "tool_calls", "tool_use"),
            ("openai", "content_filter", "content_filter"),
            ("anthropic", "end_turn", "stop"),
            ("anthropic", "max_tokens", "length"),
            ("anthropic", "tool_use", "tool_use"),
            ("anthropic", "stop_sequence", "stop"),
            ("anthropic", "refusal", "content_filter"),
            ("google", "STOP", "stop"),
            ("google", "MAX_TOKENS", "length"),
            ("google", "SAFETY", "content_filter"),
            ("google", "RECITATION", "content_filter"),
            ("google", "PROHIBITED_CONTENT", "content_filter"),
        ],
    )
    def test_finish_reason_normalizes(self, served: str, raw: str, expected: str) -> None:
        from solwyn.providers._translation import normalize_finish_reason

        assert normalize_finish_reason(served, raw) == expected


# --------------------------------------------------------------------------- #
# §5.5 Fail-loudly unsupported list — every item RAISES, no value leak.        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFailLoudly:
    def test_seed_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(seed=42))
        assert ei.value.feature == "seed"
        _assert_no_value_leak(ei.value, "42")

    def test_frequency_penalty_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(frequency_penalty=0.5))
        assert ei.value.feature == "frequency_penalty"
        _assert_no_value_leak(ei.value, "0.5")

    def test_presence_penalty_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(presence_penalty=0.5))
        assert ei.value.feature == "presence_penalty"

    def test_top_k_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("anthropic", anthropic_req(top_k=5))
        assert ei.value.feature == "top_k"

    def test_response_format_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(response_format={"type": "json_object", "secret": SECRET}),
            )
        assert ei.value.feature == "response_format"
        _assert_no_value_leak(ei.value, SECRET)

    def test_response_schema_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(config={"max_output_tokens": 10, "response_schema": {"x": SECRET}}),
            )
        assert ei.value.feature == "response_schema"
        _assert_no_value_leak(ei.value, SECRET)

    def test_logprobs_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(logprobs=True))
        assert ei.value.feature == "logprobs"

    def test_top_logprobs_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(top_logprobs=5))
        assert ei.value.feature == "top_logprobs"

    def test_logit_bias_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(logit_bias={"123": -100}))
        assert ei.value.feature == "logit_bias"

    def test_service_tier_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(service_tier="flex"))
        assert ei.value.feature == "service_tier"

    def test_n_gt_one_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(n=2))
        assert ei.value.feature == "n>1"

    def test_reasoning_effort_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(reasoning_effort="high"))
        assert ei.value.feature == "reasoning_effort"

    def test_anthropic_thinking_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("anthropic", anthropic_req(thinking={"type": "enabled"}))
        assert ei.value.feature == "thinking"

    def test_google_thinking_config_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(config={"max_output_tokens": 10, "thinking_config": {"x": 1}}),
            )
        assert ei.value.feature == "thinking_config"

    def test_anthropic_cache_control_system_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    system=[
                        {
                            "type": "text",
                            "text": SECRET,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                ),
            )
        _assert_no_value_leak(ei.value, SECRET)

    def test_anthropic_cache_control_block_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": SECRET,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ]
                ),
            )
        assert ei.value.feature == "cache_control"
        _assert_no_value_leak(ei.value, SECRET)

    def test_google_cached_content_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(config={"max_output_tokens": 10, "cached_content": f"cc_{SECRET}"}),
            )
        assert ei.value.feature == "cached_content"
        _assert_no_value_leak(ei.value, SECRET)

    def test_anthropic_proprietary_tool_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(tools=[{"type": "computer_20241022", "name": "computer"}]),
            )
        assert "computer" in ei.value.feature

    def test_anthropic_web_search_tool_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
            )
        assert "web_search" in ei.value.feature

    def test_google_search_tool_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(
                    config={
                        "max_output_tokens": 10,
                        "tools": [{"google_search": {}}],
                    }
                ),
            )
        assert "google_search" in ei.value.feature

    def test_openai_non_function_tool_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(tools=[{"type": "file_search"}]),
            )
        assert ei.value.feature == "openai.file_search"

    def test_parallel_tool_calls_false_to_google_raises(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                parallel_tool_calls=False,
            ),
        )
        with pytest.raises(UntranslatableRequestError) as ei:
            from_canonical("google", canonical, model="m")
        assert ei.value.feature == "parallel_tool_calls=False"

    def test_parallel_tool_calls_false_to_anthropic_ok(self) -> None:
        # only RAISES toward google; anthropic is fine
        canonical = to_canonical(
            "openai",
            openai_req(
                tools=[{"type": "function", "function": TOOL}],
                parallel_tool_calls=False,
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        assert "tools" in out

    def test_dangling_tool_call_raises(self) -> None:
        history = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            # NO tool result follows -> dangling
        ]
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(messages=history))
        assert ei.value.feature == "dangling_tool_call"

    def test_parallel_same_name_tool_calls_raises(self) -> None:
        history = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"a"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"b"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "x"},
            {"role": "tool", "tool_call_id": "call_2", "content": "y"},
        ]
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(messages=history))
        assert ei.value.feature == "parallel_same_name_tool_calls"

    def test_parallel_distinct_name_tool_calls_ok(self) -> None:
        history = [
            {"role": "user", "content": "weather + time?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"a"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "x"},
            {"role": "tool", "tool_call_id": "call_2", "content": "y"},
        ]
        canonical = to_canonical("openai", openai_req(messages=history))
        out = from_canonical("anthropic", canonical, model="m")
        assert len(out["messages"][1]["content"]) == 2

    def test_openai_responses_api_shape_raises(self) -> None:
        # Responses API uses `input=` + `instructions=`, not messages.
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                {
                    "model": "gpt-4o",
                    "max_completion_tokens": 10,
                    "input": [{"role": "user", "content": SECRET}],
                },
            )
        assert ei.value.feature == "responses_api"
        _assert_no_value_leak(ei.value, SECRET)


# --------------------------------------------------------------------------- #
# normalize_response — duck-typed cross-provider response shaping (§5.4)        #
# --------------------------------------------------------------------------- #
class _Obj:
    """Tiny duck-typed namespace mirroring a native SDK response object."""

    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _anthropic_text_response() -> _Obj:
    block = _Obj(type="text", text="hello from claude")
    return _Obj(content=[block], stop_reason="end_turn", model="claude-3-5-sonnet")


def _anthropic_tool_response() -> _Obj:
    block = _Obj(type="tool_use", id="toolu_9", name="get_weather", input={"city": "paris"})
    return _Obj(content=[block], stop_reason="tool_use", model="claude-3-5-sonnet")


def _openai_text_response() -> _Obj:
    msg = _Obj(role="assistant", content="hello from gpt", tool_calls=None)
    choice = _Obj(index=0, message=msg, finish_reason="stop")
    return _Obj(choices=[choice], model="gpt-4o")


def _openai_tool_response() -> _Obj:
    call = _Obj(
        id="call_9",
        type="function",
        function=_Obj(name="get_weather", arguments='{"city": "paris"}'),
    )
    msg = _Obj(role="assistant", content=None, tool_calls=[call])
    choice = _Obj(index=0, message=msg, finish_reason="tool_calls")
    return _Obj(choices=[choice], model="gpt-4o")


@pytest.mark.unit
class TestNormalizeResponse:
    def test_anthropic_served_openai_requested_text(self) -> None:
        resp = normalize_response(
            served="anthropic", requested="openai", response=_anthropic_text_response()
        )
        assert resp.choices[0].message.content == "hello from claude"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.choices[0].message.role == "assistant"

    def test_anthropic_served_openai_requested_tool(self) -> None:
        import json

        resp = normalize_response(
            served="anthropic", requested="openai", response=_anthropic_tool_response()
        )
        call = resp.choices[0].message.tool_calls[0]
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "paris"}
        # OpenAI-shaped object carries the OpenAI-native finish reason.
        assert resp.choices[0].finish_reason == "tool_calls"

    def test_openai_served_anthropic_requested_text(self) -> None:
        resp = normalize_response(
            served="openai", requested="anthropic", response=_openai_text_response()
        )
        assert resp.content[0].text == "hello from gpt"
        assert resp.content[0].type == "text"
        assert resp.stop_reason == "end_turn"

    def test_openai_served_anthropic_requested_tool(self) -> None:
        resp = normalize_response(
            served="openai", requested="anthropic", response=_openai_tool_response()
        )
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.name == "get_weather"
        assert block.input == {"city": "paris"}
        assert resp.stop_reason == "tool_use"

    def test_served_openai_requested_google_text(self) -> None:
        resp = normalize_response(
            served="openai", requested="google", response=_openai_text_response()
        )
        assert resp.candidates[0].content.parts[0].text == "hello from gpt"

    def test_same_provider_is_identity(self) -> None:
        original = _openai_text_response()
        resp = normalize_response(served="openai", requested="openai", response=original)
        assert resp is original


# --------------------------------------------------------------------------- #
# Privacy — fail-loud across the boundary with `from None`, no content leak.   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestPrivacyBoundary:
    def test_untranslatable_suppresses_context(self) -> None:
        # Errors raised inside _translation use `from None` -> no __cause__ chain.
        req = openai_req(seed=1, messages=[{"role": "user", "content": SECRET}])
        try:
            to_canonical("openai", req)
        except UntranslatableRequestError as exc:
            assert exc.__cause__ is None
            assert exc.__suppress_context__ is True
            _assert_no_value_leak(exc, SECRET)
        else:  # pragma: no cover
            pytest.fail("expected UntranslatableRequestError")

    def test_no_content_in_any_raise_surface(self) -> None:
        # A request packed with the sentinel in every content slot must never
        # echo it back through the structural error.
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(
                    system=SECRET,
                    messages=[{"role": "user", "content": SECRET}],
                    response_format={"schema": SECRET},
                ),
            )
        _assert_no_value_leak(ei.value, SECRET)

    def test_module_banner_and_no_io_imports(self) -> None:
        from pathlib import Path

        import solwyn.providers._translation as mod

        src = Path(mod.__file__).read_text()
        assert "PRIVACY-CRITICAL" in src[:600]
        # Spec §7 CI form: the literal substrings must not appear ANYWHERE,
        # not just in import statements (prose mentions are forbidden too).
        assert "logging" not in src
        assert "logger" not in src
        assert "httpx" not in src


# --------------------------------------------------------------------------- #
# [A] Fail-closed: unrecognized native kwargs RAISE on a cross-provider hop.   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFailClosedUnknownKwargs:
    def test_unknown_top_level_kwarg_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(frobnicate=1))
        assert ei.value.feature == "unsupported_kwarg.frobnicate"

    def test_unknown_kwarg_value_not_leaked(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(frobnicate=SECRET))
        _assert_no_value_leak(ei.value, SECRET)

    def test_recognized_request_still_translates(self) -> None:
        canonical = to_canonical("openai", openai_req(temperature=0.5, top_p=0.9, stop=["X"]))
        out = from_canonical("anthropic", canonical, model="m")
        assert out["max_tokens"] == 256
        assert out["temperature"] == 0.5

    def test_unknown_anthropic_top_level_kwarg_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("anthropic", anthropic_req(frobnicate=1))
        assert ei.value.feature == "unsupported_kwarg.frobnicate"

    def test_unknown_google_config_kwarg_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(config={"max_output_tokens": 10, "frobnicate": 1}),
            )
        assert ei.value.feature == "unsupported_kwarg.frobnicate"

    def test_specific_label_takes_precedence_over_generic(self) -> None:
        # seed has a dedicated raise; it must win over the generic fallback.
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical("openai", openai_req(seed=42))
        assert ei.value.feature == "seed"


# --------------------------------------------------------------------------- #
# [B] Malformed content NEVER escapes as a raw ValidationError/value leak.     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestMalformedRequestGuard:
    def test_non_string_text_raises_structurally(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(
                    messages=[{"role": "user", "content": [{"type": "text", "text": 12345}]}]
                ),
            )
        assert "malformed" in ei.value.feature
        _assert_no_value_leak(ei.value, "12345")

    def test_malformed_value_not_in_cause(self) -> None:
        try:
            to_canonical(
                "openai",
                openai_req(
                    messages=[{"role": "user", "content": [{"type": "text", "text": 999888}]}]
                ),
            )
        except UntranslatableRequestError as exc:
            assert exc.__cause__ is None
            assert "999888" not in repr(exc.__cause__)
            _assert_no_value_leak(exc, "999888")
        else:  # pragma: no cover
            pytest.fail("expected UntranslatableRequestError")

    def test_malformed_tool_input_raises_structurally(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_1",
                                    "name": "x",
                                    "input": "not-a-dict",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                            ],
                        },
                    ]
                ),
            )
        assert "malformed" in ei.value.feature


# --------------------------------------------------------------------------- #
# [C] Caller-controlled discriminators map to FIXED labels (no echo-back).     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestDiscriminatorLabelsAreConstant:
    def test_exotic_block_type_does_not_echo(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "deadbeef-secret", "data": SECRET}],
                        }
                    ]
                ),
            )
        assert "deadbeef-secret" not in ei.value.feature
        assert "deadbeef-secret" not in str(ei.value)
        _assert_no_value_leak(ei.value, SECRET)

    def test_exotic_anthropic_block_type_does_not_echo(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "deadbeef-secret", "x": SECRET}],
                        }
                    ]
                ),
            )
        assert "deadbeef-secret" not in str(ei.value)
        _assert_no_value_leak(ei.value, SECRET)

    def test_exotic_tool_type_does_not_echo(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(tools=[{"type": "deadbeef-secret"}]),
            )
        assert "deadbeef-secret" not in str(ei.value)

    def test_google_exotic_mime_does_not_echo(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {"inline_data": {"mime_type": "deadbeef/secret", "data": SECRET}}
                            ],
                        }
                    ]
                ),
            )
        assert "deadbeef" not in str(ei.value)
        _assert_no_value_leak(ei.value, SECRET)


# --------------------------------------------------------------------------- #
# [E] Google tool-call response keeps the tool finish-reason (STOP+fc).        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestGoogleToolFinishReason:
    def test_google_tool_call_response_normalizes_to_tool_use(self) -> None:
        fc = _Obj(id="fc_1", name="get_weather", args={"city": "paris"})
        part = _Obj(text=None, function_call=fc)
        candidate = _Obj(content=_Obj(parts=[part], role="model"), finish_reason="STOP")
        resp = _Obj(candidates=[candidate], model="gemini-1.5-pro")
        # served google -> requested openai exposes the tool finish reason.
        out = normalize_response(served="google", requested="openai", response=resp)
        assert out.choices[0].finish_reason == "tool_calls"

    def test_google_plain_stop_stays_stop(self) -> None:
        part = _Obj(text="hi", function_call=None)
        candidate = _Obj(content=_Obj(parts=[part], role="model"), finish_reason="STOP")
        resp = _Obj(candidates=[candidate], model="gemini-1.5-pro")
        out = normalize_response(served="google", requested="openai", response=resp)
        assert out.choices[0].finish_reason == "stop"


# --------------------------------------------------------------------------- #
# [F] parallel_tool_calls=False emitted toward OpenAI/Anthropic, RAISE Google. #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestParallelToolCallsFalse:
    def _canonical_no_parallel(self) -> CanonicalRequest:
        return to_canonical(
            "anthropic",
            anthropic_req(
                tools=[
                    {"name": "get_weather", "input_schema": TOOL["parameters"]},
                ],
                tool_choice={"type": "auto"},
                parallel_tool_calls=False,
            ),
        )

    def test_toward_openai_emits_false(self) -> None:
        out = from_canonical("openai", self._canonical_no_parallel(), model="gpt-4o")
        assert out["parallel_tool_calls"] is False

    def test_toward_anthropic_disables_parallel(self) -> None:
        out = from_canonical("anthropic", self._canonical_no_parallel(), model="m")
        assert out["tool_choice"]["disable_parallel_tool_use"] is True
        assert out["tool_choice"]["type"] == "auto"

    def test_toward_google_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            from_canonical("google", self._canonical_no_parallel(), model="m")
        assert ei.value.feature == "parallel_tool_calls=False"


# --------------------------------------------------------------------------- #
# [G] Google tool-result CONTENT survives a hop into-and-out-of Google.        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestGoogleToolResultContentRoundTrip:
    def test_openai_google_openai_content_survives(self) -> None:
        original = _openai_resolved_tool_history()
        canonical = to_canonical("openai", openai_req(messages=original))
        google = from_canonical("google", canonical, model="m")
        # round back through google -> canonical -> openai
        canonical2 = to_canonical("google", google_req(contents=google["contents"]))
        back = from_canonical("openai", canonical2, model="gpt-4o")
        assert back["messages"][2]["content"] == "sunny"

    def test_google_to_anthropic_content_survives(self) -> None:
        original = _openai_resolved_tool_history()
        canonical = to_canonical("openai", openai_req(messages=original))
        google = from_canonical("google", canonical, model="m")
        canonical2 = to_canonical("google", google_req(contents=google["contents"]))
        anthro = from_canonical("anthropic", canonical2, model="m")
        result_block = anthro["messages"][2]["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["content"] == "sunny"


# --------------------------------------------------------------------------- #
# [H] Multiple system messages concatenate; insecure image URL RAISEs.        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestSystemAndImageUrl:
    def test_multiple_openai_system_messages_concatenate(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {"role": "system", "content": "first"},
                    {"role": "system", "content": "second"},
                    {"role": "user", "content": "hi"},
                ]
            ),
        )
        assert canonical.system == "first\n\nsecond"

    def test_http_image_url_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "openai",
                openai_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "http://ex.com/a.png"}}
                            ],
                        }
                    ]
                ),
            )
        assert ei.value.feature == "image.insecure_url"


# --------------------------------------------------------------------------- #
# [I] Empty assistant turn crossing to Anthropic is dropped (no content=[]).   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestEmptyAssistantTurn:
    def test_empty_assistant_turn_dropped_for_anthropic(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "still there?"},
                ]
            ),
        )
        out = from_canonical("anthropic", canonical, model="m")
        # No message may carry an empty content list (Anthropic 400s on it).
        for msg in out["messages"]:
            assert msg["content"] != []


# --------------------------------------------------------------------------- #
# [J] Google candidate_count>1 fails loud.                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestGoogleCandidateCount:
    def test_candidate_count_gt_one_raises(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "google",
                google_req(config={"max_output_tokens": 10, "candidate_count": 2}),
            )
        assert ei.value.feature == "candidate_count>1"

    def test_candidate_count_one_ok(self) -> None:
        canonical = to_canonical(
            "google",
            google_req(config={"max_output_tokens": 10, "candidate_count": 1}),
        )
        assert canonical.max_tokens == 10


# --------------------------------------------------------------------------- #
# [K] Public URL image targeting Google fails loud (base64 still works).       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestUrlImageTowardGoogle:
    def test_url_image_to_google_raises(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://ex.com/a.png"}}
                        ],
                    }
                ]
            ),
        )
        with pytest.raises(UntranslatableRequestError) as ei:
            from_canonical("google", canonical, model="m")
        assert ei.value.feature == "image.url_unsupported_google"

    def test_base64_image_to_google_ok(self) -> None:
        canonical = to_canonical(
            "openai",
            openai_req(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                            }
                        ],
                    }
                ]
            ),
        )
        out = from_canonical("google", canonical, model="m")
        part = out["contents"][0]["parts"][0]
        assert part["inline_data"]["data"] == PNG_B64


# --------------------------------------------------------------------------- #
# [L] document/audio/video block types route to multimodal.* fixed labels.    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestMultimodalLabelConsistency:
    def test_anthropic_user_document_uses_multimodal_label(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "document", "source": {"data": SECRET}}],
                        }
                    ]
                ),
            )
        assert ei.value.feature == "multimodal.document"
        _assert_no_value_leak(ei.value, SECRET)

    def test_anthropic_assistant_document_uses_multimodal_label(self) -> None:
        with pytest.raises(UntranslatableRequestError) as ei:
            to_canonical(
                "anthropic",
                anthropic_req(
                    messages=[
                        {
                            "role": "assistant",
                            "content": [{"type": "document", "source": {"data": SECRET}}],
                        }
                    ]
                ),
            )
        assert ei.value.feature == "multimodal.document"
        _assert_no_value_leak(ei.value, SECRET)


@pytest.mark.unit
class TestProviderValidation:
    def test_unknown_provider_to_canonical_raises(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            to_canonical("cohere", {"messages": []})

    def test_unknown_provider_from_canonical_raises(self) -> None:
        canonical = to_canonical("openai", openai_req())
        with pytest.raises(ValueError, match="provider"):
            from_canonical("cohere", canonical, model="x")
