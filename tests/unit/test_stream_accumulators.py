"""Tests for provider-specific streaming usage accumulators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers.anthropic import AnthropicAdapter, AnthropicStreamAccumulator
from solwyn.providers.bedrock import BedrockAdapter, BedrockStreamAccumulator
from solwyn.providers.google import GoogleAdapter, GoogleStreamAccumulator
from solwyn.providers.openai import OpenAIAdapter, OpenAIStreamAccumulator
from solwyn.providers.openai_compatible import CompatStreamAccumulator, build_compat_adapters

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAIStreamAccumulator:
    """OpenAI sends usage only in the final chunk (with stream_options)."""

    def test_extracts_usage_from_final_chunk(self) -> None:
        acc = OpenAIStreamAccumulator()

        # Content chunks — no usage
        acc.observe(
            SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
            )
        )
        acc.observe(
            SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]
            )
        )

        # Final chunk — has usage
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=150,
                    completion_tokens=83,
                    prompt_tokens_details=SimpleNamespace(
                        cached_tokens=20,
                        cache_write_tokens=30,
                        audio_tokens=0,
                    ),
                    completion_tokens_details=SimpleNamespace(
                        reasoning_tokens=10,
                        audio_tokens=0,
                        accepted_prediction_tokens=0,
                        rejected_prediction_tokens=0,
                    ),
                ),
                choices=[],
            )
        )

        result = acc.finalize()
        assert result.input_tokens == 150
        assert result.output_tokens == 83
        assert result.cached_input_tokens == 20
        assert result.cache_creation_5m_tokens == 30
        assert result.cache_creation_1h_tokens == 0
        assert result.reasoning_tokens == 10

    def test_returns_zeros_when_no_usage_chunk(self) -> None:
        acc = OpenAIStreamAccumulator()
        acc.observe(SimpleNamespace(usage=None, choices=[]))
        result = acc.finalize()
        assert result == TokenDetails()

    def test_returns_zeros_on_empty_stream(self) -> None:
        acc = OpenAIStreamAccumulator()
        result = acc.finalize()
        assert result == TokenDetails()

    def test_responses_api_shape(self) -> None:
        """Handles Responses API usage shape (input_tokens, not prompt_tokens)."""
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=200,
                    output_tokens=100,
                    input_tokens_details=SimpleNamespace(
                        cached_tokens=50,
                        cache_write_tokens=75,
                    ),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=15),
                ),
                choices=[],
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.cached_input_tokens == 50
        assert result.cache_creation_5m_tokens == 75
        assert result.cache_creation_1h_tokens == 0
        assert result.reasoning_tokens == 15

    def test_responses_api_stream_extracts_full_breakdown(self) -> None:
        """Streaming Responses API surfaces all 8 token sub-fields, same as non-streaming."""
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=500,
                    input_tokens_details=SimpleNamespace(cached_tokens=200, audio_tokens=50),
                    output_tokens_details=SimpleNamespace(
                        reasoning_tokens=300,
                        audio_tokens=20,
                        accepted_prediction_tokens=10,
                        rejected_prediction_tokens=5,
                    ),
                ),
                choices=[],
            )
        )
        result = acc.finalize()
        assert result.audio_input_tokens == 50
        assert result.audio_output_tokens == 20
        assert result.accepted_prediction_tokens == 10
        assert result.rejected_prediction_tokens == 5

    def test_response_completed_event_extracts_nested_usage_and_service_tier(self) -> None:
        acc = OpenAIStreamAccumulator()
        acc.observe(SimpleNamespace(type="response.output_text.delta", delta="partial"))
        acc.observe(
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=240,
                        output_tokens=80,
                        input_tokens_details=SimpleNamespace(cached_tokens=60),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=25),
                    ),
                    service_tier="flex",
                ),
            )
        )

        result = acc.finalize()

        assert result.input_tokens == 240
        assert result.output_tokens == 80
        assert result.cached_input_tokens == 60
        assert result.reasoning_tokens == 25
        assert acc.get_service_tier() == "flex"

    def test_pre_terminal_response_without_usage_is_ignored(self) -> None:
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                type="response.created",
                response=SimpleNamespace(usage=None, service_tier="flex"),
            )
        )

        assert acc.finalize() == TokenDetails()
        assert acc.get_service_tier() is None


@pytest.mark.unit
class TestOpenAIStreamAccumulatorServiceTier:
    """OpenAI streaming accumulator exposes service_tier from the final usage chunk."""

    @pytest.mark.parametrize("tier", ["flex", "priority"])
    def test_openai_stream_accumulator_extracts_service_tier(self, tier: str) -> None:
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                service_tier=tier,
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                choices=[],
            )
        )
        assert acc.get_service_tier() == tier

    def test_openai_stream_accumulator_service_tier_none_when_no_usage_chunk(self) -> None:
        acc = OpenAIStreamAccumulator()
        assert acc.get_service_tier() is None

    def test_openai_stream_accumulator_service_tier_absent_on_chunk(self) -> None:
        """When final chunk has no service_tier attribute, returns None."""
        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
                choices=[],
            )
        )
        assert acc.get_service_tier() is None

    def test_openai_stream_accumulator_truncates_service_tier(self) -> None:
        from solwyn._types import SERVICE_TIER_MAX_LENGTH

        acc = OpenAIStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                service_tier="x" * (SERVICE_TIER_MAX_LENGTH + 1),
                usage=SimpleNamespace(prompt_tokens=200, completion_tokens=100),
                choices=[],
            )
        )
        assert acc.get_service_tier() == "x" * SERVICE_TIER_MAX_LENGTH


@pytest.mark.unit
class TestOpenAIPrepareStreaming:
    """OpenAI adapter injects stream_options for usage in streaming."""

    def test_injects_stream_options(self) -> None:
        adapter = OpenAIAdapter()
        kwargs = {"model": "gpt-5.5", "messages": [], "stream": True}
        result = adapter.prepare_streaming(kwargs)
        assert result["stream_options"] == {"include_usage": True}
        # Original not mutated
        assert "stream_options" not in kwargs

    def test_preserves_existing_stream_options(self) -> None:
        adapter = OpenAIAdapter()
        kwargs = {"model": "gpt-5.5", "stream": True, "stream_options": {"include_usage": False}}
        result = adapter.prepare_streaming(kwargs)
        # We override include_usage to True
        assert result["stream_options"]["include_usage"] is True

    def test_creates_accumulator(self) -> None:
        adapter = OpenAIAdapter()
        acc = adapter.create_stream_accumulator()
        assert isinstance(acc, OpenAIStreamAccumulator)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicStreamAccumulator:
    """Anthropic splits usage across message_start (input) and message_delta (output)."""

    def test_extracts_input_and_output(self) -> None:
        acc = AnthropicStreamAccumulator()

        # message_start — carries input tokens
        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=150,
                        cache_read_input_tokens=0,
                    )
                ),
            )
        )

        # content_block_delta — text, no usage
        acc.observe(
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hello"))
        )

        # message_delta — carries output tokens
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=83),
            )
        )

        # message_stop
        acc.observe(SimpleNamespace(type="message_stop"))

        result = acc.finalize()
        assert result.input_tokens == 150
        assert result.output_tokens == 83
        assert result.cached_input_tokens == 0

    def test_extracts_cache_fields(self) -> None:
        acc = AnthropicStreamAccumulator()

        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=800,
                        cache_read_input_tokens=300,
                        cache_creation=SimpleNamespace(
                            ephemeral_5m_input_tokens=100,
                            ephemeral_1h_input_tokens=0,
                        ),
                    )
                ),
            )
        )

        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=200),
            )
        )

        result = acc.finalize()
        # Normalized: base + cache_read + cache_5m + cache_1h
        assert result.input_tokens == 800 + 300 + 100
        assert result.cached_input_tokens == 300
        assert result.cache_creation_5m_tokens == 100
        assert result.cache_creation_1h_tokens == 0
        assert result.output_tokens == 200

    def test_stream_input_tokens_normalized_with_both_5m_and_1h_nonzero(self) -> None:
        acc = AnthropicStreamAccumulator()

        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=1000,
                        cache_read_input_tokens=200,
                        cache_creation=SimpleNamespace(
                            ephemeral_5m_input_tokens=300,
                            ephemeral_1h_input_tokens=150,
                        ),
                    )
                ),
            )
        )
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=500),
            )
        )

        result = acc.finalize()
        assert result.cache_creation_5m_tokens == 300
        assert result.cache_creation_1h_tokens == 150
        assert result.cached_input_tokens == 200
        assert result.input_tokens == 1000 + 200 + 300 + 150
        assert result.output_tokens == 500

    def test_cache_creation_present_with_zero_values_ignores_aggregate_fallback(self) -> None:
        acc = AnthropicStreamAccumulator()

        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=0,
                        cache_read_input_tokens=0,
                        cache_creation=SimpleNamespace(
                            ephemeral_5m_input_tokens=0,
                            ephemeral_1h_input_tokens=0,
                        ),
                        cache_creation_input_tokens=999,
                    )
                ),
            )
        )
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=50),
            )
        )

        result = acc.finalize()
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_aggregate_only_cache_creation_falls_back_to_5m_bucket(self) -> None:
        acc = AnthropicStreamAccumulator()

        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=1000,
                        cache_read_input_tokens=200,
                        cache_creation_input_tokens=300,
                    )
                ),
            )
        )
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=500),
            )
        )

        result = acc.finalize()
        assert result.cache_creation_5m_tokens == 300
        assert result.cache_creation_1h_tokens == 0
        assert result.input_tokens == 1000 + 200 + 300

    def test_missing_message_start_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        acc = AnthropicStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=83),
            )
        )

        result = acc.finalize()

        assert result.input_tokens == 0
        assert result.output_tokens == 83
        assert "without message_start" in caplog.text

    def test_missing_message_delta_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        acc = AnthropicStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=150)),
            )
        )

        result = acc.finalize()

        assert result.input_tokens == 150
        assert result.output_tokens == 0
        assert "without message_delta" in caplog.text

    def test_returns_zeros_on_empty_stream(self) -> None:
        acc = AnthropicStreamAccumulator()
        result = acc.finalize()
        assert result == TokenDetails()

    def test_missing_cache_fields_default_to_zero(self) -> None:
        """Older API versions may not include cache fields."""
        acc = AnthropicStreamAccumulator()
        acc.observe(
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=100)),
            )
        )
        acc.observe(
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=50),
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cached_input_tokens == 0


@pytest.mark.unit
class TestAnthropicPrepareStreaming:
    """Anthropic needs no special kwargs preparation."""

    def test_returns_copy_unchanged(self) -> None:
        adapter = AnthropicAdapter()
        kwargs = {"model": "claude-sonnet-5", "messages": [], "stream": True}
        result = adapter.prepare_streaming(kwargs)
        assert result == kwargs
        assert result is not kwargs  # Must be a copy

    def test_creates_accumulator(self) -> None:
        adapter = AnthropicAdapter()
        acc = adapter.create_stream_accumulator()
        assert isinstance(acc, AnthropicStreamAccumulator)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoogleStreamAccumulator:
    """Google includes usage_metadata on chunks; last chunk is most complete."""

    def test_uses_last_chunk_metadata(self) -> None:
        acc = GoogleStreamAccumulator()

        # Early chunk — partial metadata
        acc.observe(
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=150,
                    candidates_token_count=20,
                    thoughts_token_count=0,
                ),
            )
        )

        # Final chunk — complete metadata
        acc.observe(
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=150,
                    candidates_token_count=83,
                    thoughts_token_count=10,
                    cached_content_token_count=30,
                    tool_use_prompt_token_count=5,
                ),
            )
        )

        result = acc.finalize()
        assert result.input_tokens == 150
        assert result.output_tokens == 83 + 10  # candidates + thoughts
        assert result.reasoning_tokens == 10
        assert result.cached_input_tokens == 30
        assert result.tool_use_input_tokens == 5

    def test_returns_zeros_on_empty_stream(self) -> None:
        acc = GoogleStreamAccumulator()
        result = acc.finalize()
        assert result == TokenDetails()

    def test_chunk_without_metadata(self) -> None:
        acc = GoogleStreamAccumulator()
        acc.observe(SimpleNamespace())  # No usage_metadata attribute
        result = acc.finalize()
        assert result == TokenDetails()


@pytest.mark.unit
class TestGooglePrepareStreaming:
    """Google needs no special kwargs preparation."""

    def test_returns_copy_unchanged(self) -> None:
        adapter = GoogleAdapter()
        kwargs = {"model": "gemini-2.5-pro", "contents": [], "stream": True}
        result = adapter.prepare_streaming(kwargs)
        assert result == kwargs
        assert result is not kwargs

    def test_creates_accumulator(self) -> None:
        adapter = GoogleAdapter()
        acc = adapter.create_stream_accumulator()
        assert isinstance(acc, GoogleStreamAccumulator)


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


def _bedrock_metadata(usage: dict, **extra) -> dict:
    return {"metadata": {"usage": usage, **extra}}


@pytest.mark.unit
class TestBedrockStreamAccumulator:
    """Bedrock ConverseStream: usage arrives only in the terminal metadata event."""

    def test_extracts_usage_from_terminal_metadata_event(self) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe({"messageStart": {"role": "assistant"}})
        acc.observe({"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}})
        acc.observe(
            _bedrock_metadata(
                {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "totalTokens": 150,
                    "cacheReadInputTokens": 20,
                    "cacheWriteInputTokens": 30,
                }
            )
        )
        result = acc.finalize()
        # AWS-documented normalization: input = base + cacheRead + cacheWrite
        assert result.input_tokens == 100 + 20 + 30
        assert result.output_tokens == 50
        assert result.cached_input_tokens == 20
        assert result.cache_creation_5m_tokens == 30  # no cacheDetails -> 5m bucket
        assert result.cache_creation_1h_tokens == 0

    def test_cache_details_splits_write_by_ttl(self) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe(
            _bedrock_metadata(
                {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cacheWriteInputTokens": 30,
                    "cacheDetails": [{"inputTokens": 12, "ttl": "1h"}],
                }
            )
        )
        result = acc.finalize()
        assert result.cache_creation_1h_tokens == 12
        assert result.cache_creation_5m_tokens == 30 - 12

    def test_returns_zeros_on_empty_stream(self) -> None:
        acc = BedrockStreamAccumulator()
        assert acc.finalize() == TokenDetails()

    def test_metadata_without_usage_warns_and_returns_zeros(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe({"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}})
        result = acc.finalize()
        assert result == TokenDetails()
        assert "settled at zero tokens" in caplog.text

    def test_non_mapping_chunks_are_ignored(self) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe("not-a-dict")
        acc.observe(None)
        assert acc.finalize() == TokenDetails()

    def test_service_tier_from_service_tier_type(self) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe(
            _bedrock_metadata({"inputTokens": 1, "outputTokens": 1}, serviceTier={"type": "flex"})
        )
        assert acc.get_service_tier() == "flex"

    def test_service_tier_falls_back_to_performance_config_latency(self) -> None:
        acc = BedrockStreamAccumulator()
        acc.observe(
            _bedrock_metadata(
                {"inputTokens": 1, "outputTokens": 1},
                performanceConfig={"latency": "optimized"},
            )
        )
        assert acc.get_service_tier() == "optimized"


@pytest.mark.unit
class TestBedrockPrepareStreaming:
    """ConverseStream always emits terminal metadata — no kwargs preparation."""

    def test_returns_copy_unchanged(self) -> None:
        adapter = BedrockAdapter()
        kwargs = {"model": "us.anthropic.claude-sonnet-5", "messages": []}
        result = adapter.prepare_streaming(kwargs)
        assert result == kwargs
        assert result is not kwargs

    def test_creates_accumulator(self) -> None:
        adapter = BedrockAdapter()
        acc = adapter.create_stream_accumulator(estimated_input_tokens=99)
        assert isinstance(acc, BedrockStreamAccumulator)


# ---------------------------------------------------------------------------
# OpenAI-compatible (CompatStreamAccumulator)
# ---------------------------------------------------------------------------


def _compat_adapter(name: str = "groq"):
    """A FRESH adapter instance (never the registry singleton) so the
    once-per-instance missing-usage warning latch starts clean per test."""
    return next(a for a in build_compat_adapters() if a.name == name)


def _compat_acc(estimated_input_tokens: int = 0, name: str = "groq") -> CompatStreamAccumulator:
    return _compat_adapter(name).create_stream_accumulator(
        estimated_input_tokens=estimated_input_tokens
    )


def _compat_content_chunk(text: str) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, reasoning_content=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None, x_groq=None)


def _compat_usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )
    return SimpleNamespace(choices=[], usage=usage, x_groq=None)


@pytest.mark.unit
class TestCompatStreamAccumulator:
    """Compat tiers: standard usage -> x_groq.usage -> explicit estimation."""

    def test_content_accumulation_stops_after_usage_latch(self) -> None:
        acc = _compat_acc(estimated_input_tokens=10)

        acc.observe(_compat_usage_chunk(10, 5))
        acc.observe(_compat_content_chunk("a" * 500))
        acc.observe(_compat_content_chunk("b" * 500))

        assert acc._content_chars == 0
        result = acc.finalize()
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.is_estimated is False

    def test_content_accumulation_runs_until_usage_latch(self) -> None:
        acc = _compat_acc()

        acc.observe(_compat_content_chunk("a" * 100))
        assert acc._content_chars == 100
        acc.observe(_compat_usage_chunk(10, 5))
        acc.observe(_compat_content_chunk("b" * 100))

        assert acc._content_chars == 100

    def test_estimation_tier_still_accumulates_without_usage(self) -> None:
        acc = _compat_acc()

        acc.observe(_compat_content_chunk("a" * 400))
        acc.observe(_compat_content_chunk("b" * 400))

        result = acc.finalize()
        assert result.is_estimated is True
        assert result.output_tokens > 0

    def test_tier1_last_nonzero_usage_wins(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]
            )
        )
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45), choices=[]
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 120
        assert result.output_tokens == 45
        assert result.is_estimated is False

    def test_tier1_zeroed_placeholder_usage_never_latches(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40), choices=[]
            )
        )
        # Trailing zero-placeholder block (some providers attach one per chunk)
        acc.observe(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0), choices=[])
        )
        result = acc.finalize()
        assert result.input_tokens == 100
        assert result.output_tokens == 40

    def test_tier2_x_groq_usage_dict_shape(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                usage=None,
                choices=[],
                x_groq={"usage": {"prompt_tokens": 33, "completion_tokens": 11}},
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 33
        assert result.output_tokens == 11
        assert result.is_estimated is False

    def test_tier2_x_groq_usage_attr_shape(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                usage=None,
                choices=[],
                x_groq=SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=21, completion_tokens=7)
                ),
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 21
        assert result.output_tokens == 7

    def test_tier1_beats_tier2_when_both_present(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=25),
                choices=[],
                x_groq={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            )
        )
        result = acc.finalize()
        assert result.input_tokens == 50
        assert result.output_tokens == 25

    def test_tier3_estimates_and_marks_is_estimated(self, caplog: pytest.LogCaptureFixture) -> None:
        acc = _compat_acc(estimated_input_tokens=42)
        acc.observe(
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello world"))],
            )
        )
        result = acc.finalize()
        assert result.is_estimated is True
        assert result.input_tokens == 42  # pre-call estimate
        assert result.output_tokens > 0  # from accumulated delta lengths — never zero
        assert "no usage data" in caplog.text

    def test_tier3_empty_stream_estimates_input_only(self) -> None:
        acc = _compat_acc(estimated_input_tokens=42)
        result = acc.finalize()
        assert result.is_estimated is True
        assert result.input_tokens == 42
        assert result.output_tokens == 0

    def test_garbage_usage_never_raises_and_degrades_to_estimation(self) -> None:
        acc = _compat_acc(estimated_input_tokens=10)
        acc.observe(SimpleNamespace(usage="garbage", choices=[]))  # must not raise
        result = acc.finalize()
        assert result.is_estimated is True
        assert result.input_tokens == 10

    def test_service_tier_extracted_from_chunk(self) -> None:
        acc = _compat_acc()
        acc.observe(
            SimpleNamespace(
                service_tier="flex",
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
                choices=[],
            )
        )
        assert acc.get_service_tier() == "flex"


@pytest.mark.unit
class TestCompatCreateAccumulator:
    """The compat adapter wires itself + the input estimate into the accumulator."""

    def test_creates_accumulator_with_estimate(self) -> None:
        acc = _compat_adapter().create_stream_accumulator(estimated_input_tokens=7)
        assert isinstance(acc, CompatStreamAccumulator)
