"""Tests for provider-specific streaming usage accumulators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers.anthropic import AnthropicAdapter, AnthropicStreamAccumulator
from solwyn.providers.google import GoogleAdapter, GoogleStreamAccumulator
from solwyn.providers.openai import OpenAIAdapter, OpenAIStreamAccumulator

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
        kwargs = {"model": "gpt-4o", "messages": [], "stream": True}
        result = adapter.prepare_streaming(kwargs)
        assert result["stream_options"] == {"include_usage": True}
        # Original not mutated
        assert "stream_options" not in kwargs

    def test_preserves_existing_stream_options(self) -> None:
        adapter = OpenAIAdapter()
        kwargs = {"model": "gpt-4o", "stream": True, "stream_options": {"include_usage": False}}
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
        kwargs = {"model": "claude-sonnet-4-5", "messages": [], "stream": True}
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
        kwargs = {"model": "gemini-pro", "contents": [], "stream": True}
        result = adapter.prepare_streaming(kwargs)
        assert result == kwargs
        assert result is not kwargs

    def test_creates_accumulator(self) -> None:
        adapter = GoogleAdapter()
        acc = adapter.create_stream_accumulator()
        assert isinstance(acc, GoogleStreamAccumulator)
