"""Tests for Anthropic provider adapter — token extraction only, no pricing.

Key subtlety: Anthropic's input_tokens and cache_read_input_tokens are separate
additive fields (base does not include cache). Cache writes are broken out by
TTL tier via usage.cache_creation.ephemeral_5m_input_tokens and
usage.cache_creation.ephemeral_1h_input_tokens. Normalized input_tokens is the
sum of all four components.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers.anthropic import AnthropicAdapter

# ---------------------------------------------------------------------------
# Helpers — build fake Anthropic response objects
# ---------------------------------------------------------------------------


def _anthropic_response(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_5m: int = 0,
    cache_1h: int = 0,
    include_cache_fields: bool = True,
) -> Any:
    """Build a fake Anthropic messages.create() response.

    By default includes cache_read and (when nonzero) cache_creation sub-object.
    Set include_cache_fields=False to simulate older responses with no cache info at all.
    """
    if include_cache_fields:
        kwargs: dict[str, object] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
        if cache_5m or cache_1h:
            kwargs["cache_creation"] = SimpleNamespace(
                ephemeral_5m_input_tokens=cache_5m,
                ephemeral_1h_input_tokens=cache_1h,
            )
        usage = SimpleNamespace(**kwargs)
    else:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return SimpleNamespace(usage=usage)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(AnthropicAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert AnthropicAdapter().name == "anthropic"


@pytest.mark.unit
class TestAnthropicAdapterDetect:
    def test_detect_model_claude_3(self) -> None:
        assert AnthropicAdapter().detect_model("claude-sonnet-5") is True

    def test_detect_model_claude_opus(self) -> None:
        assert AnthropicAdapter().detect_model("claude-opus-4-6") is True

    def test_detect_model_claude_haiku(self) -> None:
        assert AnthropicAdapter().detect_model("claude-haiku-3-5") is True

    def test_detect_model_does_not_match_gpt(self) -> None:
        assert AnthropicAdapter().detect_model("gpt-5.5") is False

    def test_detect_model_does_not_match_gemini(self) -> None:
        assert AnthropicAdapter().detect_model("gemini-2.5-flash") is False

    def test_detect_client_anthropic_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "anthropic.resources"
        assert AnthropicAdapter().detect_client(FakeClient()) is True

    def test_detect_client_non_anthropic(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai"
        assert AnthropicAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestAnthropicAdapterExtractUsage:
    def test_basic_tokens(self) -> None:
        response = _anthropic_response(input_tokens=1000, output_tokens=500)
        result = AnthropicAdapter().extract_usage(response)
        assert result.output_tokens == 500

    def test_input_tokens_normalized_to_sum_of_components(self) -> None:
        """input_tokens = base + cache_read + cache_5m + cache_1h (all additive)."""
        response = _anthropic_response(
            input_tokens=1000,
            cache_read_input_tokens=300,
            cache_5m=150,
            cache_1h=50,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 1500  # 1000 + 300 + 150 + 50

    def test_base_input_only_no_cache(self) -> None:
        """With no cache, input_tokens == base."""
        response = _anthropic_response(input_tokens=800)
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 800

    def test_cache_read_mapped_to_cached_input_tokens(self) -> None:
        response = _anthropic_response(
            input_tokens=1000,
            cache_read_input_tokens=400,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.cached_input_tokens == 400

    def test_cache_creation_mapped_to_cache_creation_5m_tokens(self) -> None:
        response = _anthropic_response(
            input_tokens=1000,
            cache_5m=250,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 250
        assert result.cache_creation_1h_tokens == 0

    def test_reasoning_tokens_always_zero(self) -> None:
        """Anthropic folds thinking tokens into output_tokens — documented blind spot."""
        response = _anthropic_response(input_tokens=1000, output_tokens=600)
        result = AnthropicAdapter().extract_usage(response)
        assert result.reasoning_tokens == 0

    def test_full_cache_response(self) -> None:
        """All cache fields present: normalized input = sum of all components."""
        response = _anthropic_response(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=400,
            cache_5m=100,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 1500  # 1000 + 400 + 100
        assert result.output_tokens == 500
        assert result.cached_input_tokens == 400
        assert result.cache_creation_5m_tokens == 100
        assert result.cache_creation_1h_tokens == 0

    def test_missing_cache_fields_graceful(self) -> None:
        """Older Anthropic responses without cache fields return zeros for those."""
        response = _anthropic_response(
            input_tokens=500,
            output_tokens=200,
            include_cache_fields=False,
        )
        result = AnthropicAdapter().extract_usage(response)
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cached_input_tokens == 0
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_returns_token_details_instance(self) -> None:
        response = _anthropic_response(input_tokens=10, output_tokens=5)
        result = AnthropicAdapter().extract_usage(response)
        assert isinstance(result, TokenDetails)

    def test_audio_tokens_always_zero(self) -> None:
        """Anthropic doesn't have audio token fields."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0

    def test_prediction_tokens_always_zero(self) -> None:
        """Anthropic doesn't have predicted output token fields."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0

    def test_tool_use_input_tokens_always_zero(self) -> None:
        """Anthropic doesn't report tool_use_input_tokens."""
        response = _anthropic_response(input_tokens=100, output_tokens=50)
        result = AnthropicAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 0

    def test_extracts_ephemeral_5m_and_1h_cache_writes(self) -> None:
        """5m and 1h cache writes come from usage.cache_creation sub-object.

        Priced separately by the API (1.25× and 2× base input rate).
        """
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=200,
                    ephemeral_1h_input_tokens=100,
                ),
            )
        )
        result = AnthropicAdapter().extract_usage(resp)
        assert result.cache_creation_5m_tokens == 200
        assert result.cache_creation_1h_tokens == 100
        # Normalized: base + cache_read + (5m + 1h)
        assert result.input_tokens == 1000 + 200 + 200 + 100
        assert result.output_tokens == 500
        assert result.cached_input_tokens == 200

    def test_aggregate_only_cache_creation_falls_back_to_5m_bucket(self) -> None:
        """Non-beta responses may only carry cache_creation_input_tokens."""
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=300,
            )
        )
        result = AnthropicAdapter().extract_usage(resp)
        assert result.cache_creation_5m_tokens == 300
        assert result.cache_creation_1h_tokens == 0
        assert result.input_tokens == 1000 + 200 + 300

    def test_cache_creation_present_with_zero_values_ignores_aggregate_fallback(self) -> None:
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=0,
                    ephemeral_1h_input_tokens=0,
                ),
                cache_creation_input_tokens=999,
            )
        )

        result = AnthropicAdapter().extract_usage(resp)

        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0
        assert result.input_tokens == 100
        assert result.output_tokens == 50


@pytest.mark.unit
class TestAnthropicAdapterNoneHandling:
    def test_none_usage_returns_zeros(self) -> None:
        """When response.usage is None, return all-zero TokenDetails."""
        response = SimpleNamespace(usage=None)
        result = AnthropicAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_no_usage_attr_returns_zeros(self) -> None:
        """When response has no usage attribute, return all-zero TokenDetails."""
        result = AnthropicAdapter().extract_usage(SimpleNamespace())
        assert result == TokenDetails()


@pytest.mark.unit
class TestAnthropicAdapterServiceTierScope:
    def test_anthropic_adapter_returns_none_for_service_tier(self) -> None:
        """Anthropic has no service-tier concept."""
        assert AnthropicAdapter().extract_service_tier(SimpleNamespace()) is None


@pytest.mark.unit
class TestAnthropicAdapterDispatchSeams:
    def test_prepare_call_selects_messages_create(self) -> None:
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        kwargs: dict[str, Any] = {"model": "claude-sonnet-4-6"}

        method, prepared = AnthropicAdapter().prepare_call(
            client, kwargs, is_streaming=False, timeout=30.0, max_retries=0
        )

        assert method is create
        assert prepared == {"model": "claude-sonnet-4-6"}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_call_streaming_sets_stream_kwarg_without_mutation(self) -> None:
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: kw))
        kwargs: dict[str, Any] = {"model": "claude-sonnet-4-6"}

        _, prepared = AnthropicAdapter().prepare_call(
            client, kwargs, is_streaming=True, timeout=30.0, max_retries=0
        )

        assert prepared["stream"] is True
        assert "stream" not in kwargs

    def test_stream_shape_seams_are_identity(self) -> None:
        adapter = AnthropicAdapter()
        response, wrapper = object(), object()
        assert adapter.unwrap_stream_source(response) is response
        assert adapter.wrap_stream_result(wrapper, response) is wrapper
