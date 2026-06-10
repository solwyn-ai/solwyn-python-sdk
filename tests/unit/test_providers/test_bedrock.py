"""Tests for Bedrock provider adapter — token extraction only, no pricing.

Key subtleties:
- Bedrock Converse responses are plain DICTS (botocore deserializes JSON),
  not attribute objects — extraction must use mapping access, never getattr.
- Converse usage is ADDITIVE like Anthropic: usage.inputTokens excludes
  cacheReadInputTokens / cacheWriteInputTokens. Normalized input_tokens sums
  all three. Cache writes map to the 5m bucket (Bedrock's prompt-cache TTL is
  ~5 minutes; no 1h tier exists).
- Streaming usage arrives in the terminal ``metadata`` event of the
  ConverseStream event stream.
- detect_client is duck-typed on botocore's client shape — boto3 is never
  imported by the SDK.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._constants import SERVICE_TIER_MAX_LENGTH
from solwyn.providers.bedrock import BedrockAdapter

# ---------------------------------------------------------------------------
# Helpers — build fake Bedrock clients and Converse response dicts
# ---------------------------------------------------------------------------


def _bedrock_client(
    *,
    module: str = "botocore.client",
    service_name: str = "bedrock-runtime",
    region_name: str = "us-east-1",
) -> Any:
    """Build a fake boto3/aioboto3 bedrock-runtime client (duck-typed shape)."""

    class FakeBedrockRuntime:
        pass

    FakeBedrockRuntime.__module__ = module
    client = FakeBedrockRuntime()
    client.meta = SimpleNamespace(  # type: ignore[attr-defined]
        service_model=SimpleNamespace(service_name=service_name),
        region_name=region_name,
    )
    return client


def _usage(
    input_tokens: int,
    output_tokens: int,
    cache_read: int | None,
    cache_write: int | None,
    cache_details: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a TokenUsage dict per the documented shape.

    inputTokens/outputTokens/totalTokens are contractually required;
    cacheReadInputTokens/cacheWriteInputTokens/cacheDetails are optional.
    """
    usage: dict[str, Any] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
    }
    if cache_read is not None:
        usage["cacheReadInputTokens"] = cache_read
    if cache_write is not None:
        usage["cacheWriteInputTokens"] = cache_write
    if cache_details is not None:
        usage["cacheDetails"] = cache_details
    return usage


def _converse_response(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int | None = None,
    cache_write: int | None = None,
    cache_details: list[dict[str, Any]] | None = None,
    performance_latency: str | None = None,
    service_tier: str | None = None,
) -> dict[str, Any]:
    """Build a fake Converse response dict (the documented uniform shape)."""
    response: dict[str, Any] = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": _usage(input_tokens, output_tokens, cache_read, cache_write, cache_details),
        "metrics": {"latencyMs": 100},
    }
    if performance_latency is not None:
        response["performanceConfig"] = {"latency": performance_latency}
    if service_tier is not None:
        response["serviceTier"] = {"type": service_tier}
    return response


def _metadata_event(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int | None = None,
    cache_write: int | None = None,
    cache_details: list[dict[str, Any]] | None = None,
    performance_latency: str | None = None,
    service_tier: str | None = None,
) -> dict[str, Any]:
    """Build the terminal ConverseStream ``metadata`` event."""
    metadata: dict[str, Any] = {
        "usage": _usage(input_tokens, output_tokens, cache_read, cache_write, cache_details),
        "metrics": {"latencyMs": 800},
    }
    if performance_latency is not None:
        metadata["performanceConfig"] = {"latency": performance_latency}
    if service_tier is not None:
        metadata["serviceTier"] = {"type": service_tier}
    return {"metadata": metadata}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBedrockAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(BedrockAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert BedrockAdapter().name == "bedrock"


@pytest.mark.unit
class TestBedrockAdapterDetectModel:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "eu.anthropic.claude-sonnet-4-20250514-v1:0",
            "apac.amazon.nova-pro-v1:0",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "mistral.mistral-large-2402-v1:0",
            "cohere.command-r-plus-v1:0",
            "ai21.jamba-1-5-large-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.titan-text-express-v1",
            "deepseek.r1-v1:0",
            "openai.gpt-oss-120b-1:0",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/abcdef",
            "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/"
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
        ],
    )
    def test_detect_model_matches_bedrock_ids(self, model: str) -> None:
        assert BedrockAdapter().detect_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "claude-3-5-sonnet",  # direct-Anthropic id, not a Bedrock id
            "gemini-2.5-flash",
            "anthropic",  # vendor namespace alone, no model
            "us.unknownvendor.some-model-v1:0",
            "",
        ],
    )
    def test_detect_model_rejects_non_bedrock_ids(self, model: str) -> None:
        assert BedrockAdapter().detect_model(model) is False


@pytest.mark.unit
class TestBedrockAdapterDetectClient:
    def test_detect_boto3_bedrock_runtime_client(self) -> None:
        assert BedrockAdapter().detect_client(_bedrock_client()) is True

    def test_detect_aiobotocore_bedrock_runtime_client(self) -> None:
        client = _bedrock_client(module="aiobotocore.client")
        assert BedrockAdapter().detect_client(client) is True

    def test_rejects_other_aws_service_client(self) -> None:
        client = _bedrock_client(service_name="s3")
        assert BedrockAdapter().detect_client(client) is False

    def test_rejects_bedrock_control_plane_client(self) -> None:
        # The "bedrock" control-plane client (ListFoundationModels etc.) is not
        # an inference client; wrapping it is a configuration mistake.
        client = _bedrock_client(service_name="bedrock")
        assert BedrockAdapter().detect_client(client) is False

    def test_rejects_non_botocore_client(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai._client"
        assert BedrockAdapter().detect_client(FakeClient()) is False

    def test_rejects_object_without_meta(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "botocore.client"
        assert BedrockAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestBedrockAdapterExtractUsage:
    def test_basic_tokens(self) -> None:
        details = BedrockAdapter().extract_usage(
            _converse_response(input_tokens=1000, output_tokens=500)
        )
        assert details.input_tokens == 1000
        assert details.output_tokens == 500
        assert details.cached_input_tokens == 0
        assert details.cache_creation_5m_tokens == 0
        assert details.cache_creation_1h_tokens == 0

    def test_cache_fields_are_additive(self) -> None:
        # AWS documents the formula: total input tokens =
        # inputTokens + cacheReadInputTokens + cacheWriteInputTokens
        # (inputTokens covers only NON-cached tokens — the Anthropic convention).
        details = BedrockAdapter().extract_usage(
            _converse_response(input_tokens=100, output_tokens=50, cache_read=30, cache_write=20)
        )
        assert details.input_tokens == 150
        assert details.cached_input_tokens == 30
        # Aggregate-only cache writes land in the 5m bucket — Bedrock's default
        # prompt-cache TTL (mirrors the Anthropic adapter's aggregate fallback).
        assert details.cache_creation_5m_tokens == 20
        assert details.cache_creation_1h_tokens == 0

    def test_cache_details_split_write_buckets_by_ttl(self) -> None:
        # TokenUsage.cacheDetails breaks down cache writes by TTL (1h sorted
        # before 5m). Claude Opus/Sonnet/Haiku 4.5 on Bedrock support the 1h
        # tier, which prices differently — the split must be preserved.
        details = BedrockAdapter().extract_usage(
            _converse_response(
                input_tokens=100,
                output_tokens=50,
                cache_read=0,
                cache_write=25,
                cache_details=[
                    {"inputTokens": 15, "ttl": "1h"},
                    {"inputTokens": 10, "ttl": "5m"},
                ],
            )
        )
        assert details.cache_creation_1h_tokens == 15
        assert details.cache_creation_5m_tokens == 10
        assert details.input_tokens == 125

    def test_cache_details_partial_preserves_aggregate(self) -> None:
        # If cacheDetails only itemizes the 1h share, the remainder of the
        # aggregate cacheWriteInputTokens stays in the 5m bucket so the total
        # write count is never lost.
        details = BedrockAdapter().extract_usage(
            _converse_response(
                input_tokens=10,
                output_tokens=5,
                cache_write=20,
                cache_details=[{"inputTokens": 8, "ttl": "1h"}],
            )
        )
        assert details.cache_creation_1h_tokens == 8
        assert details.cache_creation_5m_tokens == 12
        assert details.input_tokens == 30

    def test_cache_fields_absent(self) -> None:
        # Model families without prompt caching omit the cache keys entirely.
        details = BedrockAdapter().extract_usage(
            _converse_response(input_tokens=10, output_tokens=5)
        )
        assert details.input_tokens == 10
        assert details.cached_input_tokens == 0

    def test_none_valued_usage_fields_default_to_zero(self) -> None:
        response = _converse_response(input_tokens=10, output_tokens=5)
        response["usage"]["cacheReadInputTokens"] = None
        details = BedrockAdapter().extract_usage(response)
        assert details.input_tokens == 10
        assert details.cached_input_tokens == 0

    def test_missing_usage_returns_zeros(self) -> None:
        response = _converse_response()
        del response["usage"]
        details = BedrockAdapter().extract_usage(response)
        assert details.input_tokens == 0
        assert details.output_tokens == 0

    def test_non_mapping_response_returns_zeros(self) -> None:
        details = BedrockAdapter().extract_usage(object())
        assert details.input_tokens == 0
        assert details.output_tokens == 0

    def test_non_mapping_usage_returns_zeros(self) -> None:
        response = _converse_response()
        response["usage"] = "garbage"
        details = BedrockAdapter().extract_usage(response)
        assert details.input_tokens == 0


@pytest.mark.unit
class TestBedrockAdapterServiceTierAndRegion:
    def test_service_tier_from_service_tier_field(self) -> None:
        # Converse responses echo serviceTier ({"type": ...}); tiers price
        # differently so the API can reprice per tier.
        response = _converse_response(service_tier="flex")
        assert BedrockAdapter().extract_service_tier(response) == "flex"

    def test_service_tier_falls_back_to_performance_config(self) -> None:
        # Latency-optimized inference also prices differently; when no
        # serviceTier is echoed, performanceConfig.latency is the tier signal.
        response = _converse_response(performance_latency="optimized")
        assert BedrockAdapter().extract_service_tier(response) == "optimized"

    def test_service_tier_prefers_service_tier_over_performance_config(self) -> None:
        response = _converse_response(service_tier="priority", performance_latency="optimized")
        assert BedrockAdapter().extract_service_tier(response) == "priority"

    def test_service_tier_absent_returns_none(self) -> None:
        assert BedrockAdapter().extract_service_tier(_converse_response()) is None

    def test_service_tier_non_string_returns_none(self) -> None:
        response = _converse_response()
        response["performanceConfig"] = {"latency": 123}
        assert BedrockAdapter().extract_service_tier(response) is None

    def test_service_tier_overlong_is_truncated(self) -> None:
        response = _converse_response(performance_latency="x" * (SERVICE_TIER_MAX_LENGTH + 10))
        tier = BedrockAdapter().extract_service_tier(response)
        assert tier == "x" * SERVICE_TIER_MAX_LENGTH

    def test_extract_region_from_client_meta(self) -> None:
        assert BedrockAdapter().extract_region(_bedrock_client(region_name="eu-west-1")) == (
            "eu-west-1"
        )

    def test_extract_region_missing_meta_returns_none(self) -> None:
        assert BedrockAdapter().extract_region(object()) is None

    def test_extract_region_non_string_returns_none(self) -> None:
        client = _bedrock_client()
        client.meta.region_name = None
        assert BedrockAdapter().extract_region(client) is None


@pytest.mark.unit
class TestBedrockAdapterPrepareStreaming:
    def test_returns_copy_without_mutation(self) -> None:
        kwargs = {"model": "amazon.nova-pro-v1:0", "messages": []}
        prepared = BedrockAdapter().prepare_streaming(kwargs)
        assert prepared == kwargs
        assert prepared is not kwargs


@pytest.mark.unit
class TestBedrockStreamAccumulator:
    def test_accumulates_usage_from_metadata_event(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe({"messageStart": {"role": "assistant"}})
        acc.observe({"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}})
        acc.observe({"messageStop": {"stopReason": "end_turn"}})
        acc.observe(_metadata_event(input_tokens=12, output_tokens=7))

        details = acc.finalize()
        assert details.input_tokens == 12
        assert details.output_tokens == 7

    def test_cache_fields_from_metadata_event(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe(
            _metadata_event(input_tokens=100, output_tokens=10, cache_read=40, cache_write=15)
        )

        details = acc.finalize()
        assert details.input_tokens == 155
        assert details.cached_input_tokens == 40
        assert details.cache_creation_5m_tokens == 15

    def test_finalize_without_metadata_returns_zeros_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Degradation is EXPLICIT: an abandoned/aborted stream that never saw
        # the terminal metadata event settles at zeros with a warning, never
        # silently wrong numbers.
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe({"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}})

        with caplog.at_level(logging.WARNING):
            details = acc.finalize()

        assert details.input_tokens == 0
        assert details.output_tokens == 0
        assert any("metadata" in rec.message for rec in caplog.records)

    def test_finalize_on_empty_stream_returns_zeros_without_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        with caplog.at_level(logging.WARNING):
            details = acc.finalize()
        assert details.input_tokens == 0
        assert not caplog.records

    def test_finalize_with_usageless_metadata_event_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A terminal metadata event that arrives WITHOUT a usage block must still
        # settle at zeros LOUDLY, not silently. Regression: _saw_event was set
        # only on non-metadata events, so a metadata-only stream (or a metadata
        # event missing usage) returned zero tokens with no warning — a silent
        # budget undercount that violates the "never silently wrong" invariant.
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe({"metadata": {"metrics": {"latencyMs": 42}}})

        with caplog.at_level(logging.WARNING):
            details = acc.finalize()

        assert details.input_tokens == 0
        assert details.output_tokens == 0
        assert any("metadata" in rec.message for rec in caplog.records)

    def test_observe_non_mapping_chunk_does_not_raise(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe(object())
        acc.observe(None)
        assert acc.finalize().input_tokens == 0

    def test_get_service_tier_from_metadata_event(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe(_metadata_event(performance_latency="optimized"))
        assert acc.get_service_tier() == "optimized"

    def test_get_service_tier_prefers_service_tier_field(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        acc.observe(_metadata_event(service_tier="flex", performance_latency="optimized"))
        assert acc.get_service_tier() == "flex"

    def test_get_service_tier_without_metadata_returns_none(self) -> None:
        acc = BedrockAdapter().create_stream_accumulator()
        assert acc.get_service_tier() is None
