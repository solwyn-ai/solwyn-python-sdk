"""Tests for Google/Gemini provider adapter — token extraction only, no pricing.

Key subtlety: Google reports candidatesTokenCount (model output) and
thoughtsTokenCount (thinking) as SEPARATE fields. Our normalized
output_tokens = candidates + thoughts.

reasoning_tokens = thoughtsTokenCount (the raw thinking count).
tool_use_input_tokens = toolUsePromptTokenCount.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._token_details import TokenDetails
from solwyn.exceptions import UnsupportedSurfaceError
from solwyn.providers.google import GoogleAdapter

# ---------------------------------------------------------------------------
# Helpers — build fake Google GenerateContentResponse objects
# ---------------------------------------------------------------------------


def _google_response(
    *,
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    thoughts_token_count: int | None = None,
    cached_content_token_count: int | None = None,
    tool_use_prompt_token_count: int | None = None,
    include_usage: bool = True,
) -> Any:
    """Build a fake Google GenerateContentResponse.

    By default includes usage_metadata. Set include_usage=False to simulate
    responses without usage_metadata (returns all zeros).

    Optional fields are absent from the namespace when not provided, simulating
    older or simpler API responses that omit them entirely.
    """
    if not include_usage:
        return SimpleNamespace()

    kwargs: dict[str, Any] = {
        "prompt_token_count": prompt_token_count,
        "candidates_token_count": candidates_token_count,
        "total_token_count": prompt_token_count + candidates_token_count,
    }
    if thoughts_token_count is not None:
        kwargs["thoughts_token_count"] = thoughts_token_count
    if cached_content_token_count is not None:
        kwargs["cached_content_token_count"] = cached_content_token_count
    if tool_use_prompt_token_count is not None:
        kwargs["tool_use_prompt_token_count"] = tool_use_prompt_token_count

    usage_metadata = SimpleNamespace(**kwargs)
    return SimpleNamespace(usage_metadata=usage_metadata)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoogleAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(GoogleAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert GoogleAdapter().name == "google"


@pytest.mark.unit
class TestGoogleAdapterDetect:
    def test_detect_model_gemini_flash(self) -> None:
        assert GoogleAdapter().detect_model("gemini-2.5-flash") is True

    def test_detect_model_gemini_pro(self) -> None:
        assert GoogleAdapter().detect_model("gemini-2.5-pro") is True

    def test_detect_model_gemini_flash_lite(self) -> None:
        assert GoogleAdapter().detect_model("gemini-2.0-flash-lite") is True

    def test_detect_model_does_not_match_gpt(self) -> None:
        assert GoogleAdapter().detect_model("gpt-4o") is False

    def test_detect_model_does_not_match_claude(self) -> None:
        assert GoogleAdapter().detect_model("claude-3-5-sonnet") is False

    def test_detect_client_google_genai_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "google.genai.resources"
        assert GoogleAdapter().detect_client(FakeClient()) is True

    def test_detect_client_google_generativeai_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "google.generativeai"
        assert GoogleAdapter().detect_client(FakeClient()) is True

    def test_detect_client_non_google(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai"
        assert GoogleAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestGoogleAdapterExtractUsage:
    def test_basic_input_tokens(self) -> None:
        response = _google_response(prompt_token_count=1000, candidates_token_count=500)
        result = GoogleAdapter().extract_usage(response)
        assert result.input_tokens == 1000

    def test_output_tokens_without_thinking(self) -> None:
        """When no thoughts_token_count, output_tokens = candidates only."""
        response = _google_response(prompt_token_count=1000, candidates_token_count=500)
        result = GoogleAdapter().extract_usage(response)
        assert result.output_tokens == 500

    def test_output_tokens_normalized_candidates_plus_thoughts(self) -> None:
        """Critical: candidatesTokenCount does NOT include thoughtsTokenCount.
        Normalized output_tokens = candidates + thoughts.
        """
        response = _google_response(
            prompt_token_count=1000,
            candidates_token_count=400,
            thoughts_token_count=200,
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.output_tokens == 600  # 400 + 200

    def test_reasoning_tokens_from_thoughts(self) -> None:
        """thoughts_token_count maps to reasoning_tokens."""
        response = _google_response(
            prompt_token_count=1000,
            candidates_token_count=400,
            thoughts_token_count=150,
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.reasoning_tokens == 150

    def test_reasoning_tokens_zero_when_absent(self) -> None:
        """No thoughts field → reasoning_tokens = 0."""
        response = _google_response(prompt_token_count=1000, candidates_token_count=500)
        result = GoogleAdapter().extract_usage(response)
        assert result.reasoning_tokens == 0

    def test_cached_input_tokens_from_cached_content(self) -> None:
        """cached_content_token_count maps to cached_input_tokens."""
        response = _google_response(
            prompt_token_count=2000,
            candidates_token_count=500,
            cached_content_token_count=800,
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.cached_input_tokens == 800

    def test_cached_input_tokens_zero_when_absent(self) -> None:
        """No cached_content_token_count → cached_input_tokens = 0."""
        response = _google_response(prompt_token_count=1000, candidates_token_count=500)
        result = GoogleAdapter().extract_usage(response)
        assert result.cached_input_tokens == 0

    def test_tool_use_input_tokens_from_tool_use_prompt(self) -> None:
        """tool_use_prompt_token_count maps to tool_use_input_tokens."""
        response = _google_response(
            prompt_token_count=1000,
            candidates_token_count=300,
            tool_use_prompt_token_count=120,
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 120

    def test_tool_use_input_tokens_zero_when_absent(self) -> None:
        """No tool_use_prompt_token_count → tool_use_input_tokens = 0."""
        response = _google_response(prompt_token_count=1000, candidates_token_count=300)
        result = GoogleAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 0

    def test_full_response_all_fields(self) -> None:
        """All optional fields present — full normalization check."""
        response = _google_response(
            prompt_token_count=2000,
            candidates_token_count=400,
            thoughts_token_count=200,
            cached_content_token_count=500,
            tool_use_prompt_token_count=100,
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.input_tokens == 2000
        assert result.output_tokens == 600  # 400 candidates + 200 thoughts
        assert result.reasoning_tokens == 200
        assert result.cached_input_tokens == 500
        assert result.tool_use_input_tokens == 100

    def test_returns_token_details_instance(self) -> None:
        response = _google_response(prompt_token_count=10, candidates_token_count=5)
        result = GoogleAdapter().extract_usage(response)
        assert isinstance(result, TokenDetails)

    def test_audio_tokens_always_zero(self) -> None:
        """Google doesn't expose audio token fields."""
        response = _google_response(prompt_token_count=100, candidates_token_count=50)
        result = GoogleAdapter().extract_usage(response)
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0

    def test_cache_creation_split_tokens_always_zero(self) -> None:
        """Google doesn't have a cache creation token concept — both 5m/1h fields stay 0."""
        response = _google_response(prompt_token_count=100, candidates_token_count=50)
        result = GoogleAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_prediction_tokens_always_zero(self) -> None:
        """Google doesn't have predicted output token fields."""
        response = _google_response(prompt_token_count=100, candidates_token_count=50)
        result = GoogleAdapter().extract_usage(response)
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0


@pytest.mark.unit
class TestGoogleAdapterNoneHandling:
    def test_no_usage_metadata_attr_returns_zeros(self) -> None:
        """When response has no usage_metadata attribute, return all-zero TokenDetails."""
        result = GoogleAdapter().extract_usage(SimpleNamespace())
        assert result == TokenDetails()

    def test_none_usage_metadata_returns_zeros(self) -> None:
        """When usage_metadata is None, return all-zero TokenDetails."""
        response = SimpleNamespace(usage_metadata=None)
        result = GoogleAdapter().extract_usage(response)
        assert result == TokenDetails()


@pytest.mark.unit
class TestGoogleAdapterDispatchSeams:
    def _client(self) -> Any:
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: kw,
                generate_content_stream=lambda **kw: kw,
                embed_content=lambda **kw: kw,
            )
        )

    def test_prepare_call_strips_stream_and_applies_http_bound(self) -> None:
        client = self._client()
        kwargs: dict[str, Any] = {"model": "gemini-2.0-flash", "stream": True}
        original = dict(kwargs)

        method, prepared = GoogleAdapter().prepare_call(
            client, kwargs, is_streaming=False, timeout=12.5, max_retries=2
        )

        assert method is client.models.generate_content
        assert "stream" not in prepared
        # google-genai timeouts are milliseconds; attempts = retries + 1.
        assert prepared["config"]["http_options"]["timeout"] == 12500
        assert prepared["config"]["http_options"]["retry_options"]["attempts"] == 3
        assert kwargs == original  # input never mutated

    def test_prepare_call_streaming_selects_generate_content_stream(self) -> None:
        client = self._client()

        method, _ = GoogleAdapter().prepare_call(
            client, {"model": "gemini-2.0-flash"}, is_streaming=True, timeout=30.0, max_retries=0
        )

        assert method is client.models.generate_content_stream

    def test_prepare_call_preserves_caller_config_keys(self) -> None:
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": "gemini-2.0-flash",
            "config": {"temperature": 0.2, "http_options": {"headers": {"x-a": "1"}}},
        }

        _, prepared = GoogleAdapter().prepare_call(
            client, kwargs, is_streaming=False, timeout=10.0, max_retries=0
        )

        assert prepared["config"]["temperature"] == 0.2
        assert prepared["config"]["http_options"]["headers"] == {"x-a": "1"}
        assert prepared["config"]["http_options"]["timeout"] == 10000

    def test_stream_shape_seams_are_identity(self) -> None:
        adapter = GoogleAdapter()
        response, wrapper = object(), object()
        assert adapter.unwrap_stream_source(response) is response
        assert adapter.wrap_stream_result(wrapper, response) is wrapper

    def test_prepare_media_call_embeddings_selects_embed_content(self) -> None:
        # P1.8: embeddings routes to client.models.embed_content. google-genai
        # has no with_options, so the per-hop bound rides config.http_options —
        # injected here (like prepare_call) as a defensive COPY of kwargs.
        client = self._client()
        kwargs: dict[str, Any] = {"model": "gemini-embedding-001", "contents": "hi"}
        original = dict(kwargs)

        method, prepared = GoogleAdapter().prepare_media_call(
            "embeddings", client, kwargs, timeout=12.5, max_retries=2
        )

        assert method is client.models.embed_content
        assert prepared["model"] == "gemini-embedding-001"
        assert prepared["contents"] == "hi"
        # google-genai timeouts are milliseconds; attempts = retries + 1.
        assert prepared["config"]["http_options"]["timeout"] == 12500
        assert prepared["config"]["http_options"]["retry_options"]["attempts"] == 3
        assert kwargs == original  # input never mutated

    def test_prepare_media_call_raises_unsupported_surface_for_unwired(self) -> None:
        # Only embeddings is wired (P1.8); images/audio/video still fail loud
        # with the structural, content-free UnsupportedSurfaceError.
        adapter = GoogleAdapter()
        for surface in ("images", "audio", "video"):
            with pytest.raises(UnsupportedSurfaceError) as excinfo:
                adapter.prepare_media_call(
                    surface, object(), {"model": "m"}, timeout=30.0, max_retries=0
                )
            assert excinfo.value.surface == surface
            assert excinfo.value.provider == "google"
