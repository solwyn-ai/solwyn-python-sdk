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


def _modality_bucket(modality: Any, token_count: int) -> Any:
    """One ModalityTokenCount entry (snake_case ``modality`` / ``token_count``)."""
    return SimpleNamespace(modality=modality, token_count=token_count)


def _google_response(
    *,
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    thoughts_token_count: int | None = None,
    cached_content_token_count: int | None = None,
    tool_use_prompt_token_count: int | None = None,
    prompt_tokens_details: list[Any] | None = None,
    candidates_tokens_details: list[Any] | None = None,
    include_usage: bool = True,
) -> Any:
    """Build a fake Google GenerateContentResponse.

    By default includes usage_metadata. Set include_usage=False to simulate
    responses without usage_metadata (returns all zeros).

    Optional fields are absent from the namespace when not provided, simulating
    older or simpler API responses that omit them entirely. The
    ``prompt_tokens_details`` / ``candidates_tokens_details`` lists (per-modality
    ModalityTokenCount buckets) are likewise absent unless supplied.
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
    if prompt_tokens_details is not None:
        kwargs["prompt_tokens_details"] = prompt_tokens_details
    if candidates_tokens_details is not None:
        kwargs["candidates_tokens_details"] = candidates_tokens_details

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
        assert GoogleAdapter().detect_model("gemini-3.1-flash-lite") is True

    def test_detect_model_does_not_match_gpt(self) -> None:
        assert GoogleAdapter().detect_model("gpt-5.5") is False

    def test_detect_model_does_not_match_claude(self) -> None:
        assert GoogleAdapter().detect_model("claude-sonnet-5") is False

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
class TestGoogleAdapterModalityBuckets:
    """usageMetadata per-modality buckets map to image/audio token fields.

    ``prompt_tokens_details`` (input side) and ``candidates_tokens_details``
    (output side) are lists of ModalityTokenCount {modality, token_count}. IMAGE
    buckets map to image_input/image_output; AUDIO buckets to
    audio_input/audio_output. TEXT buckets need no field (they are the remainder).
    """

    def test_no_details_leaves_modality_fields_zero(self) -> None:
        # A plain chat response with no per-modality details: fields stay 0
        # exactly as before this mapping existed.
        response = _google_response(prompt_token_count=100, candidates_token_count=50)
        result = GoogleAdapter().extract_usage(response)
        assert result.image_input_tokens == 0
        assert result.image_output_tokens == 0
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0

    def test_image_output_bucket_maps_to_image_output_tokens(self) -> None:
        # gemini-3-pro-image on the chat path: candidates carry an IMAGE bucket so
        # the server prices the image output at the image rate.
        response = _google_response(
            prompt_token_count=20,
            candidates_token_count=1300,
            candidates_tokens_details=[
                _modality_bucket("TEXT", 4),
                _modality_bucket("IMAGE", 1290),
            ],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.image_output_tokens == 1290
        assert result.image_input_tokens == 0
        # Total output is unchanged — the bucket is a SUBSET of output_tokens.
        assert result.output_tokens == 1300

    def test_image_input_bucket_maps_to_image_input_tokens(self) -> None:
        # A multimodal prompt with image parts: prompt details carry an IMAGE bucket.
        response = _google_response(
            prompt_token_count=558,
            candidates_token_count=30,
            prompt_tokens_details=[
                _modality_bucket("TEXT", 300),
                _modality_bucket("IMAGE", 258),
            ],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.image_input_tokens == 258
        assert result.image_output_tokens == 0
        assert result.input_tokens == 558

    def test_audio_buckets_map_to_audio_fields_both_sides(self) -> None:
        response = _google_response(
            prompt_token_count=400,
            candidates_token_count=200,
            prompt_tokens_details=[_modality_bucket("AUDIO", 150)],
            candidates_tokens_details=[_modality_bucket("AUDIO", 80)],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.audio_input_tokens == 150
        assert result.audio_output_tokens == 80

    def test_enum_like_modality_is_read_via_value(self) -> None:
        # MediaModality is a str-enum; a non-str enum-like object exposing .value
        # must still be recognized (duck-typed, no provider SDK import).
        response = _google_response(
            prompt_token_count=10,
            candidates_token_count=500,
            candidates_tokens_details=[_modality_bucket(SimpleNamespace(value="IMAGE"), 480)],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.image_output_tokens == 480

    def test_multiple_same_modality_buckets_sum(self) -> None:
        response = _google_response(
            prompt_token_count=10,
            candidates_token_count=300,
            candidates_tokens_details=[
                _modality_bucket("IMAGE", 100),
                _modality_bucket("IMAGE", 150),
            ],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.image_output_tokens == 250

    def test_garbage_bucket_count_is_skipped(self) -> None:
        # A malformed entry (non-int / negative / bool count) contributes 0 and
        # never raises out of extraction.
        response = _google_response(
            prompt_token_count=10,
            candidates_token_count=300,
            candidates_tokens_details=[
                _modality_bucket("IMAGE", 200),
                SimpleNamespace(modality="IMAGE", token_count="oops"),
                SimpleNamespace(modality="IMAGE"),  # no token_count at all
            ],
        )
        result = GoogleAdapter().extract_usage(response)
        assert result.image_output_tokens == 200

    def test_non_list_details_ignored(self) -> None:
        response = _google_response(
            prompt_token_count=10,
            candidates_token_count=50,
            candidates_tokens_details=[],
        )
        # Empty list -> no buckets, fields stay 0.
        result = GoogleAdapter().extract_usage(response)
        assert result.image_output_tokens == 0


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
                generate_images=lambda **kw: kw,
                generate_videos=lambda **kw: kw,
            )
        )

    def _legacy_client(self) -> Any:
        class LegacyGenerativeModel:
            def __init__(self) -> None:
                self.generate_content = lambda **kwargs: kwargs

        LegacyGenerativeModel.__module__ = "google.generativeai.generative_models"
        return LegacyGenerativeModel()

    def test_prepare_call_strips_stream_and_applies_http_bound(self) -> None:
        client = self._client()
        kwargs: dict[str, Any] = {"model": "gemini-3.5-flash", "stream": True}
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
            client, {"model": "gemini-3.5-flash"}, is_streaming=True, timeout=30.0, max_retries=0
        )

        assert method is client.models.generate_content_stream

    def test_prepare_call_preserves_caller_config_keys(self) -> None:
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": "gemini-3.5-flash",
            "config": {"temperature": 0.2, "http_options": {"headers": {"x-a": "1"}}},
        }

        _, prepared = GoogleAdapter().prepare_call(
            client, kwargs, is_streaming=False, timeout=10.0, max_retries=0
        )

        assert prepared["config"]["temperature"] == 0.2
        assert prepared["config"]["http_options"]["headers"] == {"x-a": "1"}
        assert prepared["config"]["http_options"]["timeout"] == 10000

    def test_prepare_call_shapes_legacy_request_without_mutating_caller_options(self) -> None:
        client = self._legacy_client()
        kwargs: dict[str, Any] = {
            "model": "gemini-1.5-flash",
            "contents": "hi",
            "generation_config": {"temperature": 0.2},
            "stream": True,
            "request_options": {
                "metadata": (("x-test", "1"),),
                "retry": object(),
                "timeout": 999.0,
            },
        }
        original = {
            **kwargs,
            "request_options": dict(kwargs["request_options"]),
        }

        method, prepared = GoogleAdapter().prepare_call(
            client,
            kwargs,
            is_streaming=True,
            timeout=12.5,
            max_retries=2,
        )

        assert method is client.generate_content
        assert "model" not in prepared
        assert prepared["stream"] is True
        assert prepared["generation_config"] == {"temperature": 0.2}
        assert prepared["request_options"] == {
            "metadata": (("x-test", "1"),),
            "retry": None,
            "timeout": 12.5,
        }
        assert prepared["request_options"] is not kwargs["request_options"]
        assert kwargs == original

    def test_stream_shape_seams_are_identity(self) -> None:
        adapter = GoogleAdapter()
        response, wrapper = object(), object()
        assert adapter.unwrap_stream_source(response) is response
        assert adapter.wrap_stream_result(wrapper, response) is wrapper

    def test_prepare_media_call_embeddings_selects_embed_content(self) -> None:
        # embeddings routes to client.models.embed_content. google-genai
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

    def test_prepare_media_call_images_selects_generate_images(self) -> None:
        # images routes to client.models.generate_images (imagen). The
        # per-hop bound rides config.http_options (google-genai has no
        # with_options), injected here as a defensive COPY that preserves the
        # caller's own config keys (e.g. number_of_images).
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": "imagen-3.0-generate-002",
            "prompt": "a cat",
            "config": {"number_of_images": 3},
        }
        original = dict(kwargs)

        method, prepared = GoogleAdapter().prepare_media_call(
            "images", client, kwargs, timeout=12.5, max_retries=2
        )

        assert method is client.models.generate_images
        assert prepared["model"] == "imagen-3.0-generate-002"
        # The caller's config key is preserved alongside the injected http bound.
        assert prepared["config"]["number_of_images"] == 3
        assert prepared["config"]["http_options"]["timeout"] == 12500
        assert prepared["config"]["http_options"]["retry_options"]["attempts"] == 3
        assert kwargs == original  # input never mutated

    def test_prepare_media_call_video_selects_generate_videos(self) -> None:
        # video routes to client.models.generate_videos (veo). The per-hop bound
        # rides config.http_options (google-genai has no with_options), injected
        # here as a defensive COPY that preserves the caller's own config keys
        # (e.g. duration_seconds, resolution).
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": "veo-3.0-generate-001",
            "prompt": "a cat",
            "config": {"duration_seconds": 8, "resolution": "720p"},
        }
        original = {**kwargs, "config": dict(kwargs["config"])}

        method, prepared = GoogleAdapter().prepare_media_call(
            "video", client, kwargs, timeout=12.5, max_retries=2
        )

        assert method is client.models.generate_videos
        assert prepared["model"] == "veo-3.0-generate-001"
        # The caller's config keys are preserved alongside the injected http bound.
        assert prepared["config"]["duration_seconds"] == 8
        assert prepared["config"]["resolution"] == "720p"
        assert prepared["config"]["http_options"]["timeout"] == 12500
        assert prepared["config"]["http_options"]["retry_options"]["attempts"] == 3
        assert kwargs == original  # input never mutated

    def test_prepare_media_call_raises_unsupported_surface_for_unwired(self) -> None:
        # embeddings, images, and video are wired; audio still fails loud with
        # the structural, content-free UnsupportedSurfaceError.
        adapter = GoogleAdapter()
        for surface in ("audio",):
            with pytest.raises(UnsupportedSurfaceError) as excinfo:
                adapter.prepare_media_call(
                    surface, object(), {"model": "m"}, timeout=30.0, max_retries=0
                )
            assert excinfo.value.surface == surface
            assert excinfo.value.provider == "google"
