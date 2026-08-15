"""Tests for OpenAI provider adapter — token extraction only, no pricing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._token_details import TokenDetails
from solwyn.exceptions import UnsupportedSurfaceError
from solwyn.providers.openai import (
    _AUDIO_OP_KEY,
    _IMAGE_OP_KEY,
    OpenAIAdapter,
    _extract_image_usage,
)

# ---------------------------------------------------------------------------
# Helpers — build fake OpenAI response objects
# ---------------------------------------------------------------------------

_ABSENT = object()


def _chat_response(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    audio_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_output_tokens: int = 0,
    accepted_prediction_tokens: int = 0,
    rejected_prediction_tokens: int = 0,
    cache_write_tokens: object = _ABSENT,
    include_details: bool = True,
) -> Any:
    """Build a fake Chat Completions API response (prompt_tokens naming)."""
    if include_details:
        prompt_details = SimpleNamespace(
            cached_tokens=cached_tokens,
            audio_tokens=audio_input_tokens,
        )
        if cache_write_tokens is not _ABSENT:
            prompt_details.cache_write_tokens = cache_write_tokens
        completion_details = SimpleNamespace(
            reasoning_tokens=reasoning_tokens,
            audio_tokens=audio_output_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
        )
    else:
        prompt_details = None
        completion_details = None

    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        )
    )


def _responses_api_response(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    audio_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_output_tokens: int = 0,
    accepted_prediction_tokens: int = 0,
    rejected_prediction_tokens: int = 0,
    cache_write_tokens: object = _ABSENT,
) -> Any:
    """Build a fake Responses API response with all token sub-fields."""
    input_details = SimpleNamespace(
        cached_tokens=cached_tokens,
        audio_tokens=audio_input_tokens,
    )
    if cache_write_tokens is not _ABSENT:
        input_details.cache_write_tokens = cache_write_tokens

    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=input_details,
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens,
                audio_tokens=audio_output_tokens,
                accepted_prediction_tokens=accepted_prediction_tokens,
                rejected_prediction_tokens=rejected_prediction_tokens,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAIAdapterProtocol:
    def test_satisfies_provider_adapter_protocol(self) -> None:
        from solwyn.providers._protocol import ProviderAdapter

        assert isinstance(OpenAIAdapter(), ProviderAdapter)

    def test_name(self) -> None:
        assert OpenAIAdapter().name == "openai"


@pytest.mark.unit
class TestOpenAIAdapterDetect:
    def test_detect_model_gpt(self) -> None:
        assert OpenAIAdapter().detect_model("gpt-5.5") is True

    def test_detect_model_gpt_4(self) -> None:
        assert OpenAIAdapter().detect_model("gpt-4") is True

    def test_detect_model_o3(self) -> None:
        assert OpenAIAdapter().detect_model("o3-mini") is True

    def test_detect_model_o4(self) -> None:
        assert OpenAIAdapter().detect_model("o4-mini") is True

    def test_detect_model_does_not_match_claude(self) -> None:
        assert OpenAIAdapter().detect_model("claude-sonnet-5") is False

    def test_detect_model_does_not_match_gemini(self) -> None:
        assert OpenAIAdapter().detect_model("gemini-2.5-flash") is False

    def test_detect_client_openai_module(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "openai.resources"
        assert OpenAIAdapter().detect_client(FakeClient()) is True

    def test_detect_client_non_openai(self) -> None:
        class FakeClient:
            pass

        FakeClient.__module__ = "anthropic"
        assert OpenAIAdapter().detect_client(FakeClient()) is False


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageChatCompletions:
    """Chat Completions API uses prompt_tokens / completion_tokens naming."""

    def test_basic_tokens(self) -> None:
        response = _chat_response(prompt_tokens=1000, completion_tokens=500)
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_cached_tokens(self) -> None:
        response = _chat_response(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=400,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.cached_input_tokens == 400

    def test_flat_cached_tokens_when_prompt_details_absent(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=500,
                cached_tokens=400,
            )
        )

        result = OpenAIAdapter().extract_usage(response)

        assert result.cached_input_tokens == 400

    @pytest.mark.parametrize("nested_cached_tokens", [0, None])
    def test_prompt_details_cached_tokens_wins_over_flat_fallback(
        self, nested_cached_tokens: int | None
    ) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=500,
                cached_tokens=400,
                prompt_tokens_details=SimpleNamespace(cached_tokens=nested_cached_tokens),
            )
        )

        result = OpenAIAdapter().extract_usage(response)

        assert result.cached_input_tokens == 0

    def test_audio_input_tokens(self) -> None:
        response = _chat_response(prompt_tokens=100, audio_input_tokens=50)
        result = OpenAIAdapter().extract_usage(response)
        assert result.audio_input_tokens == 50

    def test_reasoning_tokens(self) -> None:
        response = _chat_response(completion_tokens=300, reasoning_tokens=100)
        result = OpenAIAdapter().extract_usage(response)
        assert result.reasoning_tokens == 100

    def test_audio_output_tokens(self) -> None:
        response = _chat_response(completion_tokens=200, audio_output_tokens=80)
        result = OpenAIAdapter().extract_usage(response)
        assert result.audio_output_tokens == 80

    def test_accepted_prediction_tokens(self) -> None:
        response = _chat_response(accepted_prediction_tokens=150)
        result = OpenAIAdapter().extract_usage(response)
        assert result.accepted_prediction_tokens == 150

    def test_rejected_prediction_tokens(self) -> None:
        response = _chat_response(rejected_prediction_tokens=25)
        result = OpenAIAdapter().extract_usage(response)
        assert result.rejected_prediction_tokens == 25

    def test_missing_detail_sub_objects_returns_zeros(self) -> None:
        """When prompt_tokens_details / completion_tokens_details are None, use zeros."""
        response = _chat_response(
            prompt_tokens=500,
            completion_tokens=200,
            include_details=False,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        assert result.cached_input_tokens == 0
        assert result.reasoning_tokens == 0
        assert result.audio_input_tokens == 0
        assert result.audio_output_tokens == 0
        assert result.accepted_prediction_tokens == 0
        assert result.rejected_prediction_tokens == 0

    def test_returns_token_details_instance(self) -> None:
        response = _chat_response(prompt_tokens=10, completion_tokens=5)
        result = OpenAIAdapter().extract_usage(response)
        assert isinstance(result, TokenDetails)

    def test_cache_write_tokens_absent_stays_zero(self) -> None:
        response = _chat_response(prompt_tokens=100, cached_tokens=50)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_cache_write_tokens_map_to_existing_5m_bucket(self) -> None:
        response = _chat_response(prompt_tokens=100, cache_write_tokens=25)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 25
        assert result.cache_creation_1h_tokens == 0

    @pytest.mark.parametrize(
        "cache_write_tokens",
        [None, -1, True, "25", 25.0, object()],
    )
    def test_unusable_cache_write_tokens_degrade_to_zero(self, cache_write_tokens: object) -> None:
        response = _chat_response(
            prompt_tokens=100,
            cache_write_tokens=cache_write_tokens,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_tool_use_input_tokens_always_zero(self) -> None:
        """OpenAI doesn't report tool_use_input_tokens — field stays 0."""
        response = _chat_response(prompt_tokens=100)
        result = OpenAIAdapter().extract_usage(response)
        assert result.tool_use_input_tokens == 0


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageResponsesAPI:
    """Responses API uses input_tokens / output_tokens naming."""

    def test_basic_tokens(self) -> None:
        response = _responses_api_response(input_tokens=800, output_tokens=300)
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 800
        assert result.output_tokens == 300

    def test_cached_tokens(self) -> None:
        response = _responses_api_response(input_tokens=800, cached_tokens=200)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cached_input_tokens == 200

    def test_cache_write_tokens_absent_stays_zero(self) -> None:
        response = _responses_api_response(input_tokens=800)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_cache_write_tokens_map_to_existing_5m_bucket(self) -> None:
        response = _responses_api_response(input_tokens=800, cache_write_tokens=200)
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 200
        assert result.cache_creation_1h_tokens == 0

    @pytest.mark.parametrize(
        "cache_write_tokens",
        [None, -1, True, "200", 200.0, object()],
    )
    def test_unusable_cache_write_tokens_degrade_to_zero(self, cache_write_tokens: object) -> None:
        response = _responses_api_response(
            input_tokens=800,
            cache_write_tokens=cache_write_tokens,
        )
        result = OpenAIAdapter().extract_usage(response)
        assert result.cache_creation_5m_tokens == 0
        assert result.cache_creation_1h_tokens == 0

    def test_reasoning_tokens(self) -> None:
        response = _responses_api_response(output_tokens=500, reasoning_tokens=150)
        result = OpenAIAdapter().extract_usage(response)
        assert result.reasoning_tokens == 150

    def test_responses_api_extracts_audio_and_prediction_tokens(self) -> None:
        """All 8 token sub-fields surface from Responses API — parity with Chat Completions."""
        resp = _responses_api_response(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
            audio_input_tokens=50,
            reasoning_tokens=300,
            audio_output_tokens=20,
            accepted_prediction_tokens=10,
            rejected_prediction_tokens=5,
        )
        details = OpenAIAdapter().extract_usage(resp)
        assert details.input_tokens == 1000
        assert details.output_tokens == 500
        assert details.cached_input_tokens == 200
        assert details.audio_input_tokens == 50
        assert details.reasoning_tokens == 300
        assert details.audio_output_tokens == 20
        assert details.accepted_prediction_tokens == 10
        assert details.rejected_prediction_tokens == 5


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageNoneHandling:
    def test_none_usage_returns_zeros(self) -> None:
        """When response.usage is None, return all-zero TokenDetails."""
        response = SimpleNamespace(usage=None)
        result = OpenAIAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_no_usage_attr_returns_zeros(self) -> None:
        """When response has no usage attribute, return all-zero TokenDetails."""
        result = OpenAIAdapter().extract_usage(SimpleNamespace())
        assert result == TokenDetails()


@pytest.mark.unit
class TestOpenAIAdapterExtractUsageGarbageValues:
    """Value-level garbage must degrade to 0, never raise (never-raise contract;
    TokenDetails fields are ge=0, so a passed-through negative would
    ValidationError out of paths that sit after breaker success accounting)."""

    def test_chat_completions_negative_counts_degrade_to_zeros(self) -> None:
        response = _chat_response(prompt_tokens=-1, completion_tokens=-5)
        result = OpenAIAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_responses_api_negative_counts_degrade_to_zeros(self) -> None:
        response = _responses_api_response(input_tokens=-1, output_tokens=-2)
        result = OpenAIAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_non_int_counts_degrade_to_zeros(self) -> None:
        """`'abc' or 0` would pass the string through; the coercion must not."""
        response = _chat_response()
        response.usage.prompt_tokens = "abc"
        response.usage.completion_tokens = True  # bool is not a count
        result = OpenAIAdapter().extract_usage(response)
        assert result == TokenDetails()

    def test_negative_detail_fields_degrade_but_valid_totals_survive(self) -> None:
        response = _chat_response(prompt_tokens=10, completion_tokens=5, cached_tokens=-3)
        result = OpenAIAdapter().extract_usage(response)
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.cached_input_tokens == 0


@pytest.mark.unit
class TestOpenAIServiceTier:
    """OpenAI-only service_tier extraction."""

    @pytest.mark.parametrize("tier", ["priority", "flex", "batch", "default"])
    def test_openai_extract_service_tier_known_values(self, tier: str) -> None:
        resp = SimpleNamespace(
            service_tier=tier,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        assert OpenAIAdapter().extract_service_tier(resp) == tier

    def test_openai_extract_service_tier_absent_returns_none(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        assert OpenAIAdapter().extract_service_tier(resp) is None

    def test_openai_extract_service_tier_none_returns_none(self) -> None:
        resp = SimpleNamespace(
            service_tier=None,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        assert OpenAIAdapter().extract_service_tier(resp) is None

    def test_openai_extract_service_tier_non_string_returns_none(self) -> None:
        resp = SimpleNamespace(
            service_tier=42,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        assert OpenAIAdapter().extract_service_tier(resp) is None

    def test_openai_extract_service_tier_truncates_to_event_limit(self) -> None:
        from solwyn._types import SERVICE_TIER_MAX_LENGTH

        tier = "x" * (SERVICE_TIER_MAX_LENGTH + 1)
        resp = SimpleNamespace(
            service_tier=tier,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        assert OpenAIAdapter().extract_service_tier(resp) == "x" * SERVICE_TIER_MAX_LENGTH


@pytest.mark.unit
class TestOpenAIAdapterDispatchSeams:
    def test_prepare_call_selects_chat_completions_create(self) -> None:
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        kwargs: dict[str, Any] = {"model": "gpt-5.5"}

        method, prepared = OpenAIAdapter().prepare_call(
            client, kwargs, is_streaming=False, timeout=30.0, max_retries=0
        )

        assert method is create
        assert prepared == {"model": "gpt-5.5"}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_call_streaming_sets_stream_kwarg_without_mutation(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: kw))
        )
        kwargs: dict[str, Any] = {"model": "gpt-5.5"}

        _, prepared = OpenAIAdapter().prepare_call(
            client, kwargs, is_streaming=True, timeout=30.0, max_retries=0
        )

        assert prepared["stream"] is True
        assert "stream" not in kwargs

    def test_stream_shape_seams_are_identity(self) -> None:
        adapter = OpenAIAdapter()
        response, wrapper = object(), object()
        assert adapter.unwrap_stream_source(response) is response
        assert adapter.wrap_stream_result(wrapper, response) is wrapper

    def test_prepare_media_call_embeddings_selects_embeddings_create(self) -> None:
        # embeddings routes to client.embeddings.create with a COPY of
        # kwargs — the same (method, shaped_kwargs) shape prepare_call gives chat.
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        kwargs: dict[str, Any] = {"model": "text-embedding-3-small", "input": "hi"}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "embeddings", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is create
        assert prepared == {"model": "text-embedding-3-small", "input": "hi"}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_media_call_images_selects_generate_by_default(self) -> None:
        # images routes to client.images.generate with a COPY of kwargs.
        def generate(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(images=SimpleNamespace(generate=generate, edit=lambda **k: k))
        kwargs: dict[str, Any] = {"model": "gpt-image-2", "prompt": "a cat", "n": 2}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "images", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is generate
        assert prepared == {"model": "gpt-image-2", "prompt": "a cat", "n": 2}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_media_call_images_edit_marker_selects_edit_and_is_stripped(self) -> None:
        # The private op marker routes edit vs generate and is NEVER sent to the SDK.
        def edit(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(images=SimpleNamespace(generate=lambda **k: k, edit=edit))
        kwargs: dict[str, Any] = {"model": "gpt-image-2", "prompt": "a cat", _IMAGE_OP_KEY: "edit"}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "images", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is edit
        assert _IMAGE_OP_KEY not in prepared  # marker stripped before the SDK call
        assert prepared == {"model": "gpt-image-2", "prompt": "a cat"}

    def test_prepare_media_call_audio_selects_transcriptions_create(self) -> None:
        # audio routes to client.audio.transcriptions.create by default (no op
        # marker) with a COPY of kwargs.
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
        )
        kwargs: dict[str, Any] = {"model": "whisper-1", "file": b"audio-bytes"}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "audio", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is create
        assert prepared == {"model": "whisper-1", "file": b"audio-bytes"}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_media_call_audio_speech_marker_selects_speech_and_is_stripped(self) -> None:
        # The audio op marker routes speech vs transcriptions on the ONE "audio"
        # surface and is NEVER sent to the SDK.
        def speech_create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=lambda **k: k),
                speech=SimpleNamespace(create=speech_create),
            )
        )
        kwargs: dict[str, Any] = {"model": "tts-1", "input": "hello", _AUDIO_OP_KEY: "speech"}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "audio", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is speech_create
        assert _AUDIO_OP_KEY not in prepared  # marker stripped before the SDK call
        assert prepared == {"model": "tts-1", "input": "hello"}

    def test_prepare_media_call_video_routes_to_videos_create(self) -> None:
        # video (Sora) routes to client.videos.create with a defensive COPY of
        # kwargs (never mutates / aliases the caller's dict).
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(videos=SimpleNamespace(create=create))
        kwargs: dict[str, Any] = {"model": "sora-2", "seconds": "8", "size": "1280x720"}

        method, prepared = OpenAIAdapter().prepare_media_call(
            "video", client, kwargs, timeout=30.0, max_retries=0
        )

        assert method is create
        assert prepared == {"model": "sora-2", "seconds": "8", "size": "1280x720"}
        assert prepared is not kwargs  # never mutates / aliases the input

    def test_prepare_media_call_raises_unsupported_surface_for_unwired(self) -> None:
        # embeddings + images + audio + video are wired; an unrecognized surface
        # fails loud with the structural, content-free UnsupportedSurfaceError.
        adapter = OpenAIAdapter()
        for surface in ("translations",):
            with pytest.raises(UnsupportedSurfaceError) as excinfo:
                adapter.prepare_media_call(
                    surface, object(), {"model": "m"}, timeout=30.0, max_retries=0
                )
            assert excinfo.value.surface == surface
            assert excinfo.value.provider == "openai"


@pytest.mark.unit
class TestOpenAIAdapterResponsesDispatch:
    def test_non_streaming_selects_responses_create_with_defensive_copy(self) -> None:
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        kwargs: dict[str, Any] = {"model": "gpt-5.5", "metadata": {"request_id": "req-1"}}

        method, shaped = OpenAIAdapter().prepare_responses_call(client, kwargs, is_streaming=False)

        assert method is create
        assert shaped == kwargs
        assert shaped is not kwargs

    def test_parse_leaf_selects_responses_parse_with_defensive_copy(self) -> None:
        # Arrange.
        def parse(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        kwargs: dict[str, Any] = {
            "model": "gpt-5.5",
            "input": "hello",
            "text_format": dict,
        }

        # Act.
        method, shaped = OpenAIAdapter().prepare_responses_call(
            client,
            kwargs,
            is_streaming=False,
            leaf="parse",
        )

        # Assert.
        assert method is parse
        assert shaped == kwargs
        assert shaped is not kwargs

    def test_unsupported_internal_leaf_fails_loud(self) -> None:
        # Arrange.
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: kwargs))

        # Act and assert.
        with pytest.raises(RuntimeError, match="unsupported OpenAI Responses leaf"):
            OpenAIAdapter().prepare_responses_call(
                client,
                {"model": "gpt-5.5"},
                is_streaming=False,
                leaf="retrieve",
            )

    def test_streaming_adds_stream_true_without_mutating_input(self) -> None:
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: kwargs))
        kwargs: dict[str, Any] = {"model": "gpt-5.5"}

        _, shaped = OpenAIAdapter().prepare_responses_call(client, kwargs, is_streaming=True)

        assert shaped == {"model": "gpt-5.5", "stream": True}
        assert "stream" not in kwargs

    def test_streaming_does_not_inject_stream_options(self) -> None:
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: kwargs))

        _, shaped = OpenAIAdapter().prepare_responses_call(
            client, {"model": "gpt-5.5"}, is_streaming=True
        )

        assert "stream_options" not in shaped


@pytest.mark.unit
class TestExtractImageUsage:
    """gpt-image images.generate/edit usage extraction."""

    def _image_response(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        input_image_tokens: int = 0,
        output_image_tokens: int = 0,
    ) -> Any:
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=SimpleNamespace(
                    image_tokens=input_image_tokens,
                    text_tokens=input_tokens - input_image_tokens,
                ),
                output_tokens_details=SimpleNamespace(
                    image_tokens=output_image_tokens,
                    text_tokens=output_tokens - output_image_tokens,
                ),
            )
        )

    def test_reads_input_output_and_image_buckets(self) -> None:
        # Probe-observed: input 222 = 194 image + 28 text; output all image.
        resp = self._image_response(
            input_tokens=222,
            output_tokens=1024,
            input_image_tokens=194,
            output_image_tokens=1024,
        )
        details = _extract_image_usage(resp)
        assert details is not None
        assert details.input_tokens == 222
        assert details.output_tokens == 1024
        assert details.image_input_tokens == 194  # image ⊂ input
        assert details.image_output_tokens == 1024  # image_output ⊂ output
        assert details.is_estimated is False

    def test_none_usage_returns_none(self) -> None:
        # Compat FLUX returns usage: null -> None so the per-image MediaUsage is
        # the sole billable basis (never a silent $0).
        assert _extract_image_usage(SimpleNamespace(usage=None)) is None

    def test_no_usage_attr_returns_none(self) -> None:
        assert _extract_image_usage(SimpleNamespace()) is None

    def test_zero_token_usage_returns_none(self) -> None:
        # A usage block with zeroed input/output (dall-e-style / empty) yields None.
        resp = self._image_response(input_tokens=0, output_tokens=0)
        assert _extract_image_usage(resp) is None

    def test_missing_detail_buckets_default_to_zero(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=50))
        details = _extract_image_usage(resp)
        assert details is not None
        assert details.input_tokens == 100
        assert details.output_tokens == 50
        assert details.image_input_tokens == 0
        assert details.image_output_tokens == 0

    def test_garbage_counts_degrade_without_raising(self) -> None:
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=-5,
                output_tokens="abc",
                input_tokens_details=SimpleNamespace(image_tokens=None),
                output_tokens_details=None,
            )
        )
        # Both totals degrade to 0 -> no usable usage -> None.
        assert _extract_image_usage(resp) is None
