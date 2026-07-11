"""Tests for OpenAI-compatible provider adapters — token extraction only.

Covers the three things that make a compat provider "supported":
detection (base_url host / local port / class name / explicit name),
streaming-usage policy (include_usage injection vs stripping), and the
cost path (usage extraction with the x_groq and estimated fallbacks).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solwyn._types import ProviderName
from solwyn.exceptions import UnsupportedSurfaceError
from solwyn.providers import get_adapter_by_name, get_adapter_for_client, get_adapter_for_model
from solwyn.providers._protocol import ProviderAdapter
from solwyn.providers.openai import _IMAGE_OP_KEY, OpenAIAdapter
from solwyn.providers.openai_compatible import (
    COMPAT_PROFILES,
    CompatStreamAccumulator,
    OpenAICompatibleAdapter,
    build_compat_adapters,
)
from solwyn.providers.together import TogetherAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    module_path: str = "openai._client",
    base_url: str | None = None,
    class_name: str = "OpenAI",
) -> Any:
    """An instance with controlled __module__/__name__ and a base_url attr."""
    FakeClient = type(class_name, (), {"__module__": module_path})
    client = FakeClient()
    if base_url is not None:
        client.base_url = base_url  # type: ignore[attr-defined]
    return client


def _usage_chunk(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    service_tier: str | None = None,
) -> Any:
    """A final stream chunk carrying a standard usage block."""
    chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            completion_tokens_details=None,
        ),
    )
    if service_tier is not None:
        chunk.service_tier = service_tier
    return chunk


def _text_chunk(text: str) -> Any:
    """A content delta chunk with no usage."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)],
        usage=None,
    )


def _adapter(name: str) -> OpenAICompatibleAdapter:
    adapter = get_adapter_by_name(name)
    assert isinstance(adapter, OpenAICompatibleAdapter)
    return adapter


# ---------------------------------------------------------------------------
# Profile table invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProfileTable:
    def test_profile_names_are_unique(self) -> None:
        names = [p.name for p in COMPAT_PROFILES]
        assert len(names) == len(set(names))

    def test_profile_names_are_provider_name_values(self) -> None:
        """Attribution identity must be wire-valid: every profile name parses."""
        for profile in COMPAT_PROFILES:
            assert ProviderName(profile.name).value == profile.name

    def test_catch_all_is_last_profile(self) -> None:
        """The generic catch-all must never shadow a named profile."""
        assert COMPAT_PROFILES[-1].catch_all is True
        assert all(not p.catch_all for p in COMPAT_PROFILES[:-1])

    def test_all_adapters_satisfy_protocol(self) -> None:
        for adapter in build_compat_adapters():
            assert isinstance(adapter, ProviderAdapter), adapter.name

    def test_all_adapters_speak_openai_dialect(self) -> None:
        for adapter in build_compat_adapters():
            assert adapter.dialect == "openai"

    def test_prepare_media_call_embeddings_selects_embeddings_create_for_every_profile(
        self,
    ) -> None:
        # one embeddings branch covers every compat adapter (incl. the
        # first-class Together adapter, which inherits it) since they share the
        # openai dialect. Each routes to client.embeddings.create with a COPY.
        def create(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        for adapter in build_compat_adapters():
            kwargs: dict[str, Any] = {"model": "m", "input": "hi"}
            method, prepared = adapter.prepare_media_call(
                "embeddings", client, kwargs, timeout=30.0, max_retries=0
            )
            assert method is create, adapter.name
            assert prepared == {"model": "m", "input": "hi"}, adapter.name
            assert prepared is not kwargs, adapter.name  # never mutates / aliases input

    def test_prepare_media_call_images_selects_images_method_for_every_profile(self) -> None:
        # one images branch covers every compat adapter (incl. Together) —
        # generate() by default, edit() via the marker (stripped before the call).
        def generate(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        def edit(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        client = SimpleNamespace(images=SimpleNamespace(generate=generate, edit=edit))
        for adapter in build_compat_adapters():
            gen_method, gen_prepared = adapter.prepare_media_call(
                "images", client, {"model": "flux", "n": 2}, timeout=30.0, max_retries=0
            )
            assert gen_method is generate, adapter.name
            assert gen_prepared == {"model": "flux", "n": 2}, adapter.name

            edit_method, edit_prepared = adapter.prepare_media_call(
                "images",
                client,
                {"model": "flux", "prompt": "x", _IMAGE_OP_KEY: "edit"},
                timeout=30.0,
                max_retries=0,
            )
            assert edit_method is edit, adapter.name
            assert _IMAGE_OP_KEY not in edit_prepared, adapter.name  # marker stripped

    def test_prepare_media_call_raises_unsupported_surface_for_unwired_every_profile(self) -> None:
        # embeddings + images are wired; audio/video still fail loud
        # with the provider's own name attached.
        for adapter in build_compat_adapters():
            for surface in ("audio", "video"):
                with pytest.raises(UnsupportedSurfaceError) as excinfo:
                    adapter.prepare_media_call(
                        surface, object(), {"model": "m"}, timeout=30.0, max_retries=0
                    )
                assert excinfo.value.surface == surface
                assert excinfo.value.provider == adapter.name

    def test_together_profile_slot_uses_first_class_adapter(self) -> None:
        adapters = build_compat_adapters()

        assert [adapter.name for adapter in adapters] == [
            profile.name for profile in COMPAT_PROFILES
        ]
        assert sum(isinstance(adapter, TogetherAdapter) for adapter in adapters) == 1
        assert all(
            isinstance(adapter, TogetherAdapter)
            if adapter.name == "together"
            else type(adapter) is OpenAICompatibleAdapter
            for adapter in adapters
        )


# ---------------------------------------------------------------------------
# Client detection — base_url hosts
# ---------------------------------------------------------------------------

_HOST_CASES = [
    ("xai", "https://api.x.ai/v1"),
    ("deepseek", "https://api.deepseek.com/v1"),
    ("mistral", "https://api.mistral.ai/v1"),
    ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("qwen", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"),
    ("zai", "https://api.z.ai/api/paas/v4"),
    ("zai", "https://API.Z.AI/api/paas/v4"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("together", "https://api.together.xyz/v1"),
    ("together", "https://api.together.ai/v1"),
    ("fireworks", "https://api.fireworks.ai/inference/v1"),
    ("perplexity", "https://api.perplexity.ai"),
    ("azure_openai", "https://my-resource.openai.azure.com/openai"),
    ("azure_openai", "https://my-resource.cognitiveservices.azure.com/openai"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("ollama", "http://localhost:11434/v1"),
    ("ollama", "http://127.0.0.1:11434/v1"),
    ("vllm", "http://localhost:8000/v1"),
    ("lmstudio", "http://localhost:1234/v1"),
]


@pytest.mark.unit
class TestDetectClientByBaseUrl:
    @pytest.mark.parametrize(("expected_name", "base_url"), _HOST_CASES)
    def test_known_host_detects_named_provider(self, expected_name: str, base_url: str) -> None:
        client = _make_client(base_url=base_url)
        adapter = get_adapter_for_client(client)
        assert adapter.name == expected_name

    def test_unknown_remote_host_detects_generic_compat(self) -> None:
        client = _make_client(base_url="https://api.some-new-vendor.example/v1")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai_compatible"

    def test_localhost_nonstandard_port_detects_generic_compat(self) -> None:
        client = _make_client(base_url="http://localhost:9999/v1")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai_compatible"

    def test_openai_default_base_url_stays_openai(self) -> None:
        client = _make_client(base_url="https://api.openai.com/v1")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.name == "openai"

    @pytest.mark.parametrize(
        "base_url",
        ["https://eu.api.openai.com/v1", "https://us.api.openai.com/v1"],
    )
    def test_openai_regional_hosts_stay_openai(self, base_url: str) -> None:
        """OpenAI's data-residency endpoints are OpenAI, not generic compat."""
        client = _make_client(base_url=base_url)
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, OpenAIAdapter)

    def test_missing_base_url_stays_openai(self) -> None:
        client = _make_client(base_url=None)
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, OpenAIAdapter)

    def test_unparseable_base_url_stays_openai(self) -> None:
        """A garbage base_url (e.g. a mock repr) must never crash detection."""
        client = _make_client(base_url="<MagicMock id='4399732496'>")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, OpenAIAdapter)

    def test_non_openai_module_client_never_matches_compat(self) -> None:
        """Non-OpenAI clients never match compat, except native Together clients."""
        client = _make_client(module_path="anthropic._client", base_url="https://api.groq.com/v1")
        for adapter in build_compat_adapters():
            assert adapter.detect_client(client) is False

    def test_azure_detected_by_class_name_without_base_url(self) -> None:
        client = _make_client(class_name="AzureOpenAI", base_url=None)
        adapter = get_adapter_for_client(client)
        assert adapter.name == "azure_openai"

    def test_async_azure_detected_by_class_name(self) -> None:
        client = _make_client(class_name="AsyncAzureOpenAI", base_url=None)
        adapter = get_adapter_for_client(client)
        assert adapter.name == "azure_openai"


# ---------------------------------------------------------------------------
# Model detection — only unambiguous prefixes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectModel:
    @pytest.mark.parametrize(
        ("model", "expected_name"),
        [
            ("grok-4", "xai"),
            ("deepseek-chat", "deepseek"),
            ("mistral-large-latest", "mistral"),
            ("codestral-2405", "mistral"),
            ("qwen-max", "qwen"),
            ("qwq-32b", "qwen"),
            ("glm-4.6", "zai"),
            ("sonar-pro", "perplexity"),
            ("accounts/fireworks/models/llama-v3p1-70b-instruct", "fireworks"),
        ],
    )
    def test_distinctive_prefix_detects_provider(self, model: str, expected_name: str) -> None:
        adapter = get_adapter_for_model(model)
        assert adapter.name == expected_name

    def test_shared_open_weight_model_still_raises(self) -> None:
        """llama et al. are served by many providers — never silently claimed."""
        with pytest.raises(ValueError):
            get_adapter_for_model("llama-3-8b")

    def test_gpt_model_still_detects_openai(self) -> None:
        adapter = get_adapter_for_model("gpt-4o")
        assert adapter.name == "openai"


# ---------------------------------------------------------------------------
# prepare_streaming — the include_usage policy
# ---------------------------------------------------------------------------

_INJECTING = ["deepseek", "qwen", "zai", "groq", "azure_openai", "ollama", "vllm", "lmstudio"]
_NON_INJECTING = [
    "xai",
    "mistral",
    "together",
    "fireworks",
    "perplexity",
    "openrouter",
    "openai_compatible",
]


@pytest.mark.unit
class TestPrepareStreaming:
    @pytest.mark.parametrize("name", _INJECTING)
    def test_supported_providers_inject_include_usage(self, name: str) -> None:
        kwargs = {"model": "m", "messages": []}
        prepared = _adapter(name).prepare_streaming(kwargs)
        assert prepared["stream_options"] == {"include_usage": True}
        assert "stream_options" not in kwargs  # input never mutated

    @pytest.mark.parametrize("name", _NON_INJECTING)
    def test_unsupported_providers_do_not_inject(self, name: str) -> None:
        prepared = _adapter(name).prepare_streaming({"model": "m", "messages": []})
        assert "stream_options" not in prepared

    @pytest.mark.parametrize("name", _NON_INJECTING)
    def test_unsupported_providers_strip_stream_options_on_failover_hop(self, name: str) -> None:
        """A failover hop into a strict-validation provider must not 4xx on a
        stream_options the original target accepted."""
        kwargs = {"model": "m", "messages": [], "stream_options": {"include_usage": True}}
        prepared = _adapter(name).prepare_streaming(kwargs, cross_provider=True)
        assert "stream_options" not in prepared
        assert kwargs["stream_options"] == {"include_usage": True}  # input never mutated

    @pytest.mark.parametrize("name", _NON_INJECTING)
    def test_caller_stream_options_preserved_on_own_target(self, name: str) -> None:
        """Drop-in contract: an option the caller explicitly chose for THEIR
        OWN endpoint is never silently removed (they may know their gateway
        supports it better than our point-in-time profile does)."""
        kwargs = {"model": "m", "messages": [], "stream_options": {"include_usage": True}}
        prepared = _adapter(name).prepare_streaming(kwargs)
        assert prepared["stream_options"] == {"include_usage": True}

    def test_supported_provider_merges_existing_stream_options(self) -> None:
        kwargs = {"stream_options": {"other": 1}}
        prepared = _adapter("groq").prepare_streaming(kwargs)
        assert prepared["stream_options"] == {"other": 1, "include_usage": True}

    def test_azure_skips_injection_for_data_sources(self) -> None:
        """Azure 'on your data' rejects stream_options with a 422."""
        prepared = _adapter("azure_openai").prepare_streaming(
            {"model": "m", "data_sources": [{"type": "azure_search"}]}
        )
        assert "stream_options" not in prepared

    def test_azure_skips_injection_for_extra_body_data_sources(self) -> None:
        prepared = _adapter("azure_openai").prepare_streaming(
            {"model": "m", "extra_body": {"data_sources": [{"type": "azure_search"}]}}
        )
        assert "stream_options" not in prepared

    def test_azure_data_sources_preserves_caller_stream_options_on_own_target(self) -> None:
        """Drop-in contract: even on the data_sources pipeline, an option the
        caller explicitly chose for THEIR OWN Azure endpoint is untouched."""
        prepared = _adapter("azure_openai").prepare_streaming(
            {
                "model": "m",
                "data_sources": [{"type": "azure_search"}],
                "stream_options": {"include_usage": True},
            }
        )
        assert prepared["stream_options"] == {"include_usage": True}

    def test_azure_data_sources_strips_caller_stream_options_on_failover_hop(self) -> None:
        """On a cross-provider hop the caller's stream_options was authored
        for the original target — strip it so the Azure pipeline doesn't 422."""
        prepared = _adapter("azure_openai").prepare_streaming(
            {
                "model": "m",
                "data_sources": [{"type": "azure_search"}],
                "stream_options": {"include_usage": True},
            },
            cross_provider=True,
        )
        assert "stream_options" not in prepared

    def test_non_azure_extra_body_data_sources_key_still_injects(self) -> None:
        """The data_sources caveat is Azure-specific: another injecting profile
        whose extra_body happens to carry that key keeps normal injection."""
        prepared = _adapter("groq").prepare_streaming(
            {"model": "m", "extra_body": {"data_sources": [{"type": "custom"}]}}
        )
        assert prepared["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Usage extraction — non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractUsage:
    def test_standard_usage_block_extracts(self) -> None:
        response = _usage_chunk(prompt_tokens=10, completion_tokens=5)
        details = _adapter("groq").extract_usage(response)
        assert details.input_tokens == 10
        assert details.output_tokens == 5
        assert details.is_estimated is False

    def test_zai_cached_tokens_extract_without_reasoning_tokens(self) -> None:
        response = _usage_chunk(prompt_tokens=10, completion_tokens=5, cached_tokens=4)
        details = _adapter("zai").extract_usage(response)
        assert details.cached_input_tokens == 4
        assert details.reasoning_tokens == 0

    def test_missing_usage_extracts_zeros(self) -> None:
        details = _adapter("groq").extract_usage(SimpleNamespace(usage=None))
        assert details.input_tokens == 0
        assert details.output_tokens == 0

    def test_negative_usage_extracts_zeros_not_raise(self) -> None:
        """Garbage counts (ge=0 would ValidationError) degrade to zeros."""
        response = _usage_chunk(prompt_tokens=-1, completion_tokens=-7)
        details = _adapter("groq").extract_usage(response)
        assert details.input_tokens == 0
        assert details.output_tokens == 0


@pytest.mark.unit
class TestEstimateMissingUsage:
    def test_usage_present_returns_none(self) -> None:
        response = _usage_chunk(prompt_tokens=10, completion_tokens=5)
        assert _adapter("ollama").estimate_missing_usage(response, estimated_input_tokens=7) is None

    def test_provider_reported_zero_usage_is_not_estimated(self) -> None:
        """A usage block with zeros is provider truth, not a gap to estimate."""
        response = _usage_chunk(prompt_tokens=0, completion_tokens=0)
        assert _adapter("ollama").estimate_missing_usage(response, estimated_input_tokens=7) is None

    def test_absent_usage_estimates_and_flags(self) -> None:
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="x" * 40))],
        )
        details = _adapter("ollama").estimate_missing_usage(response, estimated_input_tokens=7)
        assert details is not None
        assert details.is_estimated is True
        assert details.input_tokens == 7
        assert details.output_tokens == 10  # 40 chars / 4.0

    def test_absent_usage_with_no_content_estimates_zero_output(self) -> None:
        response = SimpleNamespace(usage=None, choices=[])
        details = _adapter("ollama").estimate_missing_usage(response, estimated_input_tokens=7)
        assert details is not None
        assert details.is_estimated is True
        assert details.output_tokens == 0

    def test_unparseable_usage_shape_with_content_estimates(self) -> None:
        """A usage block in a foreign shape (e.g. a raw dict from a naive
        gateway) must not settle real output as silent zero spend."""
        response = SimpleNamespace(
            usage={"input_tokens": 512, "output_tokens": 128},  # dict, not attrs
            choices=[SimpleNamespace(message=SimpleNamespace(content="y" * 40))],
        )
        details = _adapter("openai_compatible").estimate_missing_usage(
            response, estimated_input_tokens=9
        )
        assert details is not None
        assert details.is_estimated is True
        assert details.input_tokens == 9
        assert details.output_tokens == 10

    def test_negative_usage_with_content_estimates_not_raise(self) -> None:
        """Negative counts are garbage, not truth — degrade to the flagged
        estimate instead of raising ValidationError out of a never-raise path."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=-1,
                completion_tokens=-7,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content="y" * 40))],
        )
        details = _adapter("openai_compatible").estimate_missing_usage(
            response, estimated_input_tokens=9
        )
        assert details is not None
        assert details.is_estimated is True
        assert details.input_tokens == 9
        assert details.output_tokens == 10

    def test_zeroed_usage_with_real_content_estimates(self) -> None:
        """All-zero usage alongside visible content is garbage, not truth."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content="y" * 40))],
        )
        details = _adapter("openai_compatible").estimate_missing_usage(
            response, estimated_input_tokens=9
        )
        assert details is not None
        assert details.is_estimated is True
        assert details.output_tokens == 10

    def test_tool_call_arguments_count_toward_estimate(self) -> None:
        """Tool-only responses must not estimate to zero output."""
        tool_call = SimpleNamespace(function=SimpleNamespace(arguments="x" * 40))
        response = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))
            ],
        )
        details = _adapter("ollama").estimate_missing_usage(response, estimated_input_tokens=7)
        assert details is not None
        assert details.output_tokens == 10

    def test_reasoning_content_counts_toward_estimate(self) -> None:
        """DeepSeek-style reasoning rides a separate field — a reasoning-only
        response from a usage-less endpoint must not estimate to zero output."""
        response = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None, reasoning_content="r" * 40))
            ],
        )
        details = _adapter("deepseek").estimate_missing_usage(response, estimated_input_tokens=7)
        assert details is not None
        assert details.is_estimated is True
        assert details.input_tokens == 7
        assert details.output_tokens == 10  # 40 chars / 4.0


# ---------------------------------------------------------------------------
# Streaming accumulator — the three usage tiers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompatStreamAccumulator:
    def _accumulator(self, name: str = "groq", est_in: int = 0) -> CompatStreamAccumulator:
        return _adapter(name).create_stream_accumulator(estimated_input_tokens=est_in)

    def test_usage_in_final_chunk_extracts_exactly(self) -> None:
        acc = self._accumulator()
        acc.observe(_text_chunk("Hello"))
        acc.observe(_text_chunk(" world"))
        acc.observe(_usage_chunk(prompt_tokens=12, completion_tokens=4))
        details = acc.finalize()
        assert details.input_tokens == 12
        assert details.output_tokens == 4
        assert details.is_estimated is False

    def test_usage_on_every_chunk_last_wins(self) -> None:
        """Some providers attach (cumulative) usage to every chunk."""
        acc = self._accumulator()
        acc.observe(_usage_chunk(prompt_tokens=12, completion_tokens=1))
        acc.observe(_usage_chunk(prompt_tokens=12, completion_tokens=2))
        acc.observe(_usage_chunk(prompt_tokens=12, completion_tokens=9))
        details = acc.finalize()
        assert details.output_tokens == 9

    def test_usage_in_second_to_last_chunk_still_captured(self) -> None:
        """xAI emits the usage chunk BEFORE the terminal chunk — last-non-None
        must not require usage on the literal final chunk."""
        acc = self._accumulator("xai")
        acc.observe(_text_chunk("hi"))
        acc.observe(_usage_chunk(prompt_tokens=8, completion_tokens=3))
        acc.observe(SimpleNamespace(choices=[], usage=None))  # terminal chunk
        details = acc.finalize()
        assert details.input_tokens == 8
        assert details.output_tokens == 3
        assert details.is_estimated is False

    def test_x_groq_dict_usage_fallback(self) -> None:
        """Groq's legacy final chunk carries usage under x_groq as a raw dict."""
        acc = self._accumulator("groq")
        acc.observe(_text_chunk("hello"))
        final = SimpleNamespace(
            choices=[],
            usage=None,
            x_groq={"id": "req_1", "usage": {"prompt_tokens": 21, "completion_tokens": 6}},
        )
        acc.observe(final)
        details = acc.finalize()
        assert details.input_tokens == 21
        assert details.output_tokens == 6
        assert details.is_estimated is False

    def test_x_groq_attr_usage_fallback(self) -> None:
        """x_groq parsed into an attr object (not a raw dict) still extracts."""
        acc = self._accumulator("groq")
        acc.observe(_text_chunk("hello"))
        final = SimpleNamespace(
            choices=[],
            usage=None,
            x_groq=SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=21, completion_tokens=6, total_tokens=27)
            ),
        )
        acc.observe(final)
        details = acc.finalize()
        assert details.input_tokens == 21
        assert details.output_tokens == 6
        assert details.is_estimated is False

    def test_standard_usage_wins_over_x_groq(self) -> None:
        acc = self._accumulator("groq")
        acc.observe(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                x_groq={"usage": {"prompt_tokens": 99, "completion_tokens": 99}},
            )
        )
        details = acc.finalize()
        assert details.input_tokens == 10
        assert details.output_tokens == 5

    def test_no_usage_anywhere_estimates_and_flags(self) -> None:
        acc = self._accumulator("lmstudio", est_in=15)
        acc.observe(_text_chunk("x" * 30))
        acc.observe(_text_chunk("y" * 10))
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 15
        assert details.output_tokens == 10  # 40 chars / 4.0

    def test_empty_stream_estimates_zero_output(self) -> None:
        acc = self._accumulator("openai_compatible", est_in=15)
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 15
        assert details.output_tokens == 0

    def test_zeroed_placeholder_usage_chunks_never_latch(self) -> None:
        """A gateway attaching zero-count usage placeholders to interim chunks
        (with no real final usage) must fall through to estimation, not settle
        as silent zero spend."""
        acc = self._accumulator("openai_compatible", est_in=15)
        chunk = _text_chunk("z" * 40)
        chunk.usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        acc.observe(chunk)
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 15
        assert details.output_tokens == 10

    def test_malformed_chunk_never_raises(self) -> None:
        """Arbitrary endpoint output (truthy non-iterable choices) must not
        convert a deliverable stream into a failure."""
        acc = self._accumulator("openai_compatible", est_in=3)
        acc.observe(SimpleNamespace(choices=42, usage=None))
        details = acc.finalize()
        assert details.is_estimated is True

    def test_negative_usage_chunk_falls_to_estimation_tier(self) -> None:
        """Negative counts never latch AND never raise out of observe (a raise
        would settle a content-healthy stream as a breaker failure)."""
        acc = self._accumulator("openai_compatible", est_in=15)
        acc.observe(_text_chunk("z" * 40))
        acc.observe(_usage_chunk(prompt_tokens=-1, completion_tokens=-2))
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 15
        assert details.output_tokens == 10

    def test_non_int_usage_chunk_falls_to_estimation_tier(self) -> None:
        acc = self._accumulator("openai_compatible", est_in=15)
        chunk = _usage_chunk()
        chunk.usage.prompt_tokens = "abc"
        chunk.usage.completion_tokens = "def"
        acc.observe(chunk)
        acc.observe(_text_chunk("z" * 40))
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.output_tokens == 10

    def test_x_groq_negative_usage_falls_to_estimation_tier(self) -> None:
        acc = self._accumulator("groq", est_in=15)
        acc.observe(_text_chunk("z" * 40))
        acc.observe(
            SimpleNamespace(
                choices=[],
                usage=None,
                x_groq={"usage": {"prompt_tokens": -21, "completion_tokens": -6}},
            )
        )
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 15
        assert details.output_tokens == 10

    def test_stream_tool_call_arguments_count_toward_estimate(self) -> None:
        tool_call = SimpleNamespace(function=SimpleNamespace(arguments="x" * 20))
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=[tool_call]),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        acc = self._accumulator("ollama", est_in=5)
        acc.observe(chunk)
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.output_tokens == 5  # 20 chars / 4.0

    def test_stream_reasoning_content_counts_toward_estimate(self) -> None:
        """A usage-less stream of reasoning-only deltas (DeepSeek-style
        reasoning_content field) must estimate from the reasoning text."""
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning_content="r" * 40),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        acc = self._accumulator("deepseek", est_in=5)
        acc.observe(chunk)
        details = acc.finalize()
        assert details.is_estimated is True
        assert details.input_tokens == 5
        assert details.output_tokens == 10  # 40 chars / 4.0

    def test_service_tier_from_usage_chunk(self) -> None:
        acc = self._accumulator("groq")
        acc.observe(_usage_chunk(prompt_tokens=1, completion_tokens=1, service_tier="on_demand"))
        assert acc.get_service_tier() == "on_demand"

    def test_service_tier_none_without_usage_chunk(self) -> None:
        acc = self._accumulator("groq")
        acc.observe(_text_chunk("hi"))
        assert acc.get_service_tier() is None
