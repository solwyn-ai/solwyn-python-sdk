"""Integration tests for the provider adapter registry.

These tests call the public registry functions using the REAL loaded adapters
(no patching). They verify all three adapters are registered and dispatch
correctly by name, model prefix, and client module path.

Client detection uses fake classes with controlled __module__ values — the
adapters detect by checking type(client).__module__, so no real SDK install
is required.
"""

from __future__ import annotations

from typing import Any

import pytest

import solwyn.providers as provider_registry
from solwyn._types import ProviderName
from solwyn.providers import get_adapter_by_name, get_adapter_for_client, get_adapter_for_model
from solwyn.providers._protocol import ProviderAdapter
from solwyn.providers.anthropic import AnthropicAdapter
from solwyn.providers.bedrock import BedrockAdapter
from solwyn.providers.google import GoogleAdapter
from solwyn.providers.openai import OpenAIAdapter
from solwyn.providers.openai_compatible import COMPAT_PROFILES
from solwyn.providers.together import TogetherAdapter


def _make_client(
    module_path: str,
    *,
    class_name: str = "FakeClient",
    base_url: str | None = None,
) -> Any:
    """Return a fake client with controlled type metadata and optional base URL.

    This lets us test detect_client() without installing the real SDK packages.
    """
    FakeClient = type(class_name, (), {"__module__": module_path})
    client = FakeClient()
    if base_url is not None:
        client.base_url = base_url  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# get_adapter_by_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAdapterByName:
    def test_returns_openai_adapter_for_openai(self) -> None:
        adapter = get_adapter_by_name("openai")
        assert isinstance(adapter, OpenAIAdapter)

    def test_returns_anthropic_adapter_for_anthropic(self) -> None:
        adapter = get_adapter_by_name("anthropic")
        assert isinstance(adapter, AnthropicAdapter)

    def test_returns_google_adapter_for_google(self) -> None:
        adapter = get_adapter_by_name("google")
        assert isinstance(adapter, GoogleAdapter)

    def test_returns_together_adapter_for_together(self) -> None:
        adapter = get_adapter_by_name("together")
        assert isinstance(adapter, TogetherAdapter)

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown_provider"):
            get_adapter_by_name("unknown_provider")

    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_adapter_by_name("")


# ---------------------------------------------------------------------------
# get_adapter_for_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAdapterForModel:
    def test_gpt_prefix_returns_openai_adapter(self) -> None:
        adapter = get_adapter_for_model("gpt-5.5")
        assert isinstance(adapter, OpenAIAdapter)

    def test_o3_prefix_returns_openai_adapter(self) -> None:
        adapter = get_adapter_for_model("o3-mini")
        assert isinstance(adapter, OpenAIAdapter)

    def test_o4_prefix_returns_openai_adapter(self) -> None:
        adapter = get_adapter_for_model("o4-mini")
        assert isinstance(adapter, OpenAIAdapter)

    def test_claude_prefix_returns_anthropic_adapter(self) -> None:
        adapter = get_adapter_for_model("claude-sonnet-5")
        assert isinstance(adapter, AnthropicAdapter)

    def test_gemini_prefix_returns_google_adapter(self) -> None:
        adapter = get_adapter_for_model("gemini-3.5-flash")
        assert isinstance(adapter, GoogleAdapter)

    def test_unknown_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="completely-unknown-model"):
            get_adapter_for_model("completely-unknown-model")

    def test_unknown_model_never_silently_falls_back(self) -> None:
        with pytest.raises(ValueError):
            get_adapter_for_model("llama-3-8b")


# ---------------------------------------------------------------------------
# get_adapter_for_client
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAdapterForClient:
    def test_openai_module_path_returns_openai_adapter(self) -> None:
        client = _make_client("openai.lib._base_client")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, OpenAIAdapter)

    def test_anthropic_module_path_returns_anthropic_adapter(self) -> None:
        client = _make_client("anthropic._client")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, AnthropicAdapter)

    def test_google_genai_module_path_returns_google_adapter(self) -> None:
        client = _make_client("google.genai.client")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, GoogleAdapter)

    def test_google_generativeai_module_path_returns_google_adapter(self) -> None:
        client = _make_client("google.generativeai")
        adapter = get_adapter_for_client(client)
        assert isinstance(adapter, GoogleAdapter)

    def test_unknown_client_raises_value_error(self) -> None:
        client = _make_client("unknown_vendor.sdk.client")
        with pytest.raises(ValueError):
            get_adapter_for_client(client)

    def test_unknown_client_never_silently_falls_back(self) -> None:
        client = _make_client("huggingface_hub")
        with pytest.raises(ValueError):
            get_adapter_for_client(client)


# ---------------------------------------------------------------------------
# All adapters registered
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllAdaptersRegistered:
    def test_registry_preserves_together_singleton_and_detection_order(self) -> None:
        together_by_name = get_adapter_by_name("together")
        assert provider_registry._ADAPTERS is not None
        assert provider_registry._ADAPTER_BY_NAME is not None
        adapters = provider_registry._ADAPTERS

        together_adapters = [
            adapter for adapter in adapters if isinstance(adapter, TogetherAdapter)
        ]
        assert len(together_adapters) == 1
        together_adapter = together_adapters[0]
        assert provider_registry._ADAPTER_BY_NAME["together"] is together_adapter
        assert together_by_name is together_adapter

        compat_names = [profile.name for profile in COMPAT_PROFILES]
        assert COMPAT_PROFILES[-1].catch_all is True
        assert all(not profile.catch_all for profile in COMPAT_PROFILES[:-1])
        assert [adapter.name for adapter in adapters] == [
            *compat_names,
            "openai",
            "anthropic",
            "google",
            "bedrock",
        ]
        assert [type(adapter) for adapter in adapters[len(COMPAT_PROFILES) :]] == [
            OpenAIAdapter,
            AnthropicAdapter,
            GoogleAdapter,
            BedrockAdapter,
        ]

        native_client = _make_client("together", class_name="Together")
        openai_client = _make_client(
            "openai._client",
            class_name="OpenAI",
            base_url="https://api.together.xyz/v1",
        )
        assert get_adapter_for_client(native_client) is together_adapter
        assert get_adapter_for_client(openai_client) is together_adapter

    def test_all_four_providers_registered(self) -> None:
        """All four expected provider names resolve without error."""
        for name in ("openai", "anthropic", "google", "bedrock"):
            adapter = get_adapter_by_name(name)
            assert adapter is not None

    def test_all_adapters_satisfy_provider_adapter_protocol(self) -> None:
        """Every registered adapter is a runtime ProviderAdapter instance."""
        for name in ("openai", "anthropic", "google", "bedrock"):
            adapter = get_adapter_by_name(name)
            assert isinstance(adapter, ProviderAdapter), (
                f"Adapter '{name}' does not satisfy ProviderAdapter protocol"
            )

    def test_adapter_names_match_registry_keys(self) -> None:
        """Each adapter's name property matches the key it was registered under."""
        for name in ("openai", "anthropic", "google", "bedrock"):
            adapter = get_adapter_by_name(name)
            assert adapter.name == name

    def test_adapter_names_are_provider_name_values(self) -> None:
        """CostPolicy price hints key by ProviderName value; adapter names must match."""
        for name in ("openai", "anthropic", "google", "bedrock"):
            adapter = get_adapter_by_name(name)
            assert ProviderName(adapter.name) is ProviderName(name)

    def test_bedrock_model_id_routes_to_bedrock_adapter(self) -> None:
        from solwyn.providers import get_adapter_for_model

        adapter = get_adapter_for_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert adapter.name == "bedrock"

    def test_direct_anthropic_model_id_still_routes_to_anthropic(self) -> None:
        # The Bedrock patterns must never shadow the native adapters.
        from solwyn.providers import get_adapter_for_model

        assert get_adapter_for_model("claude-sonnet-5").name == "anthropic"

    def test_bedrock_runtime_client_routes_to_bedrock_adapter(self) -> None:
        from types import SimpleNamespace

        from solwyn.providers import get_adapter_for_client

        class FakeBedrockRuntime:
            pass

        FakeBedrockRuntime.__module__ = "botocore.client"
        client = FakeBedrockRuntime()
        client.meta = SimpleNamespace(  # type: ignore[attr-defined]
            service_model=SimpleNamespace(service_name="bedrock-runtime"),
            region_name="us-east-1",
        )
        assert get_adapter_for_client(client).name == "bedrock"
