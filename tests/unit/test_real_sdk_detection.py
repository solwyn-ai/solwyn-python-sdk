"""Real-SDK detection tests — genuine provider client objects, no stubs.

Every other detection test in this suite (test_registry.py,
test_openai_compatible.py, ...) constructs synthetic fakes with a controlled
``__module__``/``__name__``/``base_url`` — convenient, but blind to drift in
the REAL SDKs: a renamed class, a changed base_url shape, a botocore internal
that stops matching. These tests construct genuine ``openai``, ``anthropic``,
``together``, and ``boto3`` client objects (all dev dependencies, installed
here) and assert each resolves to the Solwyn adapter the design intends.

No network calls — every client here constructs fully offline given an
explicit fake API key / credentials. Each SDK is imported via its own
module-scoped fixture (``pytest.importorskip`` inside the fixture body, not
at module level) so a missing package skips only the tests that requested
that fixture — collection of the module itself never fails, and the other
SDKs' tests still run.
"""

from __future__ import annotations

from typing import Any

import pytest

from solwyn.providers import get_adapter_for_client

# ---------------------------------------------------------------------------
# Per-SDK import fixtures — each importorskip is scoped to its own fixture so
# a missing SDK only skips the tests that depend on it, never the whole file.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_mod() -> Any:
    return pytest.importorskip("openai")


@pytest.fixture(scope="module")
def anthropic_mod() -> Any:
    return pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def together_mod() -> Any:
    return pytest.importorskip("together")


@pytest.fixture(scope="module")
def boto3_mod() -> Any:
    return pytest.importorskip("boto3")


# ---------------------------------------------------------------------------
# openai (+ Azure + OpenAI-compatible base_urls)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAIRealClient:
    def test_openai_sync_default_base_url(self, openai_mod: Any) -> None:
        client = openai_mod.OpenAI(api_key="sk-test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai"
        assert adapter.dialect == "openai"

    def test_openai_async_default_base_url(self, openai_mod: Any) -> None:
        client = openai_mod.AsyncOpenAI(api_key="sk-test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai"
        assert adapter.dialect == "openai"

    def test_openai_explicit_default_base_url(self, openai_mod: Any) -> None:
        client = openai_mod.OpenAI(api_key="sk-test", base_url="https://api.openai.com/v1")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai"
        assert adapter.dialect == "openai"

    def test_openai_regional_data_residency_host(self, openai_mod: Any) -> None:
        # eu.api.openai.com is a suffix match against api.openai.com — the
        # regional data-residency endpoints must still resolve as "openai".
        client = openai_mod.OpenAI(api_key="sk-test", base_url="https://eu.api.openai.com/v1")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "openai"
        assert adapter.dialect == "openai"


_COMPAT_BASE_URL_CASES = [
    ("https://api.x.ai/v1", "xai"),
    ("https://api.deepseek.com", "deepseek"),
    ("https://api.mistral.ai/v1", "mistral"),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen"),
    ("https://api.z.ai/api/paas/v4", "zai"),
    ("https://api.groq.com/openai/v1", "groq"),
    ("https://api.together.xyz/v1", "together"),
    ("https://api.fireworks.ai/inference/v1", "fireworks"),
    ("https://api.perplexity.ai", "perplexity"),
    ("https://openrouter.ai/api/v1", "openrouter"),
    ("http://localhost:11434/v1", "ollama"),
    ("http://127.0.0.1:11434/v1", "ollama"),
    ("http://localhost:8000/v1", "vllm"),
    ("http://localhost:1234/v1", "lmstudio"),
    ("https://llm.internal.example.com/v1", "openai_compatible"),
]


@pytest.mark.unit
class TestOpenAICompatibleRealClient:
    @pytest.mark.parametrize(("base_url", "expected_name"), _COMPAT_BASE_URL_CASES)
    def test_sync_client_base_url_resolves_expected_provider(
        self, openai_mod: Any, base_url: str, expected_name: str
    ) -> None:
        client = openai_mod.OpenAI(api_key="sk-test", base_url=base_url)
        adapter = get_adapter_for_client(client)
        assert adapter.name == expected_name
        assert adapter.dialect == "openai"

    @pytest.mark.parametrize(
        ("base_url", "expected_name"),
        [
            ("https://api.x.ai/v1", "xai"),
            ("https://api.together.xyz/v1", "together"),
        ],
    )
    def test_async_client_base_url_resolves_expected_provider(
        self, openai_mod: Any, base_url: str, expected_name: str
    ) -> None:
        client = openai_mod.AsyncOpenAI(api_key="sk-test", base_url=base_url)
        adapter = get_adapter_for_client(client)
        assert adapter.name == expected_name
        assert adapter.dialect == "openai"


@pytest.mark.unit
class TestAzureOpenAIRealClient:
    def test_azure_openai_endpoint(self, openai_mod: Any) -> None:
        client = openai_mod.AzureOpenAI(
            api_key="sk-test",
            api_version="2024-06-01",
            azure_endpoint="https://myres.openai.azure.com",
        )
        adapter = get_adapter_for_client(client)
        assert adapter.name == "azure_openai"
        assert adapter.dialect == "openai"

    def test_azure_cognitiveservices_endpoint(self, openai_mod: Any) -> None:
        client = openai_mod.AzureOpenAI(
            api_key="sk-test",
            api_version="2024-06-01",
            azure_endpoint="https://myres.cognitiveservices.azure.com",
        )
        adapter = get_adapter_for_client(client)
        assert adapter.name == "azure_openai"
        assert adapter.dialect == "openai"

    def test_async_azure_openai_endpoint(self, openai_mod: Any) -> None:
        client = openai_mod.AsyncAzureOpenAI(
            api_key="sk-test",
            api_version="2024-06-01",
            azure_endpoint="https://myres.openai.azure.com",
        )
        adapter = get_adapter_for_client(client)
        assert adapter.name == "azure_openai"
        assert adapter.dialect == "openai"


# ---------------------------------------------------------------------------
# anthropic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicRealClient:
    def test_anthropic_sync_client(self, anthropic_mod: Any) -> None:
        client = anthropic_mod.Anthropic(api_key="sk-ant-test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "anthropic"
        assert adapter.dialect == "anthropic"

    def test_anthropic_async_client(self, anthropic_mod: Any) -> None:
        client = anthropic_mod.AsyncAnthropic(api_key="sk-ant-test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "anthropic"
        assert adapter.dialect == "anthropic"


# ---------------------------------------------------------------------------
# together (native clients admitted into the together compat slot)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTogetherRealClient:
    def test_together_sync_client(self, together_mod: Any) -> None:
        client = together_mod.Together(api_key="test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "together"
        assert adapter.dialect == "openai"

    def test_together_async_client(self, together_mod: Any) -> None:
        client = together_mod.AsyncTogether(api_key="test")
        adapter = get_adapter_for_client(client)
        assert adapter.name == "together"
        assert adapter.dialect == "openai"


# ---------------------------------------------------------------------------
# boto3 / botocore (bedrock-runtime data plane vs bedrock control plane)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBedrockRealClient:
    def test_bedrock_runtime_client_resolves_bedrock_adapter(self, boto3_mod: Any) -> None:
        client = boto3_mod.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        adapter = get_adapter_for_client(client)
        assert adapter.name == "bedrock"
        assert adapter.dialect == "bedrock"
        assert adapter.extract_region(client) == "us-east-1"

    def test_bedrock_control_plane_client_is_not_claimed(self, boto3_mod: Any) -> None:
        # The control-plane "bedrock" client (model listing, provisioning) is
        # NOT an inference surface — it must never resolve to any adapter.
        client = boto3_mod.client(
            "bedrock",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        with pytest.raises(ValueError):
            get_adapter_for_client(client)
