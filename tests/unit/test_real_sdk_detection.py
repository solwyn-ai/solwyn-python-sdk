"""Real-SDK detection tests — genuine provider client objects, no stubs.

Every other detection test in this suite (test_registry.py,
test_openai_compatible.py, ...) constructs synthetic fakes with a controlled
``__module__``/``__name__``/``base_url`` — convenient, but blind to drift in
the REAL SDKs: a renamed class, a changed base_url shape, a botocore internal
that stops matching. These tests construct genuine ``openai``, ``anthropic``,
``together``, legacy ``google.generativeai``, and ``boto3`` client objects (all
dev dependencies, installed here) and assert each resolves to the Solwyn
adapter the design intends.

No network calls — every client here constructs fully offline given an
explicit fake API key / credentials. Each SDK is imported via its own
module-scoped fixture (``pytest.importorskip`` inside the fixture body, not
at module level) so a missing package skips only the tests that requested
that fixture — collection of the module itself never fails, and the other
SDKs' tests still run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

from solwyn.client import Solwyn
from solwyn.exceptions import UntranslatableRequestError
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


@pytest.fixture(scope="module")
def google_generativeai_mod() -> Any:
    return pytest.importorskip("google.generativeai")


@pytest.fixture(scope="module")
def google_genai_mod() -> Any:
    return pytest.importorskip("google.genai")


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
# google.generativeai (legacy GenerativeModel root generation surface)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyGoogleRealClient:
    @pytest.mark.parametrize(
        ("configured_model", "expected_model", "positional_contents"),
        [
            (None, "gemini-1.5-flash", False),
            ("gemini-configured", "gemini-configured", True),
        ],
    )
    def test_public_generation_dispatches_and_settles_through_real_client_shape(
        self,
        google_generativeai_mod: Any,
        configured_model: str | None,
        expected_model: str,
        positional_contents: bool,
    ) -> None:
        model = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        response = SimpleNamespace(
            text="Hello",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=7,
                thoughts_token_count=0,
                cached_content_token_count=0,
                tool_use_prompt_token_count=0,
            ),
        )
        request_options = {
            "metadata": (("x-test", "1"),),
            "retry": object(),
            "timeout": 999.0,
        }
        config: dict[str, Any] = {
            "api_key": VALID_API_KEY,
            "failover_hop_read_timeout": 12.5,
        }
        if configured_model is not None:
            config["model"] = configured_model
        solwyn = Solwyn(model, **config)
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None
        settlements: list[tuple[Any, Any]] = []

        with (
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(model, "generate_content", return_value=response) as generate_content,
            patch.object(
                solwyn._reporter,
                "report_settlement",
                side_effect=lambda confirm, event: settlements.append((confirm, event)),
            ),
        ):
            if positional_contents:
                result = solwyn.generate_content(
                    "Hello",
                    request_options=request_options,
                )
            else:
                result = solwyn.generate_content(
                    contents="Hello",
                    request_options=request_options,
                )

        assert result is response
        generate_content.assert_called_once()
        provider_kwargs = generate_content.call_args.kwargs
        assert provider_kwargs["contents"] == "Hello"
        assert "model" not in provider_kwargs
        assert provider_kwargs["request_options"] == {
            "metadata": (("x-test", "1"),),
            "retry": None,
            "timeout": 12.5,
        }
        assert provider_kwargs["request_options"] is not request_options
        assert request_options["timeout"] == 999.0
        assert len(settlements) == 1
        confirm, event = settlements[0]
        assert confirm.model == expected_model
        assert event.model == expected_model
        assert event.status == "success"
        assert not hasattr(model, "models")
        solwyn.close()

    def test_budget_preflight_uses_legacy_constructor_defaults_and_call_overrides(
        self,
        google_generativeai_mod: Any,
    ) -> None:
        system_instruction = "S" * 40_000
        call_generation_config = {"temperature": 0.7}
        call_tool_config = {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["lookup"],
            }
        }
        model = google_generativeai_mod.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={
                "max_output_tokens": 20_000,
                "temperature": 0.2,
                "top_k": 7,
                "response_mime_type": "application/json",
            },
            tools=[
                {
                    "function_declarations": [
                        {
                            "name": "lookup",
                            "description": "Look up a value",
                        }
                    ]
                }
            ],
            system_instruction=system_instruction,
        )
        response = SimpleNamespace(
            text="ok",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10_000,
                candidates_token_count=1,
                thoughts_token_count=0,
                cached_content_token_count=0,
                tool_use_prompt_token_count=0,
            ),
        )
        solwyn = Solwyn(model, api_key=VALID_API_KEY)
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._budget,
                "check_budget",
                wraps=solwyn._budget.check_budget,
            ) as check_budget,
            patch.object(model, "generate_content", return_value=response) as generate_content,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            result = solwyn.generate_content(
                "Hi",
                generation_config=call_generation_config,
                tool_config=call_tool_config,
            )

        assert result is response
        assert check_budget.call_args.kwargs["estimated_input_tokens"] == 10_000
        assert check_budget.call_args.kwargs["estimated_output_bound"] == 20_000
        provider_kwargs = generate_content.call_args.kwargs
        assert provider_kwargs["contents"] == "Hi"
        assert provider_kwargs["generation_config"] == {"temperature": 0.7}
        assert provider_kwargs["tool_config"] == call_tool_config
        assert "config" not in provider_kwargs
        assert call_generation_config == {"temperature": 0.7}
        assert call_tool_config == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["lookup"],
            }
        }
        solwyn.close()

    def test_cross_dialect_fallback_into_legacy_normalizes_modern_config(
        self,
        google_generativeai_mod: Any,
        openai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        response_proto = google_generativeai_mod.protos.GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"text": "fallback"}]},
                    "finish_reason": 1,
                }
            ],
            usage_metadata={"prompt_token_count": 8, "candidates_token_count": 3},
        )

        class LegacyTransport:
            def __init__(self) -> None:
                self.requests: list[Any] = []

            def generate_content(self, request: Any, **request_options: Any) -> Any:
                self.requests.append(request)
                return response_proto

        primary = openai_mod.OpenAI(api_key="sk-test")
        fallback = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        legacy_transport = LegacyTransport()
        fallback._client = legacy_transport
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            model="gpt-5.5",
            fallback=[
                (
                    fallback,
                    "gemini-1.5-flash",
                    {"generation_config": {"temperature": 0.1}},
                )
            ],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(primary, "with_options", return_value=primary),
            patch.object(
                primary.chat.completions,
                "create",
                side_effect=RetryableError("primary unavailable"),
            ),
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=64,
                temperature=0.7,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                                "required": ["key"],
                            },
                        },
                    }
                ],
                tool_choice="required",
            )

        assert response.choices[0].message.content == "fallback"
        assert len(legacy_transport.requests) == 1
        request = legacy_transport.requests[0]
        assert request.generation_config.max_output_tokens == 64
        assert request.generation_config.temperature == pytest.approx(0.7)
        declaration = request.tools[0].function_declarations[0]
        assert declaration.name == "lookup"
        assert declaration.parameters.type_.name == "OBJECT"
        assert request.tool_config.function_calling_config.mode.name == "ANY"
        solwyn.close()

    def test_same_dialect_modern_to_legacy_normalizes_to_legacy_signature(
        self,
        google_generativeai_mod: Any,
        google_genai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        response_proto = google_generativeai_mod.protos.GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"text": "fallback"}]},
                    "finish_reason": 1,
                }
            ],
            usage_metadata={"prompt_token_count": 8, "candidates_token_count": 3},
        )

        class LegacyTransport:
            def __init__(self) -> None:
                self.requests: list[Any] = []

            def generate_content(self, request: Any, **request_options: Any) -> Any:
                self.requests.append(request)
                return response_proto

        primary = google_genai_mod.Client(api_key="test")
        fallback = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        legacy_transport = LegacyTransport()
        fallback._client = legacy_transport
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            model="gemini-2.5-flash",
            fallback=[
                (
                    fallback,
                    "gemini-1.5-flash",
                    {
                        "generation_config": {
                            "max_output_tokens": 20_000,
                            "temperature": 0.1,
                        }
                    },
                )
            ],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(
                primary.models,
                "generate_content",
                side_effect=RetryableError("primary unavailable"),
            ),
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._budget,
                "check_budget",
                wraps=solwyn._budget.check_budget,
            ) as check_budget,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello",
                config={
                    "temperature": 0.7,
                    "tools": [
                        {
                            "function_declarations": [
                                {
                                    "name": "lookup",
                                    "description": "Look up a value",
                                    "parameters_json_schema": {"type": "object"},
                                }
                            ]
                        }
                    ],
                    "tool_config": {"function_calling_config": {"mode": "ANY"}},
                },
            )

        assert response.text == "fallback"
        assert len(legacy_transport.requests) == 1
        request = legacy_transport.requests[0]
        assert request.generation_config.max_output_tokens == 20_000
        assert request.generation_config.temperature == pytest.approx(0.7)
        declaration = request.tools[0].function_declarations[0]
        assert declaration.name == "lookup"
        assert declaration.parameters.type_.name == "OBJECT"
        assert request.tool_config.function_calling_config.mode.name == "ANY"
        assert check_budget.call_args.kwargs["estimated_output_bound"] == 20_000
        solwyn.close()

    def test_legacy_fallback_constructor_cap_is_included_in_preflight(
        self,
        google_generativeai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        class FailingLegacyTransport:
            def generate_content(self, request: Any, **request_options: Any) -> Any:
                raise RetryableError("primary unavailable")

        response_proto = google_generativeai_mod.protos.GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"text": "fallback"}]},
                    "finish_reason": 1,
                }
            ],
            usage_metadata={"prompt_token_count": 8, "candidates_token_count": 3},
        )

        class RecordingLegacyTransport:
            def __init__(self) -> None:
                self.requests: list[Any] = []

            def generate_content(self, request: Any, **request_options: Any) -> Any:
                self.requests.append(request)
                return response_proto

        primary = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        primary._client = FailingLegacyTransport()
        fallback = google_generativeai_mod.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={"max_output_tokens": 20_000},
        )
        fallback_transport = RecordingLegacyTransport()
        fallback._client = fallback_transport
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[(fallback, "gemini-1.5-pro")],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._budget,
                "check_budget",
                wraps=solwyn._budget.check_budget,
            ) as check_budget,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.generate_content(
                "Hello",
                generation_config={"temperature": 0.7},
            )

        assert response.text == "fallback"
        assert len(fallback_transport.requests) == 1
        request = fallback_transport.requests[0]
        assert request.generation_config.max_output_tokens == 20_000
        assert request.generation_config.temperature == pytest.approx(0.7)
        assert check_budget.call_args.kwargs["estimated_output_bound"] == 20_000
        solwyn.close()

    def test_incompatible_legacy_fallback_is_validated_only_when_reached(
        self,
        google_generativeai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        primary = google_generativeai_mod.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"max_output_tokens": 64},
            system_instruction="Follow the system instruction",
        )
        fallback = google_generativeai_mod.GenerativeModel("gemini-1.5-pro")
        success = SimpleNamespace(
            text="primary",
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=3,
                thoughts_token_count=0,
                cached_content_token_count=0,
                tool_use_prompt_token_count=0,
            ),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[(fallback, "gemini-1.5-pro")],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(primary, "generate_content", return_value=success) as primary_call,
            patch.object(fallback, "generate_content") as fallback_call,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            assert solwyn.generate_content("Hello") is success
            primary_call.side_effect = RetryableError("primary unavailable")
            with pytest.raises(
                UntranslatableRequestError,
                match="google.system_instruction_per_call",
            ):
                solwyn.generate_content("Hello")

        fallback_call.assert_not_called()
        solwyn.close()

    def test_legacy_fallback_metering_error_does_not_block_healthy_primary(
        self,
        google_generativeai_mod: Any,
        google_genai_mod: Any,
    ) -> None:
        primary = google_genai_mod.Client(api_key="test")
        fallback = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        fallback._cached_content = "cachedContents/example"
        primary_response = SimpleNamespace(
            text="primary",
            candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            model="gemini-2.5-flash",
            fallback=[(fallback, "gemini-1.5-flash")],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(
                primary.models,
                "generate_content",
                return_value=primary_response,
            ) as primary_call,
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._budget,
                "check_budget",
                wraps=solwyn._budget.check_budget,
            ) as check_budget,
            patch.object(fallback, "generate_content") as fallback_call,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello",
                config={
                    "max_output_tokens": 64,
                    "tools": [
                        {
                            "function_declarations": [
                                {
                                    "name": "lookup",
                                    "description": "Look up a value",
                                    "parameters_json_schema": {},
                                }
                            ]
                        }
                    ],
                },
            )

        assert response is primary_response
        primary_call.assert_called_once()
        check_budget.assert_called_once()
        fallback_call.assert_not_called()
        solwyn.close()

    def test_same_dialect_legacy_to_modern_normalizes_to_modern_signature(
        self,
        google_generativeai_mod: Any,
        google_genai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        class FailingLegacyTransport:
            def generate_content(self, request: Any, **request_options: Any) -> Any:
                raise RetryableError("primary unavailable")

        primary = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        primary._client = FailingLegacyTransport()
        fallback = google_genai_mod.Client(api_key="test")
        fallback_response = SimpleNamespace(
            text="fallback",
            candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[
                (
                    fallback,
                    "gemini-2.5-flash",
                    {"config": {"temperature": 0.1}},
                )
            ],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(
                fallback.models,
                "generate_content",
                autospec=True,
                return_value=fallback_response,
            ) as generate_content,
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.generate_content(
                contents="Hello",
                generation_config={"max_output_tokens": 64, "temperature": 0.7},
                tools=[
                    {
                        "function_declarations": [
                            {
                                "name": "lookup",
                                "description": "Look up a value",
                                "parameters": {"type": "object"},
                            }
                        ]
                    }
                ],
                tool_config={"function_calling_config": {"mode": "ANY"}},
            )

        assert response is fallback_response
        generate_content.assert_called_once()
        provider_kwargs = generate_content.call_args.kwargs
        assert provider_kwargs["model"] == "gemini-2.5-flash"
        assert provider_kwargs["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert "generation_config" not in provider_kwargs
        assert "tools" not in provider_kwargs
        assert "tool_config" not in provider_kwargs
        config = provider_kwargs["config"]
        assert config["max_output_tokens"] == 64
        assert config["temperature"] == pytest.approx(0.7)
        assert config["tools"][0]["function_declarations"][0]["name"] == "lookup"
        assert config["tool_config"]["function_calling_config"]["mode"] == "ANY"
        solwyn.close()

    def test_cross_dialect_fallback_away_from_legacy_normalizes_string_contents(
        self,
        google_generativeai_mod: Any,
        openai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        class FailingLegacyTransport:
            def generate_content(self, request: Any, **request_options: Any) -> Any:
                raise RetryableError("primary unavailable")

        primary = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        primary._client = FailingLegacyTransport()
        fallback = openai_mod.OpenAI(api_key="sk-test")
        message = SimpleNamespace(role="assistant", content="fallback", tool_calls=None)
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
            model="gpt-5.5",
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[(fallback, "gpt-5.5")],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(fallback, "with_options", return_value=fallback),
            patch.object(
                fallback.chat.completions,
                "create",
                autospec=True,
                return_value=fallback_response,
            ) as create,
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.generate_content(
                "Hello",
                generation_config={"max_output_tokens": 64, "temperature": 0.7},
            )

        assert response.text == "fallback"
        create.assert_called_once()
        provider_kwargs = create.call_args.kwargs
        assert provider_kwargs["model"] == "gpt-5.5"
        assert provider_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert provider_kwargs["max_completion_tokens"] == 64
        assert provider_kwargs["temperature"] == pytest.approx(0.7)
        solwyn.close()

    def test_legacy_constructor_defaults_reach_openai_fallback(
        self,
        google_generativeai_mod: Any,
        openai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        class FailingLegacyTransport:
            def generate_content(self, request: Any, **request_options: Any) -> Any:
                raise RetryableError("primary unavailable")

        primary = google_generativeai_mod.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"max_output_tokens": 64, "temperature": 0.2},
            system_instruction="Follow the system instruction",
        )
        primary._client = FailingLegacyTransport()
        fallback = openai_mod.OpenAI(api_key="sk-test")
        message = SimpleNamespace(role="assistant", content="fallback", tool_calls=None)
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
            model="gpt-5.5",
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[(fallback, "gpt-5.5")],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(fallback, "with_options", return_value=fallback),
            patch.object(
                fallback.chat.completions,
                "create",
                autospec=True,
                return_value=fallback_response,
            ) as create,
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.generate_content("Hello")

        assert response.text == "fallback"
        provider_kwargs = create.call_args.kwargs
        assert provider_kwargs["messages"] == [
            {"role": "system", "content": "Follow the system instruction"},
            {"role": "user", "content": "Hello"},
        ]
        assert provider_kwargs["max_completion_tokens"] == 64
        assert provider_kwargs["temperature"] == pytest.approx(0.2)
        solwyn.close()

    def test_legacy_constructor_defaults_reach_modern_google_fallback(
        self,
        google_generativeai_mod: Any,
        google_genai_mod: Any,
    ) -> None:
        class RetryableError(Exception):
            status_code = 429

        class FailingLegacyTransport:
            def generate_content(self, request: Any, **request_options: Any) -> Any:
                raise RetryableError("primary unavailable")

        primary = google_generativeai_mod.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"max_output_tokens": 64, "temperature": 0.2},
            safety_settings={"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"},
            tools=[
                {
                    "function_declarations": [
                        {
                            "name": "lookup",
                            "description": "Look up a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                                "required": ["key"],
                            },
                        },
                        {
                            "name": "ping",
                            "description": "Return service health",
                        },
                    ]
                }
            ],
            tool_config={"function_calling_config": {"mode": "ANY"}},
            system_instruction="Follow the system instruction",
        )
        primary._client = FailingLegacyTransport()
        fallback = google_genai_mod.Client(api_key="test")
        fallback_response = SimpleNamespace(
            text="fallback",
            candidates=[],
            usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=3),
        )
        solwyn = Solwyn(
            primary,
            api_key=VALID_API_KEY,
            fallback=[
                (
                    fallback,
                    "gemini-2.5-flash",
                    {"config": {"max_output_tokens": 20_000}},
                )
            ],
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None

        with (
            patch.object(
                fallback.models,
                "generate_content",
                autospec=True,
                return_value=fallback_response,
            ) as generate_content,
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._budget,
                "check_budget",
                wraps=solwyn._budget.check_budget,
            ) as check_budget,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._reporter, "report_settlement"),
        ):
            response = solwyn.generate_content(
                "Hello",
                generation_config={"temperature": 0.7},
            )

        assert response is fallback_response
        provider_kwargs = generate_content.call_args.kwargs
        assert provider_kwargs["model"] == "gemini-2.5-flash"
        assert provider_kwargs["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        config = provider_kwargs["config"]
        assert config["max_output_tokens"] == 20_000
        assert config["temperature"] == pytest.approx(0.7)
        assert config["system_instruction"] == "Follow the system instruction"
        assert config["safety_settings"] == [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
        ]
        declaration = config["tools"][0]["function_declarations"][0]
        assert declaration == {
            "name": "lookup",
            "description": "Look up a value",
            "parameters_json_schema": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        }
        assert config["tools"][0]["function_declarations"][1] == {
            "name": "ping",
            "description": "Return service health",
            "parameters_json_schema": {},
        }
        assert config["tool_config"] == {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": []}
        }
        assert check_budget.call_args.kwargs["estimated_output_bound"] == 20_000
        solwyn.close()

    def test_public_stream_generation_uses_legacy_root_method_and_settles_on_consumption(
        self,
        google_generativeai_mod: Any,
    ) -> None:
        model = google_generativeai_mod.GenerativeModel("gemini-1.5-flash")
        first_response = google_generativeai_mod.protos.GenerateContentResponse(
            candidates=[
                {
                    "content": {"role": "model", "parts": [{"text": "Hello"}]},
                    "finish_reason": 1,
                }
            ],
            usage_metadata={"prompt_token_count": 12, "candidates_token_count": 7},
        )

        class LaterProviderError(Exception):
            status_code = 429

        class PrimaryIterator:
            def __init__(self) -> None:
                self.pulls = 0

            def __iter__(self) -> PrimaryIterator:
                return self

            def __next__(self) -> Any:
                self.pulls += 1
                if self.pulls == 1:
                    return first_response
                raise LaterProviderError("second provider response failed")

        primary_iterator = PrimaryIterator()
        primary_transport = SimpleNamespace(
            stream_generate_content=MagicMock(return_value=primary_iterator)
        )
        model._client = primary_transport

        fallback = google_generativeai_mod.GenerativeModel("gemini-1.5-pro")
        fallback_responses = iter([first_response, first_response, first_response])
        fallback_transport = SimpleNamespace(
            stream_generate_content=MagicMock(return_value=fallback_responses)
        )
        fallback._client = fallback_transport
        solwyn = Solwyn(
            model,
            api_key=VALID_API_KEY,
            fallback=[(fallback, "gemini-1.5-pro")],
            failover_hop_read_timeout=12.5,
        )
        budget_response = MagicMock(spec=httpx.Response)
        budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        budget_response.raise_for_status.return_value = None
        settlements: list[tuple[Any, Any]] = []

        with (
            patch.object(solwyn._budget._http, "post", return_value=budget_response),
            patch.object(
                solwyn._reporter,
                "report_settlement",
                side_effect=lambda confirm, event: settlements.append((confirm, event)),
            ),
        ):
            stream = solwyn.generate_content(contents="Hello", stream=True)
            assert primary_iterator.pulls == 1
            fallback_transport.stream_generate_content.assert_not_called()
            stream_iterator = iter(stream)
            first_chunk = next(stream_iterator)
            assert first_chunk.text == "Hello"
            with pytest.raises(LaterProviderError, match="second provider response failed"):
                next(stream_iterator)

        primary_transport.stream_generate_content.assert_called_once()
        assert primary_transport.stream_generate_content.call_args.kwargs == {
            "retry": None,
            "timeout": 12.5,
        }
        assert len(settlements) == 0
        fallback_transport.stream_generate_content.assert_not_called()
        assert primary_iterator.pulls == 2
        assert not hasattr(model, "models")
        solwyn.close()


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
