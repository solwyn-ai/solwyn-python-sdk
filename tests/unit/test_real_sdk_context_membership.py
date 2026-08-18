"""Real-SDK check: undeclared client/mode pairings are rejected at construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY

from solwyn import AsyncSolwyn, Solwyn
from solwyn._base import _client_mode
from solwyn.exceptions import ConfigurationError

ANTHROPIC_SYNC_VARIANTS = (
    "AnthropicAWS",
    "AnthropicBedrock",
    "AnthropicBedrockMantle",
    "AnthropicFoundry",
    "AnthropicGoogleCloud",
    "AnthropicVertex",
)
ANTHROPIC_ASYNC_VARIANTS = (
    "AsyncAnthropicAWS",
    "AsyncAnthropicBedrock",
    "AsyncAnthropicBedrockMantle",
    "AsyncAnthropicFoundry",
    "AsyncAnthropicGoogleCloud",
    "AsyncAnthropicVertex",
)
OPENAI_SYNC_MODULE_CLIENTS = (
    "_ModuleClient",
    "_AzureModuleClient",
    "_BedrockModuleClient",
)


@pytest.fixture(scope="module")
def openai_mod() -> Any:
    return pytest.importorskip("openai")


@pytest.fixture(scope="module")
def anthropic_mod() -> Any:
    return pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def boto3_mod() -> Any:
    return pytest.importorskip("boto3")


def _uninitialized_client(module: Any, class_name: str) -> object:
    return object.__new__(getattr(module, class_name))


def _bedrock_client(boto3_mod: Any) -> object:
    return boto3_mod.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.mark.unit
def test_async_wrapper_rejects_a_sync_boto3_bedrock_client(monkeypatch, boto3_mod: Any) -> None:
    # Arrange
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    client = _bedrock_client(boto3_mod)

    # Act / Assert
    with pytest.raises(ConfigurationError) as exc_info:
        AsyncSolwyn(client, api_key=VALID_API_KEY)
    assert "aioboto3" in str(exc_info.value)
    assert exc_info.value.field == "client"

    # The matched pairing still constructs and releases owned resources.
    with Solwyn(_bedrock_client(boto3_mod), api_key=VALID_API_KEY) as wrapper:
        assert wrapper._solwyn_surface_context.client_shape == "bedrock_boto3"


@pytest.mark.unit
def test_explicit_pin_rejects_sync_openai_client_in_async_mode() -> None:
    sync_openai_type = type("OpenAI", (), {"__module__": "openai._client"})

    with pytest.raises(ConfigurationError) as exc_info:
        AsyncSolwyn(sync_openai_type(), api_key=VALID_API_KEY, provider="openai")

    assert exc_info.value.field == "client"
    assert "synchronous" in str(exc_info.value)


@pytest.mark.unit
def test_explicit_pin_rejects_async_openai_client_in_sync_mode() -> None:
    async_openai_type = type("AsyncOpenAI", (), {"__module__": "openai._client"})

    with pytest.raises(ConfigurationError) as exc_info:
        Solwyn(async_openai_type(), api_key=VALID_API_KEY, provider="openai")

    assert exc_info.value.field == "client"
    assert "asynchronous" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("wrapper", "module"),
    [
        (Solwyn, "botocore.client"),
        (AsyncSolwyn, "aiobotocore.client"),
    ],
)
def test_explicit_bedrock_pin_rejects_a_different_aws_service(wrapper, module: str) -> None:
    client_type = type(
        "AwsServiceClient",
        (),
        {
            "__module__": module,
            "meta": type(
                "ClientMeta",
                (),
                {"service_model": type("ServiceModel", (), {"service_name": "s3"})()},
            )(),
        },
    )

    with pytest.raises(ConfigurationError) as exc_info:
        wrapper(client_type(), api_key=VALID_API_KEY, provider="bedrock")

    assert exc_info.value.field == "client"
    assert "bedrock-runtime" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("client_name", ANTHROPIC_ASYNC_VARIANTS)
def test_sync_wrapper_rejects_pinned_async_anthropic_variant(
    anthropic_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pin must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = Solwyn(
                    _uninitialized_client(anthropic_mod, client_name),
                    api_key=VALID_API_KEY,
                    provider="anthropic",
                )
        finally:
            if wrapper is not None:
                wrapper.close()

    assert exc_info.value.field == "client"
    assert "asynchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", ANTHROPIC_SYNC_VARIANTS)
async def test_async_wrapper_rejects_pinned_sync_anthropic_variant(
    anthropic_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pin must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = AsyncSolwyn(
                    _uninitialized_client(anthropic_mod, client_name),
                    api_key=VALID_API_KEY,
                    provider="anthropic",
                )
        finally:
            if wrapper is not None:
                await wrapper.close()

    assert exc_info.value.field == "client"
    assert "synchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("client_name", ANTHROPIC_ASYNC_VARIANTS)
def test_sync_wrapper_rejects_pinned_async_anthropic_fallback(
    anthropic_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pins must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = Solwyn(
                    _uninitialized_client(anthropic_mod, "Anthropic"),
                    api_key=VALID_API_KEY,
                    provider="anthropic",
                    fallback=[
                        (
                            _uninitialized_client(anthropic_mod, client_name),
                            "fallback-model",
                            {},
                            "anthropic",
                        )
                    ],
                )
        finally:
            if wrapper is not None:
                wrapper.close()

    assert exc_info.value.field == "client"
    assert "asynchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", ANTHROPIC_SYNC_VARIANTS)
async def test_async_wrapper_rejects_pinned_sync_anthropic_fallback(
    anthropic_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pins must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = AsyncSolwyn(
                    _uninitialized_client(anthropic_mod, "AsyncAnthropic"),
                    api_key=VALID_API_KEY,
                    provider="anthropic",
                    fallback=[
                        (
                            _uninitialized_client(anthropic_mod, client_name),
                            "fallback-model",
                            {},
                            "anthropic",
                        )
                    ],
                )
        finally:
            if wrapper is not None:
                await wrapper.close()

    assert exc_info.value.field == "client"
    assert "synchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("client_name", ANTHROPIC_SYNC_VARIANTS)
def test_sync_anthropic_variant_pins_accept_primary_and_fallback_without_detection(
    anthropic_mod: Any, client_name: str
) -> None:
    with (
        patch(
            "solwyn._registry.get_adapter_for_client",
            side_effect=AssertionError("explicit pins must bypass detection"),
        ) as detector,
        Solwyn(
            _uninitialized_client(anthropic_mod, client_name),
            api_key=VALID_API_KEY,
            provider="anthropic",
            fallback=[
                (
                    _uninitialized_client(anthropic_mod, client_name),
                    "fallback-model",
                    {},
                    "anthropic",
                )
            ],
        ) as wrapper,
    ):
        assert all(runtime.adapter.name == "anthropic" for runtime in wrapper._solwyn_runtimes)

    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", ANTHROPIC_ASYNC_VARIANTS)
async def test_async_anthropic_variant_pins_accept_primary_and_fallback_without_detection(
    anthropic_mod: Any, client_name: str
) -> None:
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pins must bypass detection"),
    ) as detector:
        wrapper = AsyncSolwyn(
            _uninitialized_client(anthropic_mod, client_name),
            api_key=VALID_API_KEY,
            provider="anthropic",
            fallback=[
                (
                    _uninitialized_client(anthropic_mod, client_name),
                    "fallback-model",
                    {},
                    "anthropic",
                )
            ],
        )
        try:
            assert all(runtime.adapter.name == "anthropic" for runtime in wrapper._solwyn_runtimes)
        finally:
            await wrapper.close()

    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("wrapper", "client_name"),
    [
        (Solwyn, "AsyncBedrockOpenAI"),
        (AsyncSolwyn, "BedrockOpenAI"),
    ],
)
def test_openai_bedrock_variant_pin_rejects_wrong_mode(
    openai_mod: Any, wrapper, client_name: str
) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        wrapper(
            _uninitialized_client(openai_mod, client_name),
            api_key=VALID_API_KEY,
            provider="openai",
        )

    assert exc_info.value.field == "client"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", OPENAI_SYNC_MODULE_CLIENTS)
async def test_async_wrapper_rejects_pinned_sync_openai_module_client(
    openai_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pin must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = AsyncSolwyn(
                    _uninitialized_client(openai_mod, client_name),
                    api_key=VALID_API_KEY,
                    provider="openai",
                )
        finally:
            if wrapper is not None:
                await wrapper.close()

    assert exc_info.value.field == "client"
    assert "synchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("client_name", OPENAI_SYNC_MODULE_CLIENTS)
async def test_async_wrapper_rejects_pinned_sync_openai_module_fallback(
    openai_mod: Any, client_name: str
) -> None:
    wrapper = None
    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("explicit pins must bypass detection"),
    ) as detector:
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                wrapper = AsyncSolwyn(
                    _uninitialized_client(openai_mod, "AsyncOpenAI"),
                    api_key=VALID_API_KEY,
                    provider="openai",
                    fallback=[
                        (
                            _uninitialized_client(openai_mod, client_name),
                            "fallback-model",
                            {},
                            "openai",
                        )
                    ],
                )
        finally:
            if wrapper is not None:
                await wrapper.close()

    assert exc_info.value.field == "client"
    assert "synchronous" in str(exc_info.value)
    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("client_name", OPENAI_SYNC_MODULE_CLIENTS)
def test_sync_openai_module_client_pins_accept_primary_and_fallback_without_detection(
    openai_mod: Any, client_name: str
) -> None:
    with (
        patch(
            "solwyn._registry.get_adapter_for_client",
            side_effect=AssertionError("explicit pins must bypass detection"),
        ) as detector,
        Solwyn(
            _uninitialized_client(openai_mod, client_name),
            api_key=VALID_API_KEY,
            provider="openai",
            fallback=[
                (
                    _uninitialized_client(openai_mod, client_name),
                    "fallback-model",
                    {},
                    "openai",
                )
            ],
        ) as wrapper,
    ):
        assert all(runtime.adapter.name == "openai" for runtime in wrapper._solwyn_runtimes)

    detector.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("client_name", ["Responses", "AsyncResponses"])
def test_openai_resources_are_not_classified_as_sdk_clients(
    openai_mod: Any, client_name: str
) -> None:
    del openai_mod  # fixture scopes this test to the OpenAI SDK's availability
    resources = import_module("openai.resources.responses.responses")

    assert _client_mode(_uninitialized_client(resources, client_name), "openai_sdk") is None


@pytest.mark.unit
@pytest.mark.parametrize("class_name", ["AnthropicError", "AsyncAnthropicMessages"])
def test_anthropic_named_helpers_are_not_classified_as_sdk_clients(class_name: str) -> None:
    helper_type = type(class_name, (), {"__module__": "anthropic.resources"})

    assert _client_mode(helper_type(), "anthropic_sdk") is None
