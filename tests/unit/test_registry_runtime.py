"""Unit tests for solwyn._registry.build_runtimes (sans-I/O client holder).

The registry detects each client's adapter by *type* only (no network), builds
a ProviderEntry per spec, and returns ProviderRuntime objects in
[primary, *fallbacks] order. Clients are faked with MagicMock whose
__class__.__module__ matches a provider SDK so get_adapter_for_client detects
them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solwyn._registry import ProviderRuntime, build_runtimes
from solwyn._types import ProviderName
from solwyn.exceptions import ConfigurationError


def _mock_client(module: str) -> MagicMock:
    """Return a MagicMock whose type module path triggers adapter detection."""
    client = MagicMock()
    client.__class__.__module__ = module
    return client


@pytest.mark.unit
def test_build_runtimes_primary_only() -> None:
    # Arrange
    primary = _mock_client("openai._client")

    # Act
    runtimes = build_runtimes(primary, "gpt-5.5", [])

    # Assert
    assert len(runtimes) == 1
    assert isinstance(runtimes[0], ProviderRuntime)
    assert runtimes[0].entry.provider == ProviderName.OPENAI
    assert runtimes[0].entry.model == "gpt-5.5"
    assert runtimes[0].entry.default_params == {}
    assert runtimes[0].sdk_client is primary
    assert runtimes[0].adapter.name == "openai"


@pytest.mark.unit
def test_build_runtimes_order_is_primary_then_fallbacks() -> None:
    # Arrange
    primary = _mock_client("openai._client")
    fb_anthropic = _mock_client("anthropic._client")
    fb_google = _mock_client("google.genai")

    # Act
    runtimes = build_runtimes(
        primary,
        "gpt-5.5",
        [(fb_anthropic, "claude-sonnet-5"), (fb_google, "gemini-2.5-pro")],
    )

    # Assert — order preserved: [primary, *fallbacks]
    assert [r.entry.provider for r in runtimes] == [
        ProviderName.OPENAI,
        ProviderName.ANTHROPIC,
        ProviderName.GOOGLE,
    ]
    assert [r.entry.model for r in runtimes] == [
        "gpt-5.5",
        "claude-sonnet-5",
        "gemini-2.5-pro",
    ]
    assert runtimes[1].sdk_client is fb_anthropic
    assert runtimes[2].sdk_client is fb_google


@pytest.mark.unit
def test_entry_provider_matches_detected_adapter() -> None:
    # Arrange — each client maps to a distinct provider
    primary = _mock_client("anthropic._client")
    fb = _mock_client("google.genai")

    # Act
    runtimes = build_runtimes(primary, "claude-sonnet-5", [(fb, "gemini-2.5-pro")])

    # Assert — entry.provider is derived from adapter.name, not the caller
    for runtime in runtimes:
        assert runtime.entry.provider.value == runtime.adapter.name


@pytest.mark.unit
def test_two_tuple_fallback_spec_has_empty_default_params() -> None:
    # Arrange
    primary = _mock_client("openai._client")
    fb = _mock_client("anthropic._client")

    # Act
    runtimes = build_runtimes(primary, "gpt-5.5", [(fb, "claude-sonnet-5")])

    # Assert
    assert runtimes[1].entry.default_params == {}


@pytest.mark.unit
def test_three_tuple_fallback_spec_carries_default_params() -> None:
    # Arrange
    primary = _mock_client("openai._client")
    fb = _mock_client("anthropic._client")
    params = {"max_tokens": 1024, "temperature": 0.2}

    # Act
    runtimes = build_runtimes(primary, "gpt-5.5", [(fb, "claude-sonnet-5", params)])

    # Assert
    assert runtimes[1].entry.default_params == params


@pytest.mark.unit
def test_primary_model_none_yields_empty_model_string() -> None:
    # Arrange
    primary = _mock_client("openai._client")

    # Act
    runtimes = build_runtimes(primary, None, [])

    # Assert — per-call model wins for primary, "" placeholder is fine
    assert runtimes[0].entry.model == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_spec",
    [
        (_mock_client("anthropic._client"),),  # 1-tuple: missing model
        (_mock_client("anthropic._client"), "m", {}, "extra"),  # 4-tuple: too long
        "not-a-tuple",  # not a sequence pair at all
        (_mock_client("anthropic._client"), 123),  # model not a str
        (_mock_client("anthropic._client"), "m", "not-a-dict"),  # params not a dict
    ],
)
def test_malformed_fallback_spec_raises_configuration_error(bad_spec: object) -> None:
    # Arrange
    primary = _mock_client("openai._client")

    # Act / Assert
    with pytest.raises(ConfigurationError):
        build_runtimes(primary, "gpt-5.5", [bad_spec])


@pytest.mark.unit
def test_runtime_is_frozen_dataclass() -> None:
    # Arrange
    primary = _mock_client("openai._client")
    runtimes = build_runtimes(primary, "gpt-5.5", [])

    # Act / Assert — frozen dataclass rejects attribute assignment
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError subclass varies
        runtimes[0].entry = None  # type: ignore[misc]


@pytest.mark.unit
def test_explicit_provider_pin_bypasses_client_detection() -> None:
    """A provider pin is an identity assertion, not detected-client relabeling."""
    primary = _mock_client("openai._client")
    primary.__class__.__name__ = "OpenAI"

    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("provider detection must not run"),
    ) as detect:
        runtimes = build_runtimes(
            primary,
            "gpt-5.5",
            [],
            primary_provider="openai",
        )

    detect.assert_not_called()
    assert runtimes[0].adapter.name == "openai"
    assert runtimes[0].entry.provider == ProviderName.OPENAI


@pytest.mark.unit
def test_fallback_provider_pin_bypasses_client_detection() -> None:
    primary = _mock_client("openai._client")
    fallback = _mock_client("openai._client")

    with patch(
        "solwyn._registry.get_adapter_for_client",
        side_effect=AssertionError("provider detection must not run for a named fallback"),
    ):
        runtimes = build_runtimes(
            primary,
            "gpt-5.5",
            [(fallback, "fallback-model", {}, "vllm")],
            primary_provider="openai",
        )

    assert [runtime.adapter.name for runtime in runtimes] == ["openai", "vllm"]
    assert runtimes[1].entry.provider == ProviderName.VLLM


@pytest.mark.unit
def test_unknown_primary_provider_pin_uses_provider_field() -> None:
    primary = _mock_client("openai._client")

    with pytest.raises(ConfigurationError) as exc_info:
        build_runtimes(primary, "gpt-5.5", [], primary_provider="not-a-provider")

    assert exc_info.value.field == "provider"
    assert "Unknown provider" in str(exc_info.value)
