"""Runtime posture and guarded-resource behavior for untracked capabilities."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY, foreground_records

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError, UntrackedSpendSurfaceError


class _SafeURL:
    pass


class _ResponsesResource:
    def create(self) -> str:
        return "created"

    def delete(self) -> str:
        return "deleted"


class _FutureResource:
    def launch(self) -> str:
        return "launched"

    def sibling(self) -> str:
        return "sibling"


class _ContextResource:
    def __enter__(self) -> _ContextResource:
        return self

    def __exit__(self, *args: object) -> None:
        return None


_ResponsesResource.__module__ = "openai.resources.responses"
_FutureResource.__module__ = "openai.resources.future"
_ContextResource.__module__ = "openai.resources.future"


class _OpenAIClient:
    def __init__(self) -> None:
        self.responses_evaluations = 0
        self.future_evaluations = 0
        self.base_url_evaluations = 0
        self.custom_auth_evaluations = 0
        self._safe_url = _SafeURL()

    @functools.cached_property
    def responses(self) -> _ResponsesResource:
        self.responses_evaluations += 1
        return _ResponsesResource()

    @functools.cached_property
    def future(self) -> _FutureResource:
        self.future_evaluations += 1
        return _FutureResource()

    @functools.cached_property
    def future_context(self) -> _ContextResource:
        return _ContextResource()

    @property
    def base_url(self) -> _SafeURL:
        self.base_url_evaluations += 1
        return self._safe_url

    @property
    def custom_auth(self) -> str:
        self.custom_auth_evaluations += 1
        return "custom-auth"

    def post(self) -> str:
        return "posted"

    def close(self) -> None:
        return None


class _MissingResponsesClient:
    def __init__(self) -> None:
        self._safe_url = _SafeURL()

    @property
    def base_url(self) -> _SafeURL:
        return self._safe_url

    def close(self) -> None:
        return None


class _DynamicOpenAIClient(_MissingResponsesClient):
    def __init__(self) -> None:
        super().__init__()
        self.dynamic_evaluations = 0

    def __getattr__(self, name: str) -> Any:
        self.dynamic_evaluations += 1
        return lambda: name


class _DriftedBaseURLClient(_OpenAIClient):
    @property
    def base_url(self) -> Any:
        self.base_url_evaluations += 1
        return lambda: "dispatch"


class _DriftedPostClient(_OpenAIClient):
    @property
    def post(self) -> str:
        self.post_evaluations += 1
        return "drifted-post"


class _OpaqueClient(_OpenAIClient):
    def __init__(self) -> None:
        super().__init__()
        self.opaque = object()


for _client_type in (
    _OpenAIClient,
    _MissingResponsesClient,
    _DynamicOpenAIClient,
    _DriftedBaseURLClient,
    _DriftedPostClient,
    _OpaqueClient,
):
    _client_type.__module__ = "openai._client"


def _make_solwyn(client: object, **kwargs: object) -> Solwyn:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        wrapper = Solwyn(client, api_key=VALID_API_KEY, **kwargs)
    wrapper._reporter._shutdown.set()
    wrapper._reporter._thread.join(timeout=2.0)
    return wrapper


def _close(wrapper: Solwyn) -> None:
    wrapper.close()


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Iterator[None]:
    _reset_unmetered_spend_warnings()
    yield
    _reset_unmetered_spend_warnings()


@pytest.mark.unit
def test_default_warn_logs_once_and_returns_the_terminal_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrapper = _make_solwyn(_OpenAIClient())

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        first = wrapper.post
        second = wrapper.post

    assert first() == "posted"
    assert second() == "posted"
    records = foreground_records(caplog)
    assert len(records) == 1
    assert records[0].args == ("openai", "openai_sdk", "post", "arbitrary_endpoint")
    _close(wrapper)


@pytest.mark.unit
def test_allow_is_silent_and_returns_the_terminal_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="allow")

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert wrapper.post() == "posted"

    assert foreground_records(caplog) == []
    _close(wrapper)


@pytest.mark.unit
def test_raise_refuses_a_known_untracked_surface_with_its_scope() -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.post

    assert exc_info.value.surface == "post"
    assert exc_info.value.token == "post"
    assert exc_info.value.capability_scope == "arbitrary_endpoint"
    _close(wrapper)


@pytest.mark.unit
def test_exact_acknowledgment_allows_only_the_terminal_leaf() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"responses.create"},
    )

    responses = wrapper.responses

    assert responses is wrapper.responses
    assert responses is not client.responses
    assert responses.create() == "created"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = responses.delete
    _close(wrapper)


@pytest.mark.unit
def test_unknown_acknowledgment_keeps_prefix_guarded_and_future_sibling_refused() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"future.launch"},
    )

    future = wrapper.future

    assert future is wrapper.future
    assert future is not client.future
    assert future.launch() == "launched"
    with pytest.raises(UntrackedSpendSurfaceError):
        _ = future.sibling
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.parametrize(
    "token",
    [
        "responses",
        "responses.*",
        "missing",
        "messages.stream",
        "chat.completions.create",
        "future",
    ],
)
def test_invalid_acknowledgments_are_rejected_even_under_allow(token: str) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        _make_solwyn(
            _OpenAIClient(),
            on_unmetered="allow",
            acknowledge_untracked={token},
        )

    assert exc_info.value.field == "acknowledge_untracked"


@pytest.mark.unit
def test_unsupported_acknowledgment_is_rejected_before_live_graph_validation() -> None:
    with pytest.raises(ConfigurationError, match="classified as unsupported"):
        _make_solwyn(
            _OpenAIClient(),
            provider="groq",
            on_unmetered="allow",
            acknowledge_untracked={"videos.create"},
        )


@pytest.mark.unit
def test_blocked_acknowledgment_is_rejected_before_live_graph_validation() -> None:
    BedrockClient = type(
        "BedrockRuntime",
        (),
        {
            "__module__": "botocore.client",
            "invoke_model": lambda self: None,
        },
    )
    client = BedrockClient()
    client.meta = SimpleNamespace(
        service_model=SimpleNamespace(service_name="bedrock-runtime"),
        region_name="us-east-1",
    )

    with pytest.raises(ConfigurationError, match="classified as blocked"):
        _make_solwyn(
            client,
            on_unmetered="allow",
            acknowledge_untracked={"invoke_model"},
        )


@pytest.mark.unit
def test_exported_conditional_acknowledgment_is_valid_without_a_raw_path() -> None:
    wrapper = _make_solwyn(
        _OpenAIClient(),
        on_unmetered="raise",
        acknowledge_untracked={"audio.speech.create:gpt-4o-mini-tts"},
    )

    assert wrapper._config.acknowledge_untracked == frozenset(
        {"audio.speech.create:gpt-4o-mini-tts"}
    )
    _close(wrapper)


@pytest.mark.unit
def test_strict_invisible_dynamic_unknown_refuses_before_descriptor_evaluation() -> None:
    client = _DynamicOpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.dynamic_evaluations = 0

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.future_operation

    assert client.dynamic_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_strict_invisible_dynamic_known_path_refuses_before_dynamic_lookup() -> None:
    client = _DynamicOpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.dynamic_evaluations = 0

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.responses

    assert client.dynamic_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_strict_untracked_property_is_not_evaluated_before_refusal() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.custom_auth

    assert client.custom_auth_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_untracked_static_shape_drift_reenters_unknown_before_evaluation() -> None:
    client = _DriftedPostClient()
    client.post_evaluations = 0
    wrapper = _make_solwyn(client, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.post

    assert exc_info.value.kind == "unknown"
    assert client.post_evaluations == 0
    _close(wrapper)


@pytest.mark.unit
def test_acknowledged_unevaluated_property_is_not_touched_during_validation() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(
        client,
        on_unmetered="raise",
        acknowledge_untracked={"custom_auth"},
    )

    assert client.custom_auth_evaluations == 0
    assert wrapper.custom_auth == "custom-auth"
    assert client.custom_auth_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_known_missing_path_preserves_provider_attribute_error() -> None:
    wrapper = _make_solwyn(_MissingResponsesClient(), on_unmetered="raise")

    with pytest.raises(AttributeError) as exc_info:
        _ = wrapper.responses

    assert type(exc_info.value) is AttributeError
    _close(wrapper)


@pytest.mark.unit
def test_feature_probe_helpers_treat_strict_refusal_as_absent() -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="raise")
    sentinel = object()

    assert hasattr(wrapper, "post") is False
    assert getattr(wrapper, "post", sentinel) is sentinel
    _close(wrapper)


@pytest.mark.unit
def test_private_names_preserve_raw_passthrough() -> None:
    client = _OpenAIClient()
    client._provider_private = object()
    wrapper = _make_solwyn(client, on_unmetered="raise")

    assert wrapper._provider_private is client._provider_private
    _close(wrapper)


@pytest.mark.unit
def test_safe_metadata_preserves_identity_when_both_shapes_match() -> None:
    client = _OpenAIClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.base_url_evaluations = 0

    assert wrapper.base_url is client._safe_url
    assert client.base_url_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_safe_metadata_return_shape_drift_reenters_strict_unknown_posture() -> None:
    client = _DriftedBaseURLClient()
    wrapper = _make_solwyn(client, on_unmetered="raise")
    client.base_url_evaluations = 0

    with pytest.raises(UntrackedSpendSurfaceError) as exc_info:
        _ = wrapper.base_url

    assert exc_info.value.kind == "unknown"
    assert client.base_url_evaluations == 1
    _close(wrapper)


@pytest.mark.unit
def test_allow_guards_recognized_unknown_resources_but_returns_opaque_values() -> None:
    client = _OpaqueClient()
    wrapper = _make_solwyn(client, on_unmetered="allow")

    future = wrapper.future

    assert future is not client.future
    assert future.launch() == "launched"
    assert wrapper.opaque is client.opaque
    _close(wrapper)


@pytest.mark.unit
def test_guard_does_not_forward_context_manager_special_methods() -> None:
    wrapper = _make_solwyn(_OpenAIClient(), on_unmetered="allow")

    guarded = wrapper.future_context

    with pytest.raises(TypeError), guarded:
        pass
    _close(wrapper)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_wrapper_uses_the_same_strict_resolver() -> None:
    wrapper = AsyncSolwyn(_OpenAIClient(), api_key=VALID_API_KEY, on_unmetered="raise")

    with pytest.raises(UntrackedSpendSurfaceError):
        _ = wrapper.post

    await wrapper.close()
