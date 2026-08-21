"""Real-SDK coverage for provider-native per-hop timeout objects.

Hermetic: provider requests terminate at MockTransport and never open a socket.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn.client import AsyncSolwyn, Solwyn, _hop_client_timeout

_CONNECT_SLICE = 0.25
_READ_WRITE_BOUND = 12.5
_TIMEOUT_UNSET = object()


def _openai_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _assert_granular_timeout(values: dict[str, Any]) -> None:
    assert values == {
        "connect": _CONNECT_SLICE,
        "read": _READ_WRITE_BOUND,
        "write": _READ_WRITE_BOUND,
        "pool": _CONNECT_SLICE,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "configured_timeout",
    [_TIMEOUT_UNSET, pytest.param(30.0, id="scalar"), pytest.param(None, id="null")],
)
def test_sync_dispatch_uses_real_openai_native_httpx_timeout(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: object,
) -> None:
    openai = pytest.importorskip("openai")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx.Response(200, request=request, json=_openai_response())

    timeout_kwargs = {} if configured_timeout is _TIMEOUT_UNSET else {"timeout": configured_timeout}
    provider = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **timeout_kwargs,
    )
    native_http_client = inspect.getattr_static(provider, "_client")
    native_timeout_type = type(inspect.getattr_static(native_http_client, "_timeout"))
    original_with_options = provider.with_options

    def capture_with_options(**kwargs: Any) -> Any:
        seen["option_timeout"] = kwargs["timeout"]
        return original_with_options(**kwargs)

    monkeypatch.setattr(provider, "with_options", capture_with_options)
    wrapper = Solwyn(provider, api_key=VALID_API_KEY)
    try:
        response = wrapper._sync_dispatch(
            wrapper._solwyn_runtimes[0],
            {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            is_streaming=False,
            timeout=_CONNECT_SLICE,
            read_timeout=_READ_WRITE_BOUND,
            max_retries=0,
        )
        assert response.choices[0].message.content == "ok"
    finally:
        wrapper.close()

    assert type(seen["option_timeout"]) is native_timeout_type
    _assert_granular_timeout(seen["request_timeout"])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_timeout",
    [_TIMEOUT_UNSET, pytest.param(30.0, id="scalar"), pytest.param(None, id="null")],
)
async def test_async_dispatch_uses_real_openai_native_httpx_timeout(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: object,
) -> None:
    openai = pytest.importorskip("openai")
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx.Response(200, request=request, json=_openai_response())

    timeout_kwargs = {} if configured_timeout is _TIMEOUT_UNSET else {"timeout": configured_timeout}
    provider = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **timeout_kwargs,
    )
    native_http_client = inspect.getattr_static(provider, "_client")
    native_timeout_type = type(inspect.getattr_static(native_http_client, "_timeout"))
    original_with_options = provider.with_options

    def capture_with_options(**kwargs: Any) -> Any:
        seen["option_timeout"] = kwargs["timeout"]
        return original_with_options(**kwargs)

    monkeypatch.setattr(provider, "with_options", capture_with_options)
    wrapper = AsyncSolwyn(provider, api_key=VALID_API_KEY)
    try:
        response = await wrapper._async_dispatch(
            wrapper._solwyn_runtimes[0],
            {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            is_streaming=False,
            timeout=_CONNECT_SLICE,
            read_timeout=_READ_WRITE_BOUND,
            max_retries=0,
        )
        assert response.choices[0].message.content == "ok"
    finally:
        await wrapper.close()

    assert type(seen["option_timeout"]) is native_timeout_type
    _assert_granular_timeout(seen["request_timeout"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "configured_timeout",
    [_TIMEOUT_UNSET, pytest.param(30.0, id="scalar"), pytest.param(None, id="null")],
)
def test_sync_dispatch_uses_real_anthropic_native_httpx2_timeout(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: object,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx2.Response(200, request=request, json=_anthropic_response())

    timeout_kwargs = {} if configured_timeout is _TIMEOUT_UNSET else {"timeout": configured_timeout}
    provider = anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        **timeout_kwargs,
    )
    original_with_options = provider.with_options

    def capture_with_options(**kwargs: Any) -> Any:
        seen["option_timeout"] = kwargs["timeout"]
        return original_with_options(**kwargs)

    monkeypatch.setattr(provider, "with_options", capture_with_options)
    wrapper = Solwyn(provider, api_key=VALID_API_KEY)
    try:
        response = wrapper._sync_dispatch(
            wrapper._solwyn_runtimes[0],
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            is_streaming=False,
            timeout=_CONNECT_SLICE,
            read_timeout=_READ_WRITE_BOUND,
            max_retries=0,
        )
        assert response.content[0].text == "ok"
    finally:
        wrapper.close()

    assert seen["option_timeout"] == (
        _CONNECT_SLICE,
        _READ_WRITE_BOUND,
        _READ_WRITE_BOUND,
        _CONNECT_SLICE,
    )
    _assert_granular_timeout(seen["request_timeout"])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_timeout",
    [_TIMEOUT_UNSET, pytest.param(30.0, id="scalar"), pytest.param(None, id="null")],
)
async def test_async_dispatch_uses_real_anthropic_native_httpx2_timeout(
    monkeypatch: pytest.MonkeyPatch,
    configured_timeout: object,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    async def handler(request: Any) -> Any:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx2.Response(200, request=request, json=_anthropic_response())

    timeout_kwargs = {} if configured_timeout is _TIMEOUT_UNSET else {"timeout": configured_timeout}
    provider = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **timeout_kwargs,
    )
    original_with_options = provider.with_options

    def capture_with_options(**kwargs: Any) -> Any:
        seen["option_timeout"] = kwargs["timeout"]
        return original_with_options(**kwargs)

    monkeypatch.setattr(provider, "with_options", capture_with_options)
    wrapper = AsyncSolwyn(provider, api_key=VALID_API_KEY)
    try:
        response = await wrapper._async_dispatch(
            wrapper._solwyn_runtimes[0],
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            is_streaming=False,
            timeout=_CONNECT_SLICE,
            read_timeout=_READ_WRITE_BOUND,
            max_retries=0,
        )
        assert response.content[0].text == "ok"
    finally:
        await wrapper.close()

    assert seen["option_timeout"] == (
        _CONNECT_SLICE,
        _READ_WRITE_BOUND,
        _READ_WRITE_BOUND,
        _CONNECT_SLICE,
    )
    _assert_granular_timeout(seen["request_timeout"])


@pytest.mark.unit
def test_mutated_httpx2_timeout_export_and_state_never_execute_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    class ExecutableTimeout:
        constructions = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).constructions += 1
            vars(self).update(kwargs)

    provider = anthropic.Anthropic(api_key="sk-ant-test")
    native_http_client = inspect.getattr_static(provider, "_client")
    configured = object.__new__(ExecutableTimeout)
    monkeypatch.setattr(httpx2, "Timeout", ExecutableTimeout)
    monkeypatch.setattr(native_http_client, "_timeout", configured)
    try:
        timeout = _hop_client_timeout(provider, _CONNECT_SLICE, _READ_WRITE_BOUND)
    finally:
        provider.close()

    assert timeout == (
        _CONNECT_SLICE,
        _READ_WRITE_BOUND,
        _READ_WRITE_BOUND,
        _CONNECT_SLICE,
    )
    assert ExecutableTimeout.constructions == 0


@pytest.mark.unit
def test_mutated_httpx2_sync_client_export_never_executes_class_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    class EqualityTrapMeta(type):
        comparisons = 0

        def __eq__(cls, other: object) -> bool:
            type(cls).comparisons += 1
            raise AssertionError("mutated httpx2.Client equality executed")

        __hash__ = type.__hash__

    class MutatedClient(metaclass=EqualityTrapMeta):
        pass

    provider = anthropic.Anthropic(api_key="sk-ant-test")
    monkeypatch.setattr(httpx2, "Client", MutatedClient)
    try:
        timeout = _hop_client_timeout(provider, _CONNECT_SLICE, _READ_WRITE_BOUND)
    finally:
        provider.close()

    assert type(timeout) is httpx.Timeout
    assert EqualityTrapMeta.comparisons == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mutated_httpx2_async_client_export_never_executes_class_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    class EqualityTrapMeta(type):
        comparisons = 0

        def __eq__(cls, other: object) -> bool:
            type(cls).comparisons += 1
            raise AssertionError("mutated httpx2.AsyncClient equality executed")

        __hash__ = type.__hash__

    class MutatedAsyncClient(metaclass=EqualityTrapMeta):
        pass

    provider = anthropic.AsyncAnthropic(api_key="sk-ant-test")
    monkeypatch.setattr(httpx2, "AsyncClient", MutatedAsyncClient)
    try:
        timeout = _hop_client_timeout(provider, _CONNECT_SLICE, _READ_WRITE_BOUND)
    finally:
        await provider.close()

    assert type(timeout) is httpx.Timeout
    assert EqualityTrapMeta.comparisons == 0


@pytest.mark.unit
def test_unknown_native_client_mro_never_executes_class_equality() -> None:
    class EqualityTrapMeta(type):
        comparisons = 0

        def __eq__(cls, other: object) -> bool:
            type(cls).comparisons += 1
            raise AssertionError("native client MRO equality executed")

        __hash__ = type.__hash__

    class NativeClient(metaclass=EqualityTrapMeta):
        pass

    timeout = _hop_client_timeout(
        SimpleNamespace(_client=NativeClient()),
        _CONNECT_SLICE,
        _READ_WRITE_BOUND,
    )

    assert type(timeout) is httpx.Timeout
    assert EqualityTrapMeta.comparisons == 0


@pytest.mark.unit
def test_unknown_client_uses_frozen_httpx_timeout_fallback() -> None:
    timeout = _hop_client_timeout(object(), _CONNECT_SLICE, _READ_WRITE_BOUND)

    assert type(timeout) is httpx.Timeout
    assert timeout == httpx.Timeout(
        connect=_CONNECT_SLICE,
        read=_READ_WRITE_BOUND,
        write=_READ_WRITE_BOUND,
        pool=_CONNECT_SLICE,
    )


@pytest.mark.unit
def test_timeout_discovery_does_not_execute_provider_descriptors() -> None:
    class DescriptorTrap:
        @property
        def _client(self) -> object:
            raise RuntimeError("provider _client descriptor executed")

        @property
        def timeout(self) -> object:
            raise RuntimeError("provider timeout descriptor executed")

    timeout = _hop_client_timeout(DescriptorTrap(), _CONNECT_SLICE, _READ_WRITE_BOUND)

    assert type(timeout) is httpx.Timeout
    assert timeout.connect == _CONNECT_SLICE
    assert timeout.read == _READ_WRITE_BOUND


@pytest.mark.unit
def test_timeout_discovery_does_not_execute_an_unproven_constructor() -> None:
    class ExecutableTimeout:
        constructions = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).constructions += 1
            vars(self).update(kwargs)

    configured = object.__new__(ExecutableTimeout)
    configured.connect = 5.0
    configured.read = 30.0
    configured.write = 30.0
    configured.pool = 5.0
    client = SimpleNamespace(timeout=configured)

    timeout = _hop_client_timeout(client, _CONNECT_SLICE, _READ_WRITE_BOUND)

    assert type(timeout) is httpx.Timeout
    assert ExecutableTimeout.constructions == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "configured",
    [
        pytest.param(SimpleNamespace(connect=True, read=1.0, write=1.0, pool=1.0), id="bool"),
        pytest.param(SimpleNamespace(connect="1", read=1.0, write=1.0, pool=1.0), id="nonnumeric"),
        pytest.param(SimpleNamespace(connect=1.0, read=1.0, write=1.0), id="malformed"),
    ],
)
def test_timeout_discovery_rejects_malformed_unproven_shapes(configured: object) -> None:
    native_http_client = httpx.Client()
    native_http_client._timeout = configured
    try:
        timeout = _hop_client_timeout(
            SimpleNamespace(_client=native_http_client), _CONNECT_SLICE, _READ_WRITE_BOUND
        )
    finally:
        native_http_client.close()

    assert type(timeout) is httpx.Timeout
    assert timeout == httpx.Timeout(
        connect=_CONNECT_SLICE,
        read=_READ_WRITE_BOUND,
        write=_READ_WRITE_BOUND,
        pool=_CONNECT_SLICE,
    )


@pytest.mark.unit
def test_identity_proven_timeout_constructor_failure_fails_closed() -> None:
    native_http_client = httpx.Client()
    client = SimpleNamespace(_client=native_http_client)
    try:
        with (
            patch.object(httpx.Timeout, "__init__", side_effect=ValueError("constructor failed")),
            pytest.raises(RuntimeError, match="native timeout construction failed"),
        ):
            _hop_client_timeout(client, _CONNECT_SLICE, _READ_WRITE_BOUND)
    finally:
        native_http_client.close()
