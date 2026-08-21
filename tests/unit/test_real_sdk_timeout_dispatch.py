"""Real-SDK coverage for provider-native per-hop timeout objects.

Hermetic: provider requests terminate at MockTransport and never open a socket.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn.client import AsyncSolwyn, Solwyn, _hop_client_timeout

_CONNECT_SLICE = 0.25
_READ_WRITE_BOUND = 12.5
_TIMEOUT_UNSET = object()
# Stainless-generated SDKs (openai, anthropic) stringify the per-request timeout
# into this header: `timeout.read if isinstance(timeout, Timeout) else timeout`.
# A non-Timeout hop timeout therefore leaks its whole repr onto the wire.
_READ_TIMEOUT_HEADER = "x-stainless-read-timeout"


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


def _granular_tuple() -> tuple[float, float, float, float]:
    """The transport-correct fallback shape: (connect, read, write, pool)."""
    return (_CONNECT_SLICE, _READ_WRITE_BOUND, _READ_WRITE_BOUND, _CONNECT_SLICE)


def _install_fake_httpx2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client_type: type,
    timeout_export: object = _TIMEOUT_UNSET,
) -> None:
    """Install a fake ``httpx2`` whose exports we control by identity.

    A fake module (rather than monkeypatching the real one) is what makes the
    httpx2 branch reachable with hostile exports: ``client_type`` becomes the
    module's ``Client``, so any instance of it is a *proven* httpx2 native client
    and discovery proceeds to the Timeout proofs instead of short-circuiting on
    the native-client MRO.
    """
    module = ModuleType("httpx2")
    module.Client = client_type
    module.AsyncClient = type("FakeHttpx2AsyncClient", (), {})
    if timeout_export is not _TIMEOUT_UNSET:
        module.Timeout = timeout_export
    monkeypatch.setitem(sys.modules, "httpx2", module)


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
        seen["read_timeout_header"] = request.headers.get(_READ_TIMEOUT_HEADER)
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
    # httpx1 control: a real native Timeout also keeps the advertised read bound
    # honest instead of stringifying a tuple into the header.
    assert seen["read_timeout_header"] == str(_READ_WRITE_BOUND)


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
        seen["read_timeout_header"] = request.headers.get(_READ_TIMEOUT_HEADER)
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
    # httpx1 control: a real native Timeout also keeps the advertised read bound
    # honest instead of stringifying a tuple into the header.
    assert seen["read_timeout_header"] == str(_READ_WRITE_BOUND)


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
        seen["read_timeout_header"] = request.headers.get(_READ_TIMEOUT_HEADER)
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

    # A real, identity-proven httpx2.Timeout - not a four-tuple, which anthropic
    # would stringify whole into x-stainless-read-timeout.
    assert type(seen["option_timeout"]) is httpx2.Timeout
    assert seen["option_timeout"] == httpx2.Timeout(
        connect=_CONNECT_SLICE,
        read=_READ_WRITE_BOUND,
        write=_READ_WRITE_BOUND,
        pool=_CONNECT_SLICE,
    )
    _assert_granular_timeout(seen["request_timeout"])
    assert seen["read_timeout_header"] == str(_READ_WRITE_BOUND)


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
        seen["read_timeout_header"] = request.headers.get(_READ_TIMEOUT_HEADER)
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

    # A real, identity-proven httpx2.Timeout - not a four-tuple, which anthropic
    # would stringify whole into x-stainless-read-timeout.
    assert type(seen["option_timeout"]) is httpx2.Timeout
    assert seen["option_timeout"] == httpx2.Timeout(
        connect=_CONNECT_SLICE,
        read=_READ_WRITE_BOUND,
        write=_READ_WRITE_BOUND,
        pool=_CONNECT_SLICE,
    )
    _assert_granular_timeout(seen["request_timeout"])
    assert seen["read_timeout_header"] == str(_READ_WRITE_BOUND)


@pytest.mark.unit
def test_identity_proven_httpx2_timeout_export_is_the_constructor_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one constructor Solwyn may call: the loaded export the client itself holds."""

    class FakeTimeout:
        constructions = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).constructions += 1
            vars(self).update(kwargs)

    class FakeHttpx2Client:
        pass

    _install_fake_httpx2(monkeypatch, client_type=FakeHttpx2Client, timeout_export=FakeTimeout)
    native_http_client = FakeHttpx2Client()
    # object.__new__ so the client's own state costs no construction count.
    native_http_client._timeout = object.__new__(FakeTimeout)

    timeout = _hop_client_timeout(
        SimpleNamespace(_client=native_http_client), _CONNECT_SLICE, _READ_WRITE_BOUND
    )

    assert type(timeout) is FakeTimeout
    assert FakeTimeout.constructions == 1
    assert vars(timeout) == {
        "connect": _CONNECT_SLICE,
        "read": _READ_WRITE_BOUND,
        "write": _READ_WRITE_BOUND,
        "pool": _CONNECT_SLICE,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "export_kind",
    ["foreign_class", "callable_instance", "absent"],
)
def test_unproven_httpx2_timeout_export_is_never_called(
    monkeypatch: pytest.MonkeyPatch,
    export_kind: str,
) -> None:
    """No export survives without matching the class the client already instantiated."""
    calls: list[str] = []

    class ExportedTimeout:
        def __init__(self, **kwargs: object) -> None:
            calls.append("foreign_class")

    class CallableExport:
        def __call__(self, **kwargs: object) -> object:
            calls.append("callable_instance")
            return object()

    class ClientTimeout:
        def __init__(self, **kwargs: object) -> None:
            calls.append("client_timeout")

    class FakeHttpx2Client:
        pass

    exports: dict[str, object] = {
        "foreign_class": ExportedTimeout,  # a real type, but not the client's
        "callable_instance": CallableExport(),  # callable, but not a type at all
        "absent": _TIMEOUT_UNSET,  # module exports no Timeout
    }
    _install_fake_httpx2(
        monkeypatch, client_type=FakeHttpx2Client, timeout_export=exports[export_kind]
    )
    native_http_client = FakeHttpx2Client()
    native_http_client._timeout = object.__new__(ClientTimeout)

    timeout = _hop_client_timeout(
        SimpleNamespace(_client=native_http_client), _CONNECT_SLICE, _READ_WRITE_BOUND
    )

    assert timeout == _granular_tuple()
    assert calls == []


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
def test_native_client_timeout_descriptor_is_never_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proving the Timeout class must not run provider code (getattr_static only)."""

    class FakeTimeout:
        constructions = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).constructions += 1

    class FakeHttpx2Client:
        @property
        def _timeout(self) -> object:
            raise AssertionError("native client _timeout descriptor executed")

    _install_fake_httpx2(monkeypatch, client_type=FakeHttpx2Client, timeout_export=FakeTimeout)

    timeout = _hop_client_timeout(
        SimpleNamespace(_client=FakeHttpx2Client()), _CONNECT_SLICE, _READ_WRITE_BOUND
    )

    assert timeout == _granular_tuple()
    assert FakeTimeout.constructions == 0


@pytest.mark.unit
def test_proven_httpx2_timeout_constructor_failure_falls_back_to_granular_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx2 has a transport-correct fallback, so it degrades instead of failing closed."""

    class ExplodingTimeout:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("constructor failed")

    class FakeHttpx2Client:
        pass

    _install_fake_httpx2(monkeypatch, client_type=FakeHttpx2Client, timeout_export=ExplodingTimeout)
    native_http_client = FakeHttpx2Client()
    native_http_client._timeout = object.__new__(ExplodingTimeout)

    timeout = _hop_client_timeout(
        SimpleNamespace(_client=native_http_client), _CONNECT_SLICE, _READ_WRITE_BOUND
    )

    assert timeout == _granular_tuple()


@pytest.mark.unit
@pytest.mark.parametrize("timeout_state", ["removed", "malformed", "foreign_type"])
def test_real_httpx2_client_without_proven_timeout_state_still_bounds_the_transport(
    monkeypatch: pytest.MonkeyPatch,
    timeout_state: str,
) -> None:
    """Fallback path, end to end: the tuple still reaches the transport as granular bounds."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["request_timeout"] = request.extensions["timeout"]
        seen["read_timeout_header"] = request.headers.get(_READ_TIMEOUT_HEADER)
        return httpx2.Response(200, request=request, json=_anthropic_response())

    native_http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    provider = anthropic.Anthropic(
        api_key="sk-ant-test", max_retries=0, http_client=native_http_client
    )
    if timeout_state == "removed":
        monkeypatch.delattr(native_http_client, "_timeout")
    elif timeout_state == "malformed":
        monkeypatch.setattr(native_http_client, "_timeout", SimpleNamespace(connect=1.0, read=1.0))
    else:
        # A real Timeout - from the wrong httpx major.
        monkeypatch.setattr(native_http_client, "_timeout", httpx.Timeout(1.0))

    timeout = _hop_client_timeout(provider, _CONNECT_SLICE, _READ_WRITE_BOUND)
    assert timeout == _granular_tuple()

    try:
        provider.with_options(timeout=timeout, max_retries=0).messages.create(
            model="claude-sonnet-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    finally:
        provider.close()

    _assert_granular_timeout(seen["request_timeout"])
    # Documented degradation of the fallback: the header carries the tuple repr.
    assert seen["read_timeout_header"] == str(_granular_tuple())


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
