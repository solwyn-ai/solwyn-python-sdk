"""Real-SDK coverage for provider-native per-hop timeout objects.

Hermetic: provider requests terminate at MockTransport and never open a socket.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from conftest import VALID_API_KEY

from solwyn.client import AsyncSolwyn, Solwyn

_CONNECT_SLICE = 0.25
_READ_WRITE_BOUND = 12.5


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
def test_sync_dispatch_uses_real_openai_native_httpx_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai = pytest.importorskip("openai")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx.Response(200, request=request, json=_openai_response())

    provider = openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    native_timeout_type = type(provider.timeout)
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
async def test_async_dispatch_uses_real_openai_native_httpx_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai = pytest.importorskip("openai")
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx.Response(200, request=request, json=_openai_response())

    provider = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    native_timeout_type = type(provider.timeout)
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
def test_sync_dispatch_uses_real_anthropic_native_httpx2_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    def handler(request: Any) -> Any:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx2.Response(200, request=request, json=_anthropic_response())

    provider = anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    native_timeout_type = type(provider.timeout)
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

    assert type(seen["option_timeout"]) is native_timeout_type
    _assert_granular_timeout(seen["request_timeout"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_dispatch_uses_real_anthropic_native_httpx2_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen: dict[str, Any] = {}

    async def handler(request: Any) -> Any:
        seen["request_timeout"] = request.extensions["timeout"]
        return httpx2.Response(200, request=request, json=_anthropic_response())

    provider = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    native_timeout_type = type(provider.timeout)
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

    assert type(seen["option_timeout"]) is native_timeout_type
    _assert_granular_timeout(seen["request_timeout"])
