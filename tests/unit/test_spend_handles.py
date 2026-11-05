"""Spend handles returned before a key refusal must not dispatch after it.

From the TypeScript SDK's round-3 review: a handle that sends its provider
request later (a Google chat session, an OpenAI or Anthropic stream helper)
is fenced where the request really starts, not where the handle was made.
Each test builds the handle, records a blocking refusal, then uses the handle,
and asserts no provider request was sent.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from solwyn._api_key_errors import Refusal
from solwyn.exceptions import ApiKeyRefusedError
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.project_validation]


class _Wire:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        return httpx.Response(500, json={"error": {"message": "unexpected dispatch"}})

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def _block(wrapped: Any) -> None:
    binding = wrapped._solwyn_binding
    binding.refuse(Refusal(code="api_key_invalid", posture="hard_deny"), binding.generation)


def _genai(wire: _Wire) -> Any:
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    return genai.Client(
        api_key="k",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.async_handler)),
        ),
    )


@pytest.mark.parametrize("method", ["send_message", "send_message_stream"])
def test_google_chat_session_made_before_a_refusal_never_sends_after_it(method: str) -> None:
    wire = _Wire()
    plane = FakeControlPlane()
    wrapped = plane.wrap(_genai(wire), model="gemini-2.5-flash", on_unmetered="allow")
    with wrapped:
        chat = wrapped.chats.create(model="gemini-2.5-flash")
        pending = getattr(chat, method)  # a reference taken before the refusal too
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            result = pending("hi")
            if method == "send_message_stream":
                next(iter(result))

    assert wire.paths == []


def test_google_chat_stream_generator_made_before_a_refusal_never_sends_after_it() -> None:
    wire = _Wire()
    plane = FakeControlPlane()
    wrapped = plane.wrap(_genai(wire), model="gemini-2.5-flash", on_unmetered="allow")
    with wrapped:
        chat = wrapped.chats.create(model="gemini-2.5-flash")
        stream = chat.send_message_stream("hi")  # nothing sent until iterated
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            next(iter(stream))

    assert wire.paths == []


@pytest.mark.asyncio
async def test_async_google_chat_session_made_before_a_refusal_never_sends_after_it() -> None:
    wire = _Wire()
    plane = FakeControlPlane()
    wrapped = plane.wrap_async(_genai(wire).aio, model="gemini-2.5-flash", on_unmetered="allow")
    try:
        chat = wrapped.chats.create(model="gemini-2.5-flash")
        pending = chat.send_message("hi")  # a coroutine: nothing sent yet
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            await pending
    finally:
        await wrapped.close()

    assert wire.paths == []


def _openai(wire: _Wire) -> Any:
    openai = pytest.importorskip("openai")
    return openai.OpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
    )


def _async_openai(wire: _Wire) -> Any:
    openai = pytest.importorskip("openai")
    return openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.async_handler)),
    )


def test_openai_chat_stream_helper_made_before_a_refusal_never_sends_after_it() -> None:
    wire = _Wire()
    plane = FakeControlPlane()
    wrapped = plane.wrap(_openai(wire), on_unmetered="allow")
    with wrapped:
        manager = wrapped.chat.completions.stream(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError), manager:
            pass

    assert wire.paths == []


@pytest.mark.asyncio
async def test_async_openai_chat_stream_helper_made_before_a_refusal_never_sends_after_it() -> None:
    wire = _Wire()
    plane = FakeControlPlane()
    wrapped = plane.wrap_async(_async_openai(wire), on_unmetered="allow")
    try:
        manager = wrapped.chat.completions.stream(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError):
            async with manager:
                pass
    finally:
        await wrapped.close()

    assert wire.paths == []


def test_anthropic_message_stream_made_before_a_refusal_never_sends_after_it() -> None:
    anthropic = pytest.importorskip("anthropic")
    try:
        import httpx2 as provider_httpx  # Anthropic 2.x ships its own httpx fork
    except ImportError:
        provider_httpx = httpx
    paths: list[str] = []

    def handler(request: Any) -> Any:
        paths.append(request.url.path)
        return provider_httpx.Response(500, json={"error": {"message": "unexpected dispatch"}})

    raw = anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=provider_httpx.Client(transport=provider_httpx.MockTransport(handler)),
    )
    plane = FakeControlPlane()
    wrapped = plane.wrap(raw, on_unmetered="allow")
    with wrapped:
        manager = wrapped.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
        _block(wrapped)
        with pytest.raises(ApiKeyRefusedError), manager:
            pass

    assert paths == []
