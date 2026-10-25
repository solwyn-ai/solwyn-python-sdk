"""Real-SDK check: native Together chat completions are metered end to end.

Drives the genuine ``together`` client through its own in-memory
``httpx.MockTransport`` and the hermetic ``FakeControlPlane``. Together 3.0
moved chat completions to a separate inference host and dropped ``/v1`` from
the client base URL; detection and settlement must not depend on either.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from solwyn.testing import FakeControlPlane

together = pytest.importorskip("together", minversion="2.0")

pytestmark = pytest.mark.unit

_PROMPT = "private prompt text that must never leave the process"
_COMPLETION = "private completion text"
_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
_USAGE = {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40}


def _completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_761_000_000,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _COMPLETION},
                "finish_reason": "stop",
            }
        ],
        "usage": _USAGE,
    }


def _sse() -> str:
    base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1, "model": _MODEL}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": _COMPLETION}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": _USAGE},
    ]
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text=_sse(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_completion())

    return handle


def _client(handler: Callable[[httpx.Request], httpx.Response], *, is_async: bool) -> Any:
    if is_async:
        return together.AsyncTogether(
            api_key="tg-test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    return together.Together(
        api_key="tg-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["message", "stream"])
async def test_native_together_chat_completion_settles_provider_usage(
    is_async: bool, streaming: bool
) -> None:
    # Arrange
    seen: list[httpx.Request] = []
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if is_async else plane.wrap)(
        _client(_handler(seen), is_async=is_async), lease_enabled=False, on_unmetered="raise"
    )

    # Act
    try:
        response = wrapper.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=16,
            stream=streaming,
        )
        if is_async:
            response = await response
        if streaming:
            chunks = [chunk async for chunk in response] if is_async else list(response)
            assert chunks[0].choices[0].delta.content == _COMPLETION
        else:
            assert response.choices[0].message.content == _COMPLETION
    finally:
        if is_async:
            await wrapper.close()
        else:
            wrapper.close()

    # Assert: the provider was called once, and the settlement is Together's own usage.
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/chat/completions")
    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert event.provider == "together"
    assert event.model == _MODEL
    assert (event.input_tokens, event.output_tokens) == (31, 9)
    assert event.token_details is None or not event.token_details.is_estimated
    assert _PROMPT not in event.model_dump_json()
    assert _COMPLETION not in event.model_dump_json()
