"""Parity checks from the TS #26 round-3 review, on the Python SDK.

(1) An ESTABLISHED stream that may have billed tokens and then ends early
(provider disconnect, provider error, or the caller breaking off and closing
it) must keep its funding: a lease claim stays spent at its bound, so the next
call in the run cannot reuse those tokens, and a per-call hold is not
confirmed down to a partial or zero usage.
(2) A caller cancellation racing a known-unbilled 400/429 must not mark that
attempt as possibly billed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import solwyn
from solwyn.testing import FakeControlPlane

openai = pytest.importorskip("openai")

MESSAGES = [{"role": "user", "content": "hi"}]
MODEL = "gpt-4o"
_REQUEST = {"model": MODEL, "messages": MESSAGES, "max_tokens": 100}


def _chunk(content: str) -> bytes:
    body = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


_CHAT_BODY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": MODEL,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _BrokenStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Two content chunks, then the connection drops (or the provider errors)."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __iter__(self) -> Any:
        yield _chunk("he")
        yield _chunk("llo")
        raise self._error

    async def __aiter__(self) -> Any:
        yield _chunk("he")
        yield _chunk("llo")
        raise self._error


class _OpenStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A long stream the caller abandons after the first chunks."""

    def __iter__(self) -> Any:
        for _ in range(50):
            yield _chunk("x")

    async def __aiter__(self) -> Any:
        for _ in range(50):
            yield _chunk("x")


class _Provider:
    """First request streams via ``first``; later requests answer normally."""

    def __init__(self, first: Any) -> None:
        self.first = first
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        if len(self.bodies) == 1:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=self.first
            )
        return httpx.Response(200, json=_CHAT_BODY)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    def client(self, *, is_async: bool) -> Any:
        if is_async:
            return openai.AsyncOpenAI(
                api_key="sk-test",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler)),
                max_retries=0,
            )
        return openai.OpenAI(
            api_key="sk-test",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


async def _consume(stream: Any, *, is_async: bool, break_after: int | None) -> None:
    seen = 0
    if break_after == -1:
        await stream.close() if is_async else stream.close()
        return
    if is_async:
        async for _ in stream:
            seen += 1
            if break_after is not None and seen >= break_after:
                break
        await stream.close()
    else:
        for _ in stream:
            seen += 1
            if break_after is not None and seen >= break_after:
                break
        stream.close()


@pytest.mark.unit
class TestEstablishedStreamKeepsItsLeaseClaim:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("ending", ["disconnect", "provider_error", "caller_break"])
    async def test_early_end_keeps_the_claim_spent(self, is_async: bool, ending: str) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        if ending == "disconnect":
            first: Any = _BrokenStream(httpx.RemoteProtocolError("peer closed connection"))
        elif ending == "provider_error":
            first = _BrokenStream(httpx.ReadError("connection reset"))
        else:
            first = _OpenStream()
        provider = _Provider(first)
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))

        async def create(**extra: Any) -> Any:
            result = wrapped.chat.completions.create(**_REQUEST, **extra)
            return await result if is_async else result

        try:
            with solwyn.run(f"stream-{ending}") as run_id:
                stream = await create(stream=True)
                if ending == "caller_break":
                    await _consume(stream, is_async=is_async, break_after=2)
                else:
                    with pytest.raises(openai.APIError):
                        await _consume(stream, is_async=is_async, break_after=None)
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                first_spent, first_reserved = state.spent_tokens_since_report, state.reserved_tokens
                await create()
        finally:
            if is_async:
                await wrapped.close()
            else:
                wrapped.close()

        assert len(provider.bodies) == 2
        assert first_reserved == 0
        # The established stream may have billed its whole bound: it stays spent.
        assert first_spent >= 100
        # So the second call no longer fits the lease and takes a per-call check.
        assert [check.model for check in plane.checks] == [MODEL]
        assert plane.lease_surrenders[-1].spent_tokens >= 100

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("break_after", [0, 2])
    async def test_per_call_break_keeps_the_hold(self, is_async: bool, break_after: int) -> None:
        plane = FakeControlPlane()
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        try:
            result = wrapped.chat.completions.create(**_REQUEST, stream=True)
            stream = await result if is_async else result
            await _consume(stream, is_async=is_async, break_after=break_after or -1)
        finally:
            if is_async:
                await wrapped.close()
            else:
                wrapped.close()

        # No confirm frees the server hold. The receipt is the stream's success
        # with its observed usage; nothing was measured, so it is the estimate.
        assert plane.confirms == []
        assert len(plane.ingested) == 1
        event = plane.ingested[0]
        assert event.status.value == "success"
        assert event.token_details is not None
        assert event.token_details.is_estimated is True
        assert event.input_tokens > 0


_RATE_LIMIT = {"error": {"message": "slow down", "type": "rate_limit_error", "code": None}}
_BAD_REQUEST = {"error": {"message": "bad", "type": "invalid_request_error", "code": None}}


class _Rejecting:
    """Every request answers ``status`` (a known-unbilled 400 or 429)."""

    def __init__(self, status: int, *, on_request: Any = None) -> None:
        self.status = status
        self.requests = 0
        self.on_request = on_request

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.on_request is not None:
            self.on_request()
        body = _RATE_LIMIT if self.status == 429 else _BAD_REQUEST
        return httpx.Response(self.status, json=body, headers={"retry-after": "1"})

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    def client(self, *, is_async: bool) -> Any:
        if is_async:
            return openai.AsyncOpenAI(
                api_key="sk-test",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler)),
                max_retries=0,
            )
        return openai.OpenAI(
            api_key="sk-test",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


def _assert_unbilled(plane: FakeControlPlane, first_spent: int, first_reserved: int) -> None:
    # The rejected attempt released its claim: nothing reserved, nothing spent.
    assert first_reserved == 0
    assert first_spent == 0
    assert [e.possibly_succeeded for e in plane.ingested if e.possibly_succeeded] == []
    assert plane.lease_surrenders[-1].spent_tokens == 0


@pytest.mark.unit
class TestCancellationRacingAKnownUnbilledRejection:
    """A caller cancel or timeout during a known-unbilled 400/429 never bills it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", ["timeout", "cancel"])
    async def test_async_cancel_during_retry_after_sleep_releases(self, how: str) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Rejecting(429)
        wrapped = plane.wrap_async(provider.client(is_async=True), same_provider_retries=1)
        try:
            with solwyn.run(f"cancel-429-{how}") as run_id:
                call = wrapped.chat.completions.create(**_REQUEST)
                if how == "timeout":
                    with pytest.raises(TimeoutError):
                        await asyncio.wait_for(call, 0.2)
                else:
                    task = asyncio.ensure_future(call)
                    while provider.requests == 0:
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.05)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert provider.requests == 1  # cancelled inside the Retry-After sleep
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
        finally:
            await wrapped.close()
        _assert_unbilled(plane, spent, reserved)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 429])
    async def test_async_cancel_requested_while_the_rejection_lands(self, status: int) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        holder: dict[str, asyncio.Task[Any]] = {}
        provider = _Rejecting(status, on_request=lambda: holder["task"].cancel())
        wrapped = plane.wrap_async(provider.client(is_async=True))
        try:
            with solwyn.run(f"cancel-race-{status}") as run_id:
                task = asyncio.ensure_future(wrapped.chat.completions.create(**_REQUEST))
                holder["task"] = task
                # Solwyn has no suspension point between classifying the
                # rejection and releasing its claim, so the pending cancel
                # cannot land in between: the caller sees the provider error.
                with pytest.raises(openai.APIStatusError):
                    await task
                assert provider.requests == 1
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
        finally:
            await wrapped.close()
        _assert_unbilled(plane, spent, reserved)

    def test_sync_interrupt_during_retry_after_sleep_releases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import solwyn.client as client_module

        def interrupted_sleep(_seconds: float) -> None:
            raise KeyboardInterrupt

        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Rejecting(429)
        wrapped = plane.wrap(provider.client(is_async=False), same_provider_retries=1)
        monkeypatch.setattr(client_module.time, "sleep", interrupted_sleep)
        try:
            with solwyn.run("interrupt-429") as run_id:
                with pytest.raises(KeyboardInterrupt):
                    wrapped.chat.completions.create(**_REQUEST)
                assert provider.requests == 1
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
        finally:
            monkeypatch.undo()
            wrapped.close()
        _assert_unbilled(plane, spent, reserved)
