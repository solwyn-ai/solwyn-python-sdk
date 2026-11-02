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
import gc
import json
import threading
import time
import weakref
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


# ── Dropped without close(): GC and client close settle once ─────────────


def _wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


async def _await_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


async def _read(stream: Any, *, is_async: bool, chunks: int) -> None:
    for _ in range(chunks):
        if is_async:
            await anext(stream)
        else:
            next(stream)


async def _close_client(wrapped: Any, *, is_async: bool) -> None:
    if is_async:
        await wrapped.close()
    else:
        wrapped.close()


async def _open_stream(wrapped: Any, *, is_async: bool) -> Any:
    # No local alias may outlive the caller's reference: GC tests drop it.
    if is_async:
        return await wrapped.chat.completions.create(**_REQUEST, stream=True)
    return wrapped.chat.completions.create(**_REQUEST, stream=True)


@pytest.mark.unit
class TestDroppedStreamSettlesOnce:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_dropped_lease_stream_keeps_its_claim_spent(self, is_async: bool) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))

        async def create(**extra: Any) -> Any:
            result = wrapped.chat.completions.create(**_REQUEST, **extra)
            return await result if is_async else result

        try:
            with solwyn.run("dropped-lease") as run_id:
                stream = await create(stream=True)
                await _read(stream, is_async=is_async, chunks=2)
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                assert state.reserved_tokens > 0
                ref = weakref.ref(stream)
                del stream
                gc.collect()
                assert ref() is None
                # GC hands settlement to the run-cleanup worker thread.
                settled = (
                    await _await_until(lambda: state.reserved_tokens == 0)
                    if is_async
                    else _wait_until(lambda: state.reserved_tokens == 0)
                )
                assert settled
                assert state.spent_tokens_since_report >= 100
                await create()
        finally:
            await _close_client(wrapped, is_async=is_async)

        # The second call no longer fits the lease, so it takes a check.
        assert [check.model for check in plane.checks] == [MODEL]
        assert plane.lease_surrenders[-1].spent_tokens >= 100
        first = plane.ingested[0]
        assert first.status.value == "success"
        assert first.token_details.is_estimated is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_dropped_per_call_stream_keeps_the_hold(self, is_async: bool) -> None:
        plane = FakeControlPlane()
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        try:
            stream = await _open_stream(wrapped, is_async=is_async)
            await _read(stream, is_async=is_async, chunks=2)
            del stream
            gc.collect()
        finally:
            # Settled by the GC handoff, or by close() if the worker has not
            # run yet: exactly one receipt either way.
            await _close_client(wrapped, is_async=is_async)

        assert plane.confirms == []
        assert [event.status.value for event in plane.ingested] == ["success"]
        assert plane.ingested[0].token_details.is_estimated is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_finalizer_keeps_nothing_alive(self, is_async: bool) -> None:
        from solwyn import _run_control

        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        client_ref = weakref.ref(wrapped)
        with solwyn.run("dropped-gc") as run_id:
            stream = await _open_stream(wrapped, is_async=is_async)
            await _read(stream, is_async=is_async, chunks=1)
            stream_ref = weakref.ref(stream)
            cell_ref = weakref.ref(stream._settlement)
            assert run_id in _run_control._STATE.active_handles
            del stream
            gc.collect()
            assert stream_ref() is None
            # Settled on the cleanup worker, which then lets go of the cell.
            assert _wait_until(lambda: (gc.collect(), cell_ref())[1] is None)
            # The stream's run-stop handle went with it.
            assert _wait_until(lambda: run_id not in _run_control._STATE.active_handles)
        await _close_client(wrapped, is_async=is_async)
        del wrapped
        gc.collect()
        assert _wait_until(lambda: (gc.collect(), client_ref())[1] is None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_client_close_settles_an_open_stream_once(self, is_async: bool) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        with solwyn.run("close-while-open") as run_id:
            stream = await _open_stream(wrapped, is_async=is_async)
            await _read(stream, is_async=is_async, chunks=2)
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            await _close_client(wrapped, is_async=is_async)
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report >= 100
            # The caller's own close afterwards settles nothing more.
            if is_async:
                await stream.close()
            else:
                stream.close()
            del stream
            gc.collect()

        assert [event.status.value for event in plane.ingested] == ["success"]
        # The settlement shipped before the reporter sealed, and the surrender
        # (sent by close) counted the claim at its bound.
        assert len(plane.confirms) == 1
        assert plane.lease_surrenders[-1].spent_tokens >= 100

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("order", ["close_first", "dropped_first", "threads"])
    async def test_close_and_finalizer_settle_once(self, is_async: bool, order: str) -> None:
        plane = FakeControlPlane()
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        try:
            stream = await _open_stream(wrapped, is_async=is_async)
            await _read(stream, is_async=is_async, chunks=1)
            settlement = stream._settlement

            async def close() -> None:
                if is_async:
                    await stream.close()
                else:
                    stream.close()

            if order == "close_first":
                await close()
                settlement.settle_dropped()
            elif order == "dropped_first":
                settlement.settle_dropped()
                await close()
            else:
                barrier = threading.Barrier(8)

                def race() -> None:
                    barrier.wait()
                    settlement.settle_dropped()

                threads = [threading.Thread(target=race) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                await close()
        finally:
            await _close_client(wrapped, is_async=is_async)

        assert len(plane.ingested) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_dropped_settlement_never_raises(self, is_async: bool) -> None:
        plane = FakeControlPlane()
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        try:
            stream = await _open_stream(wrapped, is_async=is_async)
            await _read(stream, is_async=is_async, chunks=1)
            settlement = stream._settlement
            settlement._accumulator = None  # finalize() now raises AttributeError
            settlement.settle_dropped()
            assert settlement.settled
            settlement.settle_dropped()
        finally:
            await _close_client(wrapped, is_async=is_async)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_dropped_after_reporter_shutdown_floors_and_counts(self, is_async: bool) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Provider(_OpenStream())
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))
        try:
            with solwyn.run("reporter-closed") as run_id:
                stream = await _open_stream(wrapped, is_async=is_async)
                await _read(stream, is_async=is_async, chunks=1)
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                if is_async:
                    await wrapped._solwyn_reporter.close()
                else:
                    wrapped._solwyn_reporter.close()
                settlement = stream._settlement
                del stream
                gc.collect()
                assert _wait_until(lambda: settlement.settled)
                # Locally the claim still holds its bound ...
                assert state.reserved_tokens == 0
                assert state.spent_tokens_since_report >= 100
                # ... but the sealed reporter refuses the pair and counts it.
                drops = wrapped._solwyn_reporter.dropped_counts
                assert drops.get("settlement_confirm.closed_enqueue") == 1
                assert drops.get("event.closed_enqueue") == 1
        finally:
            await _close_client(wrapped, is_async=is_async)
        assert plane.confirms == []
        assert plane.ingested == []


_EXIT_SCRIPT = r"""
import json, httpx, openai, solwyn
from solwyn.testing import FakeControlPlane

def _chunk(i):
    body = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
    return ("data: " + json.dumps(body) + "\n\n").encode()

class Body(httpx.SyncByteStream):
    def __iter__(self):
        for i in range(50):
            yield _chunk(i)

def handler(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

plane = FakeControlPlane(granted_tokens=128, final_grant=True)
client = plane.wrap(openai.OpenAI(api_key="sk-test", max_retries=0,
    http_client=httpx.Client(transport=httpx.MockTransport(handler))))
run = solwyn.run("exit-open")
run.__enter__()
KEPT = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
    max_tokens=100, stream=True)
next(KEPT)
DROPPED = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": "hi"}], max_tokens=100, stream=True)
next(DROPPED)
del DROPPED
print("ok", flush=True)
"""


@pytest.mark.unit
def test_open_and_dropped_streams_at_interpreter_exit_raise_nothing(tmp_path: Any) -> None:
    import subprocess
    import sys

    script = tmp_path / "exit_open_stream.py"
    script.write_text(_EXIT_SCRIPT)
    result = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"
    assert "Traceback" not in result.stderr
    assert "Exception ignored" not in result.stderr


# ── An unrecognized failure after dispatch is possibly billed (TS finding 4) ──


class _Failing:
    """First request fails after dispatch in ``kind``; later requests succeed."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.requests > 1:
            return httpx.Response(200, json=_CHAT_BODY)
        if self.kind == "truncated_json":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=b'{"id": "c", "cho'
            )
        if self.kind == "decode_error":
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                content=b"not gzip at all",
            )
        if self.kind == "unknown_cause":
            raise RuntimeError("transport adapter blew up after sending")
        if self.kind == "overloaded_529":
            return httpx.Response(529, json={"error": {"message": "overloaded"}})
        return httpx.Response(500, json={"error": {"message": "server error"}})

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


@pytest.mark.unit
class TestUnrecognizedPostDispatchFailureKeepsItsClaim:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("kind", ["truncated_json", "decode_error", "unknown_cause"])
    async def test_claim_stays_spent_and_next_call_is_checked(
        self, is_async: bool, kind: str
    ) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Failing(kind)
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))

        async def create() -> Any:
            result = wrapped.chat.completions.create(**_REQUEST)
            return await result if is_async else result

        try:
            with solwyn.run(f"post-dispatch-{kind}") as run_id:
                with pytest.raises(Exception) as raised:
                    await create()
                assert not isinstance(raised.value, solwyn.SolwynError)
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
                await create()
        finally:
            await _close_client(wrapped, is_async=is_async)

        assert provider.requests == 2
        assert reserved == 0
        # The provider may have billed the whole bound: it stays spent ...
        assert spent >= 100
        # ... so the second call no longer fits the 128-token lease.
        assert [check.model for check in plane.checks] == [MODEL]
        assert plane.lease_surrenders[-1].spent_tokens >= 100
        failed = plane.ingested[0]
        assert failed.status.value == "error"
        # A stopped chain flags its receipt. A body that failed to decode is
        # post-send ambiguous (no failover under "safe"); a handler's own
        # exception propagates raw and stops the chain as unknown.
        assert failed.possibly_succeeded is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_a_529_is_a_known_rejection_and_releases(self, is_async: bool) -> None:
        plane = FakeControlPlane(granted_tokens=128, final_grant=True)
        provider = _Failing("overloaded_529")
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client(is_async=is_async))

        async def create() -> Any:
            result = wrapped.chat.completions.create(**_REQUEST)
            return await result if is_async else result

        try:
            with solwyn.run("post-dispatch-529") as run_id:
                with pytest.raises(openai.APIStatusError):
                    await create()
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
                await create()
        finally:
            await _close_client(wrapped, is_async=is_async)

        assert (spent, reserved) == (0, 0)
        # The released claim funds the next call: no per-call check.
        assert plane.checks == []
        assert plane.ingested[0].possibly_succeeded is None


# ── Every funding-keeping rule applies under hard_deny only ──────────────


class _ByMode:
    """First request ends in ``case``; later requests answer normally."""

    def __init__(self, case: str, *, is_async: bool) -> None:
        self.case = case
        self.is_async = is_async
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.requests > 1:
            return httpx.Response(200, json=_CHAT_BODY)
        if self.case in ("caller_break", "dropped"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_OpenStream()
            )
        if self.case == "mid_stream_error":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_BrokenStream(httpx.ReadError("connection reset")),
            )
        if self.case == "read_timeout":
            raise httpx.ReadTimeout("no response")
        if self.case == "truncated_json":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=b'{"id": "c", "cho'
            )
        # cancel_in_flight: the caller gives up while the request is out.
        raise asyncio.CancelledError() if self.is_async else KeyboardInterrupt()

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    def client(self) -> Any:
        if self.is_async:
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


_CASES = [
    "caller_break",
    "dropped",
    "mid_stream_error",
    "read_timeout",
    "truncated_json",
    "cancel_in_flight",
]


@pytest.mark.unit
class TestFundingRulesByMode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("case", _CASES)
    @pytest.mark.parametrize("budget_mode", ["hard_deny", "alert_only"])
    async def test_possibly_billed_funding_is_kept_only_under_hard_deny(
        self, budget_mode: str, case: str, is_async: bool
    ) -> None:
        plane = FakeControlPlane(mode=budget_mode, granted_tokens=128, final_grant=True)
        provider = _ByMode(case, is_async=is_async)
        wrapped = (plane.wrap_async if is_async else plane.wrap)(provider.client())
        streaming = case in ("caller_break", "dropped", "mid_stream_error")

        async def create(**extra: Any) -> Any:
            result = wrapped.chat.completions.create(**_REQUEST, **extra)
            return await result if is_async else result

        try:
            with solwyn.run(f"by-mode-{budget_mode}-{case}") as run_id:
                state = None
                if streaming:
                    stream = await create(stream=True)
                    state = wrapped._solwyn_budget._lease.state_for(run_id)
                    if case == "caller_break":
                        await _consume(stream, is_async=is_async, break_after=2)
                    elif case == "dropped":
                        await _read(stream, is_async=is_async, chunks=2)
                        del stream
                        gc.collect()
                        assert _wait_until(lambda: state.reserved_tokens == 0)
                    else:
                        with pytest.raises(openai.APIError):
                            await _consume(stream, is_async=is_async, break_after=None)
                else:
                    expected: Any = (
                        (asyncio.CancelledError if is_async else KeyboardInterrupt)
                        if case == "cancel_in_flight"
                        else Exception
                    )
                    with pytest.raises(expected):
                        await create()
                    state = wrapped._solwyn_budget._lease.state_for(run_id)
                spent, reserved = state.spent_tokens_since_report, state.reserved_tokens
                await create()
        finally:
            await _close_client(wrapped, is_async=is_async)

        assert reserved == 0
        first = plane.ingested[0]
        if budget_mode == "hard_deny":
            # Kept at the bound: the next call no longer fits the lease.
            assert spent >= 100
            assert [check.model for check in plane.checks] == [MODEL]
        else:
            # Released (or settled at the observed zero): the lease funds it.
            assert spent == 0
            assert plane.checks == []
        if case in ("caller_break", "dropped"):
            assert first.status.value == "success"
            # A lease confirm goes out in both modes; hard_deny sends the
            # estimate (floored locally), alert_only the observed zero usage.
            assert plane.confirms[0].token_details.is_estimated is (budget_mode == "hard_deny")
        else:
            # The receipt flag does not depend on the mode.
            assert first.status.value == "error"
            assert first.possibly_succeeded is True


# ── A body that fails to decode is post-send ambiguous for routing ───────

_GROQ_URL = "https://api.groq.com/openai/v1"
_FALLBACK_MODEL = "llama-3.3-70b-versatile"


class _Answering:
    def __init__(self) -> None:
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        return httpx.Response(200, json={**_CHAT_BODY, "model": _FALLBACK_MODEL})

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)

    def client(self, *, is_async: bool) -> Any:
        if is_async:
            return openai.AsyncOpenAI(
                api_key="gsk-test",
                base_url=_GROQ_URL,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler)),
                max_retries=0,
            )
        return openai.OpenAI(
            api_key="gsk-test",
            base_url=_GROQ_URL,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            max_retries=0,
        )


@pytest.mark.unit
class TestUndecodableBodyRouting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("idempotency", ["safe", "always"])
    async def test_undecodable_body_fails_over_only_when_always(
        self, idempotency: str, is_async: bool
    ) -> None:
        plane = FakeControlPlane()
        primary, fallback = _Failing("decode_error"), _Answering()
        wrapped = (plane.wrap_async if is_async else plane.wrap)(
            primary.client(is_async=is_async),
            fallback=[(fallback.client(is_async=is_async), _FALLBACK_MODEL)],
            failover_idempotency=idempotency,
        )

        async def create() -> Any:
            result = wrapped.chat.completions.create(**_REQUEST)
            return await result if is_async else result

        try:
            if idempotency == "safe":
                with pytest.raises(openai.APIConnectionError):
                    await create()
            else:
                await create()
        finally:
            await _close_client(wrapped, is_async=is_async)

        assert primary.requests == 1
        if idempotency == "safe":
            # The provider answered: failing over could bill a second time.
            assert fallback.requests == 0
            assert plane.ingested[0].possibly_succeeded is True
        else:
            # "always" may fail over; under hard_deny the next hop is re-admitted.
            assert fallback.requests == 1
            assert [check.model for check in plane.checks] == [MODEL, _FALLBACK_MODEL]
