"""Recovery admission and lease ownership through real, offline OpenAI objects."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from solwyn import run
from solwyn._types import CircuitState
from solwyn.testing import FakeControlPlane

openai = pytest.importorskip("openai")
pytestmark = pytest.mark.unit


class _ConfiguredOrder:
    """Exercise the recovering primary even when a healthy fallback is available."""

    def order(self, candidates, _request):
        return candidates


def _chat_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.5",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _open_probe(wrapped):
    breaker = wrapped._get_circuit_breaker("openai")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    return breaker


@pytest.mark.asyncio
@pytest.mark.parametrize("half_open", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
async def test_dispatch_cancel_retires_claim_without_refunding_paid_authority(half_open, fallback):
    plane = FakeControlPlane(granted_tokens=60, final_grant=True)
    entered = asyncio.Event()
    finish = asyncio.Event()
    requests = []

    async def handle(request):
        requests.append(request.url.path)
        entered.set()
        await finish.wait()
        return _chat_response()

    raw = openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    spare_calls = []

    async def spare_handle(request):
        spare_calls.append(request.url.path)
        return _chat_response()

    spare = openai.AsyncOpenAI(
        api_key="offline",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(spare_handle)),
    )
    wrapped = plane.wrap_async(
        raw,
        fallback=[(spare, "gpt-5.5", {}, "groq")] if fallback else None,
        circuit_breaker_recovery_timeout=0,
        circuit_breaker_recovery_timeout_jitter=0,
        selection_policy=_ConfiguredOrder(),
    )
    breaker = _open_probe(wrapped) if half_open else wrapped._get_circuit_breaker("openai")
    try:
        async with run("cancel-ownership") as run_id:
            pending = asyncio.create_task(
                wrapped.chat.completions.create(
                    model="gpt-5.5", messages=[], max_completion_tokens=20
                )
            )
            await asyncio.wait_for(entered.wait(), 2)
            ledger = wrapped._solwyn_budget._lease
            state = ledger.state_for(run_id)
            call_id, reservation = next(iter(state.reservations.items()))
            remaining = state.granted_remaining_tokens
            with patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert release.call_count == 1
            assert state.reserved_tokens == 0
            assert call_id not in ledger._call_index
            assert state.granted_remaining_tokens == remaining
            assert state.spent_tokens_since_report == reservation.tokens
            assert not breaker._half_open_probe_active
            assert breaker.success_count == 0
            assert breaker.state is (CircuitState.HALF_OPEN if half_open else CircuitState.CLOSED)
            # A late refund or true-up cannot re-lend the retired draw.
            wrapped._solwyn_budget.release_reservation(call_id, reservation.claim_token)
            ledger.true_up(call_id, 0, claim_token=reservation.claim_token)
            assert state.granted_remaining_tokens == remaining
            finish.set()
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], max_completion_tokens=20
            )
            assert len(requests) == 2
            assert spare_calls == []
            assert state.reserved_tokens == 0
        await wrapped.close()
        errors = [e for e in plane.ingested if e.call_id == call_id]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].failover_error_class == "CancelledError"
        assert errors[0].attempt_index == 0
        assert errors[0].lease_id == reservation.lease_id
        assert errors[0].input_tokens == errors[0].output_tokens == 0
        assert all(c.call_id != call_id for c in plane.confirms)
        assert len(plane.lease_surrenders) == 1
    finally:
        finish.set()
        await wrapped.close()
        await spare.close()


@pytest.mark.parametrize("terminal", ["close", "bad_request"])
def test_sync_responses_neutral_terminal_releases_probe_once(terminal):
    plane = FakeControlPlane()
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if request.url.path.endswith("/responses"):
            return httpx.Response(400, json={"error": {"message": "synthetic", "type": "invalid"}})
        return _chat_response()

    raw = openai.OpenAI(
        api_key="offline", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    wrapped = plane.wrap(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_probe(wrapped)
    try:
        with run("responses-neutral") as run_id:
            manager = wrapped.responses.stream(model="gpt-5.5", input=[], max_output_tokens=20)
            assert breaker._half_open_probe_active
            with patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release:
                if terminal == "bad_request":
                    with pytest.raises(openai.BadRequestError):
                        manager.__enter__()
                manager.close()
                manager.close()
                assert release.call_count == 1
            assert not breaker._half_open_probe_active
            assert breaker.success_count == 0
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 0
            wrapped.chat.completions.create(model="gpt-5.5", messages=[], max_completion_tokens=20)
            assert requests.count("/v1/chat/completions") == 1
            assert requests.count("/v1/responses") == (terminal == "bad_request")
    finally:
        wrapped.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "bad_request", "cancel"])
async def test_async_responses_terminal_ownership(terminal):
    plane = FakeControlPlane()
    entered = asyncio.Event()
    requests = []

    async def handle(request):
        requests.append(request.url.path)
        if request.url.path.endswith("/responses"):
            entered.set()
            if terminal == "cancel":
                await asyncio.Event().wait()
            return httpx.Response(400, json={"error": {"message": "synthetic", "type": "invalid"}})
        return _chat_response()

    raw = openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_probe(wrapped)
    try:
        async with run("async-responses") as run_id:
            manager = wrapped.responses.stream(model="gpt-5.5", input=[], max_output_tokens=20)
            assert not breaker._half_open_probe_active  # deferred factory has not admitted
            with patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release:
                if terminal == "bad_request":
                    with pytest.raises(openai.BadRequestError):
                        await manager.__aenter__()
                elif terminal == "cancel":
                    pending = asyncio.create_task(manager.__aenter__())
                    await asyncio.wait_for(entered.wait(), 2)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                await manager.close()
                await manager.close()
                assert release.call_count == (terminal != "close")
            assert not breaker._half_open_probe_active
            assert breaker.success_count == 0
            assert breaker.state is (
                CircuitState.OPEN if terminal == "close" else CircuitState.HALF_OPEN
            )
            if terminal != "close":
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                assert state.reserved_tokens == 0
                assert state.spent_tokens_since_report == (20 if terminal == "cancel" else 0)
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], max_completion_tokens=20
            )
            assert requests.count("/v1/chat/completions") == 1
    finally:
        await wrapped.close()


def _usage_chunk() -> bytes:
    payload = {
        "id": "offline",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.5",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


@pytest.mark.parametrize("failure", [False, True])
def test_live_sync_stream_keeps_probe_until_one_verdict(failure):
    plane = FakeControlPlane()

    class Body(httpx.SyncByteStream):
        closes = 0

        def __iter__(self):
            if failure:
                raise httpx.ReadError("synthetic")
            yield _usage_chunk()

        def close(self):
            self.closes += 1

    body = Body()
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return _chat_response()

    raw = openai.OpenAI(
        api_key="offline", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    wrapped = plane.wrap(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_probe(wrapped)
    try:
        with run("live-sync"):
            with (
                patch.object(breaker, "record_success", wraps=breaker.record_success) as success,
                patch.object(breaker, "record_failure", wraps=breaker.record_failure) as failed,
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as released,
            ):
                stream = wrapped.chat.completions.create(
                    model="gpt-5.5", messages=[], stream=True, max_completion_tokens=20
                )
                assert breaker._half_open_probe_active
                assert not breaker.admit().allowed
                assert success.call_count == failed.call_count == released.call_count == 0
                if failure:
                    with pytest.raises(openai.APIConnectionError):
                        list(stream)
                else:
                    assert len(list(stream)) == 1
                stream.close()
                stream.close()
                assert success.call_count == (not failure)
                assert failed.call_count == failure
                assert released.call_count == 0
                assert body.closes == 1
            wrapped.chat.completions.create(model="gpt-5.5", messages=[], max_completion_tokens=20)
            assert len(calls) == 2
    finally:
        wrapped.close()
    if failure:
        event = next(event for event in plane.ingested if event.status == "error")
        assert event.possibly_succeeded is True
        assert "failover_error_class" not in event.model_dump(exclude_none=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["success", "failure", "cancel"])
async def test_live_async_stream_keeps_probe_until_one_verdict(terminal):
    plane = FakeControlPlane(granted_tokens=60, final_grant=True)
    pulling = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        closes = 0

        async def __aiter__(self):
            pulling.set()
            if terminal == "cancel":
                await asyncio.Event().wait()
            if terminal == "failure":
                raise httpx.ReadError("synthetic")
            yield _usage_chunk()

        async def aclose(self):
            self.closes += 1

    body = Body()
    calls = []

    async def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return _chat_response()

    raw = openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_probe(wrapped)
    try:
        async with run("live-async") as run_id:
            with (
                patch.object(breaker, "record_success", wraps=breaker.record_success) as success,
                patch.object(breaker, "record_failure", wraps=breaker.record_failure) as failed,
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as released,
            ):
                stream = await wrapped.chat.completions.create(
                    model="gpt-5.5", messages=[], stream=True, max_completion_tokens=20
                )
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                assert breaker._half_open_probe_active
                assert not breaker.admit().allowed
                assert success.call_count == failed.call_count == released.call_count == 0
                if terminal == "cancel":
                    pending = asyncio.create_task(anext(stream))
                    await asyncio.wait_for(pulling.wait(), 2)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                elif terminal == "failure":
                    with pytest.raises(openai.APIConnectionError):
                        await anext(stream)
                else:
                    assert len([chunk async for chunk in stream]) == 1
                await stream.close()
                await stream.close()
                assert state.reserved_tokens == 0
                assert state.spent_tokens_since_report == (2 if terminal == "success" else 20)
                assert success.call_count == (terminal == "success")
                assert failed.call_count == (terminal == "failure")
                assert released.call_count == (terminal == "cancel")
                assert body.closes == 1
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], max_completion_tokens=20
            )
            assert len(calls) == 2
    finally:
        await wrapped.close()
    if terminal == "failure":
        event = next(event for event in plane.ingested if event.status == "error")
        assert event.possibly_succeeded is True
        assert "failover_error_class" not in event.model_dump(exclude_none=True)


def test_stale_manager_cleanup_cannot_clear_successor_probe_after_sibling_verdict():
    plane = FakeControlPlane()
    raw = openai.OpenAI(
        api_key="offline",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: _chat_response())),
    )
    wrapped = plane.wrap(raw, circuit_breaker_recovery_timeout=0)
    breaker = wrapped._get_circuit_breaker("openai")
    assert breaker.admit().allowed  # a sibling admitted while CLOSED
    _open_probe(wrapped)
    try:
        manager = wrapped.responses.stream(model="gpt-5.5", input=[], max_output_tokens=20)
        old_token = breaker._half_open_probe_token
        breaker.record_success()  # the already-admitted sibling reports health
        successor = breaker.admit()
        assert successor.allowed and successor.probe_token != old_token
        manager.close()
        manager.close()
        assert breaker._half_open_probe_token == successor.probe_token
        assert not breaker.admit().allowed
        breaker.release_probe(successor)
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    finally:
        wrapped.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
async def test_cancellation_in_context_body_is_neutral_and_keeps_unknown_spend(responses):
    plane = FakeControlPlane(granted_tokens=60, final_grant=True)
    entered = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        closes = 0

        async def __aiter__(self):
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            self.closes += 1

    body = Body()

    async def handle(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    raw = openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_probe(wrapped)

    async def consume():
        stream = (
            wrapped.responses.stream(model="gpt-5.5", input=[], max_output_tokens=20)
            if responses
            else await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], stream=True, max_completion_tokens=20
            )
        )
        async with stream:
            entered.set()
            await asyncio.Event().wait()

    try:
        async with run("context-cancel") as run_id:
            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 20
            assert breaker.success_count == 0
            assert breaker.state is CircuitState.HALF_OPEN
            assert not breaker._half_open_probe_active
            assert body.closes == 1
        await wrapped.close()
        assert plane.confirms == []
        assert len(plane.ingested) == 1
        assert plane.ingested[0].possibly_succeeded is True
    finally:
        await wrapped.close()
