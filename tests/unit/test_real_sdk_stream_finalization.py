"""Invalid streamed usage cannot strand a real SDK stream's admission owner."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from solwyn import _run_control, run
from solwyn._types import CircuitState
from solwyn.testing import FakeControlPlane

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
pytestmark = pytest.mark.unit


def _invalid_usage_events():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": -1, "output_tokens": 0},
                },
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    for event, data in events:
        yield f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class _SyncBody(httpx2.SyncByteStream):
    def __init__(self) -> None:
        self.closes = 0

    def __iter__(self):
        yield from _invalid_usage_events()

    def close(self) -> None:
        self.closes += 1


class _AsyncBody(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.closes = 0

    async def __aiter__(self):
        for event in _invalid_usage_events():
            yield event

    async def aclose(self) -> None:
        self.closes += 1


def _healthy_response():
    return httpx2.Response(
        200,
        json={
            "id": "msg_healthy",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def _open_breaker(wrapper):
    breaker = wrapper._get_circuit_breaker("anthropic")
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    return breaker


def _observe_finalization(stream, errors):
    """Observe the real accumulator exception without replacing its behavior."""
    finalize = stream._accumulator.finalize

    def observed():
        try:
            return finalize()
        except ValidationError as exc:
            errors.append(exc)
            raise

    return patch.object(stream._accumulator, "finalize", autospec=True, side_effect=observed)


def _assert_retired(wrapper, run_id, state, call_id, breaker):
    assert state.reserved_tokens == 0
    assert call_id not in wrapper._solwyn_budget._lease._call_index
    assert state.granted_remaining_tokens == 20
    assert state.spent_tokens_since_report == 20
    assert not breaker._half_open_probe_active
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.success_count == breaker.failure_count == 0
    assert run_id not in _run_control._STATE.active_handles


def _assert_receipt(plane, call_id, lease_id):
    errors = [event for event in plane.ingested if event.call_id == call_id]
    assert len(errors) == 1
    assert errors[0].possibly_succeeded is True
    assert errors[0].failover_error_class == "ValidationError"
    assert errors[0].lease_id == lease_id
    assert errors[0].attempt_index == 0
    assert errors[0].input_tokens == errors[0].output_tokens == 0
    assert not any(confirm.call_id == call_id for confirm in plane.confirms)


def test_sync_real_anthropic_finalization_error_retires_owner_neutrally() -> None:
    plane = FakeControlPlane(granted_tokens=40, final_grant=True)
    body = _SyncBody()
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if len(requests) == 1:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return _healthy_response()

    raw = anthropic.Anthropic(
        api_key="offline", http_client=httpx2.Client(transport=httpx2.MockTransport(handle))
    )
    wrapped = plane.wrap(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_breaker(wrapped)
    try:
        with run("sync-finalization-error") as run_id:
            stream = wrapped.messages.create(
                model="claude-sonnet-4-5", messages=[], stream=True, max_tokens=20
            )
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state is not None
            call_id, reservation = next(iter(state.reservations.items()))
            assert state.reserved_tokens == 20
            assert breaker._half_open_probe_active
            observed_errors = []
            with _observe_finalization(stream, observed_errors) as finalize:
                with pytest.raises(ValidationError) as raised:
                    list(stream)
                assert len(observed_errors) == 1
                assert raised.value is observed_errors[0]
                assert raised.value.errors()[0]["loc"] == ("input_tokens",)
                stream.close()
                stream.close()
                finalize.assert_called_once()
            assert body.closes == 1
            _assert_retired(wrapped, run_id, state, call_id, breaker)
            wrapped.messages.create(model="claude-sonnet-4-5", messages=[], max_tokens=20)
            assert requests == ["/v1/messages", "/v1/messages"]
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 22
        wrapped.close()
        _assert_receipt(plane, call_id, reservation.lease_id)
        assert len(plane.lease_surrenders) == 1
        assert body.closes == 1
    finally:
        wrapped.close()


@pytest.mark.asyncio
async def test_async_real_anthropic_finalization_error_retires_owner_neutrally() -> None:
    plane = FakeControlPlane(granted_tokens=40, final_grant=True)
    body = _AsyncBody()
    requests = []

    async def handle(request):
        requests.append(request.url.path)
        if len(requests) == 1:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return _healthy_response()

    raw = anthropic.AsyncAnthropic(
        api_key="offline", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    )
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_breaker(wrapped)
    try:
        async with run("async-finalization-error") as run_id:
            stream = await wrapped.messages.create(
                model="claude-sonnet-4-5", messages=[], stream=True, max_tokens=20
            )
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state is not None
            call_id, reservation = next(iter(state.reservations.items()))
            assert state.reserved_tokens == 20
            assert breaker._half_open_probe_active
            observed_errors = []
            with _observe_finalization(stream, observed_errors) as finalize:
                with pytest.raises(ValidationError) as raised:
                    async for _ in stream:
                        pass
                assert len(observed_errors) == 1
                assert raised.value is observed_errors[0]
                assert raised.value.errors()[0]["loc"] == ("input_tokens",)
                await stream.close()
                await stream.close()
                finalize.assert_called_once()
            assert body.closes == 1
            _assert_retired(wrapped, run_id, state, call_id, breaker)
            await wrapped.messages.create(model="claude-sonnet-4-5", messages=[], max_tokens=20)
            assert requests == ["/v1/messages", "/v1/messages"]
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 22
        await wrapped.close()
        _assert_receipt(plane, call_id, reservation.lease_id)
        assert len(plane.lease_surrenders) == 1
        assert body.closes == 1
    finally:
        await wrapped.close()
