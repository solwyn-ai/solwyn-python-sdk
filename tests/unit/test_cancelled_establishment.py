"""Cancellation ownership at pre-send, retry, and lazy first-chunk boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from solwyn import _run_control, run
from solwyn._types import CircuitState
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _RetryAfter(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("synthetic rate limit")
        self.response = SimpleNamespace(headers={"retry-after": "2"})


class AsyncOpenAI:
    __module__ = "openai._client"

    def __init__(self, *, retry_first: bool = False) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.calls = 0
        self.retry_first = retry_first
        self.options_error: asyncio.CancelledError | None = None

    def with_options(self, **_kwargs: object) -> AsyncOpenAI:
        if self.options_error is not None:
            raise self.options_error
        return self

    async def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        if self.retry_first and self.calls == 1:
            raise _RetryAfter()
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    async def close(self) -> None:
        pass


class _HeldFirstChunk:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.pulls = 0
        self.closes = 0
        self.cancellation: asyncio.CancelledError | None = None

    def __aiter__(self) -> _HeldFirstChunk:
        return self

    async def __anext__(self) -> object:
        self.pulls += 1
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            self.cancellation = exc
            raise
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closes += 1


class AsyncClient:
    __module__ = "google.genai.client"

    def __init__(self, source: _HeldFirstChunk) -> None:
        self.models = SimpleNamespace(
            generate_content_stream=self.stream,
            generate_content=self.generate,
        )
        self.source = source
        self.stream_calls = 0
        self.generate_calls = 0

    async def stream(self, **_kwargs: object) -> _HeldFirstChunk:
        self.stream_calls += 1
        return self.source

    async def generate(self, **_kwargs: object) -> SimpleNamespace:
        self.generate_calls += 1
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1)
        )

    async def aclose(self) -> None:
        pass


def _open_breaker(wrapper, provider: str):
    breaker = wrapper._get_circuit_breaker(provider)
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    return breaker


async def test_retry_sleep_cancellation_refunds_rejected_attempt_without_retry_dispatch() -> None:
    plane = FakeControlPlane(granted_tokens=20, final_grant=True)
    raw = AsyncOpenAI(retry_first=True)
    wrapped = plane.wrap_async(raw, same_provider_retries=1, circuit_breaker_recovery_timeout=0)
    breaker = _open_breaker(wrapped, "openai")
    sleep_entered = asyncio.Event()
    original_sleep = asyncio.sleep
    seen_cancellations = []

    async def held_retry_sleep(delay: float) -> None:
        if delay != 2.0:
            await original_sleep(delay)
            return
        sleep_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            seen_cancellations.append(exc)
            raise

    try:
        async with run("retry-cancelled") as run_id:
            with patch("solwyn.client.asyncio.sleep", new=held_retry_sleep):
                pending = asyncio.create_task(
                    wrapped.chat.completions.create(
                        model="gpt-5.5", messages=[], max_completion_tokens=20
                    )
                )
                await asyncio.wait_for(sleep_entered.wait(), timeout=2)
                state = wrapped._solwyn_budget._lease.state_for(run_id)
                assert state is not None
                call_id, reservation = next(iter(state.reservations.items()))
                assert state.reserved_tokens == 20
                assert breaker._half_open_probe_active
                pending.cancel()
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await pending
                assert len(seen_cancellations) == 1
                assert seen_cancellations[0] is cancelled.value

            assert raw.calls == 1
            assert state.reserved_tokens == 0
            assert state.granted_remaining_tokens == 20
            assert state.spent_tokens_since_report == 0
            assert not breaker._half_open_probe_active
            assert breaker.state is CircuitState.HALF_OPEN
            assert breaker.success_count == 0
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], max_completion_tokens=20
            )
            assert raw.calls == 2
            assert state.reserved_tokens == 0
        await wrapped.close()
        errors = [event for event in plane.ingested if event.call_id == call_id]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is None
        assert errors[0].failover_error_class == "CancelledError"
        assert errors[0].lease_id == reservation.lease_id
        assert not any(confirm.call_id == call_id for confirm in plane.confirms)
        assert len(plane.lease_surrenders) == 1
    finally:
        await wrapped.close()


async def test_google_first_chunk_cancel_closes_source_and_conservatively_retires_draw() -> None:
    plane = FakeControlPlane(granted_tokens=40, final_grant=True)
    source = _HeldFirstChunk()
    raw = AsyncClient(source)
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_breaker(wrapped, "google")
    try:
        async with run("google-establishment-cancel") as run_id:
            pending = asyncio.create_task(
                wrapped.models.generate_content_stream(
                    model="gemini-3.5-flash", contents=[], config={"max_output_tokens": 20}
                )
            )
            await asyncio.wait_for(source.entered.wait(), timeout=2)
            ledger = wrapped._solwyn_budget._lease
            state = ledger.state_for(run_id)
            assert state is not None
            call_id, reservation = next(iter(state.reservations.items()))
            assert state.reserved_tokens == 20
            pending.cancel()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await pending
            assert cancelled.value is source.cancellation
            assert state.reserved_tokens == 0
            assert call_id not in ledger._call_index
            assert state.granted_remaining_tokens == 20
            assert state.spent_tokens_since_report == 20
            assert not breaker._half_open_probe_active
            assert breaker.state is CircuitState.HALF_OPEN
            assert breaker.success_count == 0
            assert run_id not in _run_control._STATE.active_handles
            assert raw.stream_calls == source.pulls == source.closes == 1

            # Late cleanup cannot return this potentially paid allowance.
            wrapped._solwyn_budget.release_reservation(call_id, reservation.claim_token)
            assert state.granted_remaining_tokens == 20
            await wrapped.models.generate_content(
                model="gemini-3.5-flash", contents=[], config={"max_output_tokens": 20}
            )
            assert raw.generate_calls == 1
        await wrapped.close()
        errors = [event for event in plane.ingested if event.call_id == call_id]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        assert errors[0].failover_error_class == "CancelledError"
        assert errors[0].lease_id == reservation.lease_id
        assert errors[0].input_tokens == errors[0].output_tokens == 0
        assert not any(confirm.call_id == call_id for confirm in plane.confirms)
        assert source.closes == 1
    finally:
        await wrapped.close()


@pytest.mark.parametrize("boundary", ["with_options", "prepare_call"])
async def test_proven_pre_send_cancellation_refunds_draw_and_preserves_exception(boundary) -> None:
    plane = FakeControlPlane(granted_tokens=20, final_grant=True)
    raw = AsyncOpenAI()
    wrapped = plane.wrap_async(raw, circuit_breaker_recovery_timeout=0)
    breaker = _open_breaker(wrapped, "openai")
    cancellation = asyncio.CancelledError("synthetic pre-send cancellation")
    try:
        async with run("pre-send-cancel") as run_id:
            if boundary == "with_options":
                raw.options_error = cancellation
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await wrapped.chat.completions.create(
                        model="gpt-5.5", messages=[], max_completion_tokens=20
                    )
                raw.options_error = None
            else:
                adapter = wrapped._solwyn_runtimes[0].adapter
                with patch.object(
                    adapter, "prepare_call", autospec=True, side_effect=cancellation
                ) as prepare:
                    with pytest.raises(asyncio.CancelledError) as cancelled:
                        await wrapped.chat.completions.create(
                            model="gpt-5.5", messages=[], max_completion_tokens=20
                        )
                    prepare.assert_called_once()

            assert cancelled.value is cancellation
            assert raw.calls == 0
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state is not None
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 0
            assert state.granted_remaining_tokens == 20
            assert not breaker._half_open_probe_active
            assert breaker.state is CircuitState.HALF_OPEN
            assert breaker.success_count == 0
            await wrapped.chat.completions.create(
                model="gpt-5.5", messages=[], max_completion_tokens=20
            )
            assert raw.calls == 1
        await wrapped.close()
        errors = [
            event for event in plane.ingested if event.failover_error_class == "CancelledError"
        ]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is None
        assert errors[0].input_tokens == errors[0].output_tokens == 0
    finally:
        raw.options_error = None
        await wrapped.close()
