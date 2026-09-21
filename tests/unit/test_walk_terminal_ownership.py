"""Every exit from the candidate walk ends its reservation exactly once.

Two exits used to leave the claim for the reservation sweep, which REFUNDS it
after ``RESERVATION_MAX_AGE_S`` even though the provider may have billed:

* a non-``Exception`` interrupt (``KeyboardInterrupt``, ``SystemExit``, a
  greenlet kill) raised inside the SYNC walk, and
* an ``Exception`` raised while wrapping an already-open provider stream, in
  both walks.

Unknown post-send usage keeps its reserved bound behind one possibly-succeeded
receipt; a provably unsent request is refunded. Neither is a provider-health
verdict, and a HALF_OPEN probe is freed exactly once.

Seams are service boundaries only: the provider client is a stub, the control
plane is ``FakeControlPlane``, and a wrapping failure is injected at the
provider-adapter seam. The real enforcer, lease ledger, breaker, and reporter
queue run.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import solwyn
from solwyn._types import CircuitState, MetadataEvent
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane

_REQUEST: dict[str, Any] = {"model": "gpt-5.5", "messages": [], "max_completion_tokens": 20}


# ---------------------------------------------------------------------------
# Provider stubs (openai-shaped; detection is duck-typed on module + name)
# ---------------------------------------------------------------------------


class _Status429RetryAfter(Exception):
    """A rejected request the provider asked us to retry: provably unbilled."""

    status_code = 429

    def __init__(self) -> None:
        super().__init__("rate limited")
        self.response = SimpleNamespace(headers={"retry-after": "0"})


class _SyncStream:
    def __init__(self) -> None:
        self.close_calls = 0

    def __iter__(self) -> Any:
        return iter(())

    def close(self) -> None:
        self.close_calls += 1


class _AsyncStream:
    """``aclose()`` when ``native_aclose``, else only an awaitable ``close()``."""

    def __init__(self, *, native_aclose: bool) -> None:
        self.close_calls = 0
        if native_aclose:
            self.aclose = self._close
        else:
            self.close = self._close

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def _close(self) -> None:
        self.close_calls += 1


class _SyncCompletions:
    def __init__(self, outcome: Any) -> None:
        self.calls = 0
        self._outcome = outcome

    def create(self, **_kwargs: object) -> Any:
        self.calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _OpenAIStub:
    def __init__(self, outcome: Any, *, options_error: BaseException | None = None) -> None:
        self.chat = SimpleNamespace(completions=_SyncCompletions(outcome))
        self._options_error = options_error

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        if self._options_error is not None:
            raise self._options_error
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


class _AsyncCompletions:
    def __init__(self, outcome: Any) -> None:
        self.calls = 0
        self._outcome = outcome

    async def create(self, **_kwargs: object) -> Any:
        self.calls += 1
        return self._outcome


class _AsyncOpenAIStub:
    def __init__(self, outcome: Any) -> None:
        self.chat = SimpleNamespace(completions=_AsyncCompletions(outcome))

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIStub:
        return self


_AsyncOpenAIStub.__module__ = "openai._client"
_AsyncOpenAIStub.__name__ = "AsyncOpenAI"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plane() -> FakeControlPlane:
    return FakeControlPlane(granted_tokens=20, headroom_share_tokens=0, final_grant=True)


def _breaker(wrapped: Any, *, half_open: bool) -> CircuitBreaker:
    """The primary's breaker; OPEN-and-eligible so the call takes the one probe."""
    breaker: CircuitBreaker = wrapped._get_circuit_breaker("openai")
    if half_open:
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
    return breaker


def _assert_ledger(wrapped: Any, run_id: str, *, bound_retained: bool) -> str:
    """The claim is gone NOW (nothing left for the sweep); returns its call id."""
    ledger = wrapped._solwyn_budget._lease
    state = ledger.state_for(run_id)
    assert state is not None
    assert state.reservations == {}
    assert state.reserved_tokens == 0
    assert state.granted_remaining_tokens == (0 if bound_retained else 20)
    assert state.spent_tokens_since_report == (20 if bound_retained else 0)
    call_id, claim = next(iter(ledger._call_claims.items()))
    assert call_id not in ledger._call_index
    # A late refund or true-up cannot re-lend a retired bound.
    wrapped._solwyn_budget.release_reservation(call_id, claim.token)
    ledger.true_up(call_id, 0, claim_token=claim.token)
    assert state.granted_remaining_tokens == (0 if bound_retained else 20)
    return str(call_id)


def _assert_neutral_breaker(breaker: CircuitBreaker, *, half_open: bool) -> None:
    # No verdict: a recorded failure would count here and, for a HALF_OPEN
    # probe (whose admission reset the count), re-open the breaker.
    assert breaker.failure_count == 0
    assert breaker.success_count == 0
    assert not breaker._half_open_probe_active
    assert breaker.state is (CircuitState.HALF_OPEN if half_open else CircuitState.CLOSED)


def _assert_one_error_receipt(
    plane: FakeControlPlane,
    call_id: str,
    *,
    possibly_succeeded: bool | None,
    error_class: str,
) -> None:
    events: list[MetadataEvent] = [e for e in plane.ingested if e.call_id == call_id]
    assert len(events) == 1, events
    event = events[0]
    assert event.status == "error"
    assert event.possibly_succeeded is possibly_succeeded
    assert event.failover_error_class == error_class
    assert event.attempt_index == 0
    assert event.lease_id == "lse_fake1"
    assert event.input_tokens == event.output_tokens == 0
    assert all(confirm.call_id != call_id for confirm in plane.confirms)


_BREAKER_CONFIG: dict[str, Any] = {
    "circuit_breaker_recovery_timeout": 0,
    "circuit_breaker_recovery_timeout_jitter": 0,
}


# ---------------------------------------------------------------------------
# Gap 1: a non-Exception interrupt inside the sync walk
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("half_open", [False, True])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_sync_interrupt_mid_dispatch_keeps_the_bound(
    interrupt_type: type[BaseException], half_open: bool
) -> None:
    # Arrange: the interrupt lands INSIDE the provider SDK call — the request
    # may have been sent and billed, so its usage is unknown.
    plane = _plane()
    interrupt = interrupt_type()
    primary, fallback = _OpenAIStub(interrupt), _OpenAIStub(None)
    wrapped = plane.wrap(primary, fallback=[(fallback, "gpt-5.5-mini")], **_BREAKER_CONFIG)
    breaker = _breaker(wrapped, half_open=half_open)

    try:
        with solwyn.run("sync-interrupt-mid-dispatch") as run_id:
            # Act
            with (
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release,
                pytest.raises(interrupt_type) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST)

            # Assert
            assert raised.value is interrupt
            assert release.call_count == 1
            assert primary.chat.completions.calls == 1
            assert fallback.chat.completions.calls == 0
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
            _assert_neutral_breaker(breaker, half_open=half_open)
    finally:
        wrapped.close()

    _assert_one_error_receipt(
        plane, call_id, possibly_succeeded=True, error_class=interrupt_type.__name__
    )
    assert [s.spent_tokens for s in plane.lease_surrenders] == [20]


@pytest.mark.unit
@pytest.mark.parametrize("half_open", [False, True])
def test_sync_interrupt_before_the_provider_is_invoked_refunds(half_open: bool) -> None:
    # Arrange: the interrupt lands in the hop's client preparation, BEFORE the
    # provider method is invoked — provably unsent.
    plane = _plane()
    interrupt = KeyboardInterrupt()
    primary = _OpenAIStub(None, options_error=interrupt)
    wrapped = plane.wrap(primary, **_BREAKER_CONFIG)
    breaker = _breaker(wrapped, half_open=half_open)

    try:
        with solwyn.run("sync-interrupt-pre-dispatch") as run_id:
            # Act
            with (
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release,
                pytest.raises(KeyboardInterrupt) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST)

            # Assert
            assert raised.value is interrupt
            assert release.call_count == 1
            assert primary.chat.completions.calls == 0
            call_id = _assert_ledger(wrapped, run_id, bound_retained=False)
            _assert_neutral_breaker(breaker, half_open=half_open)
    finally:
        wrapped.close()

    _assert_one_error_receipt(
        plane, call_id, possibly_succeeded=None, error_class="KeyboardInterrupt"
    )
    assert [s.spent_tokens for s in plane.lease_surrenders] == [0]


@pytest.mark.unit
def test_sync_interrupt_during_retry_after_sleep_refunds() -> None:
    # Arrange: the provider REJECTED the request (429) and asked for a retry;
    # the interrupt lands in the sleep, so there is no paid work to keep.
    plane = _plane()
    interrupt = KeyboardInterrupt()
    primary = _OpenAIStub(_Status429RetryAfter())
    wrapped = plane.wrap(primary, same_provider_retries=1)
    breaker = _breaker(wrapped, half_open=False)

    try:
        with solwyn.run("sync-interrupt-retry-sleep") as run_id:
            # Act
            with (
                patch("solwyn.client.time.sleep", side_effect=interrupt) as sleep,
                pytest.raises(KeyboardInterrupt) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST)

            # Assert
            assert raised.value is interrupt
            sleep.assert_called_once_with(0.0)
            assert primary.chat.completions.calls == 1
            call_id = _assert_ledger(wrapped, run_id, bound_retained=False)
            _assert_neutral_breaker(breaker, half_open=False)
    finally:
        wrapped.close()

    _assert_one_error_receipt(
        plane, call_id, possibly_succeeded=None, error_class="KeyboardInterrupt"
    )


@pytest.mark.unit
def test_sync_interrupt_cleanup_failure_never_masks_the_interrupt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the receipt cannot be enqueued while the interrupt unwinds.
    plane = _plane()
    interrupt = KeyboardInterrupt()
    wrapped = plane.wrap(_OpenAIStub(interrupt))
    cleanup_error = RuntimeError("reporter unavailable")

    try:
        with solwyn.run("sync-interrupt-cleanup-failure") as run_id:
            # Act
            with (
                patch.object(wrapped._solwyn_reporter, "report", side_effect=cleanup_error),
                caplog.at_level("WARNING", logger="solwyn.client"),
                pytest.raises(KeyboardInterrupt) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST)

            # Assert: the interrupt survives, the bound is still retired, and
            # the log names only the cleanup failure's class.
            assert raised.value is interrupt
            _assert_ledger(wrapped, run_id, bound_retained=True)
    finally:
        wrapped.close()

    messages = [record.getMessage() for record in caplog.records]
    assert "call.interrupt_cleanup_failed: RuntimeError" in messages
    assert all("reporter unavailable" not in message for message in messages)


# ---------------------------------------------------------------------------
# Gap 2: an Exception while wrapping an already-open provider stream
# ---------------------------------------------------------------------------

_WRAP_SEAMS = ["create_stream_accumulator", "unwrap_stream_source", "wrap_stream_result"]


@pytest.mark.unit
@pytest.mark.parametrize("half_open", [False, True])
@pytest.mark.parametrize("seam", _WRAP_SEAMS)
def test_sync_stream_wrap_failure_keeps_the_bound_and_closes_the_stream(
    seam: str, half_open: bool
) -> None:
    # Arrange: dispatch succeeds (the request was sent, a stream is open), then
    # the served adapter fails while the SDK builds its wrapper.
    plane = _plane()
    stream = _SyncStream()
    original = ValueError("wrapping failed")
    primary = _OpenAIStub(stream)
    wrapped = plane.wrap(primary, **_BREAKER_CONFIG)
    breaker = _breaker(wrapped, half_open=half_open)
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("sync-stream-wrap-failure") as run_id:
            # Act
            with (
                patch.object(adapter, seam, side_effect=original),
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release,
                pytest.raises(ValueError) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST, stream=True)

            # Assert
            assert raised.value is original
            assert release.call_count == 1
            assert stream.close_calls == 1
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
            _assert_neutral_breaker(breaker, half_open=half_open)
    finally:
        wrapped.close()

    _assert_one_error_receipt(plane, call_id, possibly_succeeded=True, error_class="ValueError")
    assert [s.spent_tokens for s in plane.lease_surrenders] == [20]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("half_open", [False, True])
@pytest.mark.parametrize("native_aclose", [True, False])
@pytest.mark.parametrize("seam", _WRAP_SEAMS)
async def test_async_stream_wrap_failure_keeps_the_bound_and_closes_the_stream(
    seam: str, native_aclose: bool, half_open: bool
) -> None:
    # Arrange
    plane = _plane()
    stream = _AsyncStream(native_aclose=native_aclose)
    original = ValueError("wrapping failed")
    primary = _AsyncOpenAIStub(stream)
    wrapped = plane.wrap_async(primary, **_BREAKER_CONFIG)
    breaker = _breaker(wrapped, half_open=half_open)
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("async-stream-wrap-failure") as run_id:
            # Act
            with (
                patch.object(adapter, seam, side_effect=original),
                patch.object(breaker, "release_probe", wraps=breaker.release_probe) as release,
                pytest.raises(ValueError) as raised,
            ):
                await wrapped.chat.completions.create(**_REQUEST, stream=True)

            # Assert
            assert raised.value is original
            assert release.call_count == 1
            assert stream.close_calls == 1
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
            _assert_neutral_breaker(breaker, half_open=half_open)
    finally:
        await wrapped.close()

    _assert_one_error_receipt(plane, call_id, possibly_succeeded=True, error_class="ValueError")
    assert [s.spent_tokens for s in plane.lease_surrenders] == [20]


@pytest.mark.unit
def test_sync_stream_wrap_failure_survives_a_failing_provider_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange: the provider's own close raises while the wrap error unwinds.
    plane = _plane()
    stream = _SyncStream()
    original = ValueError("wrapping failed")
    wrapped = plane.wrap(_OpenAIStub(stream))
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("sync-stream-wrap-close-failure") as run_id:
            # Act
            with (
                patch.object(adapter, "create_stream_accumulator", side_effect=original),
                patch.object(stream, "close", side_effect=OSError("socket already gone")),
                caplog.at_level("WARNING", logger="solwyn.client"),
                pytest.raises(ValueError) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST, stream=True)

            # Assert: the original error survives and the bound is still retired.
            assert raised.value is original
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
    finally:
        wrapped.close()

    _assert_one_error_receipt(plane, call_id, possibly_succeeded=True, error_class="ValueError")
    messages = [record.getMessage() for record in caplog.records]
    assert "stream.wrap_close_failed: OSError" in messages
    assert all("socket already gone" not in message for message in messages)


@pytest.mark.unit
def test_sync_interrupt_during_the_wrap_failure_close_settles_exactly_once() -> None:
    # Arrange: an interrupt raised by the provider close is never swallowed;
    # the walk's interrupt handler is then the ONE owner of the settlement.
    plane = _plane()
    stream = _SyncStream()
    interrupt = KeyboardInterrupt()
    wrapped = plane.wrap(_OpenAIStub(stream))
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("sync-stream-wrap-close-interrupt") as run_id:
            # Act
            with (
                patch.object(adapter, "create_stream_accumulator", side_effect=ValueError("wrap")),
                patch.object(stream, "close", side_effect=interrupt),
                pytest.raises(KeyboardInterrupt) as raised,
            ):
                wrapped.chat.completions.create(**_REQUEST, stream=True)

            # Assert
            assert raised.value is interrupt
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
            _assert_neutral_breaker(wrapped._get_circuit_breaker("openai"), half_open=False)
    finally:
        wrapped.close()

    _assert_one_error_receipt(
        plane, call_id, possibly_succeeded=True, error_class="KeyboardInterrupt"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_cancel_during_the_wrap_failure_close_settles_exactly_once() -> None:
    # Arrange: the provider's aclose() blocks until the caller cancels.
    plane = _plane()
    closing = asyncio.Event()

    class _HangingStream(_AsyncStream):
        async def _close(self) -> None:
            closing.set()
            await asyncio.Event().wait()

    wrapped = plane.wrap_async(_AsyncOpenAIStub(_HangingStream(native_aclose=True)))
    adapter = wrapped._solwyn_runtimes[0].adapter

    try:
        with solwyn.run("async-stream-wrap-close-cancel") as run_id:
            # Act
            with patch.object(adapter, "create_stream_accumulator", side_effect=ValueError("wrap")):
                pending = asyncio.create_task(
                    wrapped.chat.completions.create(**_REQUEST, stream=True)
                )
                await asyncio.wait_for(closing.wait(), 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending

            # Assert
            call_id = _assert_ledger(wrapped, run_id, bound_retained=True)
            _assert_neutral_breaker(wrapped._get_circuit_breaker("openai"), half_open=False)
    finally:
        await wrapped.close()

    _assert_one_error_receipt(plane, call_id, possibly_succeeded=True, error_class="CancelledError")
