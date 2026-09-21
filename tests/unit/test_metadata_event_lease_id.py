"""``MetadataEvent.lease_id``: every event of a lease-admitted call names its lease.

The Cloud API's lease float cannot tell that an ingested cost event belongs
to a lease unless the event says so; only the confirm used to. When the API
holds the event and not its confirm (a dropped, exhausted, or breaker-held
confirm) it counted the call twice. The SDK's half of the fix is to tag the
event with the SAME lease id the confirm carries, taken from the admission
result — never from the ledger's live state — on success and error events
alike. Every unfunded or reservation-funded event stays untagged.

Seams are service boundaries only: the provider client is a stub and the
control plane is ``FakeControlPlane`` (zero network, production wire
models). The real enforcer, lease ledger, reporter queue, and adapters run.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import solwyn
from solwyn._types import BudgetConfirmRequest, MetadataEvent
from solwyn.testing import FakeControlPlane

_LEASE_GRANT_PATH = "/api/v1/budgets/lease"
_MESSAGES = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# Provider stubs (openai-shaped; detection is duck-typed on module + name)
# ---------------------------------------------------------------------------


class _Status429(Exception):
    """Rate limit: a FAILOVER disposition, so the chain advances."""

    status_code = 429


class _Status503(Exception):
    """5xx: POST_SEND_AMBIGUOUS — re-raised, never failed over by default."""

    status_code = 503


class _Status400(Exception):
    """Request refusal: FAIL_FAST must return unused lease authority."""

    status_code = 400


def _response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _stream_chunks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
            choices=[],
        ),
    ]


class _SyncCompletions:
    def __init__(self, error: Exception | None, before_return: Any) -> None:
        self.calls = 0
        self._error = error
        self._before_return = before_return

    def create(self, **kwargs: object) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._before_return is not None:
            self._before_return()
        if kwargs.get("stream"):
            return iter(_stream_chunks())
        return _response()


class _SyncEmbeddings:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7))


class _OpenAIStub:
    def __init__(self, error: Exception | None = None, before_return: Any = None) -> None:
        self.chat = SimpleNamespace(completions=_SyncCompletions(error, before_return))
        self.embeddings = _SyncEmbeddings()

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


class _AsyncCompletions:
    def __init__(self, error: Exception | None, before_return: Any) -> None:
        self.calls = 0
        self._error = error
        self._before_return = before_return

    async def create(self, **kwargs: object) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._before_return is not None:
            await self._before_return()
        if kwargs.get("stream"):
            return _async_iter(_stream_chunks())
        return _response()


async def _async_iter(chunks: list[SimpleNamespace]) -> Any:
    for chunk in chunks:
        yield chunk


class _AsyncEmbeddings:
    async def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7))


class _AsyncOpenAIStub:
    def __init__(self, error: Exception | None = None, before_return: Any = None) -> None:
        self.chat = SimpleNamespace(completions=_AsyncCompletions(error, before_return))
        self.embeddings = _AsyncEmbeddings()

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIStub:
        return self


_AsyncOpenAIStub.__module__ = "openai._client"
_AsyncOpenAIStub.__name__ = "AsyncOpenAI"


# ---------------------------------------------------------------------------
# Assertion helpers over the plane's recordings
# ---------------------------------------------------------------------------


def _events(plane: FakeControlPlane) -> list[MetadataEvent]:
    return sorted(plane.ingested, key=lambda event: event.attempt_index)


def _granted_lease_id(plane: FakeControlPlane) -> str:
    assert len(plane.lease_grants) == 1, "expected exactly one lease grant"
    # The double mints ids in grant order; the ledger keeps the id across renewals.
    return "lse_fake1"


def _only_confirm(plane: FakeControlPlane) -> BudgetConfirmRequest:
    assert len(plane.confirms) == 1, plane.confirms
    return plane.confirms[0]


def _assert_lease_settled_once(plane: FakeControlPlane) -> MetadataEvent:
    """One lease confirm; one SUCCESS event; both carry the grant's lease id."""
    lease_id = _granted_lease_id(plane)
    confirm = _only_confirm(plane)
    assert confirm.reservation_id is None
    assert confirm.lease_id == lease_id
    successes = [event for event in plane.ingested if event.status == "success"]
    assert len(successes) == 1, plane.ingested
    event = successes[0]
    assert event.call_id == confirm.call_id
    assert event.lease_id == lease_id
    assert event.lease_id == confirm.lease_id
    return event


def _assert_reservation_settled_once(plane: FakeControlPlane) -> MetadataEvent:
    """One reservation confirm; one SUCCESS event; neither names a lease."""
    confirm = _only_confirm(plane)
    assert confirm.reservation_id is not None
    assert confirm.lease_id is None
    successes = [event for event in plane.ingested if event.status == "success"]
    assert len(successes) == 1, plane.ingested
    event = successes[0]
    assert event.call_id == confirm.call_id
    assert event.lease_id is None
    return event


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "error_type,ambiguous",
    [
        pytest.param(httpx.ReadTimeout, True, id="read-timeout"),
        pytest.param(_Status503, True, id="server-error"),
        pytest.param(httpx.RemoteProtocolError, True, id="protocol-drop"),
        pytest.param(_Status400, False, id="fail-fast-control"),
    ],
)
async def test_dispatch_abort_retires_draw_without_refunding_unknown_spend(
    mode: str, error_type: type[Exception], ambiguous: bool
) -> None:
    plane = FakeControlPlane(granted_tokens=20, headroom_share_tokens=0, final_grant=True)
    error = error_type("synthetic dispatch failure")
    stub = _AsyncOpenAIStub if mode == "async" else _OpenAIStub
    primary, fallback = stub(error), stub()
    wrap = plane.wrap_async if mode == "async" else plane.wrap
    wrapped = wrap(primary, fallback=[(fallback, "gpt-5.5-mini")])

    async def call() -> None:
        result = wrapped.chat.completions.create(
            model="gpt-5.5", messages=[], max_completion_tokens=20
        )
        if mode == "async":
            await result

    async def close() -> None:
        if mode == "async":
            await wrapped.close()
        else:
            wrapped.close()

    try:
        with solwyn.run("dispatch-abort-accounting") as run_id:
            with pytest.raises(error_type) as raised:
                await call()
            assert raised.value is error
            assert primary.chat.completions.calls == 1
            assert fallback.chat.completions.calls == 0
            assert wrapped._get_circuit_breaker("openai").failure_count == int(ambiguous)
            ledger = wrapped._solwyn_budget._lease
            state = ledger.state_for(run_id)
            assert state is not None
            assert state.reservations == {}
            assert state.reserved_tokens == 0
            assert state.granted_remaining_tokens == (0 if ambiguous else 20)
            assert state.spent_tokens_since_report == (20 if ambiguous else 0)
            call_id, claim = next(iter(ledger._call_claims.items()))
            assert call_id not in ledger._call_index
            # A late cleanup cannot refund the retired paid draw.
            wrapped._solwyn_budget.release_reservation(call_id, claim.token)
            ledger.true_up(call_id, 0, claim_token=claim.token)
            assert state.granted_remaining_tokens == (0 if ambiguous else 20)

            primary.chat.completions._error = None
            await call()
            # Unknown spend exhausted the final grant and needs fresh server
            # authorization. FAIL_FAST still funds the following call locally.
            assert len(plane.checks) == int(ambiguous)
            assert state.spent_tokens_since_report == (20 if ambiguous else 15)
            assert state.granted_remaining_tokens == (0 if ambiguous else 5)
            assert primary.chat.completions.calls == 2
            assert fallback.chat.completions.calls == 0
            assert plane.lease_renewals == []
    finally:
        await close()

    event = next(event for event in plane.ingested if event.call_id == call_id)
    assert event.status == "error"
    assert event.possibly_succeeded is (True if ambiguous else None)
    # The receipt carries the wire-normalized class name: the underscore-prefixed
    # doubles lose the prefix the API's pattern rejects (`_Status503` ->
    # `Status503`); the public httpx names are reported unchanged.
    assert event.failover_error_class == error_type.__name__.removeprefix("_")
    assert event.attempt_index == 0
    assert not event.is_provider_fallback
    assert event.lease_id == _granted_lease_id(plane)
    assert event.input_tokens == event.output_tokens == 0
    assert all(confirm.call_id != call_id for confirm in plane.confirms)
    assert len(plane.ingested) == 2
    assert len(plane.lease_surrenders) == 1
    assert plane.lease_surrenders[0].spent_tokens == (20 if ambiguous else 15)
    await close()
    assert len(plane.lease_surrenders) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "primary_error,fallback_error,spent",
    [
        pytest.param(_Status503, _Status503, 20, id="ambiguous-then-exhausted"),
        pytest.param(_Status503, _Status400, 20, id="ambiguous-then-refused"),
        pytest.param(_Status503, None, 20, id="ambiguous-then-served"),
        pytest.param(_Status429, _Status429, 0, id="pre-send-exhausted-control"),
        pytest.param(_Status429, None, 15, id="pre-send-then-served-control"),
    ],
)
async def test_failed_over_ambiguous_hop_pins_the_bound_for_every_later_exit(
    mode: str,
    primary_error: type[Exception],
    fallback_error: type[Exception] | None,
    spent: int,
) -> None:
    # failover_idempotency="always" walks past a post-send-ambiguous hop on the
    # SAME reservation. Failing over does not un-send that hop, so no later exit
    # may re-lend its bound; a provably pre-send 429 still refunds as before.
    plane = FakeControlPlane(granted_tokens=20, headroom_share_tokens=0, final_grant=True)
    stub = _AsyncOpenAIStub if mode == "async" else _OpenAIStub
    primary = stub(primary_error("synthetic primary failure"))
    fallback = stub(fallback_error("synthetic fallback failure") if fallback_error else None)
    wrap = plane.wrap_async if mode == "async" else plane.wrap
    wrapped = wrap(
        primary,
        fallback=[(fallback, "gpt-5.5-mini")],
        failover_idempotency="always",
    )

    async def call() -> None:
        result = wrapped.chat.completions.create(
            model="gpt-5.5", messages=[], max_completion_tokens=20
        )
        if mode == "async":
            await result

    try:
        with solwyn.run("failed-over-ambiguous-accounting") as run_id:
            if fallback_error is None:
                await call()
            else:
                with pytest.raises(fallback_error):
                    await call()
            assert primary.chat.completions.calls == 1
            assert fallback.chat.completions.calls == 1
            state = wrapped._solwyn_budget._lease.state_for(run_id)
            assert state is not None
            assert state.reservations == {}
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == spent
            assert state.granted_remaining_tokens == 20 - spent
    finally:
        if mode == "async":
            await wrapped.close()
        else:
            wrapped.close()

    # The wire is unchanged: a failed-over hop is still not a possibly-succeeded
    # receipt, and a served hop confirms its MEASURED usage, not the bound.
    errors = [event for event in plane.ingested if event.status == "error"]
    assert errors and all(event.possibly_succeeded is None for event in errors)
    if fallback_error is None:
        assert _only_confirm(plane).token_details.total_tokens == 15
    else:
        assert plane.confirms == []
    assert [s.spent_tokens for s in plane.lease_surrenders] == [spent]


@pytest.mark.unit
class TestWireShape:
    def test_none_lease_id_puts_no_bytes_on_the_wire(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        event = _assert_reservation_settled_once(plane)
        assert "lease_id" not in event.model_dump(mode="json")

    def test_lease_id_is_serialized_when_set(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("lease-wire"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        event = _assert_lease_settled_once(plane)
        assert event.model_dump(mode="json")["lease_id"] == "lse_fake1"


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncLeaseId:
    def test_leased_untagged_run_tags_every_success_event_with_the_grant(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("leased"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        lease_id = _granted_lease_id(plane)
        assert len(plane.confirms) == 2
        assert len(plane.ingested) == 2
        assert {confirm.lease_id for confirm in plane.confirms} == {lease_id}
        assert {event.lease_id for event in plane.ingested} == {lease_id}
        assert all(event.status == "success" for event in plane.ingested)
        assert all(confirm.reservation_id is None for confirm in plane.confirms)
        # Each event pairs with exactly its own confirm on the call_id join key.
        assert {event.call_id for event in plane.ingested} == {
            confirm.call_id for confirm in plane.confirms
        }
        assert plane.checks == []

    def test_streaming_settlement_carries_the_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("leased-stream"):
                stream = wrapped.chat.completions.create(
                    model="gpt-5.5", messages=_MESSAGES, stream=True
                )
                chunks = list(stream)
        finally:
            wrapped.close()

        assert len(chunks) == 2
        event = _assert_lease_settled_once(plane)
        assert event.input_tokens == 100
        assert event.output_tokens == 50

    def test_tagged_run_never_leases_so_events_carry_no_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("tagged", tags={"team": "billing"}):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        assert plane.lease_grants == []
        event = _assert_reservation_settled_once(plane)
        assert event.tags == {"team": "billing"}

    def test_call_tags_inside_a_leased_run_skip_the_lease(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("leased-with-call-tags"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
                wrapped.chat.completions.create(
                    model="gpt-5.5", messages=_MESSAGES, solwyn_tags={"step": "b"}
                )
        finally:
            wrapped.close()

        lease_id = _granted_lease_id(plane)
        by_tags = {
            (None if event.tags is None else event.tags.get("step")): event
            for event in plane.ingested
        }
        assert set(by_tags) == {None, "b"}
        assert by_tags[None].lease_id == lease_id
        assert by_tags["b"].lease_id is None
        confirms = {confirm.call_id: confirm for confirm in plane.confirms}
        assert confirms[by_tags[None].call_id].lease_id == lease_id
        assert confirms[by_tags["b"].call_id].lease_id is None
        assert confirms[by_tags["b"].call_id].reservation_id is not None

    def test_allow_cache_hit_outside_a_run_has_no_lease_and_no_confirm(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub(), budget_check_cache_ttl=60.0)
        try:
            wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
            wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        assert plane.lease_grants == []
        assert len(plane.checks) == 1
        # The first call settled its reservation; the cache hit had nothing to settle.
        assert len(plane.confirms) == 1
        assert len(plane.ingested) == 2
        assert all(event.lease_id is None for event in plane.ingested)

    def test_failed_over_leased_attempt_shares_lease_id_across_attempts(self) -> None:
        plane = FakeControlPlane()
        primary = _OpenAIStub(_Status429("busy"))
        fallback = _OpenAIStub()
        wrapped = plane.wrap(
            primary,
            model="gpt-5.5",
            fallback=[(fallback, "gpt-5.5-mini")],
            same_provider_retries=0,
        )
        try:
            with solwyn.run("leased-failover"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        lease_id = _granted_lease_id(plane)
        events = _events(plane)
        assert [event.status for event in events] == ["error", "success"]
        error, success = events
        assert error.attempt_index == 0
        assert success.attempt_index == 1
        assert error.call_id == success.call_id
        # Error and served events of one lease-admitted call name the SAME lease.
        assert error.lease_id == lease_id
        assert success.lease_id == lease_id
        assert success.is_model_fallback is True
        confirm = _only_confirm(plane)
        assert confirm.lease_id == lease_id
        assert confirm.call_id == success.call_id

    def test_post_send_ambiguous_abort_error_event_carries_the_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub(_Status503("upstream")), same_provider_retries=0)
        try:
            with solwyn.run("leased-ambiguous"), pytest.raises(_Status503):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        lease_id = _granted_lease_id(plane)
        assert plane.confirms == []
        assert len(plane.ingested) == 1
        event = plane.ingested[0]
        assert event.status == "error"
        assert event.possibly_succeeded is True
        assert event.lease_id == lease_id

    def test_admit_uncounted_event_carries_no_lease_id(self) -> None:
        # Cold start with the grant endpoint unreachable and fail_open: the
        # ledger admits UNCOUNTED, deliberately with no lease id — nothing
        # settles the call, and a tagged event would be counted twice.
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub(), fail_open=True)
        try:
            with solwyn.run("uncounted"), plane.outage(path=_LEASE_GRANT_PATH):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        assert plane.lease_grants == []
        assert plane.checks == []
        assert plane.confirms == []
        assert len(plane.ingested) == 1
        event = plane.ingested[0]
        assert event.status == "success"
        assert event.lease_id is None

    def test_renewal_between_admission_and_settlement_keeps_the_admission_id(self) -> None:
        # A renewal is started by the first admission past the refresh
        # deadline and lands off the hot path. The second call's admission
        # triggers it; its stub then blocks until the renewal has reached the
        # plane, so the renewal lands BETWEEN that call's admission and its
        # settlement.
        plane = FakeControlPlane(refresh_interval_s=0.001, lease_length_s=60.0)
        calls = 0

        def wait_for_renewal_on_second_call() -> None:
            nonlocal calls
            calls += 1
            if calls < 2:
                return
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not plane.lease_renewals:
                time.sleep(0.005)
            assert plane.lease_renewals, "renewal never fired during the second call"

        wrapped = plane.wrap(_OpenAIStub(before_return=wait_for_renewal_on_second_call))
        try:
            with solwyn.run("leased-renewal"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
                time.sleep(0.01)  # past the refresh deadline
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        # A renewal keeps the lease id and bumps the generation; both events
        # and both confirms carry the id their ADMISSION was funded by.
        assert plane.lease_renewals
        assert len(plane.lease_grants) == 1
        assert len(plane.confirms) == 2
        assert len(plane.ingested) == 2
        assert {confirm.lease_id for confirm in plane.confirms} == {"lse_fake1"}
        assert {event.lease_id for event in plane.ingested} == {"lse_fake1"}

    def test_regrant_between_admission_and_settlement_keeps_the_admission_id(self) -> None:
        # A renewal keeps the id, so it cannot tell the admission result apart
        # from the ledger's live lease. A FRESH GRANT can: the lease expires
        # while the first call is at the provider, and a second call on the
        # same run, made before the first settles, takes a new lease. The
        # first call's event and confirm must still carry the id its own
        # admission was funded by.
        plane = FakeControlPlane(refresh_interval_s=0.001, lease_length_s=0.02)
        calls = 0
        wrapped: Any = None

        def regrant_during_first_call() -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                return
            # Both sides forget the first lease: the plane's record (so the
            # re-grant mints a fresh id instead of replaying the holder's
            # active one) and the SDK ledger (past its monotonic deadline).
            plane.expire_leases()
            time.sleep(0.03)
            wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)

        wrapped = plane.wrap(_OpenAIStub(before_return=regrant_during_first_call))
        try:
            with solwyn.run("leased-regrant"):
                wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            wrapped.close()

        assert len(plane.lease_grants) == 2, "the expired lease must be replaced by a fresh grant"
        assert len(plane.confirms) == 2
        assert len(plane.ingested) == 2
        confirm_lease_by_call = {confirm.call_id: confirm.lease_id for confirm in plane.confirms}
        assert set(confirm_lease_by_call.values()) == {"lse_fake1", "lse_fake2"}
        for event in plane.ingested:
            assert event.lease_id == confirm_lease_by_call[event.call_id]
        # The outer call settled while the ledger held lse_fake2, yet its
        # event names lse_fake1: the id came from its admission, not the ledger.
        outer = next(event for event in plane.ingested if event.lease_id == "lse_fake1")
        assert outer.status == "success"

    def test_media_surface_inside_a_leased_run_is_reservation_funded(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap(_OpenAIStub())
        try:
            with solwyn.run("leased-media"):
                wrapped.embeddings.create(model="text-embedding-3-small", input="x")
        finally:
            wrapped.close()

        event = _assert_reservation_settled_once(plane)
        assert event.modality == "embedding"
        assert plane.lease_grants == []


# ---------------------------------------------------------------------------
# Async mirrors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncLeaseId:
    @pytest.mark.asyncio
    async def test_leased_untagged_run_tags_the_success_event(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(_AsyncOpenAIStub())
        try:
            with solwyn.run("leased-async"):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        _assert_lease_settled_once(plane)
        assert plane.checks == []

    @pytest.mark.asyncio
    async def test_streaming_settlement_carries_the_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(_AsyncOpenAIStub())
        try:
            with solwyn.run("leased-async-stream"):
                stream = await wrapped.chat.completions.create(
                    model="gpt-5.5", messages=_MESSAGES, stream=True
                )
                chunks = [chunk async for chunk in stream]
        finally:
            await wrapped.close()

        assert len(chunks) == 2
        event = _assert_lease_settled_once(plane)
        assert event.input_tokens == 100

    @pytest.mark.asyncio
    async def test_tagged_run_never_leases(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(_AsyncOpenAIStub())
        try:
            with solwyn.run("tagged-async", tags={"team": "billing"}):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        assert plane.lease_grants == []
        _assert_reservation_settled_once(plane)

    @pytest.mark.asyncio
    async def test_failed_over_leased_attempt_shares_lease_id_across_attempts(self) -> None:
        plane = FakeControlPlane()
        primary = _AsyncOpenAIStub(_Status429("busy"))
        fallback = _AsyncOpenAIStub()
        wrapped = plane.wrap_async(
            primary,
            model="gpt-5.5",
            fallback=[(fallback, "gpt-5.5-mini")],
            same_provider_retries=0,
        )
        try:
            with solwyn.run("leased-async-failover"):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        lease_id = _granted_lease_id(plane)
        events = _events(plane)
        assert [event.status for event in events] == ["error", "success"]
        assert {event.call_id for event in events} == {events[0].call_id}
        assert {event.lease_id for event in events} == {lease_id}
        assert _only_confirm(plane).lease_id == lease_id

    @pytest.mark.asyncio
    async def test_post_send_ambiguous_abort_error_event_carries_the_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(
            _AsyncOpenAIStub(_Status503("upstream")), same_provider_retries=0
        )
        try:
            with solwyn.run("leased-async-ambiguous"), pytest.raises(_Status503):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        assert plane.confirms == []
        assert len(plane.ingested) == 1
        event = plane.ingested[0]
        assert event.possibly_succeeded is True
        assert event.lease_id == _granted_lease_id(plane)

    @pytest.mark.asyncio
    async def test_admit_uncounted_event_carries_no_lease_id(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(_AsyncOpenAIStub(), fail_open=True)
        try:
            with solwyn.run("uncounted-async"), plane.outage(path=_LEASE_GRANT_PATH):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        assert plane.lease_grants == []
        assert plane.confirms == []
        assert len(plane.ingested) == 1
        assert plane.ingested[0].lease_id is None

    @pytest.mark.asyncio
    async def test_renewal_between_admission_and_settlement_keeps_the_admission_id(
        self,
    ) -> None:
        plane = FakeControlPlane(refresh_interval_s=0.001, lease_length_s=60.0)
        calls = 0

        async def wait_for_renewal_on_second_call() -> None:
            nonlocal calls
            calls += 1
            if calls < 2:
                return
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not plane.lease_renewals:
                await asyncio.sleep(0.005)
            assert plane.lease_renewals, "renewal never fired during the second call"

        wrapped = plane.wrap_async(_AsyncOpenAIStub(before_return=wait_for_renewal_on_second_call))
        try:
            with solwyn.run("leased-async-renewal"):
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
                await asyncio.sleep(0.01)  # past the refresh deadline
                await wrapped.chat.completions.create(model="gpt-5.5", messages=_MESSAGES)
        finally:
            await wrapped.close()

        assert plane.lease_renewals
        assert len(plane.lease_grants) == 1
        assert len(plane.confirms) == 2
        assert {confirm.lease_id for confirm in plane.confirms} == {"lse_fake1"}
        assert {event.lease_id for event in plane.ingested} == {"lse_fake1"}

    @pytest.mark.asyncio
    async def test_media_surface_inside_a_leased_run_is_reservation_funded(self) -> None:
        plane = FakeControlPlane()
        wrapped = plane.wrap_async(_AsyncOpenAIStub())
        try:
            with solwyn.run("leased-async-media"):
                await wrapped.embeddings.create(model="text-embedding-3-small", input="x")
        finally:
            await wrapped.close()

        event = _assert_reservation_settled_once(plane)
        assert event.modality == "embedding"
