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
