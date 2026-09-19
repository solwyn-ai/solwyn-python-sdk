"""Unknown paid work retires local ownership without refunding lease authority."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from conftest import call_uuid

from solwyn import run
from solwyn._token_details import TokenDetails
from solwyn.budget import AsyncBudgetEnforcer, BudgetCheckResult
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _HeldRenewalTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """The plane advances the generation before the holder receives its answer."""

    def __init__(self, plane: FakeControlPlane) -> None:
        self.plane = plane
        self.renewed = asyncio.Event()
        self.deliver_renewal = asyncio.Event()
        self.surrendered = asyncio.Event()
        self.surrender_outcomes: list[tuple[int, int, int]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.plane.transport.handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.plane.transport.handle_async_request(request)
        if request.url.path.endswith("/renew"):
            self.renewed.set()
            await self.deliver_renewal.wait()
        elif request.url.path.endswith("/surrender"):
            body = json.loads(request.content)
            self.surrender_outcomes.append(
                (body["generation"], body["spent_tokens"], response.status_code)
            )
            self.surrendered.set()
        return response


async def _check(
    enforcer: AsyncBudgetEnforcer, run_id: str, label: str, tokens: int
) -> BudgetCheckResult:
    return await enforcer.check_budget(
        estimated_input_tokens=0,
        estimated_output_bound=tokens,
        model="gpt-5.5",
        provider="openai",
        agent_run_id=run_id,
        call_id=call_uuid(label),
    )


def _late_callbacks(
    enforcer: AsyncBudgetEnforcer, label: str, admission: BudgetCheckResult
) -> None:
    """A retired capability cannot refund, debit again, or settle a newer draw."""
    call_id = call_uuid(label)
    enforcer.release_reservation(call_id, admission.lease_claim_token)
    enforcer.abandon_reservation(call_id, lease_claim_token=admission.lease_claim_token)
    for usage in (0, 999):
        enforcer.build_confirm_request(
            model="gpt-5.5",
            provider="openai",
            call_id=call_id,
            lease_id=admission.lease_id,
            lease_claim_token=admission.lease_claim_token,
            token_details=TokenDetails(input_tokens=usage),
        )


@pytest.mark.parametrize("close_during_renewal", [False, True])
async def test_cancelled_draw_survives_renewal_snapshot_and_terminal_cleanup(
    close_during_renewal: bool,
) -> None:
    plane = FakeControlPlane(granted_tokens=100)
    transport = _HeldRenewalTransport(plane)
    enforcer = AsyncBudgetEnforcer(
        api_url=plane.api_url,
        api_key=plane.api_key,
        holder_id="cancelled-renewal-holder",
        transport=transport,
    )
    try:
        async with run("cancelled-renewal-accounting") as run_id:
            before = await _check(enforcer, run_id, "before-renewal", 20)
            enforcer.abandon_reservation(
                call_uuid("before-renewal"), lease_claim_token=before.lease_claim_token
            )
            ledger = enforcer._lease
            state = ledger.state_for(run_id)
            assert state is not None
            assert state.granted_remaining_tokens == 80
            assert state.spent_tokens_since_report == 20
            # The initial grant remains renewable; its successor is final so
            # subsequent admissions cannot hide an unsafe refund by renewing.
            plane.final_grant = True
            crossing = await _check(enforcer, run_id, "crossing-renewal", 55)
            await asyncio.wait_for(transport.renewed.wait(), timeout=2)
            assert len(plane.lease_renewals) == 1
            renewal = plane.lease_renewals[0]
            assert renewal.spent_tokens == 20
            assert renewal.reserved_tokens == 55
            assert state.generation == 1

            # An old claim capability cannot retire a different live call.
            enforcer.abandon_reservation(
                call_uuid("crossing-renewal"), lease_claim_token=before.lease_claim_token
            )
            assert state.reserved_tokens == 55
            assert state.spent_tokens_since_report == 20
            enforcer.abandon_reservation(
                call_uuid("crossing-renewal"), lease_claim_token=crossing.lease_claim_token
            )
            assert state.reserved_tokens == 0
            assert state.granted_remaining_tokens == 25
            assert state.spent_tokens_since_report == 75
            for label, admission in (
                ("before-renewal", before),
                ("crossing-renewal", crossing),
            ):
                call_id = call_uuid(label)
                assert call_id not in ledger._call_index
                assert ledger._call_claims[call_id].token == admission.lease_claim_token
                _late_callbacks(enforcer, label, admission)
            assert state.granted_remaining_tokens == 25
            assert state.spent_tokens_since_report == 75

            if close_during_renewal:
                closing = asyncio.create_task(enforcer.close())
                await asyncio.wait_for(transport.surrendered.wait(), timeout=2)
                # The old-generation release is refused: the server already
                # installed generation 2, whose delayed answer must carry the
                # 55 tokens spent after its renewal snapshot when released.
                assert transport.surrender_outcomes == [(1, 75, 409)]
                transport.deliver_renewal.set()
                await asyncio.wait_for(closing, timeout=2)
                assert ledger.state_for(run_id) is None
                assert transport.surrender_outcomes == [(1, 75, 409), (2, 55, 200)]
            else:
                transport.deliver_renewal.set()
                await asyncio.gather(*list(enforcer._renewal_tasks))
                assert state.generation == 2
                assert state.final_grant
                assert state.reserved_tokens == 0
                assert state.granted_remaining_tokens == 45
                assert state.spent_tokens_since_report == 55
                _late_callbacks(enforcer, "crossing-renewal", crossing)
                assert state.granted_remaining_tokens == 45
                assert state.spent_tokens_since_report == 55
                with pytest.raises(RuntimeError, match="already been used"):
                    await _check(enforcer, run_id, "crossing-renewal", 1)
                last = await _check(enforcer, run_id, "last-local-draw", 45)
                assert last.allowed and last.lease_id == before.lease_id
                assert state.granted_remaining_tokens == 0
                enforcer.abandon_reservation(
                    call_uuid("last-local-draw"), lease_claim_token=crossing.lease_claim_token
                )
                assert state.reserved_tokens == 45
                enforcer.release_reservation(call_uuid("last-local-draw"), last.lease_claim_token)
                assert state.granted_remaining_tokens == 45
                assert state.spent_tokens_since_report == 55

        # The normal scope exit sends only unreported spend. Repeated close
        # and stale callbacks cannot create a second accounting owner.
        await enforcer.close()
        assert len(plane.lease_grants) == len(plane.lease_renewals) == 1
        assert plane.checks == []
        assert plane.confirms == []
        accepted_spend = sum(
            spend for _generation, spend, status in transport.surrender_outcomes if status == 200
        )
        assert accepted_spend + renewal.spent_tokens == 75
        outcomes = list(transport.surrender_outcomes)
        _late_callbacks(enforcer, "before-renewal", before)
        _late_callbacks(enforcer, "crossing-renewal", crossing)
        await enforcer.close()
        assert transport.surrender_outcomes == outcomes
        assert enforcer._lease.state_for(run_id) is None
    finally:
        transport.deliver_renewal.set()
        await enforcer.close()


@pytest.mark.parametrize("control_plane_available", [False, True])
async def test_cancelled_final_grant_stays_spent_without_leaking_an_active_claim(
    control_plane_available: bool,
) -> None:
    plane = FakeControlPlane(granted_tokens=20, headroom_share_tokens=0, final_grant=True)
    breaker = CircuitBreaker(name="control-plane", recovery_timeout=3600)
    enforcer = AsyncBudgetEnforcer(
        api_url=plane.api_url,
        api_key=plane.api_key,
        holder_id="cancelled-final-holder",
        transport=plane.transport,
        control_plane_breaker=breaker,
    )
    try:
        async with run("cancelled-final-grant") as run_id:
            first = await _check(enforcer, run_id, "exhausted-cancelled-call", 20)
            assert first.allowed and first.lease_id is not None
            enforcer.abandon_reservation(
                call_uuid("exhausted-cancelled-call"), lease_claim_token=first.lease_claim_token
            )
            ledger = enforcer._lease
            state = ledger.state_for(run_id)
            assert state is not None
            assert state.reservations == {}
            assert call_uuid("exhausted-cancelled-call") not in ledger._call_index
            assert ledger._call_claims[call_uuid("exhausted-cancelled-call")].token == (
                first.lease_claim_token
            )
            _late_callbacks(enforcer, "exhausted-cancelled-call", first)
            assert state.granted_remaining_tokens == 0
            assert state.spent_tokens_since_report == 20
            if not control_plane_available:
                for _ in range(breaker.failure_threshold):
                    breaker.record_failure()

            following = await _check(enforcer, run_id, "after-exhaustion", 1)
            if control_plane_available:
                # A reachable plane can authorize a fresh per-call check;
                # the spent local grant supplies no reusable authority.
                assert following.allowed and following.lease_id is None
                assert len(plane.checks) == 1
            else:
                assert not following.allowed
                assert following.deny_source == "lease_exhausted"
                assert following.deny_reason == "lease_share_exhausted"
                assert plane.checks == []
            assert state.reservations == {}
            assert state.granted_remaining_tokens == 0
            assert state.spent_tokens_since_report == 20
            assert plane.lease_renewals == []
            # Restore control-plane health so terminal delivery can be
            # asserted independently of the outage admission branch above.
            breaker.replace_tuning(
                failure_threshold=breaker.failure_threshold,
                recovery_timeout=0,
                success_threshold=breaker.success_threshold,
                recovery_timeout_jitter=0,
            )

        await enforcer.close()
        assert len(plane.lease_surrenders) == 1
        assert plane.lease_surrenders[0].spent_tokens == 20
        assert plane.lease_surrenders[0].lease_id == first.lease_id
        assert plane.confirms == []
        _late_callbacks(enforcer, "exhausted-cancelled-call", first)
        await enforcer.close()
        assert len(plane.lease_surrenders) == 1
    finally:
        await enforcer.close()
