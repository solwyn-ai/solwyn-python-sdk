"""Public async lease grants keep their slot through every terminal path."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from conftest import call_uuid

from solwyn.budget import AsyncBudgetEnforcer, BudgetCheckResult
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _GatedTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Hold selected control-plane requests without replacing enforcer logic."""

    def __init__(self, plane: FakeControlPlane) -> None:
        self.plane = plane
        self.gates: dict[tuple[str, str], asyncio.Event] = {}
        self.entered: dict[tuple[str, str], asyncio.Event] = {}
        self.requests: list[tuple[str, str]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.lease_runs: dict[str, str] = {}

    def hold(self, operation: str, run_id: str) -> tuple[asyncio.Event, asyncio.Event]:
        key = operation, run_id
        self.entered[key] = asyncio.Event()
        self.gates[key] = asyncio.Event()
        return self.entered[key], self.gates[key]

    def unblock(self) -> None:
        for gate in self.gates.values():
            gate.set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.plane.transport.handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        operation = request.url.path.rsplit("/", 1)[-1]
        run_id = (
            self.lease_runs[body["lease_id"]] if operation == "surrender" else body["agent_run_id"]
        )
        key = operation, run_id
        self.requests.append(key)
        gate = self.gates.get(key)
        if gate is not None:
            self.entered[key].set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                self.cancelled.append(key)
                raise
        response = await self.plane.transport.handle_async_request(request)
        if operation == "lease" and response.status_code == 200:
            self.lease_runs[response.json()["lease_id"]] = run_id
        return response


def _enforcer(
    plane: FakeControlPlane,
    transport: _GatedTransport,
    breaker: CircuitBreaker | None = None,
) -> AsyncBudgetEnforcer:
    return AsyncBudgetEnforcer(
        api_url=plane.api_url,
        api_key=plane.api_key,
        holder_id="grant-ownership-test",
        transport=transport,
        control_plane_breaker=breaker,
    )


async def _check(enforcer: AsyncBudgetEnforcer, run_id: str, label: str) -> BudgetCheckResult:
    return await enforcer.check_budget(
        estimated_input_tokens=1,
        estimated_output_bound=20,
        model="gpt-5.5",
        provider="openai",
        agent_run_id=run_id,
        call_id=call_uuid(f"{run_id}-{label}"),
    )


async def test_fence_cancellation_preserves_release_and_future_local_admission() -> None:
    plane = FakeControlPlane()
    transport = _GatedTransport(plane)
    enforcer = _enforcer(plane, transport)
    try:
        for index in range(3):
            run_id = f"fenced-run-{index}"
            first = await _check(enforcer, run_id, "first")
            assert first.lease_id is not None
            release_entered, release_finish = transport.hold("surrender", run_id)
            enforcer.surrender_run(run_id)
            await asyncio.wait_for(release_entered.wait(), timeout=2)
            release_task = next(iter(enforcer._release_tasks))

            pending = asyncio.create_task(_check(enforcer, run_id, "cancelled"))
            # The predecessor is held, so this checkpoint puts the winner
            # inside its release fence, before a successor HTTP request.
            await asyncio.sleep(0)
            assert enforcer._lease_grants_in_flight == {run_id}
            assert transport.requests.count(("lease", run_id)) == 1
            loser = await _check(enforcer, run_id, "concurrent-loser")
            assert loser.allowed and loser.lease_id is None
            assert transport.requests.count(("check", run_id)) == 1
            assert enforcer._lease_grants_in_flight == {run_id}

            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert enforcer._lease_grants_in_flight == set()
            assert not release_task.done()
            assert transport.cancelled == []
            assert transport.requests.count(("lease", run_id)) == 1

            release_finish.set()
            await asyncio.wait_for(release_task, timeout=2)
            next_call = await _check(enforcer, run_id, "new-grant")
            assert next_call.lease_id is not None
            assert next_call.lease_id != first.lease_id
            for local_index in range(3):
                local = await _check(enforcer, run_id, f"local-{local_index}")
                assert local.lease_id == next_call.lease_id
            assert transport.requests.count(("lease", run_id)) == 2
            assert transport.requests.count(("check", run_id)) == 1
            assert enforcer._lease_grants_in_flight == set()

        assert len(plane.lease_grants) == 6
        assert len(plane.checks) == 3
        assert len(plane.lease_surrenders) == 3
        assert enforcer._release_tasks == set()
        assert enforcer._run_releases == {}
    finally:
        transport.unblock()
        await enforcer.close()


@pytest.mark.parametrize("recovery_probe", [False, True])
async def test_grant_http_cancellation_releases_slot_and_probe(recovery_probe: bool) -> None:
    plane = FakeControlPlane()
    transport = _GatedTransport(plane)
    breaker = CircuitBreaker(recovery_timeout=0, name="control-plane")
    if recovery_probe:
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
    enforcer = _enforcer(plane, transport, breaker)
    run_id = "http-cancelled-run"
    grant_entered, grant_finish = transport.hold("lease", run_id)
    try:
        pending = asyncio.create_task(_check(enforcer, run_id, "cancelled"))
        await asyncio.wait_for(grant_entered.wait(), timeout=2)
        assert enforcer._lease_grants_in_flight == {run_id}
        assert breaker._half_open_probe_active is recovery_probe

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert enforcer._lease_grants_in_flight == set()
        assert breaker._half_open_probe_active is False
        assert breaker.get_state().success_count == 0
        assert transport.cancelled == [("lease", run_id)]
        assert plane.lease_grants == []

        grant_finish.set()
        fresh = await _check(enforcer, run_id, "new-grant")
        local = await _check(enforcer, run_id, "local")
        assert fresh.lease_id is not None and local.lease_id == fresh.lease_id
        assert len(plane.lease_grants) == 1
        assert plane.checks == []
        assert transport.requests.count(("lease", run_id)) == 2
        assert enforcer._lease_grants_in_flight == set()
    finally:
        transport.unblock()
        await enforcer.close()


@pytest.mark.parametrize("failing_method", ["admit", "release_probe"])
async def test_breaker_exception_cannot_strand_grant_slot(failing_method: str) -> None:
    plane = FakeControlPlane()
    transport = _GatedTransport(plane)
    breaker = CircuitBreaker(name="control-plane")
    enforcer = _enforcer(plane, transport, breaker)
    run_id = "breaker-exception-run"
    try:
        with patch.object(
            breaker,
            failing_method,
            autospec=True,
            side_effect=RuntimeError("synthetic breaker hook failure"),
        ) as failing_hook:
            with pytest.raises(RuntimeError, match="synthetic breaker hook failure"):
                await _check(enforcer, run_id, "hook-failure")
            failing_hook.assert_called_once()
            assert enforcer._lease_grants_in_flight == set()

        # A cleanup failure may occur after installation. Surrender that
        # authority before proving this run can acquire another grant.
        enforcer.surrender_run(run_id)
        await asyncio.gather(*list(enforcer._release_tasks))
        fresh = await _check(enforcer, run_id, "new-grant")
        assert fresh.lease_id is not None
        assert len(plane.lease_grants) == (1 if failing_method == "admit" else 2)
        assert plane.checks == []
        assert enforcer._lease_grants_in_flight == set()
    finally:
        await enforcer.close()


async def test_close_racing_grant_surrenders_late_authority_exactly_once() -> None:
    plane = FakeControlPlane()
    transport = _GatedTransport(plane)
    enforcer = _enforcer(plane, transport)
    run_id = "closed-grant-run"
    grant_entered, grant_finish = transport.hold("lease", run_id)
    pending = asyncio.create_task(_check(enforcer, run_id, "late-grant"))
    try:
        await asyncio.wait_for(grant_entered.wait(), timeout=2)
        assert enforcer._lease_grants_in_flight == {run_id}
        await enforcer.close()
        grant_finish.set()
        result = await asyncio.wait_for(pending, timeout=2)
        await asyncio.gather(*enforcer._release_tasks)

        assert result.lease_id is None
        assert enforcer._lease.state_for(run_id) is None
        assert enforcer._lease_grants_in_flight == set()
        assert len(plane.lease_grants) == 1
        assert len(plane.lease_surrenders) == 1
        surrender = plane.lease_surrenders[0]
        assert transport.lease_runs[surrender.lease_id] == run_id
        assert surrender.holder_id == "grant-ownership-test"
        assert surrender.generation == 1
        assert surrender.spent_tokens == 0
        await enforcer.close()
        assert len(plane.lease_surrenders) == 1
    finally:
        transport.unblock()
        await asyncio.gather(pending, return_exceptions=True)
        await enforcer.close()
