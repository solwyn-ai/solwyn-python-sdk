"""Run-control directive surface of the in-process control-plane double."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

import solwyn as solwyn_pkg
from solwyn._types import LeaseRenewRequest, ProviderName
from solwyn.budget import BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker, CircuitState
from solwyn.testing import FakeControlPlane

STOPPED_DIRECTIVE_REASON = "manual_kill"
LEASE_BLOCK_KEYS = {
    "lease_id",
    "generation",
    "granted_tokens",
    "refresh_interval_s",
    "lease_length_s",
    "headroom_share_tokens",
    "posture",
    "final_grant",
}


def _check_payload(**overrides: object) -> dict[str, object]:
    """Build the check body the SDK actually sends: both v1 opt-ins present."""
    payload: dict[str, object] = {
        "estimated_input_tokens": 17,
        "model": "gpt-5.5",
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
        "failover_directive_version": "1",
        "run_directive_version": "1",
    }
    payload.update(overrides)
    return payload


def _post_check(
    client: httpx.Client,
    plane: FakeControlPlane,
    **overrides: object,
) -> dict[str, object]:
    response = client.post(
        f"{plane.api_url}/api/v1/budgets/check",
        json=_check_payload(**overrides),
    )
    body: dict[str, object] = response.json()
    return body


def _grant_payload(
    *,
    run_id: str = "run-lease",
    holder_id: str = "holder-1",
    model: str = "gpt-5.5",
) -> dict[str, object]:
    return {
        "agent_run_id": run_id,
        "holder_id": holder_id,
        "model": model,
        "provider": ProviderName.OPENAI.value,
        "fallback_providers": [],
        "fallback_models": [],
        "fail_open": True,
        "estimated_input_tokens": 123,
        "run_directive_version": "1",
    }


def _make_enforcer(plane: FakeControlPlane, **overrides: object) -> BudgetEnforcer:
    options: dict[str, object] = {
        "api_url": plane.api_url,
        "api_key": plane.api_key,
        "cache_ttl": 0,
        "transport": plane.transport,
        "holder_id": "holder-1",
    }
    options.update(overrides)
    return BudgetEnforcer(**options)  # type: ignore[arg-type]


class _SyncCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


class _SyncChat:
    def __init__(self) -> None:
        self.completions = _SyncCompletions()


class _OpenAIStub:
    def __init__(self) -> None:
        self.chat = _SyncChat()

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


# ── check channel ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_stop_run_attaches_v1_directive_to_opted_in_check() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")

    with httpx.Client(transport=plane.transport) as client:
        body = _post_check(client, plane, agent_run_id="run-7")

    assert body["allowed"] is False
    assert body["denied_by_period"] == "run_stopped"
    assert body["mode"] == "hard_deny"
    assert body["run_control"] == {
        "version": "1",
        "action": "terminate",
        "agent_run_id": "run-7",
        "reason": STOPPED_DIRECTIVE_REASON,
    }
    assert "reservation_id" not in body


@pytest.mark.unit
def test_stop_run_without_opt_in_denies_but_omits_directive() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")

    with httpx.Client(transport=plane.transport) as client:
        body = _post_check(client, plane, agent_run_id="run-7", run_directive_version=None)

    assert body["allowed"] is False
    assert body["denied_by_period"] == "run_stopped"
    assert "run_control" not in body


@pytest.mark.unit
def test_stop_beats_deny_next_and_other_runs_stay_allowed() -> None:
    plane = FakeControlPlane()
    plane.deny_next(5, period="monthly")
    plane.stop_run("run-7")

    with httpx.Client(transport=plane.transport) as client:
        stopped = _post_check(client, plane, agent_run_id="run-7")
        other = _post_check(client, plane, agent_run_id="run-8")

    assert stopped["denied_by_period"] == "run_stopped"
    assert other["denied_by_period"] == "monthly"
    assert "run_control" not in other


@pytest.mark.unit
def test_a_check_without_a_run_never_receives_a_directive() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")

    with httpx.Client(transport=plane.transport) as client:
        body = _post_check(client, plane)

    assert body["allowed"] is True
    assert "run_control" not in body


@pytest.mark.unit
def test_scripted_run_stopped_period_stays_directive_free() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="run_stopped")

    with httpx.Client(transport=plane.transport) as client:
        body = _post_check(client, plane, agent_run_id="run-7")

    assert body["denied_by_period"] == "run_stopped"
    assert "run_control" not in body
    assert plane.stopped_runs == {}


@pytest.mark.unit
def test_second_stop_run_with_different_reason_raises() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7", reason=STOPPED_DIRECTIVE_REASON)

    with pytest.raises(RuntimeError, match="already stopped"):
        plane.stop_run("run-7", reason="velocity:repeat_size")

    assert plane.stopped_runs == {"run-7": STOPPED_DIRECTIVE_REASON}


@pytest.mark.unit
def test_repeating_the_same_stop_reason_is_a_no_op() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7", reason="velocity:repeat_size")
    plane.stop_run("run-7", reason="velocity:repeat_size")

    assert plane.stopped_runs == {"run-7": "velocity:repeat_size"}


@pytest.mark.unit
def test_clear_stop_restores_ordinary_allows() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")
    plane.clear_stop("run-7")

    with httpx.Client(transport=plane.transport) as client:
        body = _post_check(client, plane, agent_run_id="run-7")

    assert body["allowed"] is True
    assert "run_control" not in body
    assert plane.stopped_runs == {}


@pytest.mark.unit
def test_reset_recording_preserves_scenario_stops() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7", reason="operator_stop")

    with httpx.Client(transport=plane.transport) as client:
        _post_check(client, plane, agent_run_id="run-7")
        plane.reset_recording()
        body = _post_check(client, plane, agent_run_id="run-7")

    # Only the post-reset check survives the recording wipe; the stop does not.
    assert len(plane.checks) == 1
    assert plane.stopped_runs == {"run-7": "operator_stop"}
    assert body["denied_by_period"] == "run_stopped"
    assert body["run_control"]["reason"] == "operator_stop"  # type: ignore[index]


# ── lease channel ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_stopped_lease_grant_omits_the_lease_block_and_carries_the_directive() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-lease", reason="operator_stop")

    with httpx.Client(transport=plane.transport) as client:
        body = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(),
        ).json()

    assert body["allowed"] is False
    assert body["eligible"] is True
    assert body["denied_by_period"] == "run_stopped"
    assert body["mode"] == "hard_deny"
    assert body["remaining_budget"] == 0.0
    assert LEASE_BLOCK_KEYS.isdisjoint(body)
    assert body["run_control"] == {
        "version": "1",
        "action": "terminate",
        "agent_run_id": "run-lease",
        "reason": "operator_stop",
    }


@pytest.mark.unit
def test_stopped_lease_grant_without_opt_in_omits_the_directive() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-lease")
    payload = _grant_payload()
    del payload["run_directive_version"]

    with httpx.Client(transport=plane.transport) as client:
        body = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=payload).json()

    assert body["allowed"] is False
    assert body["denied_by_period"] == "run_stopped"
    assert "run_control" not in body


@pytest.mark.unit
def test_stopped_renewal_drops_the_lease_and_terminates_the_run() -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    try:
        with solwyn_pkg.run("lease-stop") as run_id:
            enforcer.check_budget(
                estimated_input_tokens=10,
                estimated_output_bound=20,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=run_id,
                call_id=str(uuid4()),
            )
            granted = plane.lease_grants[0]
            plane.stop_run(run_id)

            renewal = enforcer._build_renewal(
                run_id,
                model="gpt-5.5",
                provider="openai",
                fallback_providers=[],
                fallback_models=[],
            )
            if renewal is None:
                pytest.fail("held lease did not produce a renewal request")
            enforcer._renew_lease(run_id, renewal, ["gpt-5.5"])

            terminated = solwyn_pkg.current_run_terminated()
            state = enforcer._lease.state_for(run_id)
            denied = enforcer.check_budget(
                estimated_input_tokens=10,
                estimated_output_bound=20,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=run_id,
                call_id=str(uuid4()),
            )
    finally:
        enforcer.close()

    assert granted.agent_run_id == run_id
    assert plane.lease_renewals[-1].run_directive_version == "1"
    assert terminated is True
    # The stop drops the lease outright: a killed run has no authority left to
    # surrender, so no release ever reaches the plane.
    assert state is not None
    assert state.lease_id is None
    assert plane.lease_surrenders == []
    assert denied.allowed is False
    assert denied.denied_by_period == "run_stopped"
    assert denied.deny_reason == STOPPED_DIRECTIVE_REASON


# ── real enforcer, real directive machinery ──────────────────────────────


@pytest.mark.unit
def test_enforcer_check_marks_the_sdk_run_registry_from_the_double_bytes() -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane, lease_enabled=False)
    try:
        with solwyn_pkg.run("enforcer-stop") as run_id:
            plane.stop_run(run_id)
            result = enforcer.check_budget(
                estimated_input_tokens=10,
                model="gpt-5.5",
                provider="openai",
                agent_run_id=run_id,
                call_id=str(uuid4()),
            )
            terminated = solwyn_pkg.current_run_terminated()
    finally:
        enforcer.close()

    assert result.allowed is False
    assert result.denied_by_period == "run_stopped"
    assert result.deny_source == "server"
    assert result.deny_reason == STOPPED_DIRECTIVE_REASON
    assert terminated is True
    assert plane.checks[0].run_directive_version == "1"


@pytest.mark.unit
def test_misrouted_stops_degrade_one_call_without_opening_the_breaker() -> None:
    plane = FakeControlPlane()
    breaker = CircuitBreaker(failure_threshold=3, name="control-plane")
    enforcer = _make_enforcer(plane, lease_enabled=False, control_plane_breaker=breaker)
    try:
        with solwyn_pkg.run("misrouted") as run_id, plane.misroute_stops():
            plane.stop_run(run_id)
            results = [
                enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=run_id,
                    call_id=str(uuid4()),
                )
                for _ in range(3)
            ]
            terminated = solwyn_pkg.current_run_terminated()
    finally:
        enforcer.close()

    assert len(plane.checks) == 3
    assert all(
        check.model_dump(mode="json").get("run_directive_version") == "1" for check in plane.checks
    )
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert terminated is False
    # Drift degrades each call down the unreachable ladder; fail_open admits.
    assert all(result.allowed for result in results)


# ── solwyn-test/kill magic model ─────────────────────────────────────────


@pytest.mark.unit
def test_kill_magic_model_allows_once_then_stops_the_run_through_wrap() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(provider, lease_enabled=False)

    try:
        with solwyn_pkg.run("kill-probe") as run_id:
            wrapped.chat.completions.create(model="solwyn-test/kill", messages=[])
            with pytest.raises(solwyn_pkg.RunStoppedError) as captured:
                wrapped.chat.completions.create(model="solwyn-test/kill", messages=[])
    finally:
        wrapped.close()

    assert not isinstance(captured.value, solwyn_pkg.BudgetExceededError)
    assert captured.value.agent_run_id == run_id
    assert captured.value.reason == STOPPED_DIRECTIVE_REASON
    assert captured.value.source == "server"
    assert provider.chat.completions.calls == 1
    assert plane.stopped_runs == {run_id: STOPPED_DIRECTIVE_REASON}


@pytest.mark.unit
def test_kill_magic_model_requires_an_open_run_before_dispatch() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(provider)

    try:
        with pytest.raises(RuntimeError, match="requires an agent_run_id"):
            wrapped.chat.completions.create(model="solwyn-test/kill", messages=[])
    finally:
        wrapped.close()

    assert provider.chat.completions.calls == 0
    assert plane.checks == []
    assert plane.stopped_runs == {}


@pytest.mark.unit
def test_kill_stop_spreads_to_every_channel_for_that_run() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        allowed = _post_check(client, plane, agent_run_id="run-k", model="solwyn-test/kill")
        stopped = _post_check(client, plane, agent_run_id="run-k", model="solwyn-test/kill")
        other_model = _post_check(client, plane, agent_run_id="run-k")
        renew_source = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(run_id="run-other"),
        ).json()
        stopped_grant = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(run_id="run-k"),
        ).json()

    assert allowed["allowed"] is True
    assert stopped["denied_by_period"] == "run_stopped"
    assert other_model["denied_by_period"] == "run_stopped"
    assert other_model["run_control"]["agent_run_id"] == "run-k"  # type: ignore[index]
    assert renew_source["allowed"] is True
    assert stopped_grant["allowed"] is False
    assert stopped_grant["run_control"]["reason"] == STOPPED_DIRECTIVE_REASON  # type: ignore[index]


@pytest.mark.unit
def test_misroute_window_rewrites_only_while_active_and_within_its_budget() -> None:
    plane = FakeControlPlane()
    plane.stop_run("run-7")

    with httpx.Client(transport=plane.transport) as client:
        with plane.misroute_stops(requests=1):
            misrouted = _post_check(client, plane, agent_run_id="run-7")
            exhausted = _post_check(client, plane, agent_run_id="run-7")
        after = _post_check(client, plane, agent_run_id="run-7")

    assert misrouted["run_control"]["agent_run_id"] == "solwyn-test-misrouted-run"  # type: ignore[index]
    assert exhausted["run_control"]["agent_run_id"] == "run-7"  # type: ignore[index]
    assert after["run_control"]["agent_run_id"] == "run-7"  # type: ignore[index]


@pytest.mark.unit
def test_renewal_of_a_stopped_run_returns_the_stopped_lease_shape() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(),
        ).json()
        plane.stop_run("run-lease", reason="operator_stop")
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=grant["generation"],
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
                run_directive_version="1",
            ).model_dump(mode="json"),
        ).json()

    assert grant["allowed"] is True
    assert renewed["allowed"] is False
    assert renewed["denied_by_period"] == "run_stopped"
    assert LEASE_BLOCK_KEYS.isdisjoint(renewed)
    assert renewed["run_control"] == {
        "version": "1",
        "action": "terminate",
        "agent_run_id": "run-lease",
        "reason": "operator_stop",
    }
