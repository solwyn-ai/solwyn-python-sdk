"""Deterministic control-plane game-day recipes for every CI run."""

from __future__ import annotations

import time
import uuid
from contextlib import closing
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import solwyn
from solwyn import BudgetExceededError
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, BudgetMode, CircuitState
from solwyn.budget import BudgetCheckResult, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane

_CHECK_PATH = "/api/v1/budgets/check"
# A body with real length: the receipt fold keys on estimated input size, so a
# zero-token call would make the aggregate's token sum prove nothing.
_REPEATED_MESSAGES = [{"role": "user", "content": "summarize the quarterly ledger"}]


def _legacy_enforcer(
    plane: FakeControlPlane,
    *,
    breaker: CircuitBreaker | None = None,
) -> BudgetEnforcer:
    return BudgetEnforcer(
        plane.api_url,
        plane.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
        cache_ttl=0,
        control_plane_breaker=breaker,
        lease_enabled=False,
        transport=plane.transport,
    )


def _legacy_check(
    enforcer: BudgetEnforcer,
    *,
    agent_run_id: str | None = None,
) -> BudgetCheckResult:
    return enforcer.check_budget(
        estimated_input_tokens=10,
        model="gpt-5.5",
        provider="openai",
        agent_run_id=agent_run_id,
    )


def _lease_check(enforcer: BudgetEnforcer, run_id: str) -> BudgetCheckResult:
    return enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=20,
        model="gpt-5.5",
        provider="openai",
        agent_run_id=run_id,
        call_id=str(uuid.uuid4()),
    )


def _reservation(plane: FakeControlPlane) -> str:
    with httpx.Client(transport=plane.transport, base_url=plane.api_url) as http:
        response = http.post(
            "/api/v1/budgets/check",
            json={
                "estimated_input_tokens": 10,
                "model": "gpt-5.5",
                "provider": "openai",
                "fallback_providers": [],
                "fallback_models": [],
                "failover_directive_version": "1",
            },
        )
    value = response.json()["reservation_id"]
    if not isinstance(value, str):
        raise RuntimeError("game-day setup expected a reservation")
    return value


def _confirm(reservation_id: str) -> BudgetConfirmRequest:
    return BudgetConfirmRequest(
        reservation_id=reservation_id,
        model="gpt-5.5",
        provider="openai",
        call_id=str(uuid.uuid4()),
        token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )


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
    """Minimal duck-typed OpenAI client: the double never mocks providers."""

    def __init__(self) -> None:
        self.chat = _SyncChat()

    def with_options(self, **_kwargs: object) -> _OpenAIStub:
        return self


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


def _wait_until(
    predicate: Any,
    *,
    timeout: float = 30.0,
    interval: float = 0.005,
    nudge: Any = None,
) -> None:
    # Generous: background reporter and lease threads pace these cycles, so
    # this returns the moment the condition holds and only a genuinely stalled
    # SDK ever spends the deadline. ``nudge`` runs once per poll for conditions
    # the SDK reaches only when fed more work.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if nudge is not None:
            nudge()
        time.sleep(interval)
    raise AssertionError("condition was not reached before the deadline")


def _call_until_stopped(client: Any, *, timeout: float = 30.0) -> solwyn.RunStoppedError:
    """Offer ordinary calls until a background stop lands, and return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.chat.completions.create(model="gpt-5.5", messages=[])
        except solwyn.RunStoppedError as stopped:
            return stopped
        time.sleep(0.01)
    raise AssertionError("the run was never stopped before the deadline")


@pytest.mark.unit
@pytest.mark.chaos
def test_deny_outage_recovery_preserves_then_clears_sticky_run_denial() -> None:
    plane = FakeControlPlane(budget_limit=5.0, current_usage=6.0, remaining_budget=-1.0)
    plane.deny_run("gameday-sticky-run")
    with closing(_legacy_enforcer(plane)) as enforcer:
        denied = _legacy_check(enforcer, agent_run_id="gameday-sticky-run")
        plane.clear_denials()
        with plane.outage(requests=1):
            preserved = _legacy_check(enforcer, agent_run_id="gameday-sticky-run")

        recovered = _legacy_check(enforcer, agent_run_id="gameday-sticky-run")
        with plane.outage(requests=1):
            post_recovery_outage = _legacy_check(
                enforcer,
                agent_run_id="gameday-sticky-run",
            )

    assert denied.allowed is False
    assert denied.denied_by_period == "agent_run"
    assert preserved.allowed is False
    assert preserved.denied_by_period == "agent_run"
    assert preserved.warning is not None
    assert "preserving prior hard deny" in preserved.warning
    assert recovered.allowed is True
    assert post_recovery_outage.allowed is True
    assert post_recovery_outage.reservation_id is None
    assert len(plane.checks) == 2


@pytest.mark.unit
@pytest.mark.chaos
def test_slow_confirm_is_bounded_by_reporter_shutdown_deadline_and_counted_once() -> None:
    plane = FakeControlPlane()
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        shutdown_deadline=0.05,
        transport=plane.transport,
    )
    try:
        confirm = _confirm(_reservation(plane))
        reporter.report_confirm(confirm)
        sleep_entered = Event()
        release_sleep = Event()
        sleep_finished = Event()

        def blocked_sleep(_seconds: float) -> None:
            sleep_entered.set()
            release_sleep.wait(timeout=1.0)
            sleep_finished.set()

        started = time.monotonic()
        with (
            plane.slow(1.0, path="/api/v1/budgets/confirm", requests=1),
            patch("solwyn.testing._transport.time.sleep", side_effect=blocked_sleep),
        ):
            reporter.close()
        elapsed = time.monotonic() - started
        release_sleep.set()
        assert sleep_finished.wait(timeout=1.0)
    finally:
        reporter.close()

    assert sleep_entered.is_set()
    # The point of this assertion is boundedness (the deadline cut the confirm
    # off rather than the full 1.0s slow-request window), not scheduler
    # precision — a tight 0.2s margin is flaky under CI load.
    assert elapsed < 1.0
    assert plane.confirms == [confirm]
    assert reporter.dropped_counts == {"confirm.shutdown_deadline": 1}
    assert len(reporter._confirm_queue) == 0
    assert reporter._in_hand.get("confirm", 0) == 0


@pytest.mark.unit
@pytest.mark.chaos
def test_lease_refusal_ladder_uses_legacy_then_outage_drawdown_and_hard_deny() -> None:
    plane = FakeControlPlane(
        mode=BudgetMode.HARD_DENY,
        granted_tokens=30,
        refresh_interval_s=60.0,
        lease_length_s=120.0,
    )
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=60,
        success_threshold=1,
        name="control-plane",
    )
    with closing(
        BudgetEnforcer(
            plane.api_url,
            plane.api_key,
            fail_open=True,
            cache_ttl=0,
            control_plane_breaker=breaker,
            holder_id="gameday-holder",
            transport=plane.transport,
        )
    ) as enforcer:
        with plane.refuse_leases(status=503, code="lease_unavailable", requests=1):
            unavailable_fallback = _lease_check(enforcer, "gameday-unavailable")
        with plane.refuse_leases(
            status=409,
            code="lease_holder_cap_exceeded",
            requests=1,
        ):
            holder_cap_fallback = _lease_check(enforcer, "gameday-holder-cap")

        granted = _lease_check(enforcer, "gameday-outage")
        with plane.outage(requests=1):
            from_share = _lease_check(enforcer, "gameday-outage")
            exhausted = _lease_check(enforcer, "gameday-outage")

    assert unavailable_fallback.allowed is True
    assert unavailable_fallback.lease_id is None
    assert holder_cap_fallback.allowed is True
    assert holder_cap_fallback.lease_id is None
    assert granted.allowed is True
    assert granted.lease_id == "lse_fake1"
    assert from_share.allowed is True
    assert from_share.lease_id == "lse_fake1"
    assert from_share.warning is not None
    assert "headroom share" in from_share.warning
    assert exhausted.allowed is False
    assert exhausted.denied_by_period == "agent_run"
    assert breaker.get_state().state is CircuitState.OPEN
    assert len(plane.lease_grants) == 1
    assert len(plane.checks) == 2


@pytest.mark.unit
@pytest.mark.chaos
def test_breaker_held_confirm_recovers_and_drains_exactly_once() -> None:
    plane = FakeControlPlane()
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=60,
        success_threshold=1,
        name="control-plane",
    )
    enforcer = _legacy_enforcer(plane, breaker=breaker)
    reporter = MetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        control_plane_breaker=breaker,
        retry_backoff_base=0.001,
        retry_backoff_cap=0.001,
        transport=plane.transport,
    )
    try:
        admitted = _legacy_check(enforcer)
        assert admitted.reservation_id is not None
        confirm = _confirm(admitted.reservation_id)
        with plane.outage(requests=1):
            unavailable = _legacy_check(enforcer)
        assert unavailable.reservation_id is None
        assert breaker.get_state().state is CircuitState.OPEN

        reporter.report_confirm(confirm)
        reporter._flush_remaining()

        assert len(reporter._confirm_queue) == 1
        assert plane.confirms == []
        assert reporter.dropped_counts == {}

        breaker.replace_tuning(
            failure_threshold=1,
            recovery_timeout=0,
            success_threshold=1,
            recovery_timeout_jitter=0.0,
        )
        reporter._flush_remaining()
        reporter._flush_remaining()
        reporter.close()
    finally:
        try:
            reporter.close()
        finally:
            enforcer.close()

    assert breaker.get_state().state is CircuitState.CLOSED
    assert len(reporter._confirm_queue) == 0
    assert plane.confirms == [confirm]
    assert reporter.dropped_counts == {}


# ── runaway-protection game days ─────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.chaos
def test_operator_kill_stays_terminal_through_a_control_plane_outage() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(provider, lease_enabled=False)
    try:
        with solwyn.run("gameday-kill") as run_id:
            wrapped.chat.completions.create(model="gpt-5.5", messages=[])
            plane.stop_run(run_id)
            with pytest.raises(solwyn.RunStoppedError) as killed:
                wrapped.chat.completions.create(model="gpt-5.5", messages=[])
            # Scoped to the check path so the outage cannot eat the reporter's
            # receipt deliveries alongside the verdict it is meant to break.
            with (
                plane.outage(path=_CHECK_PATH),
                pytest.raises(solwyn.RunStoppedError) as replayed,
            ):
                wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    finally:
        wrapped.close()

    assert killed.value.agent_run_id == run_id
    assert killed.value.reason == "manual_kill"
    assert killed.value.source == "server"
    # An unreachable plane never fails a killed run open: the SDK replays its
    # own sticky record of the stop instead of admitting the call.
    assert replayed.value.agent_run_id == run_id
    assert replayed.value.reason == "manual_kill"
    assert provider.chat.completions.calls == 1
    receipts = plane.denial_receipts
    assert len(receipts) == 2
    assert [receipt.deny_source for receipt in receipts] == ["server", "sticky_replay"]
    assert all(receipt.deny_reason == "manual_kill" for receipt in receipts)
    assert all(receipt.denied_by_period == "run_stopped" for receipt in receipts)


@pytest.mark.unit
@pytest.mark.chaos
def test_operator_kill_reaches_a_leased_run_through_its_renewal() -> None:
    plane = FakeControlPlane(refresh_interval_s=0.001, lease_length_s=60.0)
    provider = _OpenAIStub()
    wrapped = plane.wrap(provider)
    try:
        with solwyn.run("gameday-lease-kill") as run_id:
            wrapped.chat.completions.create(model="gpt-5.5", messages=[])
            plane.stop_run(run_id, reason="operator_stop")
            # The renewal flies off the hot path, so keep offering ordinary
            # calls until the stop it carries lands in the run registry.
            stopped = _call_until_stopped(wrapped)
            terminated = solwyn.current_run_terminated()
    finally:
        wrapped.close()

    assert len(plane.lease_grants) == 1
    assert plane.lease_renewals
    assert all(renewal.run_directive_version == "1" for renewal in plane.lease_renewals)
    assert terminated is True
    assert stopped.agent_run_id == run_id
    assert stopped.reason == "operator_stop"
    assert stopped.source == "server"
    # A killed run keeps no lease to hand back: the stop drops it outright, so
    # no surrender is ever owed to the plane.
    assert plane.lease_surrenders == []
    receipts = plane.denial_receipts
    assert receipts
    assert all(receipt.denied_by_period == "run_stopped" for receipt in receipts)
    assert all(receipt.deny_reason == "operator_stop" for receipt in receipts)


@pytest.mark.unit
@pytest.mark.chaos
def test_rejected_denial_receipts_fold_and_replay_as_one_aggregate() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(
        provider,
        lease_enabled=False,
        velocity_mode="warn",
        velocity_repeat_count=2,
        reporter_batch_size=1,
        reporter_flush_interval=0.01,
        report_untracked_surfaces=False,
    )
    try:
        with solwyn.run("gameday-receipt-loss"):
            with plane.reject_ingest(indices=[0], requests=None):
                for _ in range(2):
                    # Identical calls: same model, run, period, and estimated
                    # size, so both receipts share one pricing-compatible
                    # fold identity and can only replay as ONE aggregate.
                    with pytest.raises(BudgetExceededError):
                        wrapped.chat.completions.create(
                            model="solwyn-test/deny",
                            messages=_REPEATED_MESSAGES,
                        )
                # denial_receipts takes the plane lock, so the window can only
                # close once both rejecting responses are fully built.
                _wait_until(lambda: len(plane.denial_receipts) == 2)
            dropped = list(plane.denial_receipts)

            # A rejected batch is no recovery proof; a clean cycle is, and the
            # cycle AFTER a clean one replays the fold. Keep offering ordinary
            # traffic until the replay lands rather than assuming one call
            # bought a clean cycle.
            _wait_until(
                lambda: plane.aggregate_replays,
                nudge=lambda: wrapped.chat.completions.create(model="gpt-5.5", messages=[]),
                interval=0.05,
            )
    finally:
        wrapped.close()

    (replay,) = plane.aggregate_replays
    assert replay.receipt_aggregate_count == 2
    assert replay.input_tokens == sum(receipt.input_tokens for receipt in dropped)
    assert [receipt.velocity_flags for receipt in dropped] == [None, ["repeat_size"]]
    assert replay.velocity_flags == sorted(
        {flag for receipt in dropped for flag in (receipt.velocity_flags or ())}
    )
    assert replay.input_tokens > 0
    assert replay.deny_reason == "monthly"
    assert replay.call_id not in {receipt.call_id for receipt in dropped}


@pytest.mark.unit
@pytest.mark.chaos
def test_a_local_velocity_stop_is_never_lifted_by_a_permissive_plane() -> None:
    plane = FakeControlPlane()
    provider = _OpenAIStub()
    wrapped = plane.wrap(
        provider,
        lease_enabled=False,
        velocity_mode="deny",
        velocity_repeat_count=2,
    )
    try:
        with solwyn.run("gameday-velocity") as run_id:
            wrapped.chat.completions.create(model="gpt-5.5", messages=[])
            with pytest.raises(solwyn.RunStoppedError) as stopped:
                wrapped.chat.completions.create(model="gpt-5.5", messages=[])
            checks_at_stop = len(plane.checks)

            # The plane itself is entirely permissive for this run...
            with httpx.Client(transport=plane.transport) as http:
                allowed = http.post(
                    f"{plane.api_url}/api/v1/budgets/check",
                    json={
                        "estimated_input_tokens": 10,
                        "model": "gpt-5.5",
                        "provider": "openai",
                        "fallback_providers": [],
                        "fallback_models": [],
                        "agent_run_id": run_id,
                        "run_directive_version": "1",
                    },
                ).json()

            # ...and the run stays stopped anyway: a server allow can never
            # lift a local stop, and the call never reaches the plane again.
            with pytest.raises(solwyn.RunStoppedError) as after_allow:
                wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    finally:
        wrapped.close()

    assert allowed["allowed"] is True
    assert stopped.value.agent_run_id == run_id
    assert stopped.value.reason == "velocity:repeat_size"
    assert stopped.value.source == "local_velocity"
    assert after_allow.value.reason == "velocity:repeat_size"
    # The only check the plane saw after the stop is this test's own probe:
    # a stopped run never asks the server for permission again.
    assert len(plane.checks) == checks_at_stop + 1
    assert plane.stopped_runs == {}
    assert provider.chat.completions.calls == 1
    receipts = plane.denial_receipts
    assert [receipt.deny_source for receipt in receipts] == [
        "local_velocity",
        "run_terminated",
    ]
    assert all(receipt.deny_reason == "velocity:repeat_size" for receipt in receipts)
    assert all(receipt.denied_by_period == "run_stopped" for receipt in receipts)
    assert receipts[0].velocity_flags == ["repeat_size"]
