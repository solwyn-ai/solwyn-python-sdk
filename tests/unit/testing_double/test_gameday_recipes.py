"""Deterministic control-plane game-day recipes for every CI run."""

from __future__ import annotations

import time
import uuid
from contextlib import closing
from threading import Event
from unittest.mock import patch

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, BudgetMode, CircuitState
from solwyn.budget import BudgetCheckResult, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane


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
