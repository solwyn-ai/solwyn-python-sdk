"""SDK budget-check behavior dogfooded on ``FakeControlPlane``."""

from __future__ import annotations

from contextlib import closing

import pytest

from solwyn._types import BudgetMode, CircuitState
from solwyn.budget import BudgetCheckResult, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane


def _enforcer(
    plane: FakeControlPlane,
    *,
    breaker: CircuitBreaker | None = None,
    cache_ttl: int = 0,
) -> BudgetEnforcer:
    return BudgetEnforcer(
        api_url=plane.api_url,
        api_key=plane.api_key,
        budget_mode=BudgetMode.ALERT_ONLY,
        fail_open=True,
        cache_ttl=cache_ttl,
        control_plane_breaker=breaker,
        lease_enabled=False,
        transport=plane.transport,
    )


def _check(enforcer: BudgetEnforcer) -> BudgetCheckResult:
    return enforcer.check_budget(
        estimated_input_tokens=100,
        model="gpt-5.5",
        provider="openai",
    )


@pytest.mark.unit
def test_check_budget_returns_allowed_with_complete_budget_metadata() -> None:
    plane = FakeControlPlane(
        budget_limit=100.0,
        current_usage=12.5,
        remaining_budget=87.5,
    )
    with closing(_enforcer(plane)) as enforcer:
        result = _check(enforcer)

    assert result.allowed is True
    assert result.reservation_id == "res_fake_00000001"
    assert result.remaining_budget == 87.5
    assert result.budget_limit == 100.0
    assert result.current_usage == 12.5
    assert result.project_id == plane.project_id
    assert len(plane.checks) == 1
    assert plane.checks[0].failover_directive_version == "1"


@pytest.mark.unit
def test_hard_denial_preserves_server_verdict_and_never_returns_reservation() -> None:
    plane = FakeControlPlane(budget_limit=5.0, current_usage=6.0, remaining_budget=-1.0)
    plane.deny_next(period="monthly")
    with closing(_enforcer(plane)) as enforcer:
        result = _check(enforcer)

    assert result.allowed is False
    assert result.reservation_id is None
    assert result.denied_by_period == "monthly"
    assert result.budget_limit == 5.0
    assert result.current_usage == 6.0


@pytest.mark.unit
def test_alert_only_denial_allows_with_warning_but_keeps_denial_metadata() -> None:
    plane = FakeControlPlane(
        mode=BudgetMode.ALERT_ONLY,
        budget_limit=5.0,
        current_usage=6.0,
        remaining_budget=-1.0,
    )
    plane.deny_next(period="monthly")
    with closing(_enforcer(plane)) as enforcer:
        result = _check(enforcer)

    assert result.allowed is True
    assert result.mode is BudgetMode.ALERT_ONLY
    assert result.reservation_id is None
    assert result.denied_by_period == "monthly"
    assert result.warning == "Budget limit reached: $6.00/$5.00 used"


@pytest.mark.unit
def test_allow_cache_skips_second_network_check_without_reusing_reservation() -> None:
    plane = FakeControlPlane()
    with closing(_enforcer(plane, cache_ttl=60)) as enforcer:
        first = _check(enforcer)
        cached = _check(enforcer)

    assert first.allowed is True
    assert first.reservation_id is not None
    assert cached.allowed is True
    assert cached.reservation_id is None
    assert len(plane.checks) == 1


@pytest.mark.unit
def test_three_outages_fail_open_and_open_the_real_control_plane_breaker() -> None:
    plane = FakeControlPlane()
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=1,
        name="control-plane",
    )
    with closing(_enforcer(plane, breaker=breaker)) as enforcer, plane.outage(requests=3):
        unavailable = [_check(enforcer) for _ in range(3)]
        breaker_skipped = _check(enforcer)

    assert all(result.allowed is True for result in unavailable)
    assert all(result.reservation_id is None for result in unavailable)
    assert all(
        result.warning is not None and "fail-open" in result.warning.lower()
        for result in unavailable
    )
    assert breaker_skipped.allowed is True
    assert breaker_skipped.reservation_id is None
    assert breaker.get_state().state is CircuitState.OPEN
    assert breaker.get_state().failure_count == 3
    assert plane.checks == []
