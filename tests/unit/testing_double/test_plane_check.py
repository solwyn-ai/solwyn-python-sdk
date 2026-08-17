"""Budget-check behavior of the in-process control-plane double."""

from __future__ import annotations

import httpx
import pytest

from solwyn._types import BudgetMode
from solwyn.budget import BudgetEnforcer
from solwyn.testing import MAGIC_MODELS, FakeControlPlane


def _check_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "estimated_input_tokens": 17,
        "model": "gpt-5.5",
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_default_budget_enforcer_check_allows_and_records_directive_opt_in() -> None:
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(
        plane.api_url,
        plane.api_key,
        cache_ttl=0,
        lease_enabled=False,
        transport=plane.transport,
    )

    result = enforcer.check_budget(
        estimated_input_tokens=17,
        model="gpt-5.5",
        provider="openai",
    )
    enforcer.close()

    assert result.allowed is True
    assert result.reservation_id is not None
    assert result.failover_tuning_allowed is True
    assert len(plane.checks) == 1
    assert plane.checks[0].failover_directive_version == "1"


@pytest.mark.unit
def test_deny_next_is_consumed_once_and_v1_allow_omits_denied_period() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")

    with httpx.Client(transport=plane.transport) as client:
        first = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(failover_directive_version="1"),
        )
        second = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(failover_directive_version="1"),
        )

    assert first.status_code == 200
    assert first.json()["allowed"] is False
    assert first.json()["denied_by_period"] == "monthly"
    assert second.status_code == 200
    assert second.json()["allowed"] is True
    assert "denied_by_period" not in second.json()


@pytest.mark.unit
def test_check_serialization_is_legacy_nullable_until_directive_v1_opt_in() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        legacy = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(),
        )
        opted_in = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(failover_directive_version="1"),
        )

    assert legacy.json()["denied_by_period"] is None
    assert legacy.json()["price_hints"] is None
    assert legacy.json()["failover_directive"] is None
    assert "denied_by_period" not in opted_in.json()
    assert "price_hints" not in opted_in.json()
    assert opted_in.json()["failover_directive"] == {
        "version": "1",
        "failover_tuning_allowed": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize("model", sorted(MAGIC_MODELS))
def test_all_documented_magic_models_are_reserved(model: str) -> None:
    assert model.startswith("solwyn-test/")


@pytest.mark.unit
def test_unknown_magic_model_fails_loud() -> None:
    plane = FakeControlPlane()

    with (
        httpx.Client(transport=plane.transport) as client,
        pytest.raises(RuntimeError, match="unknown solwyn testing magic model"),
    ):
        client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(model="solwyn-test/not-real"),
        )


@pytest.mark.unit
def test_unknown_magic_fallback_model_fails_loud_on_the_raw_transport() -> None:
    plane = FakeControlPlane()

    with (
        httpx.Client(transport=plane.transport) as client,
        pytest.raises(RuntimeError, match="unknown solwyn testing magic model"),
    ):
        client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json=_check_payload(
                fallback_providers=["openai"],
                fallback_models=["solwyn-test/not-real"],
            ),
        )

    assert plane.checks == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fallback_models", "expected_period", "expected_mode"),
    [
        (["solwyn-test/deny"], "monthly", "hard_deny"),
        (["solwyn-test/deny-alert"], "monthly", "alert_only"),
        (["solwyn-test/deny-tag"], "tag", "hard_deny"),
        (["solwyn-test/deny-stopped"], "run_stopped", "hard_deny"),
        (
            ["solwyn-test/deny-alert", "solwyn-test/deny-tag"],
            "monthly",
            "alert_only",
        ),
    ],
)
def test_first_magic_fallback_drives_raw_check_verdict(
    fallback_models: list[str],
    expected_period: str,
    expected_mode: str,
) -> None:
    plane = FakeControlPlane()

    response = plane.handle(
        "POST",
        "/api/v1/budgets/check",
        _check_payload(
            fallback_providers=["openai"] * len(fallback_models),
            fallback_models=fallback_models,
        ),
    )

    assert response.status_code == 200
    assert response.body.allowed is False
    assert response.body.denied_by_period == expected_period
    assert response.body.mode == expected_mode


@pytest.mark.unit
def test_programmatic_denial_precedes_fallback_magic_and_is_consumed_once() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="tag")
    payload = _check_payload(
        fallback_providers=["openai"],
        fallback_models=["solwyn-test/deny-alert"],
    )

    first = plane.handle("POST", "/api/v1/budgets/check", payload)
    second = plane.handle("POST", "/api/v1/budgets/check", payload)

    assert first.body.denied_by_period == "tag"
    assert first.body.mode == "hard_deny"
    assert second.body.denied_by_period == "monthly"
    assert second.body.mode == "alert_only"


@pytest.mark.unit
def test_fallback_runaway_memory_is_ordered_and_deterministic() -> None:
    plane = FakeControlPlane()
    payload = _check_payload(
        agent_run_id="run-fallback-runaway",
        fallback_providers=["openai"],
        fallback_models=["solwyn-test/runaway"],
    )

    first = plane.handle("POST", "/api/v1/budgets/check", payload)
    second = plane.handle("POST", "/api/v1/budgets/check", payload)

    assert first.body.allowed is True
    assert second.body.allowed is False
    assert second.body.denied_by_period == "agent_run"


@pytest.mark.unit
def test_unknown_path_is_recorded_and_returns_404() -> None:
    plane = FakeControlPlane(mode=BudgetMode.ALERT_ONLY)

    with httpx.Client(transport=plane.transport) as client:
        response = client.get(f"{plane.api_url}/api/v1/not-real")

    assert response.status_code == 404
    assert plane.unmatched_requests == [("GET", "/api/v1/not-real")]


@pytest.mark.unit
def test_malformed_check_json_returns_core_like_422() -> None:
    plane = FakeControlPlane()
    request = httpx.Request(
        "POST",
        f"{plane.api_url}/api/v1/budgets/check",
        content=b"{",
        headers={"Content-Type": "application/json"},
    )

    response = plane.transport.handle_request(request)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
