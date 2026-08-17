"""Composable, request-counted control-plane scenarios."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import httpx
import pytest

from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane


def _check_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "estimated_input_tokens": 1,
        "model": "gpt-5.5",
        "provider": "openai",
        "fallback_providers": [],
        "fallback_models": [],
        "failover_directive_version": "1",
    }
    payload.update(overrides)
    return payload


def _post_check(
    client: httpx.Client,
    plane: FakeControlPlane,
    **overrides: object,
) -> httpx.Response:
    return client.post(
        f"{plane.api_url}/api/v1/budgets/check",
        json=_check_payload(**overrides),
    )


@pytest.mark.unit
def test_run_denial_and_clear_denials_are_instance_scoped() -> None:
    plane = FakeControlPlane()
    other = FakeControlPlane()
    plane.deny_run("run-a")

    with httpx.Client(transport=plane.transport) as client:
        denied = _post_check(client, plane, agent_run_id="run-a")
        allowed_other_run = _post_check(client, plane, agent_run_id="run-b")
        plane.clear_denials()
        allowed_after_clear = _post_check(client, plane, agent_run_id="run-a")
    with httpx.Client(transport=other.transport) as client:
        allowed_other_plane = _post_check(client, other, agent_run_id="run-a")

    assert denied.json()["denied_by_period"] == "agent_run"
    assert allowed_other_run.json()["allowed"] is True
    assert allowed_after_clear.json()["allowed"] is True
    assert allowed_other_plane.json()["allowed"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "allowed", "mode", "period"),
    [
        ("solwyn-test/deny", False, "hard_deny", "monthly"),
        ("solwyn-test/deny-alert", False, "alert_only", "monthly"),
        ("solwyn-test/deny-tag", False, "hard_deny", "tag"),
        ("solwyn-test/deny-stopped", False, "hard_deny", "run_stopped"),
        ("solwyn-test/lease-ineligible", True, "hard_deny", None),
    ],
)
def test_magic_models_have_deterministic_check_verdicts(
    model: str,
    allowed: bool,
    mode: str,
    period: str | None,
) -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        response = _post_check(client, plane, model=model)

    assert response.json()["allowed"] is allowed
    assert response.json()["mode"] == mode
    assert response.json().get("denied_by_period") == period


@pytest.mark.unit
def test_runaway_allows_first_check_per_run_then_denies_later_checks() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        first_a = _post_check(
            client,
            plane,
            model="solwyn-test/runaway",
            agent_run_id="run-a",
        )
        second_a = _post_check(
            client,
            plane,
            model="solwyn-test/runaway",
            agent_run_id="run-a",
        )
        first_b = _post_check(
            client,
            plane,
            model="solwyn-test/runaway",
            agent_run_id="run-b",
        )

    assert first_a.json()["allowed"] is True
    assert second_a.json()["denied_by_period"] == "agent_run"
    assert first_b.json()["allowed"] is True


@pytest.mark.unit
def test_outage_is_request_counted_and_recovers_within_context() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client, plane.outage(requests=1):
        with pytest.raises(httpx.ConnectError, match="scripted outage"):
            _post_check(client, plane)
        recovered = _post_check(client, plane)

    assert recovered.status_code == 200
    assert len(plane.checks) == 1


@pytest.mark.unit
async def test_outage_has_an_async_transport_counterpart() -> None:
    plane = FakeControlPlane()

    async with httpx.AsyncClient(transport=plane.transport) as client:
        with plane.outage(requests=1):
            with pytest.raises(httpx.ConnectError, match="scripted outage"):
                await client.post(
                    f"{plane.api_url}/api/v1/budgets/check",
                    json=_check_payload(),
                )
            recovered = await client.post(
                f"{plane.api_url}/api/v1/budgets/check",
                json=_check_payload(),
            )

    assert recovered.status_code == 200


@pytest.mark.unit
def test_slow_uses_sync_sleep_only_for_the_matching_path() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        started = time.monotonic()
        with plane.slow(0.03, path="/api/v1/budgets/check", requests=1):
            _post_check(client, plane)
            elapsed = time.monotonic() - started
            fast_started = time.monotonic()
            _post_check(client, plane)
            fast_elapsed = time.monotonic() - fast_started

    assert elapsed >= 0.025
    assert fast_elapsed < 0.025


@pytest.mark.unit
async def test_slow_uses_async_sleep_for_async_requests() -> None:
    plane = FakeControlPlane()

    async with httpx.AsyncClient(transport=plane.transport) as client:
        started = time.monotonic()
        with plane.slow(0.03, path="/api/v1/budgets/check", requests=1):
            await client.post(
                f"{plane.api_url}/api/v1/budgets/check",
                json=_check_payload(),
            )

    assert time.monotonic() - started >= 0.025


@pytest.mark.unit
def test_slow_request_keeps_refusal_after_contexts_exit_during_sleep() -> None:
    plane = FakeControlPlane()
    sleep_entered = Event()
    release_sleep = Event()

    def blocked_sleep(_seconds: float) -> None:
        sleep_entered.set()
        if not release_sleep.wait(timeout=2):
            raise RuntimeError("test did not release scripted sleep")

    with (
        httpx.Client(transport=plane.transport) as client,
        ThreadPoolExecutor(max_workers=1) as pool,
        patch("solwyn.testing._transport.time.sleep", side_effect=blocked_sleep),
    ):
        try:
            with (
                plane.slow(1, path="/api/v1/budgets/check", requests=1),
                plane.refuse_checks(status=503, requests=1),
            ):
                pending = pool.submit(_post_check, client, plane)
                assert sleep_entered.wait(timeout=1)
        finally:
            release_sleep.set()
        refused = pending.result(timeout=1)
        recorded_before_recovery = list(plane.checks)
        recovered = _post_check(client, plane)

    assert refused.status_code == 503
    assert refused.json() == {"detail": "Budget backend temporarily unavailable; retry"}
    assert recorded_before_recovery == []
    assert recovered.status_code == 200
    assert len(plane.checks) == 1


@pytest.mark.unit
async def test_slow_request_keeps_read_only_after_contexts_exit_during_sleep() -> None:
    plane = FakeControlPlane()
    sleep_entered = asyncio.Event()
    release_sleep = asyncio.Event()

    async def blocked_sleep(_seconds: float) -> None:
        sleep_entered.set()
        await release_sleep.wait()

    async with httpx.AsyncClient(transport=plane.transport) as client:
        with patch("solwyn.testing._transport.asyncio.sleep", new=blocked_sleep):
            try:
                with (
                    plane.slow(1, path="/api/v1/budgets/check", requests=1),
                    plane.read_only(requests=1),
                ):
                    pending = asyncio.create_task(
                        client.post(
                            f"{plane.api_url}/api/v1/budgets/check",
                            json=_check_payload(),
                        )
                    )
                    await asyncio.wait_for(sleep_entered.wait(), timeout=1)
            finally:
                release_sleep.set()
            refused = await asyncio.wait_for(pending, timeout=1)
            recorded_before_recovery = list(plane.checks)
            recovered = await client.post(
                f"{plane.api_url}/api/v1/budgets/check",
                json=_check_payload(),
            )

    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "read_only_key"
    assert recorded_before_recovery == []
    assert recovered.status_code == 200
    assert len(plane.checks) == 1


@pytest.mark.unit
def test_read_only_returns_the_sdk_recognized_403_shape() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client, plane.read_only(requests=1):
        response = _post_check(client, plane)
        recovered = _post_check(client, plane)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "read_only_key"
    assert recovered.status_code == 200


@pytest.mark.unit
def test_refuse_checks_429_sets_retry_after_and_is_counted() -> None:
    plane = FakeControlPlane()

    with (
        httpx.Client(transport=plane.transport) as client,
        plane.refuse_checks(status=429, requests=1, retry_after=1),
    ):
        refused = _post_check(client, plane)
        recovered = _post_check(client, plane)

    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "1"
    assert refused.json() == {"detail": "Rate limit exceeded", "retry_after": 1}
    assert recovered.status_code == 200


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (
            422,
            {
                "detail": {
                    "code": "unknown_model",
                    "model": "gpt-5.5",
                    "provider": "openai",
                    "message": (
                        "Solwyn does not have pricing for model 'gpt-5.5'. "
                        "File an issue at "
                        "https://github.com/solwyn-ai/solwyn-python-sdk/issues "
                        "or contact support — we typically add new models within 24h."
                    ),
                }
            },
        ),
        (503, {"detail": "Budget backend temporarily unavailable; retry"}),
    ],
)
def test_refuse_checks_uses_real_core_error_shapes(
    status: int,
    expected: dict[str, object],
) -> None:
    plane = FakeControlPlane()

    with (
        httpx.Client(transport=plane.transport) as client,
        plane.refuse_checks(status=status, requests=1),
    ):
        response = _post_check(client, plane)

    assert response.status_code == status
    assert response.json() == expected


@pytest.mark.unit
def test_refuse_checks_429_drives_real_enforcer_to_outage_posture() -> None:
    from solwyn.budget import BudgetEnforcer

    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(
        plane.api_url,
        plane.api_key,
        cache_ttl=0,
        lease_enabled=False,
        transport=plane.transport,
    )

    with plane.refuse_checks(status=429, requests=1, retry_after=1):
        result = enforcer.check_budget(
            estimated_input_tokens=1,
            model="gpt-5.5",
            provider="openai",
        )
    enforcer.close()

    assert result.allowed is True
    assert result.reservation_id is None
    assert result.warning is not None


@pytest.mark.unit
def test_three_scripted_outages_open_real_breaker_then_recover() -> None:
    from solwyn.budget import BudgetEnforcer

    plane = FakeControlPlane()
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=0.05,
        success_threshold=1,
        name="control-plane",
    )
    enforcer = BudgetEnforcer(
        plane.api_url,
        plane.api_key,
        cache_ttl=0,
        control_plane_breaker=breaker,
        lease_enabled=False,
        transport=plane.transport,
    )

    with plane.outage(requests=3):
        for _ in range(3):
            result = enforcer.check_budget(
                estimated_input_tokens=1,
                model="gpt-5.5",
                provider="openai",
            )
            assert result.reservation_id is None
        skipped = enforcer.check_budget(
            estimated_input_tokens=1,
            model="gpt-5.5",
            provider="openai",
        )
        assert skipped.reservation_id is None
        time.sleep(0.06)
        recovered = enforcer.check_budget(
            estimated_input_tokens=1,
            model="gpt-5.5",
            provider="openai",
        )
    enforcer.close()

    assert recovered.reservation_id is not None
    assert len(plane.checks) == 1


@pytest.mark.unit
def test_validation_failures_return_422_on_checks_and_lease_paths() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        invalid = _post_check(client, plane, unexpected="drift")
        lease = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json={},
        )

    assert invalid.status_code == 422
    assert isinstance(invalid.json()["detail"], list)
    assert lease.status_code == 422
    assert isinstance(lease.json()["detail"], list)
