"""Lease wire surface for ``FakeControlPlane``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

import solwyn as solwyn_pkg
from solwyn._types import (
    BudgetMode,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    ProviderName,
)
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.testing import FakeControlPlane

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


def _grant_payload(
    *,
    run_id: str = "run-lease",
    holder_id: str = "holder-1",
    model: str = "gpt-5.5",
    fail_open: bool = True,
) -> dict[str, object]:
    return LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=model,
        provider=ProviderName.OPENAI,
        fallback_providers=[],
        fallback_models=[],
        fail_open=fail_open,
        estimated_input_tokens=123,
    ).model_dump(mode="json")


def _post_grant(
    client: httpx.Client,
    plane: FakeControlPlane,
    **payload_overrides: object,
) -> httpx.Response:
    return client.post(
        f"{plane.api_url}/api/v1/budgets/lease",
        json=_grant_payload(**payload_overrides),  # type: ignore[arg-type]
    )


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


def _make_async_enforcer(plane: FakeControlPlane, **overrides: object) -> AsyncBudgetEnforcer:
    options: dict[str, object] = {
        "api_url": plane.api_url,
        "api_key": plane.api_key,
        "cache_ttl": 0,
        "transport": plane.transport,
        "holder_id": "holder-1",
    }
    options.update(overrides)
    return AsyncBudgetEnforcer(**options)  # type: ignore[arg-type]


def _check(
    enforcer: BudgetEnforcer,
    *,
    run_id: str = "run-lease",
    model: str = "gpt-5.5",
) -> object:
    return enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=20,
        model=model,
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
        agent_run_id=run_id,
        call_id=str(uuid4()),
    )


async def _acheck(
    enforcer: AsyncBudgetEnforcer,
    *,
    run_id: str = "run-lease",
    model: str = "gpt-5.5",
) -> object:
    return await enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=20,
        model=model,
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
        agent_run_id=run_id,
        call_id=str(uuid4()),
    )


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


async def _await_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


class _SyncCompletions:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class _SyncOpenAIStub:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SyncCompletions())

    def with_options(self, **_kwargs: object) -> _SyncOpenAIStub:
        return self


_SyncOpenAIStub.__module__ = "openai._client"
_SyncOpenAIStub.__name__ = "OpenAI"


class _AsyncCompletions:
    async def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


class _AsyncOpenAIStub:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_AsyncCompletions())

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIStub:
        return self


_AsyncOpenAIStub.__module__ = "openai._client"
_AsyncOpenAIStub.__name__ = "AsyncOpenAI"


@pytest.mark.unit
def test_grant_returns_complete_real_lease_response_and_records_request() -> None:
    plane = FakeControlPlane(
        mode=BudgetMode.ALERT_ONLY,
        budget_limit=80.0,
        current_usage=12.5,
        remaining_budget=67.5,
        granted_tokens=9_000,
        refresh_interval_s=4.0,
        lease_length_s=12.0,
    )

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(fail_open=False),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload.keys() >= LEASE_BLOCK_KEYS
    assert payload["lease_id"] == "lse_fake1"
    assert payload["generation"] == 1
    assert payload["granted_tokens"] == 9_000
    assert payload["refresh_interval_s"] == 4.0
    assert payload["lease_length_s"] == 12.0
    assert payload["refresh_interval_s"] < payload["lease_length_s"]
    assert payload["headroom_share_tokens"] == 9_000
    assert payload["posture"] == {
        "mode": "alert_only",
        "on_unreachable": "local_enforce",
    }
    assert payload["final_grant"] is False
    assert payload["project_id"] == plane.project_id
    assert payload["mode"] == "alert_only"
    assert payload["budget_limit"] == 80.0
    assert payload["current_usage"] == 12.5
    assert payload["remaining_budget"] == 67.5
    assert LeaseGrantResponse.model_validate(payload).lease_id == "lse_fake1"
    assert plane.lease_grants == [LeaseGrantRequest.model_validate(_grant_payload(fail_open=False))]


@pytest.mark.unit
def test_renew_increments_generation_and_fences_stale_or_unknown_leases() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        renewal_request = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=1,
            spent_tokens=100,
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal_request.model_dump(mode="json"),
        )
        stale = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal_request.model_dump(mode="json"),
        )
        unknown_request = LeaseRenewRequest(
            lease_id="lse_unknown",
            holder_id="holder-1",
            generation=1,
        )
        unknown = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=unknown_request.model_dump(mode="json"),
        )

    assert renewed.status_code == 200
    payload = renewed.json()
    assert payload["lease_id"] == grant["lease_id"]
    assert payload["generation"] == 2
    assert payload.keys() >= LEASE_BLOCK_KEYS
    assert LeaseGrantResponse.model_validate(payload).generation == 2
    assert stale.status_code == 409
    assert stale.json() == {
        "detail": {
            "code": "lease_generation_conflict",
            "message": "Budget lease generation conflict",
        }
    }
    assert unknown.status_code == 404
    assert unknown.json() == {
        "detail": {
            "code": "lease_not_found",
            "message": "Budget lease not found",
        }
    }
    assert plane.lease_renewals == [renewal_request, renewal_request, unknown_request]


@pytest.mark.unit
def test_surrender_is_recorded_and_idempotently_releases_tokens() -> None:
    plane = FakeControlPlane(granted_tokens=321)

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        request = LeaseSurrenderRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            spent_tokens=21,
        )
        first = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_dump(mode="json"),
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_dump(mode="json"),
        )

    assert first.status_code == 200
    assert first.json() == {"released_tokens": 321}
    assert replay.status_code == 200
    assert replay.json() == {"released_tokens": 0}
    assert plane.lease_surrenders == [request, request]


@pytest.mark.unit
def test_denied_grant_omits_lease_block_and_is_sticky_in_real_enforcer() -> None:
    raw_plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    raw_plane.deny_next(period="agent_run")
    with httpx.Client(transport=raw_plane.transport) as client:
        raw = _post_grant(client, raw_plane)

    assert raw.status_code == 200
    payload = raw.json()
    assert payload["eligible"] is True
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == "agent_run"
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)

    plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    plane.deny_next(period="agent_run")
    enforcer = _make_enforcer(plane)

    first = _check(enforcer)
    with plane.outage():
        second = _check(enforcer)
    enforcer.close()

    assert first.allowed is False  # type: ignore[attr-defined]
    assert first.mode is BudgetMode.HARD_DENY  # type: ignore[attr-defined]
    assert second.allowed is False  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "period", "mode"),
    [
        ("solwyn-test/deny", "monthly", "hard_deny"),
        ("solwyn-test/deny-alert", "monthly", "alert_only"),
        ("solwyn-test/deny-tag", "tag", "hard_deny"),
        ("solwyn-test/deny-stopped", "run_stopped", "hard_deny"),
    ],
)
def test_magic_lease_denials_match_check_verdicts(
    model: str,
    period: str,
    mode: str,
) -> None:
    plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane, model=model)

    payload = response.json()
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == period
    assert payload["mode"] == mode
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gpt-5.5", "solwyn-test/lease-ineligible"])
def test_ineligible_grant_omits_lease_block_and_falls_back_to_check(model: str) -> None:
    plane = FakeControlPlane(lease_eligible=model != "gpt-5.5")
    enforcer = _make_enforcer(plane)

    result = _check(enforcer, model=model)
    enforcer.close()

    assert result.allowed is True  # type: ignore[attr-defined]
    assert result.lease_id is None  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 1
    assert len(plane.checks) == 1

    with httpx.Client(transport=plane.transport) as client:
        raw = _post_grant(
            client,
            plane,
            run_id=f"raw-{model}",
            holder_id=f"raw-{model}",
            model=model,
        )
    payload = raw.json()
    assert payload["eligible"] is False
    assert payload["ineligible_reason"] == "zero_rate_model"
    assert payload["allowed"] is True
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (503, "lease_unavailable", "Budget lease service temporarily unavailable; retry"),
        (409, "lease_holder_cap_exceeded", "Active lease holder limit exceeded"),
    ],
)
def test_lease_refusal_has_core_shape_and_real_enforcer_uses_legacy_check(
    status: int,
    code: str,
    message: str,
) -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)

    with plane.refuse_leases(status=status, code=code, requests=1):
        result = _check(enforcer)
    enforcer.close()

    assert result.allowed is True  # type: ignore[attr-defined]
    assert result.lease_id is None  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 0
    assert len(plane.checks) == 1

    raw_plane = FakeControlPlane()
    with (
        httpx.Client(transport=raw_plane.transport) as client,
        raw_plane.refuse_leases(status=status, code=code, requests=1),
    ):
        refused = _post_grant(client, raw_plane)
        recovered = _post_grant(client, raw_plane, run_id="run-2")

    assert refused.status_code == status
    assert refused.json() == {"detail": {"code": code, "message": message}}
    assert recovered.status_code == 200


@pytest.mark.unit
def test_expire_leases_returns_not_found_until_a_new_generation_one_grant() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        first = _post_grant(client, plane).json()
        plane.expire_leases()
        expired_request = LeaseRenewRequest(
            lease_id=first["lease_id"],
            holder_id="holder-1",
            generation=first["generation"],
        )
        expired = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=expired_request.model_dump(mode="json"),
        )
        replacement = _post_grant(client, plane).json()
        stale_old = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=expired_request.model_dump(mode="json"),
        )
        renewed_replacement = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=replacement["lease_id"],
                holder_id="holder-1",
                generation=replacement["generation"],
            ).model_dump(mode="json"),
        )

    not_found = {"detail": {"code": "lease_not_found", "message": "Budget lease not found"}}
    assert expired.status_code == 404
    assert expired.json() == not_found
    assert replacement["lease_id"] == "lse_fake2"
    assert replacement["generation"] == 1
    assert stale_old.status_code == 404
    assert stale_old.json() == not_found
    assert renewed_replacement.status_code == 200
    assert renewed_replacement.json()["generation"] == 2


@pytest.mark.unit
def test_outage_after_grant_uses_share_then_applies_hard_deny_at_exhaustion() -> None:
    plane = FakeControlPlane(
        mode=BudgetMode.HARD_DENY,
        granted_tokens=30,
        refresh_interval_s=60.0,
        lease_length_s=120.0,
    )
    breaker = CircuitBreaker(failure_threshold=1, name="control-plane")
    enforcer = _make_enforcer(plane, control_plane_breaker=breaker)

    granted = _check(enforcer)
    with plane.outage():
        from_share = _check(enforcer)
        exhausted = _check(enforcer)
    enforcer.close()

    assert granted.allowed is True  # type: ignore[attr-defined]
    assert granted.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert from_share.allowed is True  # type: ignore[attr-defined]
    assert from_share.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert exhausted.allowed is False  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 1


@pytest.mark.unit
def test_expiration_forces_real_enforcer_to_drop_and_regrant() -> None:
    plane = FakeControlPlane(
        granted_tokens=1_000,
        refresh_interval_s=0.001,
        lease_length_s=10.0,
    )
    enforcer = _make_enforcer(plane)

    first = _check(enforcer)
    plane.expire_leases()
    time.sleep(0.01)
    second = _check(enforcer)
    assert _wait_for(lambda: len(plane.lease_renewals) == 1)
    third = _check(enforcer)
    enforcer.close()

    assert first.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert second.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert third.lease_id == "lse_fake2"  # type: ignore[attr-defined]
    assert [grant.agent_run_id for grant in plane.lease_grants] == [
        "run-lease",
        "run-lease",
    ]
    assert plane.lease_renewals[0].generation == 1


@pytest.mark.unit
def test_slow_request_freezes_lease_refusal_before_context_exit() -> None:
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
                plane.slow(1, path="/api/v1/budgets/lease", requests=1),
                plane.refuse_leases(requests=1),
            ):
                pending = pool.submit(_post_grant, client, plane)
                assert sleep_entered.wait(timeout=1)
        finally:
            release_sleep.set()
        refused = pending.result(timeout=1)
        recovered = _post_grant(client, plane, run_id="run-recovered")

    assert refused.status_code == 503
    assert refused.json()["detail"]["code"] == "lease_unavailable"
    assert recovered.status_code == 200


@pytest.mark.unit
def test_reset_recording_keeps_lease_state_and_planes_are_independent() -> None:
    first_plane = FakeControlPlane()
    second_plane = FakeControlPlane()
    with (
        httpx.Client(transport=first_plane.transport) as first_client,
        httpx.Client(transport=second_plane.transport) as second_client,
    ):
        first_grant = _post_grant(first_client, first_plane).json()
        second_grant = _post_grant(second_client, second_plane).json()
        first_plane.reset_recording()
        renewed = first_client.post(
            f"{first_plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=first_grant["lease_id"],
                holder_id="holder-1",
                generation=first_grant["generation"],
            ).model_dump(mode="json"),
        )

    assert first_grant["lease_id"] == "lse_fake1"
    assert second_grant["lease_id"] == "lse_fake1"
    assert renewed.status_code == 200
    assert renewed.json()["generation"] == 2
    assert first_plane.lease_grants == []
    assert len(first_plane.lease_renewals) == 1
    assert len(second_plane.lease_grants) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/budgets/lease", {**_grant_payload(), "private_prompt": "SECRET"}),
        (
            "/api/v1/budgets/lease/renew",
            {
                "lease_id": "lse_fake1",
                "holder_id": "holder-1",
                "generation": 1,
                "private_prompt": "SECRET",
            },
        ),
        (
            "/api/v1/budgets/lease/surrender",
            {
                "lease_id": "lse_fake1",
                "holder_id": "holder-1",
                "generation": 1,
                "private_prompt": "SECRET",
            },
        ),
    ],
)
def test_lease_validation_is_422_without_private_input(path: str, payload: object) -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        response = client.post(f"{plane.api_url}{path}", json=payload)

    assert response.status_code == 422
    assert "SECRET" not in response.text
    assert plane.lease_grants == []
    assert plane.lease_renewals == []
    assert plane.lease_surrenders == []


@pytest.mark.unit
async def test_async_transport_parity_for_lease_lifecycle_and_refusal() -> None:
    plane = FakeControlPlane()
    grant_request = _grant_payload()

    async with httpx.AsyncClient(transport=plane.transport) as client:
        grant_response = await client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=grant_request,
        )
        grant = grant_response.json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
        )
        renewed_response = await client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        renewed = renewed_response.json()
        surrender = LeaseSurrenderRequest(
            lease_id=renewed["lease_id"],
            holder_id="holder-1",
            generation=renewed["generation"],
        )
        surrendered = await client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=surrender.model_dump(mode="json"),
        )
        with plane.refuse_leases(requests=1):
            refused = await client.post(
                f"{plane.api_url}/api/v1/budgets/lease",
                json={**grant_request, "agent_run_id": "async-refused"},
            )

    assert grant_response.status_code == 200
    assert renewed_response.status_code == 200
    assert renewed["generation"] == 2
    assert surrendered.json()["released_tokens"] > 0
    assert refused.status_code == 503
    assert [type(item) for item in plane.lease_grants] == [LeaseGrantRequest]
    assert [type(item) for item in plane.lease_renewals] == [LeaseRenewRequest]
    assert [type(item) for item in plane.lease_surrenders] == [LeaseSurrenderRequest]


@pytest.mark.unit
def test_wrap_run_expires_renews_regrants_and_surrenders_on_close() -> None:
    plane = FakeControlPlane(
        granted_tokens=1_000,
        refresh_interval_s=0.001,
        lease_length_s=10.0,
    )
    wrapped = plane.wrap(_SyncOpenAIStub(), lease_output_bound_default=20)

    with solwyn_pkg.run("sync-lease"):
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        plane.expire_leases()
        time.sleep(0.01)
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        assert _wait_for(lambda: len(plane.lease_renewals) >= 1)
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    wrapped.close()

    assert [request.model for request in plane.lease_grants] == ["gpt-5.5", "gpt-5.5"]
    assert plane.lease_renewals[0].lease_id == "lse_fake1"
    assert plane.lease_surrenders[-1].lease_id == "lse_fake2"


@pytest.mark.unit
async def test_async_enforcer_and_wrap_cover_refusal_expiration_and_close() -> None:
    refusal_plane = FakeControlPlane()
    enforcer = _make_async_enforcer(refusal_plane)
    with refusal_plane.refuse_leases(requests=1):
        refused = await _acheck(enforcer)
    await enforcer.close()

    assert refused.allowed is True  # type: ignore[attr-defined]
    assert refused.lease_id is None  # type: ignore[attr-defined]
    assert len(refusal_plane.checks) == 1

    plane = FakeControlPlane(
        granted_tokens=1_000,
        refresh_interval_s=0.001,
        lease_length_s=10.0,
    )
    wrapped = plane.wrap_async(_AsyncOpenAIStub(), lease_output_bound_default=20)
    async with solwyn_pkg.run("async-lease"):
        await wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        plane.expire_leases()
        await asyncio.sleep(0.01)
        await wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        assert await _await_for(lambda: len(plane.lease_renewals) >= 1)
        await wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    await wrapped.close()

    assert [request.model for request in plane.lease_grants] == ["gpt-5.5", "gpt-5.5"]
    assert plane.lease_renewals[0].lease_id == "lse_fake1"
    assert plane.lease_surrenders[-1].lease_id == "lse_fake2"


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/budgets/lease",
        "/api/v1/budgets/lease/renew",
        "/api/v1/budgets/lease/surrender",
    ],
)
def test_lease_paths_never_retain_the_task_two_501_placeholder(path: str) -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        response = client.get(f"{plane.api_url}{path}")

    assert response.status_code == 404
