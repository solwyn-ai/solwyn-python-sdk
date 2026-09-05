"""Lease wire surface for ``FakeControlPlane``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

import solwyn as solwyn_pkg
from solwyn._token_details import TokenDetails
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
    provider: ProviderName = ProviderName.OPENAI,
    fallback_providers: list[ProviderName] | None = None,
    fallback_models: list[str] | None = None,
    fail_open: bool = True,
) -> dict[str, object]:
    return LeaseGrantRequest(
        agent_run_id=run_id,
        holder_id=holder_id,
        model=model,
        provider=provider,
        fallback_providers=[] if fallback_providers is None else fallback_providers,
        fallback_models=[] if fallback_models is None else fallback_models,
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
def test_plane_rejects_negative_lease_grants() -> None:
    with pytest.raises(ValueError, match="granted_tokens must not be negative"):
        FakeControlPlane(granted_tokens=-1)


@pytest.mark.unit
def test_plane_rejects_a_negative_headroom_share() -> None:
    with pytest.raises(ValueError, match="headroom_share_tokens must not be negative"):
        FakeControlPlane(headroom_share_tokens=-1)


@pytest.mark.unit
@pytest.mark.parametrize("lease_length_s", [0.0, -1.0])
def test_plane_rejects_non_positive_lease_lengths(lease_length_s: float) -> None:
    with pytest.raises(ValueError, match="lease_length_s must be greater than zero"):
        FakeControlPlane(lease_length_s=lease_length_s)


@pytest.mark.unit
@pytest.mark.parametrize("refresh_interval_s", [0.0, -1.0])
def test_plane_rejects_non_positive_lease_refresh_intervals(
    refresh_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="refresh_interval_s must be greater than zero"):
        FakeControlPlane(refresh_interval_s=refresh_interval_s)


@pytest.mark.unit
@pytest.mark.parametrize("refresh_interval_s", [10.0, 11.0])
def test_plane_requires_refresh_interval_below_lease_length(
    refresh_interval_s: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="refresh_interval_s must be less than lease_length_s",
    ):
        FakeControlPlane(refresh_interval_s=refresh_interval_s, lease_length_s=10.0)


@pytest.mark.unit
def test_plane_accepts_positive_lease_configuration_boundaries() -> None:
    plane = FakeControlPlane(
        granted_tokens=1,
        refresh_interval_s=0.5,
        lease_length_s=1.0,
    )

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane)

    assert response.json()["granted_tokens"] == 1
    assert response.json()["refresh_interval_s"] == 0.5
    assert response.json()["lease_length_s"] == 1.0


@pytest.mark.unit
def test_zero_token_grant_keeps_the_full_wire_shape_for_an_alert_only_cap() -> None:
    plane = FakeControlPlane(granted_tokens=0)

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane)

    payload = response.json()
    assert response.status_code == 200
    assert payload.keys() >= LEASE_BLOCK_KEYS
    assert payload["eligible"] is True
    assert payload["allowed"] is True
    assert payload["granted_tokens"] == 0
    assert payload["headroom_share_tokens"] == 0
    assert LeaseGrantResponse.model_validate(payload).granted_tokens == 0


@pytest.mark.unit
def test_zero_token_grant_routes_the_real_enforcer_to_the_per_call_check() -> None:
    plane = FakeControlPlane(granted_tokens=0)
    enforcer = _make_enforcer(plane)

    result = _check(enforcer)
    enforcer.close()

    assert result.allowed is True  # type: ignore[attr-defined]
    assert result.lease_id is None  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 1
    assert len(plane.checks) == 1


@pytest.mark.unit
def test_headroom_share_tokens_is_sized_independently_of_the_grant() -> None:
    plane = FakeControlPlane(granted_tokens=1_000, headroom_share_tokens=250)

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane)

    payload = response.json()
    assert payload["granted_tokens"] == 1_000
    assert payload["headroom_share_tokens"] == 250


@pytest.mark.unit
def test_final_grant_is_emitted_on_grant_and_renewal_for_the_ledger_wind_down() -> None:
    plane = FakeControlPlane(final_grant=True)

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=grant["generation"],
            ).model_dump(mode="json"),
        ).json()

    assert grant["final_grant"] is True
    assert renewed["generation"] == 2
    assert renewed["final_grant"] is True


@pytest.mark.unit
def test_final_grant_reaches_the_real_lease_ledger_state() -> None:
    plane = FakeControlPlane(final_grant=True)
    enforcer = _make_enforcer(plane)

    _check(enforcer)
    state = enforcer._lease.state_for("run-lease")
    enforcer.close()

    assert state is not None
    assert state.final_grant is True


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
def test_active_same_holder_grant_replays_current_lease_and_fences_declaration_changes() -> None:
    plane = FakeControlPlane()
    original = _grant_payload()
    same_declaration = {**original, "estimated_input_tokens": 999}
    changed_fail_open = {**original, "fail_open": False}
    changed_model = {**original, "model": "gpt-4o"}

    with httpx.Client(transport=plane.transport) as client:
        first = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=original)
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=same_declaration,
        )
        fail_open_conflict = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=changed_fail_open,
        )
        model_conflict = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=changed_model,
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=first.json()["lease_id"],
                holder_id="holder-1",
                generation=first.json()["generation"],
            ).model_dump(mode="json"),
        )
        current_replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=original,
        )
        surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=first.json()["lease_id"],
                holder_id="holder-1",
                generation=2,
            ).model_dump(mode="json"),
        )
        successor = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=original)

    holder_conflict = {
        "detail": {
            "code": "lease_holder_cap_exceeded",
            "message": "Active lease holder limit exceeded",
        }
    }
    assert first.status_code == 200
    assert replay.content == first.content
    assert fail_open_conflict.status_code == 409
    assert fail_open_conflict.json() == holder_conflict
    assert model_conflict.status_code == 409
    assert model_conflict.json() == holder_conflict
    assert renewed.status_code == 200
    assert renewed.json()["lease_id"] == "lse_fake1"
    assert renewed.json()["generation"] == 2
    assert current_replay.content == renewed.content
    assert surrender.json()["released_tokens"] > 0
    assert successor.json()["lease_id"] == "lse_fake2"
    assert successor.json()["generation"] == 1


@pytest.mark.unit
def test_duplicate_grant_pairs_replay_as_the_same_ordered_declaration() -> None:
    plane = FakeControlPlane()
    duplicated = _grant_payload(
        fallback_providers=[
            ProviderName.OPENAI,
            ProviderName.ANTHROPIC,
            ProviderName.ANTHROPIC,
            ProviderName.GOOGLE,
        ],
        fallback_models=[
            "gpt-5.5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5",
            "gemini-2.5-pro",
        ],
    )
    normalized = _grant_payload(
        fallback_providers=[ProviderName.ANTHROPIC, ProviderName.GOOGLE],
        fallback_models=["claude-sonnet-4-5", "gemini-2.5-pro"],
    )
    reordered = _grant_payload(
        fallback_providers=[ProviderName.GOOGLE, ProviderName.ANTHROPIC],
        fallback_models=["gemini-2.5-pro", "claude-sonnet-4-5"],
    )

    with httpx.Client(transport=plane.transport) as client:
        first = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=duplicated)
        replay = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=normalized)
        conflict = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=reordered)

    assert replay.content == first.content
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "lease_holder_cap_exceeded"


@pytest.mark.unit
def test_duplicate_renewal_pairs_replay_as_the_same_request() -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        duplicated = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            fallback_providers=[
                ProviderName.OPENAI,
                ProviderName.ANTHROPIC,
                ProviderName.ANTHROPIC,
            ],
            fallback_models=[
                "gpt-5.5",
                "claude-sonnet-4-5",
                "claude-sonnet-4-5",
            ],
        )
        normalized = duplicated.model_copy(
            update={
                "fallback_providers": [ProviderName.ANTHROPIC],
                "fallback_models": ["claude-sonnet-4-5"],
            }
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=duplicated.model_dump(mode="json"),
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=normalized.model_dump(mode="json"),
        )

    assert renewed.status_code == 200
    assert renewed.json()["generation"] == 2
    assert replay.content == renewed.content


@pytest.mark.unit
def test_successful_renewal_widens_ordered_declaration_for_grant_replay() -> None:
    plane = FakeControlPlane()
    narrow = _grant_payload()
    widened = _grant_payload(
        fallback_providers=[ProviderName.ANTHROPIC, ProviderName.GOOGLE],
        fallback_models=["claude-sonnet-4-5", "gemini-2.5-pro"],
    )
    with httpx.Client(transport=plane.transport) as client:
        grant = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=narrow).json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model="claude-sonnet-4-5",
            provider=ProviderName.ANTHROPIC,
            fallback_providers=[ProviderName.GOOGLE],
            fallback_models=["gemini-2.5-pro"],
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        narrow_conflict = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=narrow,
        )
        widened_replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=widened,
        )

    assert renewed.json()["generation"] == 2
    assert narrow_conflict.status_code == 409
    assert narrow_conflict.json()["detail"]["code"] == "lease_holder_cap_exceeded"
    assert widened_replay.content == renewed.content


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("solwyn-test/deny", None),
        (None, ProviderName.ANTHROPIC),
    ],
)
def test_incomplete_renewal_redeclaration_is_ignored(
    model: str | None,
    provider: ProviderName | None,
) -> None:
    plane = FakeControlPlane()
    narrow = _grant_payload()
    widened = _grant_payload(
        fallback_providers=[ProviderName.GOOGLE],
        fallback_models=["gemini-2.5-pro"],
    )
    with httpx.Client(transport=plane.transport) as client:
        grant = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=narrow).json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model=model,
            provider=provider,
            fallback_providers=[ProviderName.GOOGLE],
            fallback_models=["gemini-2.5-pro"],
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        narrow_replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=narrow,
        )
        widened_conflict = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=widened,
        )

    assert renewed.json()["generation"] == 2
    assert renewed.json()["allowed"] is True
    assert narrow_replay.content == renewed.content
    assert widened_conflict.status_code == 409
    assert widened_conflict.json()["detail"]["code"] == "lease_holder_cap_exceeded"


@pytest.mark.unit
def test_renewal_verdict_covers_the_union_of_declared_and_redeclared_models() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(
            client,
            plane,
            run_id="run-union",
            model="solwyn-test/runaway",
        ).json()
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=grant["generation"],
                model="gpt-5.5",
                provider=ProviderName.OPENAI,
            ).model_dump(mode="json"),
        ).json()

    assert grant["allowed"] is True
    assert renewed["allowed"] is False
    assert renewed["denied_by_period"] == "agent_run"
    assert LEASE_BLOCK_KEYS.isdisjoint(renewed)


@pytest.mark.unit
def test_sync_enforcer_renewal_widens_plane_declaration() -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    _check(enforcer)
    renewal = enforcer._build_renewal(
        "run-lease",
        model="claude-sonnet-4-5",
        provider="anthropic",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    enforcer._renew_lease(
        "run-lease",
        renewal,
        ["gpt-5.5", "claude-sonnet-4-5"],
    )
    with httpx.Client(transport=plane.transport) as client:
        replay = _post_grant(
            client,
            plane,
            fallback_providers=[ProviderName.ANTHROPIC],
            fallback_models=["claude-sonnet-4-5"],
        )
    state = enforcer._lease.state_for("run-lease")
    enforcer.close()

    assert state is not None
    assert state.generation == 2
    assert replay.status_code == 200
    assert replay.json()["lease_id"] == "lse_fake1"
    assert replay.json()["generation"] == 2


@pytest.mark.unit
async def test_async_enforcer_renewal_widens_plane_declaration() -> None:
    plane = FakeControlPlane()
    enforcer = _make_async_enforcer(plane)
    await _acheck(enforcer)
    renewal = enforcer._build_renewal(
        "run-lease",
        model="claude-sonnet-4-5",
        provider="anthropic",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    await enforcer._renew_lease(
        "run-lease",
        renewal,
        ["gpt-5.5", "claude-sonnet-4-5"],
    )
    async with httpx.AsyncClient(transport=plane.transport) as client:
        replay = await client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(
                fallback_providers=[ProviderName.ANTHROPIC],
                fallback_models=["claude-sonnet-4-5"],
            ),
        )
    state = enforcer._lease.state_for("run-lease")
    await enforcer.close()

    assert state is not None
    assert state.generation == 2
    assert replay.status_code == 200
    assert replay.json()["lease_id"] == "lse_fake1"
    assert replay.json()["generation"] == 2


@pytest.mark.unit
def test_sync_enforcer_can_recover_a_lost_same_holder_grant_response() -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    holder_id = enforcer._lease.holder_id
    with httpx.Client(transport=plane.transport) as client:
        lost = _post_grant(client, plane, holder_id=holder_id)

    recovered = _check(enforcer)
    enforcer.close()

    assert lost.json()["lease_id"] == "lse_fake1"
    assert recovered.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 2


@pytest.mark.unit
async def test_async_enforcer_can_recover_a_lost_same_holder_grant_response() -> None:
    plane = FakeControlPlane()
    enforcer = _make_async_enforcer(plane)
    holder_id = enforcer._lease.holder_id
    async with httpx.AsyncClient(transport=plane.transport) as client:
        lost = await client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=_grant_payload(holder_id=holder_id),
        )

    recovered = await _acheck(enforcer)
    await enforcer.close()

    assert lost.json()["lease_id"] == "lse_fake1"
    assert recovered.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert len(plane.lease_grants) == 2


@pytest.mark.unit
def test_renew_replays_the_echoed_generation_and_fences_future_or_unknown_requests() -> None:
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
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal_request.model_dump(mode="json"),
        )
        grown_tally_request = renewal_request.model_copy(update={"spent_tokens": 101})
        grown_tally = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=grown_tally_request.model_dump(mode="json"),
        )
        future_request = renewal_request.model_copy(update={"generation": 99})
        future = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=future_request.model_dump(mode="json"),
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
    assert replay.status_code == 200
    assert replay.content == renewed.content
    assert grown_tally.status_code == 200
    assert grown_tally.content == renewed.content
    assert future.status_code == 409
    assert future.json() == {
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
    assert plane.lease_renewals == [
        renewal_request,
        renewal_request,
        grown_tally_request,
        future_request,
        unknown_request,
    ]


@pytest.mark.unit
def test_renewal_replay_is_keyed_by_generation_not_by_request_body() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        first_request = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            spent_tokens=10,
        )
        renewed = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=first_request.model_dump(mode="json"),
        )
        retry_request = first_request.model_copy(update={"spent_tokens": 25})
        retried = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=retry_request.model_dump(mode="json"),
        )

    assert renewed.status_code == 200
    assert renewed.json()["generation"] == 2
    assert retried.status_code == 200
    assert retried.content == renewed.content
    assert plane.lease_renewals == [first_request, retry_request]


@pytest.mark.unit
def test_renewal_echoing_a_terminal_successor_generation_gets_a_fresh_verdict() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        plane.deny_next(period="monthly", scope="lease")
        terminal = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=grant["generation"],
            ).model_dump(mode="json"),
        )
        successor = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=LeaseRenewRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=2,
            ).model_dump(mode="json"),
        )

    assert terminal.json()["allowed"] is False
    assert terminal.json()["denied_by_period"] == "monthly"
    assert successor.status_code == 200
    assert successor.json()["allowed"] is True
    assert successor.json()["generation"] == 3


@pytest.mark.unit
def test_sync_enforcer_can_apply_replayed_success_after_lost_renewal_response() -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    granted = _check(enforcer)
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")
    with httpx.Client(transport=plane.transport) as client:
        lost = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )

    enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])
    state = enforcer._lease.state_for("run-lease")
    enforcer.close()

    assert granted.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert lost.json()["generation"] == 2
    assert state is not None
    assert state.lease_id == "lse_fake1"
    assert state.generation == 2


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
        stale = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_copy(update={"generation": 99}).model_dump(mode="json"),
        )
        wrong_holder = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_copy(update={"holder_id": "holder-2"}).model_dump(mode="json"),
        )
        missing = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_copy(update={"lease_id": "lse_missing"}).model_dump(mode="json"),
        )
        successor = _post_grant(client, plane)
        replay_after_regrant = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=request.model_dump(mode="json"),
        )

    assert first.status_code == 200
    assert first.json() == {"released_tokens": 321}
    assert replay.status_code == 200
    assert replay.json() == {"released_tokens": 0}
    assert stale.status_code == 409
    assert stale.json() == {
        "detail": {
            "code": "lease_generation_conflict",
            "message": "Budget lease generation conflict",
        }
    }
    assert wrong_holder.status_code == 404
    assert wrong_holder.json() == {
        "detail": {
            "code": "lease_not_found",
            "message": "Budget lease not found",
        }
    }
    assert missing.status_code == 404
    assert missing.json() == wrong_holder.json()
    assert successor.json()["lease_id"] == "lse_fake2"
    assert replay_after_regrant.status_code == 200
    assert replay_after_regrant.json() == {"released_tokens": 0}
    assert len(plane.lease_surrenders) == 6


@pytest.mark.unit
def test_denied_grant_omits_lease_block_and_is_sticky_in_real_enforcer() -> None:
    raw_plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    raw_plane.deny_next(period="agent_run", scope="lease")
    with httpx.Client(transport=raw_plane.transport) as client:
        raw = _post_grant(client, raw_plane)

    assert raw.status_code == 200
    payload = raw.json()
    assert payload["eligible"] is True
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == "agent_run"
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)

    plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    plane.deny_next(period="agent_run", scope="lease")
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
    ("model", "period", "remaining_budget"),
    [
        ("solwyn-test/deny", "monthly", 40.0),
        ("solwyn-test/deny-alert", "monthly", 40.0),
        ("solwyn-test/deny-stopped", "run_stopped", 0.0),
    ],
)
def test_magic_lease_denials_are_hard_denies_on_an_alert_only_plane(
    model: str,
    period: str,
    remaining_budget: float,
) -> None:
    plane = FakeControlPlane(
        mode=BudgetMode.ALERT_ONLY,
        budget_limit=100.0,
        current_usage=60.0,
        remaining_budget=40.0,
    )

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane, model=model)

    payload = response.json()
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == period
    assert payload["mode"] == "hard_deny"
    assert payload["remaining_budget"] == remaining_budget
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)


@pytest.mark.unit
def test_scripted_lease_denial_never_reports_negative_remaining_budget() -> None:
    plane = FakeControlPlane(budget_limit=5.0, current_usage=6.0, remaining_budget=-1.0)
    plane.deny_next(period="monthly", scope="lease")

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(client, plane)

    payload = response.json()
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == "monthly"
    assert payload["remaining_budget"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fallback_model", "period"),
    [
        ("solwyn-test/deny", "monthly"),
        ("solwyn-test/deny-alert", "monthly"),
        ("solwyn-test/deny-stopped", "run_stopped"),
    ],
)
def test_magic_fallback_drives_raw_lease_denial(
    fallback_model: str,
    period: str,
) -> None:
    plane = FakeControlPlane(mode=BudgetMode.ALERT_ONLY)

    with httpx.Client(transport=plane.transport) as client:
        response = _post_grant(
            client,
            plane,
            fallback_providers=[ProviderName.OPENAI],
            fallback_models=[fallback_model],
        )

    assert response.json()["allowed"] is False
    assert response.json()["denied_by_period"] == period
    assert response.json()["mode"] == "hard_deny"
    assert LEASE_BLOCK_KEYS.isdisjoint(response.json())


@pytest.mark.unit
def test_tag_scoped_trigger_makes_a_lease_grant_ineligible_rather_than_denied() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        primary = _post_grant(client, plane, model="solwyn-test/deny-tag")
        fallback = _post_grant(
            client,
            plane,
            run_id="run-tag-fallback",
            holder_id="holder-tag-fallback",
            fallback_providers=[ProviderName.OPENAI],
            fallback_models=["solwyn-test/deny-tag"],
        )

    for payload in (primary.json(), fallback.json()):
        assert payload["eligible"] is False
        assert payload["ineligible_reason"] == "scoped_rules_present"
        assert payload["allowed"] is True
        assert "denied_by_period" not in payload
        assert LEASE_BLOCK_KEYS.isdisjoint(payload)


@pytest.mark.unit
def test_lease_scoped_denials_cannot_be_scripted_for_the_tag_period() -> None:
    plane = FakeControlPlane()

    with pytest.raises(ValueError, match="lease denials cannot use the tag period"):
        plane.deny_next(period="tag", scope="lease")


@pytest.mark.unit
def test_check_scoped_denials_are_not_consumed_by_lease_traffic() -> None:
    plane = FakeControlPlane()
    plane.deny_next(1)

    with httpx.Client(transport=plane.transport) as client:
        granted = _post_grant(client, plane)
        checked = client.post(
            f"{plane.api_url}/api/v1/budgets/check",
            json={
                "estimated_input_tokens": 1,
                "model": "gpt-5.5",
                "provider": "openai",
                "fallback_providers": [],
                "fallback_models": [],
                "failover_directive_version": "1",
            },
        )

    assert granted.json()["allowed"] is True
    assert granted.json()["lease_id"] == "lse_fake1"
    assert checked.json()["allowed"] is False
    assert checked.json()["denied_by_period"] == "monthly"


@pytest.mark.unit
@pytest.mark.parametrize(
    "fallback_model",
    ["solwyn-test/lease-ineligible", "no-such-model-for-leases"],
)
def test_fallback_ineligibility_omits_raw_lease_block_and_real_enforcer_uses_legacy(
    fallback_model: str,
) -> None:
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)

    result = enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=20,
        model="gpt-5.5",
        provider="openai",
        fallback_providers=["openai"],
        fallback_models=[fallback_model],
        agent_run_id="run-fallback-ineligible",
        call_id=str(uuid4()),
    )
    enforcer.close()

    assert result.allowed is True
    assert result.lease_id is None
    assert len(plane.lease_grants) == 1
    assert len(plane.checks) == 1

    with httpx.Client(transport=plane.transport) as client:
        raw = _post_grant(
            client,
            plane,
            run_id=f"raw-{fallback_model}",
            holder_id=f"raw-{fallback_model}",
            fallback_providers=[ProviderName.OPENAI],
            fallback_models=[fallback_model],
        )

    assert raw.json()["eligible"] is False
    assert raw.json()["allowed"] is True
    assert raw.json()["ineligible_reason"] == "zero_rate_model"
    assert LEASE_BLOCK_KEYS.isdisjoint(raw.json())


@pytest.mark.unit
def test_renewal_fallback_tag_trigger_is_terminal_ineligibility() -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            fallback_providers=[ProviderName.OPENAI],
            fallback_models=["solwyn-test/deny-tag"],
        )
        response = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )

    payload = response.json()
    assert payload["eligible"] is False
    assert payload["ineligible_reason"] == "scoped_rules_present"
    assert payload["allowed"] is True
    assert "denied_by_period" not in payload
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("grant_model", "renew_model", "period"),
    [
        ("gpt-5.5", "solwyn-test/deny", "monthly"),
        ("gpt-5.5", "solwyn-test/deny-alert", "monthly"),
        ("gpt-5.5", "solwyn-test/deny-stopped", "run_stopped"),
        ("solwyn-test/runaway", None, "agent_run"),
    ],
)
def test_renewal_rechecks_magic_denials_and_replays_terminal_response(
    grant_model: str,
    renew_model: str | None,
    period: str,
) -> None:
    plane = FakeControlPlane(mode=BudgetMode.ALERT_ONLY)
    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane, model=grant_model).json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model=renew_model,
            provider=ProviderName.OPENAI if renew_model is not None else None,
        )
        denied = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )

    payload = denied.json()
    assert payload["eligible"] is True
    assert payload["allowed"] is False
    assert payload["denied_by_period"] == period
    assert payload["mode"] == "hard_deny"
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)
    assert replay.content == denied.content


@pytest.mark.unit
@pytest.mark.parametrize("terminal_kind", ["deny", "tag", "ineligible"])
def test_terminal_renewal_advances_generation_and_retains_fenced_surrender(
    terminal_kind: str,
) -> None:
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        if terminal_kind == "deny":
            plane.deny_run("run-lease")
        elif terminal_kind == "ineligible":
            plane.lease_eligible = False
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model="solwyn-test/deny-tag" if terminal_kind == "tag" else None,
            provider=ProviderName.OPENAI if terminal_kind == "tag" else None,
        )
        terminal = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        changed_predecessor = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_copy(update={"spent_tokens": 1}).model_dump(mode="json"),
        )
        future = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_copy(update={"generation": 99}).model_dump(mode="json"),
        )
        stale_surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=1,
            ).model_dump(mode="json"),
        )
        current_surrender_request = LeaseSurrenderRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=2,
        )
        current_surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=current_surrender_request.model_dump(mode="json"),
        )
        surrender_replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=current_surrender_request.model_dump(mode="json"),
        )
        plane.clear_denials()
        plane.lease_eligible = True
        successor = _post_grant(client, plane)

    payload = terminal.json()
    if terminal_kind == "deny":
        assert payload["eligible"] is True
        assert payload["allowed"] is False
        assert payload["denied_by_period"] == "agent_run"
    else:
        assert payload["eligible"] is False
        assert payload["allowed"] is True
        assert payload["ineligible_reason"] == (
            "scoped_rules_present" if terminal_kind == "tag" else "zero_rate_model"
        )
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)
    assert replay.content == terminal.content
    generation_conflict = {
        "detail": {
            "code": "lease_generation_conflict",
            "message": "Budget lease generation conflict",
        }
    }
    assert changed_predecessor.status_code == 200
    assert changed_predecessor.content == terminal.content
    assert future.status_code == 409
    assert future.json() == generation_conflict
    assert stale_surrender.status_code == 409
    assert stale_surrender.json() == generation_conflict
    assert current_surrender.json()["released_tokens"] > 0
    assert surrender_replay.json() == {"released_tokens": 0}
    assert successor.json()["lease_id"] == "lse_fake2"


@pytest.mark.unit
def test_terminal_renewal_blocks_regrant_until_expiration() -> None:
    plane = FakeControlPlane()
    original = _grant_payload()
    changed = _grant_payload(model="gpt-4o")
    with httpx.Client(transport=plane.transport) as client:
        grant = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=original).json()
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
        )
        plane.deny_next(period="agent_run", scope="lease")
        terminal = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        identical_grant = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=original,
        )
        changed_grant = client.post(
            f"{plane.api_url}/api/v1/budgets/lease",
            json=changed,
        )
        plane.expire_leases()
        successor = client.post(f"{plane.api_url}/api/v1/budgets/lease", json=original)
        expired_stale_surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=1,
            ).model_dump(mode="json"),
        )
        expired_current_surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=grant["lease_id"],
                holder_id="holder-1",
                generation=2,
            ).model_dump(mode="json"),
        )

    assert terminal.json()["allowed"] is False
    assert identical_grant.status_code == 409
    assert identical_grant.json()["detail"]["code"] == "lease_generation_conflict"
    assert changed_grant.status_code == 409
    assert changed_grant.json()["detail"]["code"] == "lease_holder_cap_exceeded"
    assert successor.json()["lease_id"] == "lse_fake2"
    assert successor.json()["generation"] == 1
    assert expired_stale_surrender.status_code == 409
    assert expired_stale_surrender.json()["detail"]["code"] == "lease_generation_conflict"
    assert expired_current_surrender.json() == {"released_tokens": 0}


@pytest.mark.unit
def test_sync_enforcer_observes_mid_run_renewal_deny_as_sticky() -> None:
    plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    enforcer = _make_enforcer(plane)
    granted = _check(enforcer)
    plane.deny_run("run-lease")
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])
    with plane.outage():
        sticky = _check(enforcer)
    enforcer.close()

    assert granted.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert sticky.allowed is False  # type: ignore[attr-defined]
    assert sticky.denied_by_period == "agent_run"  # type: ignore[attr-defined]
    assert plane.lease_renewals == [renewal]


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
    ("trigger", "reason"),
    [
        ("plane", "zero_rate_model"),
        ("magic", "zero_rate_model"),
        ("tag", "scoped_rules_present"),
    ],
)
def test_renewal_rechecks_ineligibility_and_replays_terminal_response(
    trigger: str,
    reason: str,
) -> None:
    magic_models = {
        "magic": "solwyn-test/lease-ineligible",
        "tag": "solwyn-test/deny-tag",
    }
    renewal_model = magic_models.get(trigger)
    plane = FakeControlPlane()
    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        if trigger == "plane":
            plane.lease_eligible = False
        renewal = LeaseRenewRequest(
            lease_id=grant["lease_id"],
            holder_id="holder-1",
            generation=grant["generation"],
            model=renewal_model,
            provider=ProviderName.OPENAI if renewal_model is not None else None,
        )
        ineligible = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        replay = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/renew",
            json=renewal.model_dump(mode="json"),
        )
        plane.lease_eligible = True
        regrant = _post_grant(client, plane)

    payload = ineligible.json()
    assert payload["eligible"] is False
    assert payload["ineligible_reason"] == reason
    assert payload["allowed"] is True
    assert LEASE_BLOCK_KEYS.isdisjoint(payload)
    assert replay.content == ineligible.content
    assert regrant.status_code == 409
    assert regrant.json()["detail"]["code"] == "lease_generation_conflict"


@pytest.mark.unit
@pytest.mark.parametrize("trigger", ["plane", "magic"])
async def test_async_enforcer_observes_mid_run_renewal_ineligibility(trigger: str) -> None:
    plane = FakeControlPlane()
    enforcer = _make_async_enforcer(plane)
    granted = await _acheck(enforcer)
    if trigger == "plane":
        plane.lease_eligible = False
    renewal_model = "solwyn-test/lease-ineligible" if trigger == "magic" else "gpt-5.5"
    renewal = enforcer._build_renewal(
        "run-lease",
        model=renewal_model,
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    await enforcer._renew_lease("run-lease", renewal, [renewal_model])
    plane.lease_eligible = True
    legacy = await _acheck(enforcer)
    await enforcer.close()

    assert granted.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert legacy.allowed is True  # type: ignore[attr-defined]
    assert legacy.lease_id is None  # type: ignore[attr-defined]
    assert len(plane.checks) == 1


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
def test_holder_cap_refusal_matches_grants_only_and_spends_no_other_budget() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        with plane.refuse_leases(status=409, code="lease_holder_cap_exceeded", requests=1):
            renewed = client.post(
                f"{plane.api_url}/api/v1/budgets/lease/renew",
                json=LeaseRenewRequest(
                    lease_id=grant["lease_id"],
                    holder_id="holder-1",
                    generation=grant["generation"],
                ).model_dump(mode="json"),
            )
            surrendered = client.post(
                f"{plane.api_url}/api/v1/budgets/lease/surrender",
                json=LeaseSurrenderRequest(
                    lease_id=grant["lease_id"],
                    holder_id="holder-1",
                    generation=renewed.json()["generation"],
                ).model_dump(mode="json"),
            )
            refused = _post_grant(client, plane, run_id="run-capped")
            recovered = _post_grant(client, plane, run_id="run-after-cap")

    assert renewed.status_code == 200
    assert surrendered.status_code == 200
    assert surrendered.json()["released_tokens"] > 0
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "lease_holder_cap_exceeded"
    assert recovered.status_code == 200


@pytest.mark.unit
def test_lease_unavailable_refusal_still_matches_every_lease_path() -> None:
    plane = FakeControlPlane()

    with httpx.Client(transport=plane.transport) as client:
        grant = _post_grant(client, plane).json()
        with plane.refuse_leases(requests=3):
            refused_renew = client.post(
                f"{plane.api_url}/api/v1/budgets/lease/renew",
                json=LeaseRenewRequest(
                    lease_id=grant["lease_id"],
                    holder_id="holder-1",
                    generation=grant["generation"],
                ).model_dump(mode="json"),
            )
            refused_surrender = client.post(
                f"{plane.api_url}/api/v1/budgets/lease/surrender",
                json=LeaseSurrenderRequest(
                    lease_id=grant["lease_id"],
                    holder_id="holder-1",
                    generation=grant["generation"],
                ).model_dump(mode="json"),
            )
            refused_grant = _post_grant(client, plane, run_id="run-unavailable")

    statuses = [
        refused_renew.status_code,
        refused_surrender.status_code,
        refused_grant.status_code,
    ]
    assert statuses == [503, 503, 503]
    assert plane.lease_renewals == []
    assert plane.lease_surrenders == []


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
        expired_surrender_conflict = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=first["lease_id"],
                holder_id="holder-1",
                generation=2,
            ).model_dump(mode="json"),
        )
        expired_surrender = client.post(
            f"{plane.api_url}/api/v1/budgets/lease/surrender",
            json=LeaseSurrenderRequest(
                lease_id=first["lease_id"],
                holder_id="holder-1",
                generation=1,
            ).model_dump(mode="json"),
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
    assert expired_surrender_conflict.status_code == 409
    assert expired_surrender_conflict.json() == {
        "detail": {
            "code": "lease_generation_conflict",
            "message": "Budget lease generation conflict",
        }
    }
    assert expired_surrender.status_code == 200
    assert expired_surrender.json() == {"released_tokens": 0}
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
@pytest.mark.parametrize("drop", ["ineligible", "denied", "run_stopped"])
def test_a_terminal_renewal_surrenders_the_dropped_lease_exactly_once(drop: str) -> None:
    """S1: every renewal answer that ends the lease hands it straight back.

    Today the plane answers 409 — it advanced the generation when it stored
    the terminal successor — and that is fine: the release is a courtesy, and
    the fence is the server's decision. What must hold either way is that
    exactly ONE release is sent, that it names the lease and generation the
    holder actually held, and that close() does not send a second one.
    """
    plane = FakeControlPlane(mode=BudgetMode.HARD_DENY)
    enforcer = _make_enforcer(plane)
    granted = _check(enforcer)
    if drop == "ineligible":
        plane.lease_eligible = False
    elif drop == "denied":
        plane.deny_next(scope="lease", period="monthly")
    else:
        plane.stop_run("run-lease")
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])
    enforcer.close()

    assert granted.lease_id == "lse_fake1"  # type: ignore[attr-defined]
    assert enforcer._lease.lease_id_for("run-lease") is None
    assert [
        (request.lease_id, request.holder_id, request.generation)
        for request in plane.lease_surrenders
    ] == [("lse_fake1", "holder-1", 1)]


@pytest.mark.unit
def test_a_refused_release_is_logged_once_and_never_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The plane fences the release at the successor generation, exactly as the
    # deployed API does until it learns to accept a terminal predecessor.
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    _check(enforcer)
    plane.lease_eligible = False
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    with caplog.at_level("DEBUG", logger="solwyn.budget"):
        enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])
        enforcer.close()

    assert len(plane.lease_surrenders) == 1
    failures = [r for r in caplog.records if r.getMessage().startswith("lease.surrender_failed")]
    assert len(failures) == 1
    assert failures[0].levelname == "DEBUG"


@pytest.mark.unit
def test_a_trailing_confirm_still_settles_on_the_dropped_lease_id() -> None:
    # The call was admitted under lse_fake1; its confirm must keep naming it
    # after the release, so the server settles it against the row it funded
    # (as excess once the float is gone) instead of charging it in full.
    plane = FakeControlPlane()
    enforcer = _make_enforcer(plane)
    call_id = str(uuid4())
    admitted = enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=20,
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
        agent_run_id="run-lease",
        call_id=call_id,
    )
    plane.lease_eligible = False
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")
    enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])

    confirm = enforcer.build_confirm_request(
        model="gpt-5.5",
        token_details=TokenDetails(input_tokens=8, output_tokens=4),
        provider="openai",
        call_id=call_id,
        lease_id=admitted.lease_id,  # type: ignore[attr-defined]
        lease_claim_token=admitted.lease_claim_token,  # type: ignore[attr-defined]
    )
    enforcer.close()

    assert confirm.lease_id == "lse_fake1"
    assert len(plane.lease_surrenders) == 1


@pytest.mark.unit
async def test_async_terminal_renewal_surrenders_the_dropped_lease_once() -> None:
    plane = FakeControlPlane()
    enforcer = _make_async_enforcer(plane)
    await _acheck(enforcer)
    plane.lease_eligible = False
    renewal = enforcer._build_renewal(
        "run-lease",
        model="gpt-5.5",
        provider="openai",
        fallback_providers=[],
        fallback_models=[],
    )
    if renewal is None:
        pytest.fail("held lease did not produce a renewal request")

    await enforcer._renew_lease("run-lease", renewal, ["gpt-5.5"])
    await enforcer.close()

    assert [(request.lease_id, request.generation) for request in plane.lease_surrenders] == [
        ("lse_fake1", 1)
    ]


@pytest.mark.unit
def test_run_scope_exit_surrenders_only_the_run_that_ended() -> None:
    """S2: when the block ends, the budget it held is back — and no more.

    A nested scope's exit releases the nested run's lease and leaves the
    outer one drawing; the outer exit releases its own; and close() then has
    nothing left to release, so no lease is ever handed back twice.
    """
    plane = FakeControlPlane(granted_tokens=1_000)
    wrapped = plane.wrap(_SyncOpenAIStub(), lease_output_bound_default=20)

    with solwyn_pkg.run("outer") as outer_id:
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        with solwyn_pkg.run("inner") as inner_id:
            wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        assert _wait_for(lambda: len(plane.lease_surrenders) == 1)
        inner_surrender = plane.lease_surrenders[0]
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    assert _wait_for(lambda: len(plane.lease_surrenders) == 2)
    wrapped.close()

    leases = {
        grant.agent_run_id: f"lse_fake{index + 1}" for index, grant in enumerate(plane.lease_grants)
    }
    assert leases == {outer_id: "lse_fake1", inner_id: "lse_fake2"}
    assert inner_surrender.lease_id == "lse_fake2"
    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake2", "lse_fake1"]
    # The outer run kept spending on its own lease after the inner one ended.
    assert len(plane.lease_grants) == 2


@pytest.mark.unit
def test_a_run_scope_leaving_on_an_exception_still_surrenders() -> None:
    plane = FakeControlPlane(granted_tokens=1_000)
    wrapped = plane.wrap(_SyncOpenAIStub(), lease_output_bound_default=20)

    with pytest.raises(RuntimeError, match="boom"), solwyn_pkg.run("failing"):
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        raise RuntimeError("boom")
    assert _wait_for(lambda: len(plane.lease_surrenders) == 1)
    wrapped.close()

    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake1"]


@pytest.mark.unit
def test_run_handle_finish_surrenders_but_an_activation_exit_does_not() -> None:
    # A detached identity is re-entered by design: releasing at every
    # activation would buy one blocking re-grant per binding and nothing else.
    plane = FakeControlPlane(granted_tokens=1_000)
    wrapped = plane.wrap(_SyncOpenAIStub(), lease_output_bound_default=20)

    handle = solwyn_pkg.create_run("detached")
    with handle.activate():
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    after_activation = list(plane.lease_surrenders)
    with handle.activate():
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    handle.finish()
    assert _wait_for(lambda: len(plane.lease_surrenders) == 1)

    started = solwyn_pkg.start_run("started")
    wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    started.finish()
    assert _wait_for(lambda: len(plane.lease_surrenders) == 2)
    wrapped.close()

    assert after_activation == []
    # One grant for the detached identity (re-entered, never released between
    # activations) and one for the started run.
    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake1", "lse_fake2"]


@pytest.mark.unit
def test_a_call_that_outlives_its_run_scope_pays_one_grant() -> None:
    # Documented behaviour: work created inside a scope keeps the run id after
    # the block exits. Its next call finds no lease and pays one blocking
    # grant — which the fence holds behind the release of the old lease.
    plane = FakeControlPlane(granted_tokens=1_000)
    wrapped = plane.wrap(_SyncOpenAIStub(), lease_output_bound_default=20)

    with solwyn_pkg.run("outliving") as run_id:
        wrapped.chat.completions.create(model="gpt-5.5", messages=[])
        kept = copy_context()
    assert _wait_for(lambda: len(plane.lease_surrenders) == 1)

    kept.run(lambda: wrapped.chat.completions.create(model="gpt-5.5", messages=[]))
    wrapped.close()

    assert [grant.agent_run_id for grant in plane.lease_grants] == [run_id, run_id]
    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake1", "lse_fake2"]


@pytest.mark.unit
async def test_async_run_scope_exit_surrenders_the_runs_lease() -> None:
    plane = FakeControlPlane(granted_tokens=1_000)
    wrapped = plane.wrap_async(_AsyncOpenAIStub(), lease_output_bound_default=20)

    async with solwyn_pkg.run("async-scope"):
        await wrapped.chat.completions.create(model="gpt-5.5", messages=[])
    assert await _await_for(lambda: len(plane.lease_surrenders) == 1)
    await wrapped.close()

    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake1"]


@pytest.mark.unit
async def test_an_async_holder_parks_a_release_it_cannot_schedule() -> None:
    # A run scope can end on a plain thread with no loop to schedule on (an
    # async client driven from synchronous code). The release must not be
    # silently dropped there: it stays parked and rides close().
    plane = FakeControlPlane(granted_tokens=1_000)
    enforcer = _make_async_enforcer(plane)
    await _acheck(enforcer)

    off_loop = Thread(target=enforcer.surrender_run, args=("run-lease",))
    off_loop.start()
    off_loop.join(timeout=5.0)
    parked = list(enforcer._releases_owed)
    await enforcer.close()

    assert [(run_id, request.lease_id) for run_id, request in parked] == [
        ("run-lease", "lse_fake1")
    ]
    assert [request.lease_id for request in plane.lease_surrenders] == ["lse_fake1"]


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
