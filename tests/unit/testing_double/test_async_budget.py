"""Async SDK budget and settlement behavior on ``FakeControlPlane``."""

from __future__ import annotations

import uuid

import pytest

from solwyn._token_details import TokenDetails
from solwyn.budget import AsyncBudgetEnforcer
from solwyn.reporter import AsyncMetadataReporter
from solwyn.testing import FakeControlPlane

_SAMPLE_TOKEN_DETAILS = TokenDetails(input_tokens=100, output_tokens=50)


def _components(plane: FakeControlPlane) -> tuple[AsyncBudgetEnforcer, AsyncMetadataReporter]:
    enforcer = AsyncBudgetEnforcer(
        plane.api_url,
        plane.api_key,
        cache_ttl=0,
        lease_enabled=False,
        transport=plane.transport,
    )
    reporter = AsyncMetadataReporter(
        plane.api_url,
        plane.api_key,
        flush_interval=60,
        max_send_attempts=1,
        transport=plane.transport,
    )
    return enforcer, reporter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_check_returns_allowed_with_complete_budget_metadata() -> None:
    plane = FakeControlPlane(
        budget_limit=100.0,
        current_usage=12.5,
        remaining_budget=87.5,
    )
    enforcer, reporter = _components(plane)

    result = await enforcer.check_budget(
        estimated_input_tokens=100,
        model="gpt-5.5",
        provider="openai",
    )
    await reporter.close()
    await enforcer.close()

    assert result.allowed is True
    assert result.reservation_id == "res_fake_00000001"
    assert result.remaining_budget == 87.5
    assert result.budget_limit == 100.0
    assert result.current_usage == 12.5
    assert len(plane.checks) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_valid_reservation_records_and_idempotently_replays_confirm() -> None:
    plane = FakeControlPlane()
    enforcer, reporter = _components(plane)
    result = await enforcer.check_budget(
        estimated_input_tokens=100,
        model="gpt-5.5",
        provider="openai",
    )
    assert result.reservation_id is not None
    confirm = enforcer.build_confirm_request(
        reservation_id=result.reservation_id,
        model="gpt-5.5",
        token_details=_SAMPLE_TOKEN_DETAILS,
        provider="openai",
        call_id=str(uuid.uuid4()),
    )

    await reporter._send_confirm(confirm)
    await reporter._send_confirm(confirm)
    await reporter.close()
    await enforcer.close()

    assert confirm.lease_id is None
    assert plane.confirms == [confirm]
    assert reporter.dropped_counts == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_invalid_reservation_is_best_effort_and_not_recorded() -> None:
    plane = FakeControlPlane()
    enforcer, reporter = _components(plane)
    confirm = enforcer.build_confirm_request(
        reservation_id="res_nonexistent_000",
        model="gpt-5.5",
        token_details=_SAMPLE_TOKEN_DETAILS,
        provider="openai",
        call_id=str(uuid.uuid4()),
    )

    await reporter._send_confirm(confirm)
    await reporter.close()
    await enforcer.close()

    assert plane.confirms == []
    assert reporter._consecutive_confirm_failures == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_hard_denial_and_fail_open_outage_preserve_sdk_postures() -> None:
    plane = FakeControlPlane()
    plane.deny_next(period="monthly")
    enforcer, reporter = _components(plane)

    denied = await enforcer.check_budget(
        estimated_input_tokens=100,
        model="gpt-5.5",
        provider="openai",
    )
    with plane.outage(requests=1):
        unavailable = await enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-5.5",
            provider="openai",
        )
    await reporter.close()
    await enforcer.close()

    assert denied.allowed is False
    assert denied.denied_by_period == "monthly"
    assert unavailable.allowed is False
    assert unavailable.denied_by_period == "monthly"
    assert unavailable.warning is not None
    assert "preserving prior hard deny" in unavailable.warning
