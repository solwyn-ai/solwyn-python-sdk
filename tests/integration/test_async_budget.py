"""Async integration tests for budget check and confirm."""

from __future__ import annotations

import pytest
from conftest import SAMPLE_TOKEN_DETAILS

from solwyn.budget import AsyncBudgetEnforcer


@pytest.mark.integration
class TestAsyncBudgetCheck:
    """Async budget check returns allowed when within budget."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_check_budget_returns_allowed(
        self, async_budget_enforcer: AsyncBudgetEnforcer
    ) -> None:
        result = await async_budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-4o",
            provider="openai",
        )
        assert result.allowed is True
        assert result.reservation_id is not None
        assert result.remaining_budget > 0


@pytest.mark.integration
class TestAsyncBudgetConfirm:
    """Async budget check → confirm round-trip."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_confirm_with_valid_reservation(
        self, async_budget_enforcer: AsyncBudgetEnforcer
    ) -> None:
        result = await async_budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-4o",
            provider="openai",
        )
        assert result.reservation_id is not None

        # Should not raise
        await async_budget_enforcer.confirm_cost(
            reservation_id=result.reservation_id,
            model="gpt-4o",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_async_integration_confirm_valid",
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_confirm_invalid_reservation_does_not_raise(
        self, async_budget_enforcer: AsyncBudgetEnforcer
    ) -> None:
        """Async confirm_cost is best-effort — bad reservation logs, doesn't raise."""
        await async_budget_enforcer.confirm_cost(
            reservation_id="res_nonexistent_000",
            model="gpt-4o",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_async_integration_confirm_invalid",
        )
