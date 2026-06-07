"""Integration tests for budget check → confirm round-trip."""

from __future__ import annotations

import pytest
from conftest import SAMPLE_TOKEN_DETAILS

from solwyn.budget import BudgetEnforcer


@pytest.mark.integration
class TestBudgetConfirmRoundTrip:
    """Check then confirm: the full reservation lifecycle."""

    @pytest.mark.integration
    def test_confirm_cost_with_valid_reservation(self, budget_enforcer: BudgetEnforcer) -> None:
        # Arrange — get a reservation
        result = budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-4o",
            provider="openai",
        )
        assert result.reservation_id is not None

        # Act — confirm actual usage (best-effort, should not raise)
        budget_enforcer.confirm_cost(
            reservation_id=result.reservation_id,
            model="gpt-4o",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_integration_confirm_valid",
        )

    @pytest.mark.integration
    def test_confirm_cost_invalid_reservation_does_not_raise(
        self, budget_enforcer: BudgetEnforcer
    ) -> None:
        """confirm_cost is best-effort — bad reservation_id logs, doesn't raise."""
        budget_enforcer.confirm_cost(
            reservation_id="res_nonexistent_000",
            model="gpt-4o",
            token_details=SAMPLE_TOKEN_DETAILS,
            provider="openai",
            call_id="call_integration_confirm_invalid",
        )
