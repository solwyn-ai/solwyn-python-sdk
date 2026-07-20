"""Integration tests for budget check round-trip."""

from __future__ import annotations

import pytest
from conftest import Credentials

from solwyn._types import BudgetMode
from solwyn.budget import BudgetEnforcer


@pytest.mark.integration
class TestBudgetCheckAllowed:
    """Budget check returns allowed when project is within budget."""

    @pytest.mark.integration
    def test_check_budget_returns_allowed(self, budget_enforcer: BudgetEnforcer) -> None:
        result = budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-5.5",
            provider="openai",
        )
        assert result.allowed is True
        assert result.reservation_id is not None
        assert result.remaining_budget > 0

    @pytest.mark.integration
    def test_check_budget_returns_budget_metadata(self, budget_enforcer: BudgetEnforcer) -> None:
        result = budget_enforcer.check_budget(
            estimated_input_tokens=100,
            model="gpt-5.5",
            provider="openai",
        )
        assert result.budget_limit > 0
        assert result.current_usage >= 0


@pytest.mark.integration
class TestBudgetCheckFailOpen:
    """Budget check with unreachable API and fail_open=True."""

    @pytest.mark.integration
    def test_fail_open_allows_on_bad_url(self, test_credentials: Credentials) -> None:
        enforcer = BudgetEnforcer(
            api_url="http://localhost:1",  # unreachable
            api_key=test_credentials.api_key,
            budget_mode=BudgetMode.ALERT_ONLY,
            fail_open=True,
        )
        try:
            result = enforcer.check_budget(
                estimated_input_tokens=100,
                model="gpt-5.5",
                provider="openai",
            )
            assert result.allowed is True
            assert result.warning is not None
            assert "fail-open" in result.warning.lower()
        finally:
            enforcer.close()
