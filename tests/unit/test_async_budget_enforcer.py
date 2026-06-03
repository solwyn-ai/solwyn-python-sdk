"""Tests for AsyncBudgetEnforcer — async mirror of test_budget.py sync tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer

_DENY_RESPONSE = {
    "allowed": False,
    "remaining_budget": 0.5,
    "reservation_id": None,
    "mode": "hard_deny",
    "budget_limit": 100.0,
    "current_usage": 99.5,
    "denied_by_period": "monthly",
    "project_id": VALID_PROJECT_ID,
}


def _make_async_enforcer(**overrides) -> AsyncBudgetEnforcer:
    """Create an AsyncBudgetEnforcer with sensible test defaults."""
    defaults = {
        "api_url": "https://api.test.solwyn.ai",
        "api_key": VALID_API_KEY,
        "budget_mode": BudgetMode.ALERT_ONLY,
        "fail_open": True,
        "cache_ttl": 5,
    }
    defaults.update(overrides)
    return AsyncBudgetEnforcer(**defaults)


# ---------------------------------------------------------------------------
# Cloud allow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncCloudAllow:
    """Cloud reachable and allows the request (async)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_allowed(self) -> None:
        enforcer = _make_async_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(return_value=mock_response)
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert result.allowed is True
        assert result.remaining_budget == 80.0
        assert result.reservation_id == "res_123"
        assert result.warning is None
        await enforcer.close()


# ---------------------------------------------------------------------------
# Cloud deny: hard_deny
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncCloudDenyHard:
    """Cloud reachable and denies the request in hard_deny mode (async)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_not_allowed(self) -> None:
        enforcer = _make_async_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(return_value=mock_response)
        result = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )

        assert result.allowed is False
        assert result.warning is not None
        assert "exceeded" in result.warning.lower()
        await enforcer.close()


# ---------------------------------------------------------------------------
# Cloud deny: alert_only proceeds with warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncCloudDenyAlertOnly:
    """Cloud denies but alert_only mode lets the request through (async)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allowed_with_warning(self) -> None:
        enforcer = _make_async_enforcer(budget_mode=BudgetMode.ALERT_ONLY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(return_value=mock_response)
        result = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )

        assert result.allowed is True
        assert result.warning is not None
        assert "limit" in result.warning.lower()
        await enforcer.close()


# ---------------------------------------------------------------------------
# Fail-open when cloud is unreachable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncFailOpen:
    """Cloud unreachable with fail_open=True proceeds with warning (async)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allowed_with_warning(self) -> None:
        enforcer = _make_async_enforcer(fail_open=True)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()
        await enforcer.close()


# ---------------------------------------------------------------------------
# confirm_cost
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncConfirmCost:
    """confirm_cost() sends POST to cloud API (async)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sends_confirmation(self) -> None:
        enforcer = _make_async_enforcer()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)

        enforcer._http.post = AsyncMock(return_value=mock_response)
        await enforcer.confirm_cost("res_123", "gpt-4o", token_details, provider="openai")

        enforcer._http.post.assert_called_once()
        call_args = enforcer._http.post.call_args
        assert "budgets/confirm" in call_args[0][0]
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swallows_errors(self) -> None:
        enforcer = _make_async_enforcer()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            # Should not raise
            await enforcer.confirm_cost("res_123", "gpt-4o", token_details, provider="openai")

        await enforcer.close()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncClose:
    """close() calls aclose on the httpx.AsyncClient."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_calls_aclose(self) -> None:
        enforcer = _make_async_enforcer()
        with patch.object(enforcer._http, "aclose", new_callable=AsyncMock) as mock_aclose:
            await enforcer.close()

        mock_aclose.assert_called_once()
