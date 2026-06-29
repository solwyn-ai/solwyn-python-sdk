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

_ALERT_ONLY_DENY_RESPONSE = {
    **_DENY_RESPONSE,
    "mode": "alert_only",
}


def _error_response(status_code: int) -> MagicMock:
    """A 4xx/5xx httpx.Response stand-in: raise_for_status raises HTTPStatusError.

    raise_for_status is sync even on the async client, so it is a MagicMock.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(spec=httpx.Request), response=resp
        )
    )
    return resp


def _ok_response() -> MagicMock:
    """A 2xx httpx.Response stand-in: raise_for_status is a no-op."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


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
        mock_response.json.return_value = _ALERT_ONLY_DENY_RESPONSE
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_prior_hard_deny_blocks_when_cloud_later_unreachable(self) -> None:
        enforcer = _make_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[mock_response, httpx.ConnectError("unreachable")]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert denied.allowed is False
        assert result.allowed is False
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]
        assert result.current_usage == _DENY_RESPONSE["current_usage"]
        assert result.warning is not None
        assert "cloud api unreachable" in result.warning.lower()
        assert enforcer._http.post.call_count == 2
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cloud_hard_deny_overrides_local_alert_only_when_cloud_later_unreachable(
        self,
    ) -> None:
        enforcer = _make_async_enforcer(fail_open=True, budget_mode=BudgetMode.ALERT_ONLY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[mock_response, httpx.ConnectError("unreachable")]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert denied.allowed is False
        assert denied.mode == BudgetMode.HARD_DENY
        assert result.allowed is False
        assert result.mode == BudgetMode.HARD_DENY
        assert result.warning is not None
        assert "preserving prior hard deny" in result.warning.lower()
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cloud_alert_only_deny_does_not_stick_when_local_hard_deny(
        self,
    ) -> None:
        enforcer = _make_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _ALERT_ONLY_DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[mock_response, httpx.ConnectError("unreachable")]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert denied.allowed is True
        assert denied.mode == BudgetMode.ALERT_ONLY
        assert denied.warning is not None
        assert "limit" in denied.warning.lower()
        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cloud_allow_clears_prior_hard_deny_before_later_outage(self) -> None:
        enforcer = _make_async_enforcer(
            fail_open=True, budget_mode=BudgetMode.HARD_DENY, cache_ttl=0
        )
        deny_response = MagicMock()
        deny_response.json.return_value = _DENY_RESPONSE
        deny_response.raise_for_status = MagicMock()
        allow_response = MagicMock()
        allow_response.json.return_value = ALLOW_BUDGET_RESPONSE
        allow_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[
                deny_response,
                allow_response,
                httpx.ConnectError("unreachable"),
            ]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )
        allowed = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )
        outage = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert denied.allowed is False
        assert allowed.allowed is True
        assert outage.allowed is True
        assert outage.warning is not None
        assert "fail-open" in outage.warning.lower()
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_prior_hard_deny_overrides_local_enforcement_when_cloud_unreachable(
        self,
    ) -> None:
        enforcer = _make_async_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[mock_response, httpx.ConnectError("unreachable")]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=50000, model="gpt-4o", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-4o", provider="openai"
        )

        assert denied.allowed is False
        assert result.allowed is False
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]
        assert result.current_usage == _DENY_RESPONSE["current_usage"]
        assert result.warning is not None
        assert "preserving prior hard deny" in result.warning.lower()
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
        await enforcer.confirm_cost(
            "res_123",
            "gpt-4o",
            token_details,
            provider="openai",
            call_id="call_async_budget_confirm",
        )

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
            await enforcer.confirm_cost(
                "res_123",
                "gpt-4o",
                token_details,
                provider="openai",
                call_id="call_async_budget_confirm",
            )

        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confirm_4xx_increments_failure_counter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A server 422 must surface via raise_for_status and be counted as a
        # failure, not silently treated as success.
        enforcer = _make_async_enforcer()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)
        enforcer._http.post = AsyncMock(return_value=_error_response(422))

        with caplog.at_level("WARNING"):
            await enforcer.confirm_cost(
                "res_123",
                "gpt-4o",
                token_details,
                provider="openai",
                call_id="call_async_budget_confirm",
            )

        assert enforcer._consecutive_confirm_failures == 1
        assert "budget.confirm_cost_failed" in caplog.text
        # Privacy: only the exception class name is logged, never the body.
        assert "HTTPStatusError" in caplog.text
        assert "422 Unprocessable" not in caplog.text
        await enforcer.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confirm_2xx_resets_failure_counter(self) -> None:
        # A successful confirm clears any accumulated consecutive-failure count.
        enforcer = _make_async_enforcer()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)

        enforcer._http.post = AsyncMock(return_value=_error_response(422))
        for _ in range(2):
            await enforcer.confirm_cost(
                "res_123",
                "gpt-4o",
                token_details,
                provider="openai",
                call_id="call_async_budget_confirm",
            )
        assert enforcer._consecutive_confirm_failures == 2

        enforcer._http.post = AsyncMock(return_value=_ok_response())
        await enforcer.confirm_cost(
            "res_123",
            "gpt-4o",
            token_details,
            provider="openai",
            call_id="call_async_budget_confirm",
        )
        assert enforcer._consecutive_confirm_failures == 0

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
