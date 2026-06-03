"""Tests for budget enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID
from pydantic import BaseModel

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetCheckResponse, BudgetMode
from solwyn.budget import (
    BudgetCheckResult,
    BudgetEnforcer,
    _BudgetEnforcerBase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_enforcer(**overrides):
    """Create a BudgetEnforcer with sensible test defaults."""
    defaults = {
        "api_url": "https://api.test.solwyn.ai",
        "api_key": VALID_API_KEY,
        "budget_mode": BudgetMode.ALERT_ONLY,
        "fail_open": True,
        "cache_ttl": 5,
    }
    defaults.update(overrides)
    return BudgetEnforcer(**defaults)


# ---------------------------------------------------------------------------
# Base class (sans-I/O) tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBudgetEnforcerBase:
    """Tests for _BudgetEnforcerBase sans-I/O logic."""

    def test_build_check_request(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        req = base._build_check_request(500, "gpt-4o", "openai")
        assert not hasattr(req, "project_id")
        assert req.estimated_input_tokens == 500
        assert req.model == "gpt-4o"
        assert req.provider == "openai"

    def test_local_cost_tracking(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        base._track_local_cost(10.0)
        base._track_local_cost(5.0)
        remaining = base._get_local_remaining(100.0)
        assert remaining == pytest.approx(85.0)

    def test_cache_allow_decisions(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=5,
        )

        response = BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE)
        base._cache_response(response)
        assert base._should_use_cache() is True

    def test_never_cache_deny_decisions(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=5,
        )

        response = BudgetCheckResponse(**_DENY_RESPONSE)
        base._cache_response(response)
        # Should NOT be cached
        assert base._should_use_cache() is False

    def test_cache_expires(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=0,  # Expire immediately
        )

        response = BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE)
        base._cache_response(response)
        # Cache TTL is 0, so it should expire instantly
        assert base._should_use_cache() is False


# ---------------------------------------------------------------------------
# Cloud allow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloudAllow:
    """Cloud reachable and allows the request."""

    def test_returns_allowed(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.remaining_budget == 80.0
        assert result.reservation_id == "res_123"
        assert result.warning is None


# ---------------------------------------------------------------------------
# Cloud deny: hard_deny
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloudDenyHard:
    """Cloud reachable and denies the request in hard_deny mode."""

    def test_returns_not_allowed(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response):
            result = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )

        assert result.allowed is False
        assert result.warning is not None
        assert "exceeded" in result.warning.lower()


# ---------------------------------------------------------------------------
# Cloud deny: alert_only proceeds with warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloudDenyAlertOnly:
    """Cloud denies but alert_only mode lets the request through with a warning."""

    def test_allowed_with_warning(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.ALERT_ONLY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response):
            result = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.warning is not None
        assert "limit" in result.warning.lower()


# ---------------------------------------------------------------------------
# Fail-open when cloud is unreachable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFailOpen:
    """Cloud unreachable with fail_open=True proceeds with warning."""

    def test_allowed_with_warning(self) -> None:
        enforcer = _make_enforcer(fail_open=True)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()


# ---------------------------------------------------------------------------
# Local enforcement when cloud unreachable + hard_deny
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLocalEnforcement:
    """Cloud unreachable + fail_open=False enforces budget locally."""

    def test_denies_when_cloud_never_reached(self) -> None:
        """No prior cloud contact -> no known limit -> fail-closed (deny)."""
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )

        assert result.allowed is False
        assert result.warning is not None
        assert "no prior budget limit" in result.warning.lower()

    def test_allows_within_last_known_limit(self) -> None:
        """Cloud established limit, then goes offline -> allows within limit."""
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY, cache_ttl=0)

        # Phase 1: Cloud establishes $100 limit
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "allowed": True,
            "remaining_budget": 95.0,
            "reservation_id": "res_1",
            "mode": "hard_deny",
            "budget_limit": 100.0,
            "current_usage": 5.0,
            "denied_by_period": None,
            "project_id": VALID_PROJECT_ID,
        }
        mock_response.raise_for_status = MagicMock()
        with patch.object(enforcer._http, "post", return_value=mock_response):
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-4o", provider="openai")

        # Phase 2: Cloud goes offline
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.warning is not None
        assert "locally" in result.warning.lower()

    def test_denies_when_local_exceeds_last_known_limit(self) -> None:
        """Cloud established limit, then goes offline -> denies when exceeded."""
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY, cache_ttl=0)

        # Phase 1: Cloud establishes $100 limit
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "allowed": True,
            "remaining_budget": 100.0,
            "reservation_id": "res_1",
            "mode": "hard_deny",
            "budget_limit": 100.0,
            "current_usage": 0.0,
            "denied_by_period": None,
            "project_id": VALID_PROJECT_ID,
        }
        mock_response.raise_for_status = MagicMock()
        with patch.object(enforcer._http, "post", return_value=mock_response):
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-4o", provider="openai")

        # Fill local budget past the $100 limit (directly via _track_local_cost)
        for _ in range(10):
            enforcer._track_local_cost(10.0)  # ~101.0 total

        # Phase 2: Cloud goes offline
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is False
        assert result.warning is not None
        assert "denies" in result.warning.lower()


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheBehaviour:
    """Allow decisions are cached; deny decisions are NOT cached."""

    def test_cached_allow_avoids_http_call(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response) as mock_post:
            # First call populates cache
            result1 = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )
            assert result1.allowed is True

            # Second call should use cache
            result2 = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )
            assert result2.allowed is True

            # Only one HTTP call
            assert mock_post.call_count == 1

    def test_deny_not_cached(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response) as mock_post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-4o", provider="openai")
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-4o", provider="openai")

            # Both calls should hit HTTP (deny is never cached)
            assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# BudgetCheckResult dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBudgetCheckResult:
    """BudgetCheckResult has correct defaults and fields."""

    def test_defaults(self) -> None:
        result = BudgetCheckResult(allowed=True, remaining_budget=50.0)
        assert result.allowed is True
        assert result.remaining_budget == 50.0
        assert result.reservation_id is None
        assert result.mode == BudgetMode.ALERT_ONLY
        assert result.warning is None

    def test_all_fields(self) -> None:
        result = BudgetCheckResult(
            allowed=False,
            remaining_budget=0.0,
            reservation_id="res_456",
            mode=BudgetMode.HARD_DENY,
            warning="Budget exceeded",
        )
        assert result.allowed is False
        assert result.reservation_id == "res_456"
        assert result.mode == BudgetMode.HARD_DENY


# ---------------------------------------------------------------------------
# Confirm cost
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfirmCost:
    """confirm_cost() sends POST to cloud API."""

    def test_sends_confirmation(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)

        with patch.object(enforcer._http, "post", return_value=mock_response) as mock_post:
            enforcer.confirm_cost("res_123", "gpt-4o", token_details, provider="openai")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "budgets/confirm" in call_kwargs[0][0]

    def test_swallows_errors(self) -> None:
        enforcer = _make_enforcer()
        token_details = TokenDetails(input_tokens=100, output_tokens=50)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            # Should not raise
            enforcer.confirm_cost("res_123", "gpt-4o", token_details, provider="openai")


def test_budget_check_result_is_pydantic_model() -> None:
    """BudgetCheckResult must be a Pydantic BaseModel, not a dataclass."""
    assert issubclass(BudgetCheckResult, BaseModel)
