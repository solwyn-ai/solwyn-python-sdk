"""Spec-derived tests for budget enforcement.

These tests cover the fail-open / fail-closed behaviour matrix and atomic
reservation via the Cloud API.  Each test maps to a specific cell in the
behaviour matrix or a specific requirement.

These tests would have caught:
- Bug 1.1: BudgetExceededError constructed with wrong field values
- Bug 1.2: Local enforcement using hardcoded $100 instead of last-known limit
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import BudgetMode
from solwyn.budget import BudgetEnforcer


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


def _mock_cloud_response(
    allowed: bool = True,
    remaining: float = 80.0,
    budget_limit: float = 500.0,
    current_usage: float = 420.0,
    mode: str = "alert_only",
):
    """Create a mock httpx response for a budget check."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "allowed": allowed,
        "remaining_budget": remaining,
        "reservation_id": "res_123" if allowed else None,
        "mode": mode,
        "budget_limit": budget_limit,
        "current_usage": current_usage,
        "denied_by_period": None if allowed else "monthly",
        "project_id": VALID_PROJECT_ID,
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


# ---------------------------------------------------------------------------
# Fail-Open / Fail-Closed Matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDesignDocFailMatrix:
    """Tests for the fail-open/fail-closed matrix.

    Matrix:
    | Budget Mode | Cloud Reachable | Cloud Unreachable |
    |---|---|---|
    | alert_only  | allow with warning if exceeded | fail-open |
    | hard_deny   | raise BudgetExceededError      | local enforcement |
    """

    def test_alert_only_cloud_reachable_denied_allows_with_warning(self) -> None:
        """alert_only + cloud denies -> allowed=True with warning."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.ALERT_ONLY)
        mock_resp = _mock_cloud_response(
            allowed=False,
            budget_limit=500.0,
            current_usage=500.0,
            remaining=0.0,
            mode="alert_only",
        )

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )

        # Assert
        assert result.allowed is True
        assert result.warning is not None
        assert result.budget_limit == 500.0
        assert result.current_usage == 500.0

    def test_hard_deny_cloud_reachable_denied_returns_not_allowed(self) -> None:
        """hard_deny + cloud denies -> allowed=False."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_resp = _mock_cloud_response(
            allowed=False,
            budget_limit=500.0,
            current_usage=500.0,
            remaining=0.0,
            mode="hard_deny",
        )

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )

        # Assert
        assert result.allowed is False
        assert result.budget_limit == 500.0
        assert result.current_usage == 500.0

    def test_alert_only_cloud_unreachable_fails_open(self) -> None:
        """alert_only + cloud unreachable -> fail-open (request proceeds)."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.ALERT_ONLY, fail_open=True)

        # Act
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("offline")):
            result = enforcer.check_budget(
                estimated_input_tokens=10, model="gpt-5.5", provider="openai"
            )

        # Assert
        assert result.allowed is True
        assert result.warning is not None

    def test_hard_deny_cloud_unreachable_enforces_locally_with_last_known_limit(
        self,
    ) -> None:
        """hard_deny + cloud unreachable -> local enforcement using last-known limit.

        This is the critical scenario: cloud was reachable, established a $500
        budget limit, then goes offline.  Local enforcement should use $500,
        not a hardcoded default.
        """
        # Arrange — cache_ttl=0 so Phase 2 doesn't serve from cache
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY, fail_open=False, cache_ttl=0)

        # Phase 1: Cloud is reachable — establishes the $500 limit
        allow_resp = _mock_cloud_response(
            allowed=True,
            budget_limit=500.0,
            current_usage=200.0,
            remaining=300.0,
        )
        with patch.object(enforcer._http, "post", return_value=allow_resp):
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )
        assert result.allowed is True

        # Phase 2: Cloud goes offline
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("offline")):
            # Should allow — local spend ($3 from Phase 1 via 100_000 tokens) + $3 < $500
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )
            assert result.allowed is True
            assert result.budget_limit == 500.0

    def test_hard_deny_cloud_unreachable_denies_when_local_exceeds_last_known(
        self,
    ) -> None:
        """hard_deny + cloud unreachable + local spend > last-known limit -> deny."""
        # Arrange — cache_ttl=0 so Phase 2 doesn't serve from cache
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY, fail_open=False, cache_ttl=0)

        # Phase 1: Cloud establishes a $50 limit
        allow_resp = _mock_cloud_response(
            allowed=True,
            budget_limit=50.0,
            current_usage=0.0,
            remaining=50.0,
        )
        with patch.object(enforcer._http, "post", return_value=allow_resp):
            enforcer.check_budget(estimated_input_tokens=10_000, model="gpt-5.5", provider="openai")

        # Phase 2: Cloud goes offline. Spend locally up to the limit.
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("offline")):
            # Fill local budget to ~$48.3 (0.30 from Phase 1 + 48.0 here)
            for _ in range(48):
                enforcer._track_local_cost(1.0)

            # ~$48.3 + $3.0 (100_000 tokens × $0.00003) = $51.3 > $50.0 -> deny
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )
            assert result.allowed is False
            assert "denies" in result.warning.lower()

    def test_hard_deny_cloud_never_reached_denies_fail_closed(self) -> None:
        """hard_deny + cloud NEVER reached + no last-known limit -> deny.

        If the SDK has never successfully communicated with the cloud,
        there's no known budget limit.  In hard_deny mode, this must
        fail-closed (deny) rather than allow with an arbitrary default.
        """
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY, fail_open=False)

        # Act — cloud immediately unreachable, no prior contact
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("offline")):
            result = enforcer.check_budget(
                estimated_input_tokens=50_000, model="gpt-5.5", provider="openai"
            )

        # Assert
        assert result.allowed is False
        assert "no prior budget limit" in result.warning.lower()


# ---------------------------------------------------------------------------
# Bug 1.1: BudgetExceededError field correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBudgetExceededErrorFields:
    """Verify BudgetExceededError attributes match the cloud response.

    Would have caught Bug 1.1 where budget_limit was set to
    remaining_budget and current_usage was hardcoded to 0.0.
    """

    def test_error_budget_limit_matches_cloud_response(self) -> None:
        """BudgetExceededError.budget_limit should be the configured cap, not remaining."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_resp = _mock_cloud_response(
            allowed=False,
            budget_limit=500.0,
            current_usage=499.0,
            remaining=1.0,
            mode="hard_deny",
        )

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            result = enforcer.check_budget(
                estimated_input_tokens=100_000, model="gpt-5.5", provider="openai"
            )

        # Assert — these are the fields that feed BudgetExceededError
        assert result.budget_limit == 500.0  # NOT remaining_budget (1.0)
        assert result.current_usage == 499.0  # NOT 0.0

    def test_error_budget_limit_not_remaining(self) -> None:
        """Regression: budget_limit must never be the remaining amount."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_resp = _mock_cloud_response(
            allowed=False,
            budget_limit=1000.0,
            current_usage=999.0,
            remaining=1.0,
            mode="hard_deny",
        )

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            result = enforcer.check_budget(
                estimated_input_tokens=500_000, model="gpt-5.5", provider="openai"
            )

        # Assert
        assert result.budget_limit != result.remaining_budget
        assert result.budget_limit == 1000.0
        assert result.remaining_budget == 1.0


# ---------------------------------------------------------------------------
# Bug 1.2: Last-known limit persistence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLastKnownBudgetLimit:
    """Verify last-known budget limit persists across cloud responses.

    Would have caught Bug 1.2 where local enforcement used a hardcoded
    $100.0 instead of the last-known limit from the cloud.
    """

    def test_last_known_limit_set_by_allow_response(self) -> None:
        """Allow response should update _last_known_budget_limit."""
        # Arrange
        enforcer = _make_enforcer()
        mock_resp = _mock_cloud_response(allowed=True, budget_limit=750.0, current_usage=100.0)

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            enforcer.check_budget(estimated_input_tokens=50_000, model="gpt-5.5", provider="openai")

        # Assert
        assert enforcer._last_known_budget_limit == 750.0

    def test_last_known_limit_set_by_deny_response(self) -> None:
        """Deny response should also update _last_known_budget_limit."""
        # Arrange
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_resp = _mock_cloud_response(
            allowed=False, budget_limit=200.0, current_usage=200.0, remaining=0.0
        )

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            enforcer.check_budget(estimated_input_tokens=50_000, model="gpt-5.5", provider="openai")

        # Assert
        assert enforcer._last_known_budget_limit == 200.0

    def test_last_known_limit_survives_cache_expiry(self) -> None:
        """Last-known limit should persist even after cache TTL expires."""
        # Arrange
        enforcer = _make_enforcer(cache_ttl=0)  # Expire immediately
        mock_resp = _mock_cloud_response(allowed=True, budget_limit=300.0, current_usage=50.0)

        # Act
        with patch.object(enforcer._http, "post", return_value=mock_resp):
            enforcer.check_budget(estimated_input_tokens=50_000, model="gpt-5.5", provider="openai")

        # Assert — cache expired, but last-known limit persists
        assert enforcer._should_use_cache() is False
        assert enforcer._last_known_budget_limit == 300.0

    def test_last_known_limit_is_none_before_any_cloud_contact(self) -> None:
        """Before any cloud contact, last-known limit should be None."""
        # Arrange
        enforcer = _make_enforcer()

        # Assert
        assert enforcer._last_known_budget_limit is None
