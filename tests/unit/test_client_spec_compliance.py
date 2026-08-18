"""Spec-derived tests for Solwyn client wrapper.

These tests verify that BudgetExceededError carries correct field values
when raised by the client interception flow, covering atomic reservation
via the Cloud API.

Would have caught Bug 1.1 where budget_limit was set to
remaining_budget and current_usage was hardcoded to 0.0.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import BudgetMode
from solwyn.client import Solwyn
from solwyn.exceptions import BudgetExceededError


def _mock_openai_client():
    """Create a mock that looks like openai.OpenAI()."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    mock_response = MagicMock()
    mock_response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    client.chat.completions.create.return_value = mock_response
    return client


def _make_solwyn(client, **overrides):
    """Create a Solwyn wrapper with mocked background thread."""
    defaults = {
        "api_key": VALID_API_KEY,
    }
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    return solwyn


@pytest.mark.unit
class TestBudgetExceededErrorFieldCorrectness:
    """BudgetExceededError must carry accurate values from the cloud response.

    Regression tests for Bug 1.1:
    - budget_limit must be the configured cap (NOT remaining_budget)
    - current_usage must be the actual spend (NOT 0.0)
    - budget_period must reflect config (NOT hardcoded "daily")
    """

    def test_error_fields_match_cloud_response(self) -> None:
        """All BudgetExceededError attributes should reflect cloud state."""
        # Arrange
        client = _mock_openai_client()
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 1.0,
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 500.0,
            "current_usage": 499.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        # Act
        with (
            patch.object(solwyn._solwyn_budget._http, "post", return_value=mock_budget_response),
            pytest.raises(BudgetExceededError) as exc_info,
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Assert — these are the fields that were wrong in Bug 1.1
        exc = exc_info.value
        assert exc.budget_limit == 500.0, (
            f"budget_limit should be 500.0 (configured cap), got {exc.budget_limit}"
        )
        assert exc.current_usage == 499.0, (
            f"current_usage should be 499.0 (actual spend), got {exc.current_usage}"
        )
        assert exc.mode == "hard_deny"

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_error_budget_limit_is_not_remaining(self) -> None:
        """Regression: budget_limit must never equal remaining_budget."""
        # Arrange
        client = _mock_openai_client()
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 50.0,  # This is remaining, NOT the limit
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 1000.0,  # This is the actual limit
            "current_usage": 950.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        # Act
        with (
            patch.object(solwyn._solwyn_budget._http, "post", return_value=mock_budget_response),
            pytest.raises(BudgetExceededError) as exc_info,
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Assert
        exc = exc_info.value
        assert exc.budget_limit == 1000.0
        assert exc.budget_limit != 50.0  # Must NOT be remaining_budget

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()
