"""BudgetEnforcer must track consecutive confirm failures."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from solwyn.budget import BudgetEnforcer


@pytest.mark.unit
def test_confirm_failure_emits_error_after_threshold(caplog: object) -> None:
    enforcer = BudgetEnforcer(
        api_url="http://test",
        api_key="sk_test",
    )
    # Force every confirm to fail
    enforcer._http.post = MagicMock(side_effect=RuntimeError("test failure"))

    # First 9 failures → warnings only
    with caplog.at_level(logging.WARNING):  # type: ignore[union-attr]
        for _ in range(9):
            enforcer.confirm_cost(
                reservation_id="r1", model="gpt-4o", token_details=MagicMock(), provider="openai"
            )

    assert "budget.confirm_cost_persistent_failure" not in caplog.text  # type: ignore[union-attr]

    # 10th failure → error
    caplog.clear()  # type: ignore[union-attr]
    with caplog.at_level(logging.ERROR):  # type: ignore[union-attr]
        enforcer.confirm_cost(
            reservation_id="r1", model="gpt-4o", token_details=MagicMock(), provider="openai"
        )

    assert "budget.confirm_cost_persistent_failure" in caplog.text  # type: ignore[union-attr]
    assert "consecutive_failures=10" in caplog.text  # type: ignore[union-attr]

    enforcer.close()
