"""BudgetEnforcer must track consecutive confirm failures."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn.budget import BudgetEnforcer


def _error_response(status_code: int) -> MagicMock:
    """A 4xx/5xx httpx.Response stand-in: raise_for_status raises HTTPStatusError."""
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
                reservation_id="r1",
                model="gpt-4o",
                token_details=MagicMock(),
                provider="openai",
                call_id="call_confirm_failure",
            )

    assert "budget.confirm_cost_persistent_failure" not in caplog.text  # type: ignore[union-attr]

    # 10th failure → error
    caplog.clear()  # type: ignore[union-attr]
    with caplog.at_level(logging.ERROR):  # type: ignore[union-attr]
        enforcer.confirm_cost(
            reservation_id="r1",
            model="gpt-4o",
            token_details=MagicMock(),
            provider="openai",
            call_id="call_confirm_failure",
        )

    assert "budget.confirm_cost_persistent_failure" in caplog.text  # type: ignore[union-attr]
    assert "consecutive_failures=10" in caplog.text  # type: ignore[union-attr]

    enforcer.close()


@pytest.mark.unit
def test_confirm_4xx_increments_failure_counter(caplog: pytest.LogCaptureFixture) -> None:
    # A server 422 (e.g. rejecting new provider/call_id/failover fields) must be
    # treated as a failure — raise_for_status surfaces it so the except branch
    # increments the counter instead of resetting it.
    enforcer = BudgetEnforcer(api_url="http://test", api_key="sk_test")
    token_details = TokenDetails(input_tokens=100, output_tokens=50)
    enforcer._http.post = MagicMock(return_value=_error_response(422))

    with caplog.at_level(logging.WARNING):
        enforcer.confirm_cost(
            reservation_id="r1",
            model="gpt-4o",
            token_details=token_details,
            provider="openai",
            call_id="call_confirm_failure",
        )

    assert enforcer._consecutive_confirm_failures == 1
    assert "budget.confirm_cost_failed" in caplog.text
    # Privacy: only the exception class name is logged, never the body.
    assert "HTTPStatusError" in caplog.text
    assert "422 Unprocessable" not in caplog.text

    enforcer.close()


@pytest.mark.unit
def test_confirm_2xx_resets_failure_counter() -> None:
    # A successful confirm clears any accumulated consecutive-failure count.
    enforcer = BudgetEnforcer(api_url="http://test", api_key="sk_test")
    token_details = TokenDetails(input_tokens=100, output_tokens=50)

    # Prime the counter with two failures.
    enforcer._http.post = MagicMock(return_value=_error_response(422))
    for _ in range(2):
        enforcer.confirm_cost(
            reservation_id="r1",
            model="gpt-4o",
            token_details=token_details,
            provider="openai",
            call_id="call_confirm_failure",
        )
    assert enforcer._consecutive_confirm_failures == 2

    # A 200 resets it to zero.
    enforcer._http.post = MagicMock(return_value=_ok_response())
    enforcer.confirm_cost(
        reservation_id="r1",
        model="gpt-4o",
        token_details=token_details,
        provider="openai",
        call_id="call_confirm_failure",
    )
    assert enforcer._consecutive_confirm_failures == 0

    enforcer.close()
