"""Tests for AsyncBudgetEnforcer — async mirror of test_budget.py sync tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerAdmission

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

_RUN_DENY_RESPONSE = {
    **_DENY_RESPONSE,
    "denied_by_period": "agent_run",
}


def _response(payload: dict[str, object]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


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


def _make_legacy_async_enforcer(**overrides) -> AsyncBudgetEnforcer:
    """An AsyncBudgetEnforcer with the PJ-2 lease kill switch OFF.

    Same rationale as the sync suite: these tests pin the per-call
    ``/budgets/check`` contract that the kill switch preserves byte-for-byte.
    Lease-path coverage lives in tests/unit/test_budget_lease.py.
    """
    overrides.setdefault("lease_enabled", False)
    return _make_async_enforcer(**overrides)


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
        mock_response.json.return_value = {
            **ALLOW_BUDGET_RESPONSE,
            "failover_directive": {
                "version": "1",
                "failover_tuning_allowed": False,
            },
        }
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(return_value=mock_response)
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
        )

        assert result.allowed is True
        assert result.remaining_budget == 80.0
        assert result.reservation_id == "res_123"
        assert result.warning is None
        assert result.failover_tuning_allowed is False
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()
        await enforcer.close()


@pytest.mark.unit
class TestAsyncScopedCacheAndStickyDenials:
    """Async mirror of run-scoped cache and sticky-deny behavior."""

    @pytest.mark.asyncio
    async def test_hot_global_allow_does_not_skip_or_get_overwritten_by_scoped_checks(
        self,
    ) -> None:
        enforcer = _make_legacy_async_enforcer()
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_global"}),
                _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_a"}),
                _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_b"}),
            ]
        )

        await enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
        await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_b",
        )
        unscoped = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
        )

        assert enforcer._http.post.call_count == 3
        assert enforcer._http.post.call_args_list[1].kwargs["json"]["agent_run_id"] == "run_a"
        assert enforcer._http.post.call_args_list[2].kwargs["json"]["agent_run_id"] == "run_b"
        assert enforcer._cached_response is not None
        assert enforcer._cached_response.reservation_id == "res_global"
        assert unscoped.reservation_id is None
        await enforcer.close()

    @pytest.mark.asyncio
    async def test_run_deny_is_sticky_only_for_same_run_during_outage(self) -> None:
        enforcer = _make_legacy_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
                httpx.ConnectError("unreachable"),
            ]
        )

        denied_a = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        outage_b = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_b",
        )
        outage_a = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )

        assert denied_a.allowed is False
        assert outage_b.allowed is True
        assert outage_a.allowed is False
        await enforcer.close()

    @pytest.mark.asyncio
    async def test_run_hard_deny_clears_older_global_project_deny_for_other_runs(
        self,
    ) -> None:
        enforcer = _make_legacy_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response(_DENY_RESPONSE),
                _response(_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ]
        )

        project_denied = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
        )
        run_a_denied = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        outage_b = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_b",
        )

        assert project_denied.allowed is False
        assert run_a_denied.allowed is False
        assert outage_b.allowed is True
        assert outage_b.warning is not None
        assert "fail-open" in outage_b.warning.lower()
        await enforcer.close()

    @pytest.mark.asyncio
    async def test_scoped_project_alert_only_response_clears_older_run_hard_deny(
        self,
    ) -> None:
        enforcer = _make_legacy_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                _response(_ALERT_ONLY_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ]
        )

        hard_denied = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        alert_only = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        outage = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )

        assert hard_denied.allowed is False
        assert alert_only.allowed is True
        assert outage.allowed is True
        assert outage.warning is not None
        assert "fail-open" in outage.warning.lower()
        await enforcer.close()

    @pytest.mark.asyncio
    async def test_project_period_deny_received_in_scope_stays_globally_sticky(self) -> None:
        enforcer = _make_legacy_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        enforcer._http.post = AsyncMock(
            side_effect=[
                _response(_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ]
        )

        denied = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_a",
        )
        outage_b = await enforcer.check_budget(
            estimated_input_tokens=500,
            model="gpt-5.5",
            provider="openai",
            agent_run_id="run_b",
        )

        assert denied.allowed is False
        assert outage_b.allowed is False
        await enforcer.close()


@pytest.mark.unit
class TestAsyncFailOpenSticky:
    """Legacy global sticky hard-deny behavior during cloud outages."""

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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
    async def test_prior_hard_deny_is_logged_when_cloud_unreachable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_async_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        enforcer._http.post = AsyncMock(
            side_effect=[mock_response, httpx.ConnectError("unreachable")]
        )

        with caplog.at_level("WARNING", logger="solwyn.budget"):
            await enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        preserved = [
            record
            for record in caplog.records
            if "preserving prior hard deny" in record.getMessage().lower()
        ]
        assert len(preserved) == 1
        assert preserved[0].levelname == "WARNING"
        # The logged message carries the same usage/limit figures as the field.
        assert "$99.50/$100.00 used" in preserved[0].getMessage()
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
        )
        allowed = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
        )
        outage = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
            estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
        )
        result = await enforcer.check_budget(
            estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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


# Settlement moved off the enforcer: AsyncBudgetEnforcer.confirm_cost is removed
# (settlement rides reporter.report_settlement / reporter._send_confirm). The
# confirm POST + error accounting is covered by tests/unit/test_reporter.py.


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


# ---------------------------------------------------------------------------
# Contract drift taxonomy (R6 async twin)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncContractDriftTaxonomy:
    """R6 async twin: a 2xx body the SDK cannot parse is contract drift, not an outage."""

    def _drifted_response(self) -> MagicMock:
        # spec=httpx.Response already supplies a non-raising raise_for_status;
        # a 2xx that the SDK cannot PARSE is the shape under test.
        response = MagicMock(spec=httpx.Response)
        # Valid JSON, wrong shape: model_validate raises ValidationError.
        response.json.return_value = {"totally": "unexpected"}
        return response

    def _breaker(self) -> MagicMock:
        """A closed control-plane breaker double, constrained to the real API."""
        breaker = MagicMock(spec=CircuitBreaker)
        # What a CLOSED breaker actually returns — never None, and it owns no
        # HALF_OPEN probe slot.
        breaker.admit.return_value = CircuitBreakerAdmission(allowed=True)
        return breaker

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_error_records_breaker_success_not_failure(self) -> None:
        breaker = self._breaker()
        enforcer = _make_async_enforcer(control_plane_breaker=breaker)

        with patch.object(enforcer._http, "post", AsyncMock(return_value=self._drifted_response())):
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True  # fail_open honored, same as today
        breaker.record_success.assert_called_once()
        breaker.record_failure.assert_not_called()
        breaker.release_probe.assert_called_once_with(breaker.admit.return_value)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_transport_error_still_records_breaker_failure(self) -> None:
        breaker = self._breaker()
        enforcer = _make_async_enforcer(control_plane_breaker=breaker)

        with patch.object(
            enforcer._http, "post", AsyncMock(side_effect=httpx.ConnectError("unreachable"))
        ):
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        breaker.record_failure.assert_called_once()
        breaker.record_success.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_error_logs_distinct_error_event(self, caplog) -> None:
        enforcer = _make_async_enforcer()
        with (
            patch.object(enforcer._http, "post", AsyncMock(return_value=self._drifted_response())),
            caplog.at_level("ERROR", logger="solwyn.budget"),
        ):
            await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
        assert any("budget.check_response_unreadable" in r.message for r in caplog.records)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_json_decode_error_is_also_drift(self) -> None:
        breaker = self._breaker()
        enforcer = _make_async_enforcer(control_plane_breaker=breaker)
        response = MagicMock(spec=httpx.Response)
        response.json.side_effect = ValueError("not json")

        with patch.object(enforcer._http, "post", AsyncMock(return_value=response)):
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        breaker.record_success.assert_called_once()
        breaker.record_failure.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_error_with_fail_open_false_enforces_locally(self) -> None:
        enforcer = _make_async_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        with patch.object(enforcer._http, "post", AsyncMock(return_value=self._drifted_response())):
            result = await enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
        # No prior cloud contact -> local enforcement fails closed, mirroring
        # TestLocalEnforcement::test_denies_when_cloud_never_reached.
        assert result.allowed is False
        assert result.warning is not None
        assert "no prior budget limit" in result.warning.lower()
