"""Tests for budget enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID
from pydantic import BaseModel

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
        assert req.failover_directive_version == "1"
        assert req.model_dump(mode="json")["failover_directive_version"] == "1"

    def test_build_check_request_carries_agent_run_id(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        req = base._build_check_request(
            500,
            "gpt-4o",
            "openai",
            agent_run_id="run_abc",
        )

        assert req.agent_run_id == "run_abc"

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

    @pytest.mark.parametrize(
        ("payload", "expected_allowed"),
        [
            (ALLOW_BUDGET_RESPONSE, True),
            (_ALERT_ONLY_DENY_RESPONSE, True),
            (_DENY_RESPONSE, False),
        ],
    )
    def test_cloud_result_branches_propagate_failover_tuning_allowed(
        self,
        payload: dict[str, object],
        expected_allowed: bool,
    ) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        response = BudgetCheckResponse.model_validate(
            {
                **payload,
                "failover_directive": {
                    "version": "1",
                    "failover_tuning_allowed": False,
                },
            }
        )

        result = base._build_result_from_response(response)

        assert result.allowed is expected_allowed
        assert result.failover_tuning_allowed is False


# ---------------------------------------------------------------------------
# Cloud allow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloudAllow:
    """Cloud reachable and allows the request."""

    def test_returns_allowed(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            **ALLOW_BUDGET_RESPONSE,
            "failover_directive": {
                "version": "1",
                "failover_tuning_allowed": True,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.allowed is True
        assert result.remaining_budget == 80.0
        assert result.reservation_id == "res_123"
        assert result.warning is None
        assert result.failover_tuning_allowed is True

    def test_shared_allow_fixture_propagates_non_none_price_hints(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            **ALLOW_BUDGET_RESPONSE,
            "price_hints": {"openai": 10.0, "anthropic": 2.0},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert result.price_hints == {"openai": 10.0, "anthropic": 2.0}


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
        mock_response.json.return_value = _ALERT_ONLY_DENY_RESPONSE
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

    def test_prior_hard_deny_blocks_when_cloud_later_unreachable(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[mock_response, httpx.ConnectError("unreachable")],
        ) as mock_post:
            denied = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert denied.allowed is False
        assert result.allowed is False
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]
        assert result.current_usage == _DENY_RESPONSE["current_usage"]
        assert result.warning is not None
        assert "cloud api unreachable" in result.warning.lower()
        assert mock_post.call_count == 2

    def test_prior_hard_deny_is_logged_when_cloud_unreachable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(
                enforcer._http,
                "post",
                side_effect=[mock_response, httpx.ConnectError("unreachable")],
            ),
            caplog.at_level("WARNING", logger="solwyn.budget"),
        ):
            enforcer.check_budget(estimated_input_tokens=50000, model="gpt-4o", provider="openai")
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
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

    def test_cloud_hard_deny_overrides_local_alert_only_when_cloud_later_unreachable(
        self,
    ) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.ALERT_ONLY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[mock_response, httpx.ConnectError("unreachable")],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert denied.allowed is False
        assert denied.mode == BudgetMode.HARD_DENY
        assert result.allowed is False
        assert result.mode == BudgetMode.HARD_DENY
        assert result.warning is not None
        assert "preserving prior hard deny" in result.warning.lower()

    def test_cloud_alert_only_deny_does_not_stick_when_local_hard_deny(
        self,
    ) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _ALERT_ONLY_DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[mock_response, httpx.ConnectError("unreachable")],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert denied.allowed is True
        assert denied.mode == BudgetMode.ALERT_ONLY
        assert denied.warning is not None
        assert "limit" in denied.warning.lower()
        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()

    def test_cloud_allow_clears_prior_hard_deny_before_later_outage(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY, cache_ttl=0)
        deny_response = MagicMock()
        deny_response.json.return_value = _DENY_RESPONSE
        deny_response.raise_for_status = MagicMock()
        allow_response = MagicMock()
        allow_response.json.return_value = ALLOW_BUDGET_RESPONSE
        allow_response.raise_for_status = MagicMock()

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                deny_response,
                allow_response,
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )
            allowed = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )
            outage = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert denied.allowed is False
        assert allowed.allowed is True
        assert outage.allowed is True
        assert outage.warning is not None
        assert "fail-open" in outage.warning.lower()

    def test_prior_hard_deny_overrides_local_enforcement_when_cloud_unreachable(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[mock_response, httpx.ConnectError("unreachable")],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-4o", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert denied.allowed is False
        assert result.allowed is False
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]
        assert result.current_usage == _DENY_RESPONSE["current_usage"]
        assert result.warning is not None
        assert "preserving prior hard deny" in result.warning.lower()


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


@pytest.mark.unit
class TestScopedCacheAndStickyDenials:
    """Run-scoped decisions bypass global allows and preserve deny isolation."""

    def test_hot_global_allow_does_not_skip_or_get_overwritten_by_scoped_checks(self) -> None:
        enforcer = _make_enforcer()
        global_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_global"})
        run_a_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_a"})
        run_b_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_b"})

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[global_allow, run_a_allow, run_b_allow],
        ) as mock_post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-4o", provider="openai")
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )
            unscoped = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-4o", provider="openai"
            )

        assert mock_post.call_count == 3
        assert mock_post.call_args_list[1].kwargs["json"]["agent_run_id"] == "run_a"
        assert mock_post.call_args_list[2].kwargs["json"]["agent_run_id"] == "run_b"
        assert enforcer._cached_response is not None
        assert enforcer._cached_response.reservation_id == "res_global"
        assert unscoped.reservation_id is None

    def test_run_deny_is_sticky_only_for_same_run_during_outage(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )

        assert denied_a.allowed is False
        assert outage_b.allowed is True
        assert outage_a.allowed is False
        assert outage_a.warning is not None
        assert "preserving prior hard deny" in outage_a.warning.lower()

    def test_run_hard_deny_clears_older_global_project_deny_for_other_runs(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_DENY_RESPONSE),
                _response(_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ):
            project_denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
            )
            run_a_denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )

        assert project_denied.allowed is False
        assert run_a_denied.allowed is False
        assert outage_b.allowed is True
        assert outage_b.warning is not None
        assert "fail-open" in outage_b.warning.lower()

    def test_authoritative_allow_clears_only_same_run_sticky_deny(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                _response(_RUN_DENY_RESPONSE),
                _response(ALLOW_BUDGET_RESPONSE),
                httpx.ConnectError("unreachable"),
                httpx.ConnectError("unreachable"),
            ],
        ):
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )

        assert outage_a.allowed is True
        assert outage_b.allowed is False

    def test_project_period_deny_received_in_scope_stays_globally_sticky(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_b",
            )

        assert denied.allowed is False
        assert outage_b.allowed is False

    def test_scoped_project_alert_only_response_clears_older_run_hard_deny(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                _response(_ALERT_ONLY_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ):
            hard_denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            alert_only = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )
            outage = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-4o",
                provider="openai",
                agent_run_id="run_a",
            )

        assert hard_denied.allowed is False
        assert alert_only.allowed is True
        assert outage.allowed is True
        assert outage.warning is not None
        assert "fail-open" in outage.warning.lower()

    def test_run_sticky_denials_are_bounded_and_evict_least_recently_used(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        response = BudgetCheckResponse(**_RUN_DENY_RESPONSE)

        for index in range(128):
            enforcer._cache_response(response, agent_run_id=f"run_{index}")
        assert enforcer._build_prior_hard_deny_unavailable_result("run_0") is not None
        enforcer._cache_response(response, agent_run_id="run_128")

        assert len(enforcer._run_hard_deny_responses) == 128
        assert "run_0" in enforcer._run_hard_deny_responses
        assert "run_1" not in enforcer._run_hard_deny_responses
        assert "run_128" in enforcer._run_hard_deny_responses


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
        assert result.failover_tuning_allowed is None

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


# Settlement moved off the enforcer: BudgetEnforcer.confirm_cost is removed
# (settlement rides reporter.report_settlement / reporter._send_confirm). The
# confirm POST + error accounting is covered by tests/unit/test_reporter.py.


def test_budget_check_result_is_pydantic_model() -> None:
    """BudgetCheckResult must be a Pydantic BaseModel, not a dataclass."""
    assert issubclass(BudgetCheckResult, BaseModel)
