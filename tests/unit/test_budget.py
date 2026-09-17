"""Tests for budget enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID
from pydantic import BaseModel

from solwyn._lease import LeaseAdmission, LeaseDecision
from solwyn._run_control import (
    clear_run_termination,
    mark_terminated,
    run_termination,
)
from solwyn._types import BudgetCheckResponse, BudgetMode, LeaseGrantResponse
from solwyn.budget import (
    BudgetCheckResult,
    BudgetEnforcer,
    _BudgetEnforcerBase,
)
from solwyn.circuit_breaker import CircuitBreaker, CircuitBreakerAdmission

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

_STOPPED_RUN_DENY_RESPONSE = {
    **_DENY_RESPONSE,
    "denied_by_period": "run_stopped",
}

_TAG_DENY_RESPONSE = {
    **_DENY_RESPONSE,
    "budget_limit": 25.0,
    "current_usage": 24.5,
    "denied_by_period": "tag",
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


def _make_legacy_enforcer(**overrides):
    """A BudgetEnforcer with the PJ-2 lease kill switch OFF.

    Run-scoped traffic meets the lease path first; these tests are about the
    per-call ``/budgets/check`` contract (scoped cache isolation, sticky
    denials), so they pin the legacy path explicitly rather than being
    re-pointed at ``/budgets/lease`` — the behaviour under test is the one the
    kill switch preserves byte-for-byte. Lease-path coverage of the same run
    scoping lives in tests/unit/test_budget_lease.py.
    """
    overrides.setdefault("lease_enabled", False)
    return _make_enforcer(**overrides)


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
        req = base._build_check_request(500, "gpt-5.5", "openai")
        assert not hasattr(req, "project_id")
        assert req.estimated_input_tokens == 500
        assert req.model == "gpt-5.5"
        assert req.provider == "openai"
        assert req.failover_directive_version == "1"
        assert req.run_directive_version == "1"
        assert req.price_hints_version == "1"
        assert req.model_dump(mode="json")["failover_directive_version"] == "1"
        assert req.model_dump(mode="json")["run_directive_version"] == "1"
        assert req.model_dump(mode="json")["price_hints_version"] == "1"

    def test_build_check_request_carries_agent_run_id(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        req = base._build_check_request(
            500,
            "gpt-5.5",
            "openai",
            agent_run_id="run_abc",
        )

        assert req.agent_run_id == "run_abc"

    def test_build_check_request_carries_copied_tags(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        tags = {"team": "research"}

        req = base._build_check_request(
            500,
            "gpt-5.5",
            "openai",
            tags=tags,
        )
        tags["team"] = "mutated"

        assert req.tags == {"team": "research"}

    def test_no_local_cost_ledger_or_per_token_price(self) -> None:
        """The SDK never prices a call: no dollar ledger, no per-token constant."""
        import solwyn.budget as budget_module

        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        assert not hasattr(budget_module, "DEFAULT_COST_PER_TOKEN")
        for name in (
            "_local_costs",
            "_track_local_cost",
            "_get_local_remaining",
            "_get_local_current",
            "_build_local_enforcement_result",
        ):
            assert not hasattr(base, name), name
        assert base.uncounted_tally() == (0, 0)

    def test_cache_allow_decisions(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=5,
        )

        response = BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE)
        key = base._allow_cache_key(
            provider="openai",
            model="gpt-5.5",
            fallback_providers=[],
            fallback_models=[],
            modality="text",
        )
        base._cache_response(response, cache_key=key)
        with base._state_lock:
            assert base._cached_allow_locked(key) is response

    def test_live_allow_clears_matching_server_termination(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        mark_terminated("run_server", reason="run_stopped", source="server")

        base._cache_response(
            BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE),
            agent_run_id="run_server",
        )

        assert run_termination("run_server") is None

    def test_live_allow_does_not_clear_local_velocity_termination(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        mark_terminated(
            "run_local",
            reason="velocity:repeat_size",
            source="local_velocity",
        )

        try:
            base._cache_response(
                BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE),
                agent_run_id="run_local",
            )

            assert run_termination("run_local") is not None
        finally:
            clear_run_termination("run_local")

    def test_live_allow_does_not_clear_different_server_run(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        mark_terminated("run_other", reason="run_stopped", source="server")

        try:
            base._cache_response(
                BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE),
                agent_run_id="run_allowed",
            )

            assert run_termination("run_other") is not None
        finally:
            clear_run_termination("run_other")

    def test_never_cache_deny_decisions(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=5,
        )

        response = BudgetCheckResponse(**_DENY_RESPONSE)
        base._cache_response(response)
        assert not base._allow_cache

    def test_cache_expires(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            cache_ttl=0,  # Expire immediately
        )

        response = BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE)
        key = base._allow_cache_key(
            provider="openai",
            model="gpt-5.5",
            fallback_providers=[],
            fallback_models=[],
            modality="text",
        )
        base._cache_response(response, cache_key=key)
        with base._state_lock:
            assert base._cached_allow_locked(key) is None

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
        assert result.denied_by_period == response.denied_by_period

    def test_lease_run_stopped_denial_is_filed_under_only_its_run(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        response = LeaseGrantResponse.model_validate(
            {
                "eligible": True,
                "allowed": False,
                "denied_by_period": "run_stopped",
                "project_id": VALID_PROJECT_ID,
                "mode": "hard_deny",
                "budget_limit": 100.0,
                "current_usage": 100.0,
                "remaining_budget": 0.0,
            }
        )

        result = base._lease_deny_result("run_a", response)

        assert result.denied_by_period == "run_stopped"
        assert base._last_hard_deny_response is None
        same_run = base._build_prior_hard_deny_unavailable_result("run_a")
        unrelated_run = base._build_prior_hard_deny_unavailable_result("run_b")
        assert same_run is not None
        assert same_run.denied_by_period == "run_stopped"
        assert unrelated_run is None

    @pytest.mark.parametrize("denied_by_period", ["agent_run", "run_stopped"])
    def test_run_scoped_denial_without_run_id_invalidates_allow_and_sticks_globally(
        self, denied_by_period: str
    ) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        cached_key = base._allow_cache_key(
            provider="openai",
            model="gpt-5.5",
            fallback_providers=[],
            fallback_models=[],
            modality="text",
        )
        base._cache_response(BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE), cache_key=cached_key)
        response = BudgetCheckResponse.model_validate(
            {**_DENY_RESPONSE, "denied_by_period": denied_by_period}
        )

        base._cache_response(response)

        assert not base._allow_cache
        assert base._last_hard_deny_response is response
        prior = base._build_prior_hard_deny_unavailable_result()
        assert prior is not None
        assert prior.denied_by_period == denied_by_period

    def test_run_stopped_denial_preserves_prior_project_period_sticky_deny(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        project_denial = BudgetCheckResponse(**_DENY_RESPONSE)
        stopped_denial = BudgetCheckResponse(**_STOPPED_RUN_DENY_RESPONSE)
        base._cache_response(project_denial)

        base._cache_response(stopped_denial, agent_run_id="run_a")

        assert base._last_hard_deny_response is project_denial
        assert base._run_hard_deny_responses["run_a"] is stopped_denial
        unrelated_run = base._build_prior_hard_deny_unavailable_result("run_b")
        assert unrelated_run is not None
        assert unrelated_run.denied_by_period == project_denial.denied_by_period


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
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        assert result.remaining_budget == 80.0
        assert result.reservation_id == "res_123"
        assert result.warning is None
        assert result.failover_tuning_allowed is True

    def test_ordered_live_allow_keeps_old_handle_stopped_but_new_handle_clean(self) -> None:
        from solwyn._run_control import _acquire_termination_handle

        enforcer = _make_legacy_enforcer()
        old = _acquire_termination_handle("run_server_allow")
        mark_terminated("run_server_allow", reason="old_stop", source="server")

        with patch.object(
            enforcer._http,
            "post",
            return_value=_response(ALLOW_BUDGET_RESPONSE),
        ):
            result = enforcer.check_budget(
                estimated_input_tokens=500,
                estimated_output_bound=100,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_server_allow",
                call_id="3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
            )
        new = _acquire_termination_handle("run_server_allow")

        assert result.allowed is True
        assert old.termination is not None
        assert old.termination.reason == "old_stop"
        assert new.termination is None
        new.release()
        old.release()
        enforcer.close()

    @pytest.mark.parametrize("first_stop_before_request", [False, True])
    def test_newer_stop_observation_survives_registry_eviction_during_check(
        self,
        first_stop_before_request: bool,
    ) -> None:
        from solwyn._run_control import _acquire_termination_handle

        run_id = f"run_ordered_lru_sync_{first_stop_before_request}"
        churn_prefix = f"ordered_lru_sync_{first_stop_before_request}"
        enforcer = _make_legacy_enforcer()
        old = _acquire_termination_handle(run_id)
        new = None
        if first_stop_before_request:
            with patch("solwyn._run_control.time.monotonic", return_value=1.0):
                mark_terminated(run_id, reason="first_stop", source="server")

        def allow_after_newer_stop(*_args: object, **_kwargs: object) -> MagicMock:
            with patch("solwyn._run_control.time.monotonic", return_value=3.0):
                mark_terminated(run_id, reason="newer_stop", source="server")
                for index in range(300):
                    mark_terminated(
                        f"{churn_prefix}_{index}",
                        reason="unrelated",
                        source="server",
                    )
            return _response(ALLOW_BUDGET_RESPONSE)

        try:
            with (
                patch("solwyn.budget.time.monotonic", return_value=2.0),
                patch.object(
                    enforcer._http,
                    "post",
                    side_effect=allow_after_newer_stop,
                ),
            ):
                result = enforcer.check_budget(
                    estimated_input_tokens=500,
                    estimated_output_bound=100,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id=run_id,
                    call_id="3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
                )
            new = _acquire_termination_handle(run_id)

            assert (result.allowed, result.denied_by_period, result.deny_reason) == (
                False,
                "run_stopped",
                "first_stop" if first_stop_before_request else "newer_stop",
            )
            assert old.termination is not None
            assert new.termination is old.termination
        finally:
            if new is not None:
                new.release()
            old.release()
            clear_run_termination(run_id)
            for index in range(300):
                clear_run_termination(f"{churn_prefix}_{index}")
            enforcer.close()

    def test_tagged_run_uses_per_call_check_and_sends_tags(self) -> None:
        enforcer = _make_enforcer()
        mock_response = MagicMock()
        mock_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response) as post:
            result = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_tagged",
                call_id="3f1a2b4c-5d6e-4f70-8a9b-0c1d2e3f4a5b",
                estimated_output_bound=100,
                tags={"team": "research"},
            )

        assert result.allowed is True
        post.assert_called_once()
        assert post.call_args.args[0].endswith("/api/v1/budgets/check")
        assert post.call_args.kwargs["json"]["tags"] == {"team": "research"}
        assert post.call_args.kwargs["json"]["run_directive_version"] == "1"
        assert post.call_args.kwargs["json"]["price_hints_version"] == "1"

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
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.price_hints == {"openai": 10.0, "anthropic": 2.0}

    def test_check_posts_price_hints_version(self) -> None:
        enforcer = _make_enforcer()
        response = _response(ALLOW_BUDGET_RESPONSE)

        with patch.object(enforcer._http, "post", return_value=response) as post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")

        assert post.call_args.kwargs["json"]["price_hints_version"] == "1"


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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
            enforcer.check_budget(estimated_input_tokens=50000, model="gpt-5.5", provider="openai")
            result = enforcer.check_budget(
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            allowed = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
            outage = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert denied.allowed is False
        assert allowed.allowed is True
        assert outage.allowed is True
        assert outage.warning is not None
        assert "fail-open" in outage.warning.lower()

    def test_prior_hard_deny_overrides_fail_closed_when_cloud_unreachable(self) -> None:
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
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert denied.allowed is False
        assert result.allowed is False
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]
        assert result.current_usage == _DENY_RESPONSE["current_usage"]
        assert result.warning is not None
        assert "preserving prior hard deny" in result.warning.lower()


# ---------------------------------------------------------------------------
# Cloud unreachable + fail_open=False: fail closed (no local cost estimate)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFailClosedWhenUnreachable:
    """Cloud unreachable + fail_open=False denies on the legacy path."""

    def test_denies_when_cloud_never_reached(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=50000, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        assert result.deny_source == "local_enforcement"
        assert result.deny_reason == "control_plane_unreachable"
        assert result.warning is not None
        assert "unreachable" in result.warning.lower()
        assert "lease_enabled=true" in result.warning.lower()

    def test_denies_even_with_last_known_limit(self) -> None:
        """A remembered dollar limit is not something the SDK can meter against."""
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY, cache_ttl=0)

        with patch.object(enforcer._http, "post", return_value=_response(ALLOW_BUDGET_RESPONSE)):
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        assert result.deny_source == "local_enforcement"
        assert result.deny_reason == "control_plane_unreachable"
        assert enforcer.uncounted_tally() == (0, 0)

    def test_alert_only_mode_still_fails_closed(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.ALERT_ONLY)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=10, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        assert result.mode is BudgetMode.ALERT_ONLY
        assert result.deny_reason == "control_plane_unreachable"

    def test_fail_closed_warning_is_logged_and_rate_limited(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_enforcer(fail_open=False)

        with (
            caplog.at_level("WARNING", logger="solwyn.budget"),
            patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")),
        ):
            for _ in range(3):
                enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")

        lines = [r for r in caplog.records if "budget.fail_closed_unreachable" in r.getMessage()]
        assert len(lines) == 1
        assert "lease_enabled=True" in lines[0].getMessage()

    def test_sticky_hard_deny_still_wins_over_fail_closed(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[_response(_DENY_RESPONSE), httpx.ConnectError("unreachable")],
        ):
            enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")
            result = enforcer.check_budget(
                estimated_input_tokens=10, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        assert result.deny_source == "sticky_replay"
        assert result.deny_reason == "monthly"
        assert result.budget_limit == _DENY_RESPONSE["budget_limit"]

    def test_retained_run_stop_still_wins_over_fail_closed(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        mark_terminated("run_retained_outage", reason="operator_stop", source="server")
        try:
            with patch.object(
                enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")
            ):
                result = enforcer.check_budget(
                    estimated_input_tokens=10,
                    model="gpt-5.5",
                    provider="openai",
                    agent_run_id="run_retained_outage",
                )
        finally:
            clear_run_termination("run_retained_outage")

        assert result.allowed is False
        assert result.deny_source == "sticky_replay"
        assert result.denied_by_period == "run_stopped"
        assert result.deny_reason == "operator_stop"


# ---------------------------------------------------------------------------
# Legacy-path outage tally (tokens only) under fail_open=True
# ---------------------------------------------------------------------------


def _outage_check(enforcer: BudgetEnforcer, tokens: int, call_id: str | None = None):
    with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
        return enforcer.check_budget(
            estimated_input_tokens=tokens,
            model="gpt-5.5",
            provider="openai",
            call_id=call_id,
        )


@pytest.mark.unit
class TestLegacyUncountedTally:
    def test_fail_open_admission_increments_tally(self) -> None:
        enforcer = _make_enforcer(fail_open=True)

        first = _outage_check(enforcer, 100, "call-a")
        second = _outage_check(enforcer, 250)

        assert first.allowed is True and second.allowed is True
        assert "fail-open" in (first.warning or "").lower()
        assert enforcer.uncounted_tally() == (2, 350)

    def test_breaker_open_admission_increments_tally(self) -> None:
        breaker = MagicMock(spec=CircuitBreaker)
        breaker.admit.return_value = CircuitBreakerAdmission(allowed=False)
        enforcer = _make_enforcer(fail_open=True, control_plane_breaker=breaker)

        with patch.object(enforcer._http, "post") as mock_post:
            result = enforcer.check_budget(
                estimated_input_tokens=40, model="gpt-5.5", provider="openai"
            )

        mock_post.assert_not_called()
        assert result.allowed is True
        assert enforcer.uncounted_tally() == (1, 40)

    def test_settlement_trues_estimate_up_to_actual_total(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")
        _outage_check(enforcer, 50, "call-b")

        enforcer.settle_uncounted(call_id="call-a", total_tokens=340)

        assert enforcer.uncounted_tally() == (2, 390)
        # Exactly once: a repeated settlement is a no-op.
        enforcer.settle_uncounted(call_id="call-a", total_tokens=10_000)
        assert enforcer.uncounted_tally() == (2, 390)

    def test_settlement_can_lower_an_overestimate(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        enforcer.settle_uncounted(call_id="call-a", total_tokens=30)

        assert enforcer.uncounted_tally() == (1, 30)

    def test_unmeasured_usage_never_reports_below_the_estimate(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        enforcer.settle_uncounted(call_id="call-a", total_tokens=30, usage_unmeasured=True)

        assert enforcer.uncounted_tally() == (1, 100)

    def test_surface_without_token_usage_keeps_the_estimate(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        enforcer.settle_uncounted(call_id="call-a", total_tokens=None)
        enforcer.settle_uncounted(call_id="call-a", total_tokens=5)

        assert enforcer.uncounted_tally() == (1, 100)

    def test_release_keeps_the_estimate_and_frees_the_true_up_slot(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        enforcer.release_reservation("call-a")
        enforcer.settle_uncounted(call_id="call-a", total_tokens=5)

        assert enforcer.uncounted_tally() == (1, 100)
        assert "call-a" not in enforcer._uncounted_estimates

    def test_settlement_for_an_untracked_call_is_a_noop(self) -> None:
        enforcer = _make_enforcer(fail_open=True)

        enforcer.settle_uncounted(call_id="never-admitted", total_tokens=500)

        assert enforcer.uncounted_tally() == (0, 0)

    def test_true_up_map_is_bounded(self) -> None:
        from solwyn.budget import _MAX_UNCOUNTED_TRUE_UPS

        enforcer = _make_enforcer(fail_open=True)
        for i in range(_MAX_UNCOUNTED_TRUE_UPS + 5):
            enforcer._record_legacy_uncounted(10, f"call-{i}")

        assert len(enforcer._uncounted_estimates) == _MAX_UNCOUNTED_TRUE_UPS
        assert "call-0" not in enforcer._uncounted_estimates
        # An evicted call keeps its admission estimate in the tally.
        enforcer.settle_uncounted(call_id="call-0", total_tokens=1_000)
        assert enforcer.uncounted_tally() == (
            _MAX_UNCOUNTED_TRUE_UPS + 5,
            10 * (_MAX_UNCOUNTED_TRUE_UPS + 5),
        )

    def test_tally_rides_next_check_and_resets_only_after_it_succeeds(self) -> None:
        enforcer = _make_enforcer(fail_open=True, cache_ttl=0)
        _outage_check(enforcer, 100, "call-a")
        _outage_check(enforcer, 20)
        enforcer.settle_uncounted(call_id="call-a", total_tokens=180)

        with patch.object(
            enforcer._http, "post", return_value=_response(ALLOW_BUDGET_RESPONSE)
        ) as mock_post:
            result = enforcer.check_budget(
                estimated_input_tokens=7, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        body = mock_post.call_args.kwargs["json"]
        assert body["uncounted_calls"] == 2
        assert body["uncounted_tokens"] == 200
        assert enforcer.uncounted_tally() == (0, 0)

        with patch.object(
            enforcer._http, "post", return_value=_response(ALLOW_BUDGET_RESPONSE)
        ) as mock_post:
            enforcer.check_budget(estimated_input_tokens=7, model="gpt-5.5", provider="openai")
        body = mock_post.call_args.kwargs["json"]
        assert "uncounted_calls" not in body
        assert "uncounted_tokens" not in body

    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectError("unreachable"),
            httpx.ReadTimeout("slow"),
        ],
    )
    def test_failed_check_keeps_the_tally(self, failure: Exception) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        with patch.object(enforcer._http, "post", side_effect=failure) as mock_post:
            enforcer.check_budget(estimated_input_tokens=5, model="gpt-5.5", provider="openai")

        body = mock_post.call_args.kwargs["json"]
        assert (body["uncounted_calls"], body["uncounted_tokens"]) == (1, 100)
        # The failed report is still owed, plus the call this outage just admitted.
        assert enforcer.uncounted_tally() == (2, 105)

    @pytest.mark.parametrize(
        ("status", "payload"),
        [
            # Core refuses a fold that would overflow its ledger.
            (409, {"detail": "uncounted tally would overflow the budget ledger"}),
            # Core could not record the report.
            (503, {"detail": "uncounted tally could not be recorded"}),
        ],
        ids=["409-ledger-overflow", "503-not-recorded"],
    )
    def test_non_2xx_check_keeps_the_tally(self, status: int, payload: dict[str, str]) -> None:
        breaker = MagicMock(spec=CircuitBreaker)
        breaker.admit.return_value = CircuitBreakerAdmission(allowed=True)
        enforcer = _make_enforcer(fail_open=True, control_plane_breaker=breaker)
        _outage_check(enforcer, 100)
        request = httpx.Request("POST", "https://api.test.solwyn.ai/api/v1/budgets/check")
        breaker.reset_mock()

        with patch.object(
            enforcer._http,
            "post",
            return_value=httpx.Response(status, json=payload, request=request),
        ) as mock_post:
            result = enforcer.check_budget(
                estimated_input_tokens=5, model="gpt-5.5", provider="openai"
            )

        body = mock_post.call_args.kwargs["json"]
        assert (body["uncounted_calls"], body["uncounted_tokens"]) == (1, 100)
        # Not a 2xx: the reported tally is kept, plus the call this admitted.
        assert result.allowed is True
        assert enforcer.uncounted_tally() == (2, 105)
        assert enforcer._uncounted_report_in_flight is None
        # The existing classification applies unchanged: any non-2xx other than
        # the structured read-only-key 403 is an outage for the breaker.
        breaker.record_failure.assert_called_once()
        breaker.record_success.assert_not_called()

    def test_unparseable_2xx_keeps_the_tally(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100)
        drifted = MagicMock(spec=httpx.Response)
        drifted.json.return_value = {"totally": "unexpected"}

        with patch.object(enforcer._http, "post", return_value=drifted):
            enforcer.check_budget(estimated_input_tokens=5, model="gpt-5.5", provider="openai")

        assert enforcer.uncounted_tally() == (2, 105)

    def test_breaker_held_check_keeps_accumulating(self) -> None:
        breaker = MagicMock(spec=CircuitBreaker)
        breaker.admit.return_value = CircuitBreakerAdmission(allowed=False)
        enforcer = _make_enforcer(fail_open=True, control_plane_breaker=breaker)

        for _ in range(3):
            enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")

        assert enforcer.uncounted_tally() == (3, 30)
        assert enforcer._uncounted_report_in_flight is None

    def test_calls_tallied_while_a_check_is_in_flight_stay_owed(self) -> None:
        enforcer = _make_enforcer(fail_open=True, cache_ttl=0)
        _outage_check(enforcer, 100, "call-a")

        def answer_after_more_outage_spend(*_args: object, **_kwargs: object) -> MagicMock:
            # Another thread's fail-open admission and settlement land while
            # this check is on the wire.
            enforcer._record_legacy_uncounted(40, "call-b")
            enforcer.settle_uncounted(call_id="call-a", total_tokens=130)
            return _response(ALLOW_BUDGET_RESPONSE)

        with patch.object(enforcer._http, "post", side_effect=answer_after_more_outage_spend):
            enforcer.check_budget(estimated_input_tokens=5, model="gpt-5.5", provider="openai")

        # Reported (1, 100); owed afterwards: call-b and call-a's +30 true-up.
        assert enforcer.uncounted_tally() == (1, 70)

    def test_only_one_report_is_on_the_wire_at_a_time(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        enforcer._record_legacy_uncounted(100, None)

        request = enforcer._build_check_request(5, "gpt-5.5", "openai")
        first, first_report = enforcer._with_uncounted_report(request)
        second, second_report = enforcer._with_uncounted_report(request)

        assert first_report is not None and second_report is None
        assert first.model_dump(mode="json")["uncounted_calls"] == 1
        assert "uncounted_calls" not in second.model_dump(mode="json")

        enforcer._finish_uncounted_report(second_report, acknowledged=True)
        enforcer._finish_uncounted_report(first_report, acknowledged=True)
        enforcer._finish_uncounted_report(first_report, acknowledged=True)
        assert enforcer.uncounted_tally() == (0, 0)

    def test_zero_tally_request_is_byte_identical(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        request = enforcer._build_check_request(5, "gpt-5.5", "openai", agent_run_id="run-x")

        attached, report = enforcer._with_uncounted_report(request)

        assert report is None
        assert attached is request
        body = attached.model_dump(mode="json")
        assert "uncounted_calls" not in body
        assert "uncounted_tokens" not in body
        assert (
            attached.model_dump_json()
            == request.model_copy(
                update={"uncounted_calls": 0, "uncounted_tokens": 0}
            ).model_dump_json()
        )

    def test_sticky_hard_deny_during_outage_is_not_tallied(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[_response(_DENY_RESPONSE), httpx.ConnectError("unreachable")],
        ):
            enforcer.check_budget(estimated_input_tokens=10, model="gpt-5.5", provider="openai")
            result = enforcer.check_budget(
                estimated_input_tokens=10, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is False
        assert result.deny_source == "sticky_replay"
        assert enforcer.uncounted_tally() == (0, 0)

    def test_lease_cold_start_outage_is_tallied_on_the_lease_not_twice(self) -> None:
        enforcer = _make_enforcer(fail_open=True)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=100,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run-cold",
                call_id="0b9f2d4e-5c1a-4e7b-9d3f-2a6c8e1b4f70",
            )

        assert result.allowed is True
        assert enforcer._lease._states["run-cold"].uncounted_calls == 1
        assert enforcer.uncounted_tally() == (0, 0)

    def test_fork_reset_drops_the_parent_tally(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        _outage_check(enforcer, 100, "call-a")

        enforcer._reset_after_fork_in_child()

        assert enforcer.uncounted_tally() == (0, 0)
        assert not enforcer._uncounted_estimates
        assert enforcer._uncounted_report_in_flight is None


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
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
            assert result1.allowed is True

            # Second call should use cache
            result2 = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
            assert result2.allowed is True

            # Only one HTTP call
            assert mock_post.call_count == 1

    @pytest.mark.parametrize(
        ("base_overrides", "changed"),
        [
            ({}, {"model": "gpt-5.5-mini"}),
            ({}, {"provider": "anthropic"}),
            (
                {"fallback_providers": ["anthropic"], "fallback_models": ["claude-sonnet-4"]},
                {"fallback_models": ["claude-haiku-4"]},
            ),
            (
                {"fallback_providers": ["anthropic"], "fallback_models": ["claude-sonnet-4"]},
                {"fallback_providers": ["openai"]},
            ),
            ({}, {"modality": "embedding"}),
        ],
    )
    def test_cache_hit_requires_same_provider_model_chain_modality(
        self,
        base_overrides: dict[str, object],
        changed: dict[str, object],
    ) -> None:
        enforcer = _make_enforcer()
        response = _response(ALLOW_BUDGET_RESPONSE)
        base = {
            "estimated_input_tokens": 500,
            "model": "gpt-5.5",
            "provider": "openai",
        } | base_overrides

        with patch.object(enforcer._http, "post", return_value=response) as post:
            enforcer.check_budget(**base)
            enforcer.check_budget(**base)
            enforcer.check_budget(**(base | changed))

        assert post.call_count == 2

    def test_cache_hit_replays_its_own_price_hints(self) -> None:
        enforcer = _make_enforcer()
        first_response = _response(
            {**ALLOW_BUDGET_RESPONSE, "price_hints": {"openai": 2.0, "anthropic": 1.0}}
        )
        second_response = _response(
            {**ALLOW_BUDGET_RESPONSE, "price_hints": {"openai": 1.0, "anthropic": 3.0}}
        )

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[first_response, second_response],
        ) as post:
            first = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
            cached = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
            changed_model = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5-mini", provider="openai"
            )

        assert first.price_hints == {"openai": 2.0, "anthropic": 1.0}
        assert cached.price_hints == {"openai": 2.0, "anthropic": 1.0}
        assert changed_model.price_hints == {"openai": 1.0, "anthropic": 3.0}
        assert post.call_count == 2

    def test_cache_hit_with_cleared_hints_replays_empty_dict(self) -> None:
        enforcer = _make_enforcer()
        response = _response({**ALLOW_BUDGET_RESPONSE, "price_hints": {}})

        with patch.object(enforcer._http, "post", return_value=response) as post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            cached = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert cached.price_hints == {}
        assert post.call_count == 1

    def test_cache_hit_without_hints_replays_none(self) -> None:
        enforcer = _make_enforcer()
        response = _response({**ALLOW_BUDGET_RESPONSE, "price_hints": None})

        with patch.object(enforcer._http, "post", return_value=response) as post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            cached = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert cached.price_hints is None
        assert post.call_count == 1

    def test_allow_cache_is_bounded_lru(self) -> None:
        enforcer = _make_enforcer()
        response = _response(ALLOW_BUDGET_RESPONSE)

        with patch.object(enforcer._http, "post", return_value=response) as post:
            for index in range(17):
                enforcer.check_budget(
                    estimated_input_tokens=500,
                    model=f"gpt-5.5-{index}",
                    provider="openai",
                )
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5-0", provider="openai")

        assert len(enforcer._allow_cache) <= 16
        assert post.call_count == 18

    def test_deny_clears_all_cached_allows(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        allow = _response(ALLOW_BUDGET_RESPONSE)
        deny = _response(_DENY_RESPONSE)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[allow, allow, deny, allow, allow],
        ) as post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5-mini", provider="openai"
            )
            enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5-nano", provider="openai"
            )
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5-mini", provider="openai"
            )

        assert post.call_count == 5

    def test_deny_not_cached(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        mock_response = MagicMock()
        mock_response.json.return_value = _DENY_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch.object(enforcer._http, "post", return_value=mock_response) as mock_post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")

            # Both calls should hit HTTP (deny is never cached)
            assert mock_post.call_count == 2

    def test_tagged_checks_bypass_and_do_not_replace_global_allow_cache(self) -> None:
        enforcer = _make_enforcer()
        global_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_global"})
        tagged_allow = _response(
            {
                **ALLOW_BUDGET_RESPONSE,
                "reservation_id": "res_tagged",
                "budget_limit": 75.0,
                "current_usage": 12.5,
            }
        )

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[global_allow, tagged_allow],
        ) as post:
            first = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
            )
            tagged = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                tags={"team": "research"},
            )
            cached = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
            )

        assert first.reservation_id == "res_global"
        assert tagged.reservation_id == "res_tagged"
        assert cached.reservation_id is None
        assert enforcer._allow_cache
        assert next(iter(enforcer._allow_cache.values()))[0].reservation_id == "res_global"
        assert enforcer._last_known_budget_limit == 75.0
        assert enforcer._last_known_current_usage == 12.5
        assert post.call_count == 2

    def test_tagged_allow_does_not_populate_global_cache_and_clears_project_sticky(
        self,
    ) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        tagged_allow = {
            **ALLOW_BUDGET_RESPONSE,
            "budget_limit": 80.0,
            "current_usage": 20.0,
        }

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[_response(_DENY_RESPONSE), _response(tagged_allow)],
        ) as post:
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
            )
            allowed = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                tags={"team": "research"},
            )

        assert allowed.allowed is True
        assert post.call_count == 2
        assert not enforcer._allow_cache
        assert enforcer._last_hard_deny_response is None
        assert enforcer._last_known_budget_limit == 80.0
        assert enforcer._last_known_current_usage == 20.0

    def test_tagged_allow_clears_same_run_sticky_deny(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[_response(_RUN_DENY_RESPONSE), _response(ALLOW_BUDGET_RESPONSE)],
        ):
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            allowed = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
                tags={"team": "research"},
            )

        assert allowed.allowed is True
        assert "run_a" not in enforcer._run_hard_deny_responses

    def test_tag_scoped_hard_deny_does_not_poison_global_or_run_sticky_state(
        self,
    ) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_TAG_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
                tags={"team": "research"},
            )
            outage = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )

        assert denied.allowed is False
        assert outage.allowed is True
        assert enforcer._last_hard_deny_response is None
        assert "run_a" not in enforcer._run_hard_deny_responses
        assert enforcer._last_known_budget_limit == 25.0
        assert enforcer._last_known_current_usage == 24.5

    def test_tag_scoped_hard_deny_preserves_existing_same_run_authority(self) -> None:
        enforcer = _make_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
        run_denial = BudgetCheckResponse(**_RUN_DENY_RESPONSE)
        enforcer._cache_response(run_denial, agent_run_id="run_a")
        enforcer._cache_response(BudgetCheckResponse(**_DENY_RESPONSE))

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_TAG_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ) as post:
            tagged_denial = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
                tags={"team": "research"},
            )

            assert tagged_denial.allowed is False
            assert enforcer._last_hard_deny_response is None
            assert enforcer._run_hard_deny_responses["run_a"] == run_denial
            assert enforcer._lease_path_applies("run_a") is False

            outage = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )

        assert outage.allowed is False
        assert outage.warning is not None
        assert "preserving prior hard deny" in outage.warning.lower()
        assert post.call_count == 2

    def test_tagged_project_period_hard_deny_invalidates_global_allow_and_sticks(
        self,
    ) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(ALLOW_BUDGET_RESPONSE),
                _response(_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
            ],
        ) as post:
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
            )
            denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                tags={"team": "research"},
            )
            outage = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
            )

        assert denied.allowed is False
        assert outage.allowed is False
        assert post.call_count == 3
        assert not enforcer._allow_cache
        assert enforcer._last_hard_deny_response is not None

    def test_tagged_agent_run_hard_deny_remains_run_scoped(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
                tags={"team": "research"},
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )

        assert denied.allowed is False
        assert outage_b.allowed is True
        assert outage_a.allowed is False
        assert enforcer._last_hard_deny_response is None
        assert set(enforcer._run_hard_deny_responses) == {"run_a"}


@pytest.mark.unit
class TestScopedCacheAndStickyDenials:
    """Run-scoped decisions bypass global allows and preserve deny isolation."""

    def test_hot_global_allow_does_not_skip_or_get_overwritten_by_scoped_checks(self) -> None:
        enforcer = _make_legacy_enforcer()
        global_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_global"})
        run_a_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_a"})
        run_b_allow = _response({**ALLOW_BUDGET_RESPONSE, "reservation_id": "res_run_b"})

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[global_allow, run_a_allow, run_b_allow],
        ) as mock_post:
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )
            unscoped = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert mock_post.call_count == 3
        assert mock_post.call_args_list[1].kwargs["json"]["agent_run_id"] == "run_a"
        assert mock_post.call_args_list[2].kwargs["json"]["agent_run_id"] == "run_b"
        assert enforcer._allow_cache
        assert next(iter(enforcer._allow_cache.values()))[0].reservation_id == "res_global"
        assert unscoped.reservation_id is None

    def test_run_deny_is_sticky_only_for_same_run_during_outage(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

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
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )

        assert denied_a.allowed is False
        assert outage_b.allowed is True
        assert outage_a.allowed is False
        assert outage_a.warning is not None
        assert "preserving prior hard deny" in outage_a.warning.lower()

    def test_run_stopped_is_sticky_only_for_same_run_during_outage(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

        with patch.object(
            enforcer._http,
            "post",
            side_effect=[
                _response(_STOPPED_RUN_DENY_RESPONSE),
                httpx.ConnectError("unreachable"),
                httpx.ConnectError("unreachable"),
            ],
        ):
            denied_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )

        assert denied_a.allowed is False
        assert denied_a.denied_by_period == "run_stopped"
        assert outage_b.allowed is True
        assert outage_b.denied_by_period is None
        assert outage_a.allowed is False
        assert outage_a.denied_by_period == "run_stopped"
        assert enforcer._last_hard_deny_response is None

    def test_run_hard_deny_clears_older_global_project_deny_for_other_runs(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

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
                model="gpt-5.5",
                provider="openai",
            )
            run_a_denied = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
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

    def test_authoritative_allow_clears_only_same_run_sticky_deny(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

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
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )
            enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_a = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )

        assert outage_a.allowed is True
        assert outage_b.allowed is False

    def test_project_period_deny_received_in_scope_stays_globally_sticky(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

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
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage_b = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_b",
            )

        assert denied.allowed is False
        assert outage_b.allowed is False

    def test_scoped_project_alert_only_response_clears_older_run_hard_deny(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)

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
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            alert_only = enforcer.check_budget(
                estimated_input_tokens=500,
                model="gpt-5.5",
                provider="openai",
                agent_run_id="run_a",
            )
            outage = enforcer.check_budget(
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

    def test_run_sticky_denials_are_bounded_and_evict_least_recently_used(self) -> None:
        enforcer = _make_legacy_enforcer(fail_open=True, budget_mode=BudgetMode.HARD_DENY)
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
        assert result.denied_by_period is None
        assert result.deny_source is None
        assert result.deny_reason is None
        assert result.failover_tuning_allowed is None

    def test_all_fields(self) -> None:
        result = BudgetCheckResult(
            allowed=False,
            remaining_budget=0.0,
            reservation_id="res_456",
            mode=BudgetMode.HARD_DENY,
            warning="Budget exceeded",
            denied_by_period="run_stopped",
        )
        assert result.allowed is False
        assert result.reservation_id == "res_456"
        assert result.mode == BudgetMode.HARD_DENY
        assert result.denied_by_period == "run_stopped"


@pytest.mark.unit
class TestDenialReceiptAttribution:
    """Every sans-I/O denial builder stamps its structural source."""

    def test_live_hard_deny_is_server_attributed_with_exact_period(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        result = base._build_result_from_response(
            BudgetCheckResponse.model_validate(_STOPPED_RUN_DENY_RESPONSE)
        )

        assert (result.deny_source, result.deny_reason, result.denied_by_period) == (
            "server",
            "run_stopped",
            "run_stopped",
        )

    @pytest.mark.parametrize("payload", [ALLOW_BUDGET_RESPONSE, _ALERT_ONLY_DENY_RESPONSE])
    def test_allowed_result_has_no_denial_attribution(self, payload: dict[str, object]) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        result = base._build_result_from_response(BudgetCheckResponse.model_validate(payload))

        assert result.allowed is True
        assert result.deny_source is None
        assert result.deny_reason is None

    def test_sticky_replay_preserves_exact_period_and_reason(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )
        response = BudgetCheckResponse.model_validate(_STOPPED_RUN_DENY_RESPONSE)
        base._cache_response(response, agent_run_id="run_a")

        result = base._build_prior_hard_deny_unavailable_result("run_a")

        assert result is not None
        assert (result.deny_source, result.deny_reason, result.denied_by_period) == (
            "sticky_replay",
            "run_stopped",
            "run_stopped",
        )

    def test_fail_closed_builder_uses_structural_reason(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
            fail_open=False,
        )

        cold = base._build_unreachable_result(10, None)
        base._last_known_budget_limit = 1.0
        known_limit = base._build_unreachable_result(10, None)

        for result in (cold, known_limit):
            assert (result.allowed, result.deny_source, result.deny_reason) == (
                False,
                "local_enforcement",
                "control_plane_unreachable",
            )
        assert cold.denied_by_period is None
        assert known_limit.denied_by_period is None

    def test_lease_ladder_terminal_deny_preserves_reason_and_period(self) -> None:
        base = _BudgetEnforcerBase(
            api_url="https://api.test.solwyn.ai",
            api_key=VALID_API_KEY,
        )

        result = base._result_from_admission(
            "run_a",
            LeaseAdmission(
                LeaseDecision.DENY,
                mode=BudgetMode.HARD_DENY,
                reason="lease_share_exhausted",
            ),
        )

        assert (result.deny_source, result.deny_reason, result.denied_by_period) == (
            "lease_exhausted",
            "lease_share_exhausted",
            "agent_run",
        )


# ---------------------------------------------------------------------------
# Confirm cost
# ---------------------------------------------------------------------------


# Settlement moved off the enforcer: BudgetEnforcer.confirm_cost is removed
# (settlement rides reporter.report_settlement / reporter._send_confirm). The
# confirm POST + error accounting is covered by tests/unit/test_reporter.py.


@pytest.mark.unit
class TestContractDriftTaxonomy:
    """R6: a 2xx body the SDK cannot parse is contract drift, not an outage."""

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

    def test_parse_error_records_breaker_success_not_failure(self) -> None:
        breaker = self._breaker()
        enforcer = _make_enforcer(control_plane_breaker=breaker)

        with patch.object(enforcer._http, "post", return_value=self._drifted_response()):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True  # fail_open honored, same as today
        breaker.record_success.assert_called_once()
        breaker.record_failure.assert_not_called()
        breaker.release_probe.assert_called_once_with(breaker.admit.return_value)

    def test_transport_error_still_records_breaker_failure(self) -> None:
        breaker = self._breaker()
        enforcer = _make_enforcer(control_plane_breaker=breaker)

        with patch.object(enforcer._http, "post", side_effect=httpx.ConnectError("unreachable")):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        breaker.record_failure.assert_called_once()
        breaker.record_success.assert_not_called()

    def test_parse_error_logs_distinct_error_event(self, caplog) -> None:
        enforcer = _make_enforcer()
        with (
            patch.object(enforcer._http, "post", return_value=self._drifted_response()),
            caplog.at_level("ERROR", logger="solwyn.budget"),
        ):
            enforcer.check_budget(estimated_input_tokens=500, model="gpt-5.5", provider="openai")
        assert any("budget.check_response_unreadable" in r.message for r in caplog.records)

    def test_json_decode_error_is_also_drift(self) -> None:
        breaker = self._breaker()
        enforcer = _make_enforcer(control_plane_breaker=breaker)
        response = MagicMock(spec=httpx.Response)
        response.json.side_effect = ValueError("not json")

        with patch.object(enforcer._http, "post", return_value=response):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )

        assert result.allowed is True
        breaker.record_success.assert_called_once()
        breaker.record_failure.assert_not_called()

    def test_parse_error_with_fail_open_false_fails_closed(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        with patch.object(enforcer._http, "post", return_value=self._drifted_response()):
            result = enforcer.check_budget(
                estimated_input_tokens=500, model="gpt-5.5", provider="openai"
            )
        # Mirrors TestFailClosedWhenUnreachable::test_denies_when_cloud_never_reached.
        assert result.allowed is False
        assert result.deny_reason == "control_plane_unreachable"
        assert result.warning is not None
        assert "unreachable" in result.warning.lower()


def test_budget_check_result_is_pydantic_model() -> None:
    """BudgetCheckResult must be a Pydantic BaseModel, not a dataclass."""
    assert issubclass(BudgetCheckResult, BaseModel)
