"""Enforcer-side budget-lease integration (PJ-2, task S2).

The sans-I/O ladder itself is covered by ``test_lease_state.py``; these tests
pin the ENFORCER's job: translating a call into an admission, performing the
grant/renew/surrender I/O through the shared control-plane breaker, and
mapping a ``LeaseAdmission`` onto a ``BudgetCheckResult`` / the sticky-deny
machinery. HTTP is mocked at the transport with respx, so the routes assert
what actually reached the wire.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest
import respx
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID
from httpx import Response

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker

API_URL = "https://api.test.solwyn.ai"
CHECK_URL = f"{API_URL}/api/v1/budgets/check"
GRANT_URL = f"{API_URL}/api/v1/budgets/lease"
RENEW_URL = f"{API_URL}/api/v1/budgets/lease/renew"
SURRENDER_URL = f"{API_URL}/api/v1/budgets/lease/surrender"

RUN = "run_lease"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grant_payload(**overrides: object) -> dict[str, object]:
    """An eligible, allowed grant response with generous room."""
    payload: dict[str, object] = {
        "eligible": True,
        "allowed": True,
        "lease_id": "lease_1",
        "generation": 1,
        "granted_tokens": 100_000,
        "refresh_interval_s": 300.0,
        "lease_length_s": 600.0,
        "headroom_share_tokens": 50_000,
        "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
        "final_grant": False,
        "project_id": VALID_PROJECT_ID,
        "mode": "alert_only",
        "budget_limit": 100.0,
        "current_usage": 20.0,
        "remaining_budget": 80.0,
    }
    payload.update(overrides)
    return payload


def _deny_payload(**overrides: object) -> dict[str, object]:
    """An eligible run the server refuses outright (authoritative hard deny)."""
    payload: dict[str, object] = {
        "eligible": True,
        "allowed": False,
        "denied_by_period": "agent_run",
        "project_id": VALID_PROJECT_ID,
        "mode": "hard_deny",
        "budget_limit": 100.0,
        "current_usage": 100.0,
        "remaining_budget": 0.0,
    }
    payload.update(overrides)
    return payload


def _make_enforcer(**overrides: object) -> BudgetEnforcer:
    defaults: dict[str, object] = {
        "api_url": API_URL,
        "api_key": VALID_API_KEY,
        "budget_mode": BudgetMode.ALERT_ONLY,
        "fail_open": True,
        "cache_ttl": 5,
        "holder_id": "sdk_instance_1",
    }
    defaults.update(overrides)
    return BudgetEnforcer(**defaults)  # type: ignore[arg-type]


def _make_async_enforcer(**overrides: object) -> AsyncBudgetEnforcer:
    defaults: dict[str, object] = {
        "api_url": API_URL,
        "api_key": VALID_API_KEY,
        "budget_mode": BudgetMode.ALERT_ONLY,
        "fail_open": True,
        "cache_ttl": 5,
        "holder_id": "sdk_instance_1",
    }
    defaults.update(overrides)
    return AsyncBudgetEnforcer(**defaults)  # type: ignore[arg-type]


def _check(enforcer: BudgetEnforcer, call_id: str, **overrides: object):
    kwargs: dict[str, object] = {
        "estimated_input_tokens": 100,
        "model": "gpt-5.5",
        "provider": "openai",
        "agent_run_id": RUN,
        "call_id": call_id,
        "estimated_output_bound": 500,
    }
    kwargs.update(overrides)
    return enforcer.check_budget(**kwargs)  # type: ignore[arg-type]


async def _acheck(enforcer: AsyncBudgetEnforcer, call_id: str, **overrides: object):
    kwargs: dict[str, object] = {
        "estimated_input_tokens": 100,
        "model": "gpt-5.5",
        "provider": "openai",
        "agent_run_id": RUN,
        "call_id": call_id,
        "estimated_output_bound": 500,
    }
    kwargs.update(overrides)
    return await enforcer.check_budget(**kwargs)  # type: ignore[arg-type]


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    """Poll until an off-thread renewal lands (never a bare sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------------------
# Admission: grant, drawdown, kill switch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseAdmission:
    """A run-scoped call meets the lease path before the per-call check."""

    def test_first_run_call_grants_then_admits_locally(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            result = _check(enforcer, "call_1")

        assert grant.call_count == 1
        assert check.call_count == 0
        assert result.allowed is True
        assert result.lease_id == "lease_1"
        assert result.reservation_id is None
        assert result.project_id == VALID_PROJECT_ID
        assert result.remaining_budget == 80.0
        enforcer._http.close()

    def test_grant_request_declares_the_run_holder_and_chain(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            _check(
                enforcer,
                "call_1",
                fallback_providers=["anthropic"],
                fallback_models=["claude-opus-4"],
            )

        body = grant.calls[0].request.read()
        import json as _json

        payload = _json.loads(body)
        assert payload["agent_run_id"] == RUN
        assert payload["holder_id"] == "sdk_instance_1"
        assert payload["model"] == "gpt-5.5"
        assert payload["provider"] == "openai"
        assert payload["fallback_providers"] == ["anthropic"]
        assert payload["fallback_models"] == ["claude-opus-4"]
        assert payload["fail_open"] is True
        assert payload["estimated_input_tokens"] == 100
        enforcer._http.close()

    def test_n_admissions_ride_one_grant_and_never_touch_the_check_endpoint(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            renew = respx.post(RENEW_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            results = [_check(enforcer, f"call_{index}") for index in range(20)]

        assert all(result.allowed and result.lease_id == "lease_1" for result in results)
        assert grant.call_count == 1
        assert check.call_count == 0
        assert renew.call_count == 0
        enforcer._http.close()

    def test_kill_switch_off_is_the_pure_legacy_path(self) -> None:
        enforcer = _make_enforcer(lease_enabled=False)
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            result = _check(enforcer, "call_1")

        assert grant.call_count == 0
        assert check.call_count == 1
        assert result.allowed is True
        assert result.lease_id is None
        assert result.reservation_id == "res_123"
        enforcer._http.close()

    def test_non_run_traffic_never_meets_the_lease_path(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            result = enforcer.check_budget(
                estimated_input_tokens=100, model="gpt-5.5", provider="openai"
            )

        assert grant.call_count == 0
        assert check.call_count == 1
        assert result.reservation_id == "res_123"
        enforcer._http.close()

    def test_media_call_in_a_run_takes_the_legacy_path(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            result = _check(enforcer, "call_1", modality="embedding")

        assert grant.call_count == 0
        assert check.call_count == 1
        assert result.lease_id is None
        enforcer._http.close()

    def test_exhausted_grant_falls_back_to_the_per_call_check(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=600))
            )
            respx.post(RENEW_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            first = _check(enforcer, "call_1")
            second = _check(enforcer, "call_2")

        assert first.lease_id == "lease_1"
        assert second.lease_id is None
        assert second.reservation_id == "res_123"
        assert check.call_count == 1
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Server refusals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseRefusals:
    """Ineligible / capped / denied runs, and the sticky-deny handoff."""

    def test_ineligible_run_takes_the_legacy_path_without_regranting(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(
                    200,
                    json={
                        "eligible": False,
                        "ineligible_reason": "unit_priced_model",
                        "allowed": True,
                        "project_id": VALID_PROJECT_ID,
                        "mode": "alert_only",
                        "budget_limit": 100.0,
                        "current_usage": 20.0,
                        "remaining_budget": 80.0,
                    },
                )
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            results = [_check(enforcer, f"call_{index}") for index in range(3)]

        assert grant.call_count == 1
        assert check.call_count == 3
        assert all(result.lease_id is None for result in results)
        enforcer._http.close()

    def test_holder_cap_marks_the_run_ineligible(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(409, json={"detail": {"code": "lease_holder_cap_exceeded"}})
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            results = [_check(enforcer, f"call_{index}") for index in range(3)]

        assert grant.call_count == 1
        assert check.call_count == 3
        assert all(result.allowed for result in results)
        enforcer._http.close()

    def test_lease_unavailable_retries_the_grant_only_after_the_floor(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(503, json={"detail": {"code": "lease_unavailable"}})
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            for index in range(3):
                _check(enforcer, f"call_{index}")

        assert grant.call_count == 1
        assert check.call_count == 3
        enforcer._http.close()

    def test_grant_deny_is_authoritative_and_becomes_sticky(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_deny_payload()))
            check = respx.post(CHECK_URL).mock(
                side_effect=httpx.ConnectError("control plane unreachable")
            )

            denied = _check(enforcer, "call_1")
            during_outage = _check(enforcer, "call_2")

        assert denied.allowed is False
        assert denied.mode is BudgetMode.HARD_DENY
        assert denied.current_usage == 100.0
        # Step 1: the sticky deny outranks any local lease authority — the next
        # call goes back to the authoritative per-call path, never a new grant.
        assert grant.call_count == 1
        assert check.call_count == 1
        assert during_outage.allowed is False
        assert during_outage.warning is not None
        assert "preserving prior hard deny" in during_outage.warning.lower()
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Outage postures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseOutage:
    """Solwyn being unreachable never blocks a call (T2)."""

    def test_cold_start_unreachable_fail_open_admits_and_tallies_uncounted(self) -> None:
        enforcer = _make_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            result = _check(enforcer, "call_1")

        assert result.allowed is True
        assert result.warning is not None
        assert "fail-open" in result.warning.lower()
        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.uncounted_calls == 1
        assert state.uncounted_tokens == 600
        enforcer._http.close()

    def test_cold_start_unreachable_local_enforce_denies_without_a_known_limit(self) -> None:
        enforcer = _make_enforcer(fail_open=False, budget_mode=BudgetMode.HARD_DENY)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            result = _check(enforcer, "call_1")

        assert result.allowed is False
        state = enforcer._lease.state_for(RUN)
        # local_enforce never tallies uncounted spend — nothing was admitted.
        assert state is None or state.uncounted_calls == 0
        enforcer._http.close()

    def test_outage_after_the_grant_draws_the_headroom_share(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=600, name="control-plane")
        enforcer = _make_enforcer(control_plane_breaker=breaker)
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=600))
            )
            respx.post(RENEW_URL).mock(side_effect=httpx.ConnectError("down"))
            check = respx.post(CHECK_URL).mock(side_effect=httpx.ConnectError("down"))

            first = _check(enforcer, "call_1")
            # The granted remainder is now empty. Once a control-plane failure
            # opens the shared breaker the plane is BELIEVED unreachable, so the
            # ladder draws this holder's headroom share instead of blocking.
            breaker.record_failure()
            second = _check(enforcer, "call_2")

        assert first.lease_id == "lease_1"
        assert second.allowed is True
        assert second.lease_id == "lease_1"
        assert second.warning is not None
        assert "headroom share" in second.warning.lower()
        assert check.call_count == 0
        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.share_remaining_tokens == 50_000 - 600
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseSettlement:
    """Reservations are trued up on settlement and released on error paths."""

    def test_confirm_carries_the_lease_id_and_trues_up_the_reservation(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            result = _check(enforcer, "call_1")

        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000 - 600

        confirm = enforcer.build_confirm_request(
            model="gpt-5.5",
            token_details=TokenDetails(input_tokens=120, output_tokens=80),
            provider="openai",
            call_id="call_1",
            lease_id=result.lease_id,
        )

        assert confirm.lease_id == "lease_1"
        assert confirm.reservation_id is None
        assert "reservation_id" not in confirm.model_dump(mode="json")
        # 600 reserved, 200 actually spent -> 400 handed back.
        assert state.granted_remaining_tokens == 100_000 - 200
        assert state.spent_tokens_since_report == 200
        enforcer._http.close()

    def test_release_returns_an_abandoned_reservation(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            _check(enforcer, "call_1")

        enforcer.release_reservation("call_1")

        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000
        assert state.reservations == {}
        enforcer._http.close()

    def test_reservation_settled_confirms_keep_todays_wire(self) -> None:
        enforcer = _make_enforcer()
        confirm = enforcer.build_confirm_request(
            reservation_id="res_123",
            model="gpt-5.5",
            token_details=TokenDetails(input_tokens=1, output_tokens=1),
            provider="openai",
            call_id="call_x",
        )
        assert confirm.reservation_id == "res_123"
        assert confirm.lease_id is None
        assert "lease_id" not in confirm.model_dump(mode="json")
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncLeaseAdmission:
    """The async enforcer runs the same ladder over its own client."""

    async def test_first_run_call_grants_then_admits_locally(self) -> None:
        enforcer = _make_async_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            first = await _acheck(enforcer, "call_1")
            second = await _acheck(enforcer, "call_2")

        assert grant.call_count == 1
        assert check.call_count == 0
        assert first.lease_id == "lease_1"
        assert second.lease_id == "lease_1"
        await enforcer._http.aclose()

    async def test_kill_switch_off_is_the_pure_legacy_path(self) -> None:
        enforcer = _make_async_enforcer(lease_enabled=False)
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            result = await _acheck(enforcer, "call_1")

        assert grant.call_count == 0
        assert check.call_count == 1
        assert result.reservation_id == "res_123"
        await enforcer._http.aclose()

    async def test_cold_start_unreachable_fail_open_admits_and_tallies(self) -> None:
        enforcer = _make_async_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))
            result = await _acheck(enforcer, "call_1")

        assert result.allowed is True
        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.uncounted_tokens == 600
        await enforcer._http.aclose()


# ---------------------------------------------------------------------------
# Fork safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseForkReset:
    """A forked child must never inherit its parent's lease authority."""

    def test_fork_reset_drops_all_lease_state(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            _check(enforcer, "call_1")

            assert enforcer._lease.active_run_ids() == [RUN]
            enforcer._reset_after_fork_in_child()
            assert enforcer._lease.active_run_ids() == []
            assert enforcer._lease.state_for(RUN) is None
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Thread-safety scaffolding used by the renewal tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseBurst:
    """Concurrent admissions cannot jointly overrun one grant."""

    def test_concurrent_admissions_never_exceed_the_grant(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=6_000))
            )
            respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))
            # Prime the lease on this thread so the burst below only admits.
            _check(enforcer, "warmup")

            results: list[object] = []
            lock = threading.Lock()

            def worker(index: int) -> None:
                result = _check(enforcer, f"burst_{index}")
                with lock:
                    results.append(result.lease_id)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens >= 0
        # 6000 granted, 600 per call: the warmup plus exactly 9 lease admissions.
        assert results.count("lease_1") == 9
        enforcer._http.close()
