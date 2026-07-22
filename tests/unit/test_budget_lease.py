"""Enforcer-side budget-lease integration (PJ-2, task S2).

The sans-I/O ladder itself is covered by ``test_lease_state.py``; these tests
pin the ENFORCER's job: translating a call into an admission, performing the
grant/renew/surrender I/O through the shared control-plane breaker, and
mapping a ``LeaseAdmission`` onto a ``BudgetCheckResult`` / the sticky-deny
machinery. HTTP is mocked at the transport with respx, so the routes assert
what actually reached the wire.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID, call_uuid
from httpx import Response

import solwyn as solwyn_pkg
from solwyn import _lifecycle
from solwyn._base import MediaSurfaceSpec
from solwyn._lease import INELIGIBLE_RETRY_AFTER_S
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode
from solwyn.budget import AsyncBudgetEnforcer, BudgetCheckResult, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.client import Solwyn
from solwyn.exceptions import ProviderUnavailableError

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


class _MaxJitter(random.Random):
    """An RNG whose ``uniform`` always returns the ceiling.

    The ledger samples full-jitter backoffs from ``uniform(0, ceiling)``; a
    near-zero sample would let the very next admission retry a renewal the
    test expects to be suppressed. Pinning the ceiling makes the backoff
    assertions deterministic (the ledger takes an ``rng`` seam for exactly
    this reason — the enforcer just does not plumb it through).
    """

    def uniform(self, a: float, b: float) -> float:
        return b


class _FakeClock:
    """A controllable stand-in for the ``time`` module inside ``budget.py``.

    ``budget.py`` reads ``time.monotonic()`` at call time, so swapping the
    module reference drives every deadline the enforcer computes (retry
    floors, cache expiry) without touching the real clock.
    """

    def __init__(self) -> None:
        self._now = time.monotonic()

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


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

            result = _check(enforcer, call_uuid("call_1"))

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

            result = _check(enforcer, call_uuid("call_1"))

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
        clock = _FakeClock()
        with respx.mock, patch("solwyn.budget.time", clock):
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(503, json={"detail": {"code": "lease_unavailable"}})
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            for index in range(3):
                _check(enforcer, f"call_{index}")
            suppressed = grant.call_count

            # Past the retry floor the run is eligible to ask again — the 503
            # parks the grant for 30s, it does not disable the run for good.
            clock.advance(INELIGIBLE_RETRY_AFTER_S + 1.0)
            _check(enforcer, "call_after_floor")

        assert suppressed == 1
        assert grant.call_count == 2
        assert check.call_count == 4
        enforcer._http.close()

    def test_a_drifted_lease_id_degrades_to_the_legacy_path(self) -> None:
        # A server lease id longer than the wire bound must be caught where
        # every other malformed grant body is — at install, degrading to the
        # per-call path — and must NEVER raise out of check_budget().
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(lease_id="x" * 65))
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            results = [_check(enforcer, f"call_{index}") for index in range(3)]

        assert all(result.allowed and result.lease_id is None for result in results)
        assert grant.call_count == 1  # then the 30s ineligible window holds
        assert check.call_count == 3
        assert enforcer._lease.lease_id_for(RUN) is None
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

            result = _check(enforcer, call_uuid("call_1"))

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

            result = _check(enforcer, call_uuid("call_1"))

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
# Renewals (always off the caller's thread)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseRenewal:
    """Renew ahead of need; a renewal never sits on a customer call."""

    def test_depletion_renews_once_and_admissions_keep_flowing(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            # Sized above the original: nothing settles in this harness, so
            # all three calls are still in flight when the renewal lands and
            # their bounds are carried onto the replacement grant. A 2_400
            # renewal would arrive 1_800 committed — 75% depleted on arrival,
            # and honestly due for another renewal on the spot.
            renew = respx.post(RENEW_URL).mock(
                return_value=Response(200, json=_grant_payload(generation=2, granted_tokens=9_600))
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            # 600 per call against a 2_400 grant: the 3rd call is 75% depleted.
            for index in range(3):
                assert _check(enforcer, f"call_{index}").lease_id == "lease_1"
            assert _wait_for(lambda: renew.call_count == 1)
            state = enforcer._lease.state_for(RUN)
            assert state is not None
            # The worker applies OFF this thread — wait for the install before
            # asserting on it (the admission below needs no such wait).
            assert _wait_for(lambda: state.generation == 2)
            # The renewal restored the grant, so the 4th call still admits.
            assert _check(enforcer, "call_3").lease_id == "lease_1"

        assert grant.call_count == 1
        assert renew.call_count == 1
        assert check.call_count == 0
        import json as _json

        payload = _json.loads(renew.calls[0].request.read())
        assert payload["lease_id"] == "lease_1"
        assert payload["holder_id"] == "sdk_instance_1"
        assert payload["generation"] == 1
        assert state.generation == 2
        enforcer._http.close()

    def test_a_slow_renewal_never_delays_admission(self) -> None:
        enforcer = _make_enforcer()
        release = threading.Event()
        in_flight = threading.Event()

        def _slow_renewal(request: httpx.Request) -> Response:
            in_flight.set()
            release.wait(timeout=5.0)
            return Response(200, json=_grant_payload(generation=2))

        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=6_000))
            )
            renew = respx.post(RENEW_URL).mock(side_effect=_slow_renewal)
            respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))

            # 600 per call against 6_000: the 8th call crosses 75% depletion.
            for index in range(8):
                _check(enforcer, f"warm_{index}")
            assert in_flight.wait(timeout=3.0)

            # The renewal is parked mid-flight; admissions must not wait on it.
            started = time.monotonic()
            for index in range(2):
                assert _check(enforcer, f"fast_{index}").lease_id == "lease_1"
            elapsed = time.monotonic() - started
            release.set()
            assert _wait_for(lambda: renew.call_count == 1)

        assert elapsed < 0.5
        # ...and no second renewal was stacked while the first was in flight.
        assert renew.call_count == 1
        enforcer._http.close()

    @pytest.mark.parametrize("status", [404, 409])
    def test_a_lost_lease_is_dropped_and_regranted(self, status: int) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            grant = respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            renew = respx.post(RENEW_URL).mock(return_value=Response(status, json={}))
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            for index in range(3):
                _check(enforcer, f"call_{index}")
            assert _wait_for(lambda: renew.call_count == 1)
            assert _wait_for(lambda: enforcer._lease.lease_id_for(RUN) is None)

            # The next call has no lease at all: it re-grants rather than
            # drawing on authority the server no longer recognizes.
            assert _check(enforcer, "call_after").lease_id == "lease_1"

        assert grant.call_count == 2
        assert check.call_count == 0
        enforcer._http.close()

    def test_a_failed_renewal_backs_off_instead_of_retrying_per_call(self) -> None:
        enforcer = _make_enforcer()
        enforcer._lease._rng = _MaxJitter()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            renew = respx.post(RENEW_URL).mock(side_effect=httpx.ConnectError("down"))
            respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))

            for index in range(3):
                _check(enforcer, f"call_{index}")
            assert _wait_for(lambda: renew.call_count == 1)
            state = enforcer._lease.state_for(RUN)
            assert state is not None
            # The worker records its verdict off this thread; wait for it, or
            # the "no second renewal" assertion below proves nothing.
            assert _wait_for(lambda: state.consecutive_failures == 1)
            # Every later admission is still >=75% depleted; the backoff must
            # stop them turning into a renewal per call.
            for index in range(3, 6):
                _check(enforcer, f"call_{index}")

        assert renew.call_count == 1
        assert state.lease_id == "lease_1"
        assert state.consecutive_failures == 1
        assert state.next_attempt_at > 0.0
        enforcer._http.close()

    def test_a_denied_renewal_becomes_sticky(self) -> None:
        enforcer = _make_enforcer(budget_mode=BudgetMode.HARD_DENY)
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            renew = respx.post(RENEW_URL).mock(return_value=Response(200, json=_deny_payload()))
            check = respx.post(CHECK_URL).mock(
                side_effect=httpx.ConnectError("control plane unreachable")
            )

            for index in range(3):
                _check(enforcer, f"call_{index}")
            assert _wait_for(lambda: renew.call_count == 1)
            assert _wait_for(lambda: RUN in enforcer._run_hard_deny_responses)

            after = _check(enforcer, "call_after")

        assert after.allowed is False
        assert after.warning is not None
        assert "preserving prior hard deny" in after.warning.lower()
        assert check.call_count == 1
        enforcer._http.close()

    def test_a_stale_generation_renewal_arms_the_backoff_and_keeps_the_lease(self) -> None:
        # The frozen ledger returns STALE *before* clearing renewal_in_flight,
        # so the enforcer must clear it under a backoff — otherwise the run
        # never renews again and silently coasts to expiry.
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            # The renewal re-delivers the CURRENT generation (1), which cannot
            # install over itself.
            renew = respx.post(RENEW_URL).mock(
                return_value=Response(200, json=_grant_payload(generation=1))
            )
            respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))

            for index in range(3):
                _check(enforcer, f"call_{index}")
            assert _wait_for(lambda: renew.call_count == 1)
            state = enforcer._lease.state_for(RUN)
            assert state is not None
            assert _wait_for(lambda: state.consecutive_failures == 1)
            assert state.renewal_in_flight is False

        assert state.lease_id == "lease_1"
        assert state.generation == 1
        assert state.granted_tokens == 2_400  # nothing was re-installed
        assert state.consecutive_failures == 1
        assert state.next_attempt_at > 0.0
        assert renew.call_count == 1
        enforcer._http.close()

    def test_a_renewal_worker_that_cannot_start_never_wedges_the_lease(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            renew = respx.post(RENEW_URL).mock(return_value=Response(200, json=_grant_payload()))
            respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))

            for index in range(2):
                _check(enforcer, f"warm_{index}")
            with patch.object(
                threading.Thread, "start", side_effect=RuntimeError("can't start new thread")
            ):
                # Thread exhaustion must not raise into the customer's call...
                assert _check(enforcer, "call_2").allowed is True

        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert renew.call_count == 0
        # ...and must not leave the lease believing a renewal is in flight.
        assert state.renewal_in_flight is False
        assert state.consecutive_failures == 1
        assert state.next_attempt_at > 0.0
        enforcer._http.close()

    async def test_async_renewal_runs_as_a_task(self) -> None:
        enforcer = _make_async_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(
                return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
            )
            renew = respx.post(RENEW_URL).mock(
                return_value=Response(200, json=_grant_payload(generation=2, granted_tokens=2_400))
            )
            check = respx.post(CHECK_URL).mock(
                return_value=Response(200, json=ALLOW_BUDGET_RESPONSE)
            )

            for index in range(3):
                await _acheck(enforcer, f"call_{index}")
            # The task is scheduled, not awaited: let the loop run it.
            for _ in range(50):
                if renew.call_count == 1:
                    break
                await asyncio.sleep(0.005)

        assert renew.call_count == 1
        assert check.call_count == 0
        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.generation == 2
        await enforcer._http.aclose()


# ---------------------------------------------------------------------------
# Surrender
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLeaseSurrender:
    """close() hands the float back instead of letting it expire."""

    def test_close_surrenders_every_held_lease(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            surrender = respx.post(SURRENDER_URL).mock(return_value=Response(200, json={}))

            _check(enforcer, "call_1")
            enforcer.close()

            assert surrender.call_count == 1
            import json as _json

            payload = _json.loads(surrender.calls[0].request.read())
            assert payload["lease_id"] == "lease_1"
            assert payload["holder_id"] == "sdk_instance_1"
            assert payload["generation"] == 1
            # Surrendered leases are forgotten, so the exit hook cannot send a
            # second release for the same lease.
            assert enforcer._lease.active_run_ids() == []

    def test_close_without_a_lease_sends_nothing(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            surrender = respx.post(SURRENDER_URL).mock(return_value=Response(200, json={}))
            enforcer.close()
            assert surrender.call_count == 0

    def test_a_failed_surrender_never_raises_out_of_close(self) -> None:
        enforcer = _make_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            respx.post(SURRENDER_URL).mock(side_effect=httpx.ConnectError("down"))

            _check(enforcer, "call_1")
            enforcer.close()

    async def test_async_close_surrenders_every_held_lease(self) -> None:
        enforcer = _make_async_enforcer()
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            surrender = respx.post(SURRENDER_URL).mock(return_value=Response(200, json={}))

            await _acheck(enforcer, "call_1")
            await enforcer.close()

            assert surrender.call_count == 1

    def test_close_bounds_all_renewal_joins_by_one_shared_deadline(self) -> None:
        # Three runs with parked renewals must cost ONE deadline at close, not
        # one per run (parity with the async close's single asyncio.wait).
        enforcer = _make_enforcer()
        release = threading.Event()
        in_flight = threading.Semaphore(0)

        def _parked_renewal(request: httpx.Request) -> Response:
            in_flight.release()
            release.wait(timeout=10.0)
            return Response(200, json=_grant_payload(generation=2))

        try:
            with respx.mock:
                respx.post(GRANT_URL).mock(
                    return_value=Response(200, json=_grant_payload(granted_tokens=2_400))
                )
                respx.post(RENEW_URL).mock(side_effect=_parked_renewal)
                respx.post(SURRENDER_URL).mock(return_value=Response(200, json={}))
                respx.post(CHECK_URL).mock(return_value=Response(200, json=ALLOW_BUDGET_RESPONSE))

                for run in ("run_a", "run_b", "run_c"):
                    for index in range(3):
                        _check(enforcer, f"{run}_{index}", agent_run_id=run)
                for _ in range(3):
                    assert in_flight.acquire(timeout=3.0)

                started = time.monotonic()
                enforcer.close()
                elapsed = time.monotonic() - started
        finally:
            release.set()

        assert elapsed < 2.0  # one 1s deadline, not three

    def test_exit_surrender_is_bounded_by_wall_clock_not_socket_timeouts(self) -> None:
        # A trickling server must not hold interpreter exit past the budget:
        # the drain runs on a daemon worker JOINED at the deadline, the same
        # bound the reporter's exit flush uses.
        enforcer = _make_enforcer(holder_id="slow_holder")
        release = threading.Event()

        def _trickle(request: httpx.Request) -> Response:
            release.wait(timeout=10.0)
            return Response(200, json={})

        try:
            with respx.mock:
                respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
                respx.post(SURRENDER_URL).mock(side_effect=_trickle)

                _check(enforcer, "call_1")
                started = time.monotonic()
                _lifecycle._exit_surrender_all()
                elapsed = time.monotonic() - started
        finally:
            release.set()

        assert elapsed < _lifecycle._EXIT_SURRENDER_BUDGET_S + 1.5
        enforcer._http.close()

    def test_exit_hook_surrenders_a_lease_the_process_still_holds(self) -> None:
        enforcer = _make_enforcer(holder_id="exit_holder")
        with respx.mock:
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))
            surrender = respx.post(SURRENDER_URL).mock(return_value=Response(200, json={}))

            _check(enforcer, "call_1")
            _lifecycle._exit_surrender_all()

            import json as _json

            # The hook drains every live holder in the process, so assert on
            # THIS enforcer's release rather than the global call count.
            payloads = [_json.loads(call.request.read()) for call in surrender.calls]
            assert {"lease_id": "lease_1", "holder_id": "exit_holder"} in [
                {"lease_id": p["lease_id"], "holder_id": p["holder_id"]} for p in payloads
            ]
            assert enforcer._lease.active_run_ids() == []
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
            result = _check(enforcer, call_uuid("call_1"))

        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000 - 600

        confirm = enforcer.build_confirm_request(
            model="gpt-5.5",
            token_details=TokenDetails(input_tokens=120, output_tokens=80),
            provider="openai",
            call_id=call_uuid("call_1"),
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
            call_id=call_uuid("call_x"),
        )
        assert confirm.reservation_id == "res_123"
        assert confirm.lease_id is None
        assert "lease_id" not in confirm.model_dump(mode="json")
        enforcer._http.close()


# ---------------------------------------------------------------------------
# Client plumbing
# ---------------------------------------------------------------------------


def _openai_client() -> tuple[MagicMock, object]:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
    client.chat.completions.create.return_value = response
    return client, response


def _build_solwyn(client: MagicMock, **overrides: object) -> Solwyn:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, api_key=VALID_API_KEY, **overrides)  # type: ignore[arg-type]
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


def _lease_result() -> BudgetCheckResult:
    return BudgetCheckResult(
        allowed=True,
        remaining_budget=80.0,
        project_id=VALID_PROJECT_ID,
        reservation_id=None,
        lease_id="lease_1",
        budget_limit=100.0,
        current_usage=20.0,
    )


@pytest.mark.unit
class TestClientLeasePlumbing:
    """The call's own bound, join key, and settlement key reach the enforcer."""

    def test_the_client_binds_the_ledger_to_its_instance_id_and_config(self) -> None:
        client, _ = _openai_client()
        solwyn = _build_solwyn(client, lease_output_bound_default=777)

        assert solwyn._budget._lease.holder_id == solwyn._sdk_instance_id
        assert solwyn._budget._lease.enabled is True
        assert solwyn._budget._lease.output_bound_default == 777

        disabled = _build_solwyn(client, lease_enabled=False)
        assert disabled._budget._lease.enabled is False

        for wrapper in (solwyn, disabled):
            wrapper._reporter._http.close()
            wrapper._budget._http.close()

    @pytest.mark.parametrize("cap_field", ["max_tokens", "max_completion_tokens"])
    def test_the_calls_output_cap_becomes_the_lease_bound(self, cap_field: str) -> None:
        client, _ = _openai_client()
        solwyn = _build_solwyn(client)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._reporter, "report_settlement"),
            solwyn_pkg.run("bounded-job"),
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
                **{cap_field: 256},
            )

        assert check.call_args.kwargs["estimated_output_bound"] == 256
        assert check.call_args.kwargs["call_id"]
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_an_uncapped_call_sends_no_bound(self) -> None:
        client, _ = _openai_client()
        solwyn = _build_solwyn(client)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._reporter, "report_settlement"),
            solwyn_pkg.run("uncapped-job"),
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "Hello"}]
            )

        assert check.call_args.kwargs["estimated_output_bound"] is None
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_a_lease_funded_success_settles_on_the_lease(self) -> None:
        client, _ = _openai_client()
        solwyn = _build_solwyn(client)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._reporter, "report_settlement") as settle,
            solwyn_pkg.run("leased-job"),
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "Hello"}]
            )

        settle.assert_called_once()
        confirm = settle.call_args[0][0]
        assert confirm.lease_id == "lease_1"
        assert confirm.reservation_id is None
        assert confirm.call_id == check.call_args.kwargs["call_id"]
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_a_failed_call_hands_the_reservation_back(self) -> None:
        client, _ = _openai_client()
        client.chat.completions.create.side_effect = RuntimeError("fail fast")
        solwyn = _build_solwyn(client)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._budget, "release_reservation") as release,
            patch.object(solwyn._reporter, "report"),
            solwyn_pkg.run("doomed-job"),
            pytest.raises(RuntimeError),
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "Hello"}]
            )

        release.assert_called_once_with(check.call_args.kwargs["call_id"])
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_no_provider_can_serve_hands_the_reservation_back(self) -> None:
        client, _ = _openai_client()
        solwyn = _build_solwyn(client)
        # Every candidate is filtered out: the walk raises before any dispatch.
        solwyn._get_circuit_breaker("openai").record_failure()
        solwyn._get_circuit_breaker("openai").record_failure()
        solwyn._get_circuit_breaker("openai").record_failure()

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._budget, "release_reservation") as release,
            patch.object(solwyn._reporter, "report"),
            solwyn_pkg.run("unavailable-job"),
            pytest.raises(ProviderUnavailableError),
        ):
            solwyn.chat.completions.create(
                model="gpt-5.5", messages=[{"role": "user", "content": "Hello"}]
            )

        release.assert_called_once_with(check.call_args.kwargs["call_id"])
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_media_calls_carry_the_join_key(self) -> None:
        client, _ = _openai_client()
        client.embeddings.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=42)
        )
        solwyn = _build_solwyn(client)
        spec = MediaSurfaceSpec(
            surface="embeddings",
            modality="embedding",
            extract_usage=lambda response: TokenDetails(input_tokens=response.usage.prompt_tokens),
            measure_request=lambda _kwargs: None,
        )

        def _route(surface, sdk_client, kwargs, *, timeout, max_retries):
            return sdk_client.embeddings.create, dict(kwargs)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._reporter, "report_settlement"),
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route),
            solwyn_pkg.run("media-job"),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hello")

        assert check.call_args.kwargs["call_id"]
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_a_failed_media_call_hands_the_reservation_back(self) -> None:
        client, _ = _openai_client()
        client.embeddings.create.side_effect = RuntimeError("boom")
        solwyn = _build_solwyn(client)
        spec = MediaSurfaceSpec(
            surface="embeddings",
            modality="embedding",
            extract_usage=lambda _response: None,
            measure_request=lambda _kwargs: None,
        )

        def _route(surface, sdk_client, kwargs, *, timeout, max_retries):
            return sdk_client.embeddings.create, dict(kwargs)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_lease_result()) as check,
            patch.object(solwyn._budget, "release_reservation") as release,
            patch.object(solwyn._reporter, "report"),
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route),
            solwyn_pkg.run("media-doomed"),
            pytest.raises(RuntimeError),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hello")

        release.assert_called_once_with(check.call_args.kwargs["call_id"])
        solwyn._reporter._http.close()
        solwyn._budget._http.close()


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

    def test_fork_reset_takes_a_fresh_holder_identity(self) -> None:
        # The client's _sdk_instance_id survives a fork unchanged, so the
        # enforcer must mint its own holder id in the child: core releases a
        # same-(project, run, holder) active lease as stale when a grant lands,
        # so a child re-granting under the PARENT's id would kill the parent's
        # live lease — and the parent's next regrant would kill the child's.
        enforcer = _make_enforcer()
        parent_holder = enforcer._lease.holder_id

        enforcer._reset_after_fork_in_child()

        assert enforcer._lease.holder_id != parent_holder
        assert enforcer._lease.holder_id
        enforcer._http.close()

    async def test_async_fork_reset_takes_a_fresh_holder_identity(self) -> None:
        enforcer = _make_async_enforcer()
        parent_holder = enforcer._lease.holder_id

        enforcer._reset_after_fork_in_child()

        assert enforcer._lease.holder_id != parent_holder
        await enforcer._http.aclose()


# ---------------------------------------------------------------------------
# Uncounted-mode telemetry (§8: warn on episode entry, then at most 1/30s)
# ---------------------------------------------------------------------------


def _uncounted_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Just the uncounted-mode telemetry lines (the grant failures also warn)."""
    return [r.getMessage() for r in caplog.records if "lease.uncounted" in r.getMessage()]


@pytest.mark.unit
class TestUncountedTelemetry:
    """Fail-open uncounted mode is loud on entry, then rate-limited (never silent)."""

    def test_entry_to_an_uncounted_episode_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                result = _check(enforcer, call_uuid("call_1"))

        assert result.allowed is True
        messages = _uncounted_records(caplog)
        assert len(messages) == 1, f"entry to uncounted mode must warn exactly once: {messages}"
        assert "lease.uncounted_entry" in messages[0]
        enforcer._http.close()

    def test_a_second_uncounted_call_inside_the_window_stays_quiet(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                first = _check(enforcer, call_uuid("call_1"))
                second = _check(enforcer, call_uuid("call_2"))

        assert first.allowed is True and second.allowed is True
        messages = _uncounted_records(caplog)
        assert len(messages) == 1, f"the 30s rate limit must suppress the second: {messages}"
        # Both calls are still tallied — only the LOG is rate-limited.
        state = enforcer._lease.state_for(RUN)
        assert state is not None
        assert state.uncounted_calls == 2
        enforcer._http.close()

    def test_a_new_episode_warns_on_entry_again(self, caplog: pytest.LogCaptureFixture) -> None:
        # An installed grant ENDS the episode; the next lease death is a new
        # one, and must be as loud as the first.
        enforcer = _make_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(
                side_effect=[
                    httpx.ConnectError("down"),
                    Response(200, json=_grant_payload()),
                    httpx.ConnectError("down"),
                ]
            )

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                _check(enforcer, call_uuid("call_1"))
                covered = _check(enforcer, call_uuid("call_2"))
                # The lease dies (404 drop / expiry) while the plane is down.
                enforcer._lease.drop(RUN)
                _check(enforcer, call_uuid("call_3"))

        assert covered.lease_id is not None
        messages = _uncounted_records(caplog)
        assert len(messages) == 2, f"a fresh episode must warn on entry: {messages}"
        assert all("lease.uncounted_entry" in message for message in messages)
        enforcer._http.close()

    def test_the_rate_limited_line_names_the_still_uncounted_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Past the window the episode warns again, as a CONTINUING line: an
        # hour-long outage stays visible without one line per call.
        clock = _FakeClock()
        enforcer = _make_enforcer(fail_open=True)
        with respx.mock, patch("solwyn.budget.time", clock):
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                _check(enforcer, call_uuid("call_1"))
                clock.advance(31.0)
                _check(enforcer, call_uuid("call_2"))

        messages = _uncounted_records(caplog)
        assert len(messages) == 2, f"past 30s the episode warns again: {messages}"
        assert "lease.uncounted_entry" in messages[0]
        assert "lease.uncounted_continuing" in messages[1]
        enforcer._http.close()

    def test_the_expiry_ladders_uncounted_admission_warns_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The other entry point: a lease that ran past its deadline while the
        # plane is believed down (§4 step 6), not a cold start.
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=600, name="control-plane")
        enforcer = _make_enforcer(fail_open=True, control_plane_breaker=breaker)
        clock = _FakeClock()
        with respx.mock, patch("solwyn.budget.time", clock):
            respx.post(GRANT_URL).mock(return_value=Response(200, json=_grant_payload()))

            covered = _check(enforcer, call_uuid("call_1"))
            breaker.record_failure()  # the plane is now believed unreachable
            clock.advance(601.0)  # ... and the lease outlived its deadline

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                expired = _check(enforcer, call_uuid("call_2"))

        assert covered.lease_id is not None
        assert expired.allowed is True
        assert expired.lease_id is None, "an uncounted admission settles against nothing"
        messages = _uncounted_records(caplog)
        assert len(messages) == 1, f"expiry into fail-open must warn on entry: {messages}"
        assert "lease.uncounted_entry" in messages[0]
        assert "expired_fail_open" in messages[0]
        enforcer._http.close()

    async def test_the_async_twin_warns_on_entry_and_rate_limits(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        enforcer = _make_async_enforcer(fail_open=True)
        with respx.mock:
            respx.post(GRANT_URL).mock(side_effect=httpx.ConnectError("down"))

            with caplog.at_level("WARNING", logger="solwyn.budget"):
                await _acheck(enforcer, call_uuid("call_1"))
                await _acheck(enforcer, call_uuid("call_2"))

        messages = _uncounted_records(caplog)
        assert len(messages) == 1, f"the async path shares the one discipline: {messages}"
        assert "lease.uncounted_entry" in messages[0]
        await enforcer._http.aclose()


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
