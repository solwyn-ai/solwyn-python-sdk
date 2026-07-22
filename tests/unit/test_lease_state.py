"""Unit tests for the sans-I/O budget-lease ledger (PJ-2, DoD 3 + DoD 8).

The ledger is the pure decision core the budget enforcers drive: it owns the
admission ladder, the atomic reservation math, fencing, and the renewal
bookkeeping. Every test here drives it directly — no HTTP, no threads except
the burst-atomicity test, which holds an enforcer-style lock exactly as the
sync enforcer does.
"""

from __future__ import annotations

import random
import threading
from typing import Any

import pytest

from solwyn._lease import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    INELIGIBLE_RETRY_AFTER_S,
    REFRESH_JITTER_MAX,
    REFRESH_JITTER_MIN,
    RESERVATION_MAX_AGE_S,
    GrantOutcome,
    LeaseAdmission,
    LeaseDecision,
    LeaseLedger,
    backoff_ceiling,
)
from solwyn._types import BudgetMode, LeaseGrantResponse

RUN = "run_pj2"
HOLDER = "sdk-instance-1"


def _response(**overrides: Any) -> LeaseGrantResponse:
    """A full eligible+allowed grant response."""
    payload: dict[str, Any] = {
        "eligible": True,
        "allowed": True,
        "lease_id": "lse_abc",
        "generation": 1,
        "granted_tokens": 100_000,
        "refresh_interval_s": 15.0,
        "lease_length_s": 120.0,
        "headroom_share_tokens": 500_000,
        "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
        "final_grant": False,
        "project_id": "proj_1",
        "mode": "alert_only",
        "budget_limit": 100.0,
        "current_usage": 10.0,
        "remaining_budget": 90.0,
    }
    payload.update(overrides)
    return LeaseGrantResponse.model_validate(payload)


class _FixedRandom(random.Random):
    """Deterministic uniform() for jitter/backoff assertions."""

    def __init__(self, ratio: float = 1.0) -> None:
        super().__init__(0)
        self.ratio = ratio

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.ratio


def _ledger(**kwargs: Any) -> LeaseLedger:
    kwargs.setdefault("holder_id", HOLDER)
    kwargs.setdefault("rng", _FixedRandom(1.0))
    return LeaseLedger(**kwargs)


def _granted_ledger(now: float = 1_000.0, **response_overrides: Any) -> LeaseLedger:
    """A ledger holding a live lease for RUN, installed at ``now``."""
    ledger = _ledger()
    outcome = ledger.apply_grant_response(
        RUN, _response(**response_overrides), now=now, declared_models=["gpt-5.5"]
    )
    if outcome is not GrantOutcome.APPLIED:
        raise AssertionError(f"fixture grant not applied: {outcome}")
    return ledger


def _admit(ledger: LeaseLedger, **kwargs: Any) -> LeaseAdmission:
    params: dict[str, Any] = {
        "run_id": RUN,
        "call_id": "call-1",
        "estimated_input_tokens": 1_000,
        "model": "gpt-5.5",
        "now": 1_001.0,
        "breaker_open": False,
    }
    params.update(kwargs)
    return ledger.admit(**params)


@pytest.mark.unit
class TestAdmissionGates:
    """Steps 2-3: kill switch, locally ineligible calls, server-ineligible runs."""

    def test_kill_switch_off_always_takes_the_legacy_path(self) -> None:
        # Arrange — a perfectly good live lease...
        ledger = _granted_ledger()
        ledger.enabled = False

        # Act
        admission = _admit(ledger)

        # Assert — ...is never consulted while the kill switch is off.
        assert admission.decision is LeaseDecision.LEGACY_CHECK
        assert admission.reserved_tokens == 0

    def test_non_text_modality_takes_the_legacy_path(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, modality="image")

        assert admission.decision is LeaseDecision.LEGACY_CHECK

    def test_estimated_media_takes_the_legacy_path(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, has_estimated_media=True)

        assert admission.decision is LeaseDecision.LEGACY_CHECK

    def test_model_outside_declared_set_takes_the_legacy_path(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, model="claude-4")

        assert admission.decision is LeaseDecision.LEGACY_CHECK

    def test_fallback_model_outside_declared_set_takes_the_legacy_path(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, fallback_models=["claude-4"])

        assert admission.decision is LeaseDecision.LEGACY_CHECK

    def test_declared_chain_is_admitted(self) -> None:
        ledger = _ledger()
        ledger.apply_grant_response(
            RUN, _response(), now=1_000.0, declared_models=["gpt-5.5", "claude-4"]
        )

        admission = _admit(ledger, fallback_models=["claude-4"])

        assert admission.decision is LeaseDecision.ADMIT_LOCAL

    def test_server_ineligible_run_takes_the_legacy_path(self) -> None:
        ledger = _ledger()

        ledger.apply_grant_response(
            RUN, _response(eligible=False, ineligible_reason="unit_priced_model"), now=1_000.0
        )

        assert _admit(ledger).decision is LeaseDecision.LEGACY_CHECK
        # Permanent: no retry window reopens it.
        assert _admit(ledger, now=1_000_000.0).decision is LeaseDecision.LEGACY_CHECK

    def test_temporarily_ineligible_run_retries_a_grant_after_30s(self) -> None:
        # Arrange — a 503 lease_unavailable marks the run ineligible with a
        # bounded retry (legacy per-call path meanwhile).
        ledger = _ledger()
        ledger.mark_ineligible(RUN, now=1_000.0, retry_after=INELIGIBLE_RETRY_AFTER_S)

        # Act / Assert
        assert _admit(ledger, now=1_029.0).decision is LeaseDecision.LEGACY_CHECK
        assert _admit(ledger, now=1_030.0).decision is LeaseDecision.NEED_GRANT


@pytest.mark.unit
class TestGrantApplication:
    """apply_grant_response: install, fencing, ineligible, deny, malformed."""

    def test_cold_start_has_no_state_and_needs_a_grant(self) -> None:
        ledger = _ledger()

        admission = _admit(ledger)

        assert admission.decision is LeaseDecision.NEED_GRANT
        assert ledger.lease_id_for(RUN) is None

    def test_grant_installs_counters_timers_and_posture(self) -> None:
        ledger = _ledger()

        outcome = ledger.apply_grant_response(
            RUN, _response(), now=1_000.0, declared_models=["gpt-5.5"]
        )

        assert outcome is GrantOutcome.APPLIED
        assert ledger.lease_id_for(RUN) == "lse_abc"
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.generation == 1
        assert state.granted_remaining_tokens == 100_000
        assert state.share_remaining_tokens == 500_000
        # Timers are monotonic durations measured from receipt, never wall clock.
        assert state.lease_deadline == 1_120.0
        assert state.posture_mode is BudgetMode.ALERT_ONLY
        assert state.on_unreachable == "fail_open"

    def test_refresh_deadline_carries_client_jitter_within_bounds(self) -> None:
        # Arrange — real RNG, many grants: every refresh deadline must land in
        # [now + interval*0.85, now + interval*1.15] and must not be constant.
        deadlines = set()
        for seed in range(200):
            ledger = LeaseLedger(holder_id=HOLDER, rng=random.Random(seed))
            ledger.apply_grant_response(RUN, _response(), now=1_000.0)
            state = ledger.state_for(RUN)
            assert state is not None
            deadlines.add(state.refresh_deadline)

        # Assert
        assert min(deadlines) >= 1_000.0 + 15.0 * REFRESH_JITTER_MIN
        assert max(deadlines) <= 1_000.0 + 15.0 * REFRESH_JITTER_MAX
        assert len(deadlines) > 1

    def test_stale_generation_is_ignored(self) -> None:
        # Arrange
        ledger = _granted_ledger()
        ledger.apply_grant_response(RUN, _response(generation=2, granted_tokens=7), now=1_010.0)

        # Act — a delayed generation-1 response lands after generation 2.
        outcome = ledger.apply_grant_response(
            RUN, _response(generation=1, granted_tokens=100_000), now=1_020.0
        )

        # Assert
        assert outcome is GrantOutcome.STALE
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.generation == 2
        assert state.granted_remaining_tokens == 7

    def test_duplicate_generation_is_ignored(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger)  # draw the remainder down

        outcome = ledger.apply_grant_response(RUN, _response(generation=1), now=1_010.0)

        assert outcome is GrantOutcome.STALE
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens < 100_000

    def test_deny_response_drops_the_lease_and_signals_the_enforcer(self) -> None:
        ledger = _granted_ledger()

        outcome = ledger.apply_grant_response(
            RUN,
            _response(
                allowed=False,
                denied_by_period="agent_run",
                lease_id=None,
                generation=None,
                granted_tokens=None,
                refresh_interval_s=None,
                lease_length_s=None,
                headroom_share_tokens=None,
                posture=None,
                final_grant=None,
            ),
            now=1_010.0,
        )

        assert outcome is GrantOutcome.DENIED
        assert ledger.lease_id_for(RUN) is None

    def test_ineligible_response_marks_the_run_and_drops_the_lease(self) -> None:
        ledger = _granted_ledger()

        outcome = ledger.apply_grant_response(
            RUN, _response(eligible=False, ineligible_reason="scoped_rules_present"), now=1_010.0
        )

        assert outcome is GrantOutcome.INELIGIBLE
        assert ledger.lease_id_for(RUN) is None

    def test_malformed_allowed_response_is_not_installed(self) -> None:
        # An eligible+allowed response missing the lease fields is a contract
        # break: never install half a lease, and do not hot-loop grant calls.
        ledger = _ledger()

        outcome = ledger.apply_grant_response(RUN, _response(posture=None), now=1_000.0)

        assert outcome is GrantOutcome.MALFORMED
        assert ledger.lease_id_for(RUN) is None
        assert _admit(ledger, now=1_010.0).decision is LeaseDecision.LEGACY_CHECK
        assert _admit(ledger, now=1_031.0).decision is LeaseDecision.NEED_GRANT

    def test_zero_token_grant_falls_through_to_the_legacy_path(self) -> None:
        # granted_tokens=0 (alert_only past cap) is a live lease with an empty
        # wallet: an empty wallet is not a denial — the server stays authoritative.
        ledger = _granted_ledger(granted_tokens=0)

        admission = _admit(ledger)

        assert admission.decision is LeaseDecision.LEGACY_CHECK


@pytest.mark.unit
class TestLiveLeaseAdmission:
    """Step 5, the fits branch: atomic local drawdown."""

    def test_admission_reserves_input_plus_output_bound(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, estimated_input_tokens=1_000, output_bound=500)

        assert admission.decision is LeaseDecision.ADMIT_LOCAL
        assert admission.lease_id == "lse_abc"
        assert admission.reserved_tokens == 1_500
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 98_500

    def test_absent_output_bound_uses_the_configured_default(self) -> None:
        ledger = _ledger(output_bound_default=2_048)
        ledger.apply_grant_response(RUN, _response(), now=1_000.0, declared_models=["gpt-5.5"])

        admission = _admit(ledger, estimated_input_tokens=1_000)

        assert admission.reserved_tokens == 3_048

    def test_admission_admits_only_while_the_remainder_covers_it(self) -> None:
        ledger = _granted_ledger(granted_tokens=3_000)

        first = _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)
        second = _admit(ledger, call_id="b", estimated_input_tokens=1_000, output_bound=500)
        third = _admit(ledger, call_id="c", estimated_input_tokens=1_000, output_bound=500)

        assert first.decision is LeaseDecision.ADMIT_LOCAL
        assert second.decision is LeaseDecision.ADMIT_LOCAL
        # 3_000 - 1_500 - 1_500 = 0 left: the third does not fit.
        assert third.decision is LeaseDecision.LEGACY_CHECK


@pytest.mark.unit
class TestOutageLadderLiveLease:
    """Step 5 past exhaustion: plane up → legacy, plane down → share, then mode."""

    def test_exhausted_with_plane_up_falls_to_the_legacy_check(self) -> None:
        # Arrange — an empty wallet is not a denial: the server stays
        # authoritative while it is reachable.
        ledger = _granted_ledger(granted_tokens=1_000)

        admission = _admit(ledger, estimated_input_tokens=5_000, breaker_open=False)

        assert admission.decision is LeaseDecision.LEGACY_CHECK
        assert admission.reason == "granted_exhausted_plane_up"
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.share_remaining_tokens == 500_000  # share untouched

    def test_exhausted_and_unreachable_draws_down_the_headroom_share(self) -> None:
        ledger = _granted_ledger(granted_tokens=1_000)

        admission = _admit(
            ledger, estimated_input_tokens=5_000, output_bound=1_000, breaker_open=True
        )

        assert admission.decision is LeaseDecision.ADMIT_OUTAGE_METERED
        assert admission.reserved_tokens == 6_000
        assert admission.warning is not None
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.share_remaining_tokens == 494_000
        assert state.granted_remaining_tokens == 1_000  # granted never over-drawn

    def test_share_exhausted_with_hard_deny_stops_the_call(self) -> None:
        # Arrange — the customer's own cap, conservatively enforced offline.
        ledger = _granted_ledger(
            granted_tokens=0,
            headroom_share_tokens=100,
            posture={"mode": "hard_deny", "on_unreachable": "fail_open"},
        )

        admission = _admit(ledger, estimated_input_tokens=5_000, breaker_open=True)

        assert admission.decision is LeaseDecision.DENY
        assert admission.mode is BudgetMode.HARD_DENY
        assert admission.reason == "lease_share_exhausted"
        assert admission.warning is not None

    def test_share_exhausted_with_alert_only_continues_with_a_warning(self) -> None:
        ledger = _granted_ledger(
            granted_tokens=0,
            headroom_share_tokens=100,
            posture={"mode": "alert_only", "on_unreachable": "fail_open"},
        )

        admission = _admit(ledger, estimated_input_tokens=5_000, breaker_open=True)

        assert admission.decision is LeaseDecision.ADMIT_OUTAGE_METERED
        assert admission.mode is BudgetMode.ALERT_ONLY
        assert admission.warning is not None


@pytest.mark.unit
class TestOutageLadderExpiredLease:
    """Step 6: expiry ends granted authority — it is never a deny by itself."""

    def test_expired_with_plane_up_drops_the_lease_and_needs_a_grant(self) -> None:
        ledger = _granted_ledger()

        admission = _admit(ledger, now=1_200.0, breaker_open=False)

        assert admission.decision is LeaseDecision.NEED_GRANT
        assert ledger.lease_id_for(RUN) is None

    def test_expired_and_unreachable_fail_open_admits_uncounted_and_tallies(self) -> None:
        ledger = _granted_ledger()

        first = _admit(
            ledger, call_id="a", estimated_input_tokens=1_000, now=1_200.0, breaker_open=True
        )
        second = _admit(
            ledger, call_id="b", estimated_input_tokens=2_000, now=1_201.0, breaker_open=True
        )

        assert first.decision is LeaseDecision.ADMIT_UNCOUNTED
        assert first.warning is not None
        assert second.decision is LeaseDecision.ADMIT_UNCOUNTED
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.uncounted_calls == 2
        # est + the 4096 default output bound, both calls.
        assert state.uncounted_tokens == (1_000 + 4_096) + (2_000 + 4_096)

    def test_expired_and_unreachable_local_enforce_meters_against_last_share(self) -> None:
        ledger = _granted_ledger(
            headroom_share_tokens=10_000,
            posture={"mode": "hard_deny", "on_unreachable": "local_enforce"},
        )

        admission = _admit(
            ledger,
            estimated_input_tokens=1_000,
            output_bound=1_000,
            now=1_200.0,
            breaker_open=True,
        )

        assert admission.decision is LeaseDecision.ADMIT_OUTAGE_METERED
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.share_remaining_tokens == 8_000
        # The granted counter is dead past expiry — never drawn down again.
        assert state.granted_remaining_tokens == 100_000

    def test_expired_local_enforce_bound_exceeded_applies_the_customers_mode(self) -> None:
        denying = _granted_ledger(
            headroom_share_tokens=10,
            posture={"mode": "hard_deny", "on_unreachable": "local_enforce"},
        )
        warning = _granted_ledger(
            headroom_share_tokens=10,
            posture={"mode": "alert_only", "on_unreachable": "local_enforce"},
        )

        denied = _admit(denying, estimated_input_tokens=5_000, now=1_200.0, breaker_open=True)
        allowed = _admit(warning, estimated_input_tokens=5_000, now=1_200.0, breaker_open=True)

        assert denied.decision is LeaseDecision.DENY
        assert denied.mode is BudgetMode.HARD_DENY
        assert allowed.decision is LeaseDecision.ADMIT_OUTAGE_METERED
        assert allowed.warning is not None

    def test_expired_is_not_exhausted(self) -> None:
        # DoD 8: past the deadline with a fat remainder, admissions must not
        # touch the granted or share counters — the server reclaimed that
        # float, so spending it would spend the same authority twice (R2-3).
        ledger = _granted_ledger()

        for index in range(5):
            admission = _admit(
                ledger, call_id=f"call-{index}", now=1_200.0 + index, breaker_open=True
            )
            assert admission.decision is LeaseDecision.ADMIT_UNCOUNTED

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000
        assert state.share_remaining_tokens == 500_000
        assert state.reservations == {}


@pytest.mark.unit
class TestReservationLifecycle:
    """True-up, release, sweep — reservation in, actual out."""

    def test_true_up_settles_the_reservation_against_actual_usage(self) -> None:
        # Arrange — reserve 1_500, spend 400.
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)

        # Act
        ledger.true_up("a", 400)

        # Assert — the unspent 1_100 comes back and the spend is reportable.
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000 - 400
        assert state.spent_tokens_since_report == 400
        assert state.reservations == {}

    def test_overshoot_drives_the_remainder_negative(self) -> None:
        # DoD 8 / spec §2: an uncapped call's overshoot is applied in FULL.
        ledger = _granted_ledger(granted_tokens=2_000)
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)

        ledger.true_up("a", 9_000)

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 2_000 - 9_000
        assert state.spent_tokens_since_report == 9_000

    def test_next_admission_after_overshoot_follows_the_normal_path(self) -> None:
        ledger = _granted_ledger(granted_tokens=2_000)
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)
        ledger.true_up("a", 9_000)

        plane_up = _admit(ledger, call_id="b", breaker_open=False)
        plane_down = _admit(ledger, call_id="c", breaker_open=True)

        assert plane_up.decision is LeaseDecision.LEGACY_CHECK
        assert plane_down.decision is LeaseDecision.ADMIT_OUTAGE_METERED

    def test_release_returns_the_reservation_untouched(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)

        ledger.release("a")

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000
        assert state.spent_tokens_since_report == 0

    def test_true_up_and_release_of_an_unknown_call_are_no_ops(self) -> None:
        ledger = _granted_ledger()

        ledger.true_up("never-admitted", 100)
        ledger.release("never-admitted")

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000

    def test_true_up_after_the_funding_lease_is_gone_touches_nothing(self) -> None:
        # Arrange — the lease that funded the call is dropped (404/expiry) and
        # a fresh one installed; the stale true-up must not hit new counters.
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)
        ledger.drop(RUN)
        ledger.apply_grant_response(
            RUN,
            _response(lease_id="lse_new", generation=1, granted_tokens=50_000),
            now=1_010.0,
            declared_models=["gpt-5.5"],
        )

        # Act
        ledger.true_up("a", 9_000)

        # Assert
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 50_000
        assert state.spent_tokens_since_report == 0

    def test_outage_share_reservation_trues_up_against_the_share(self) -> None:
        ledger = _granted_ledger(granted_tokens=0)
        _admit(
            ledger,
            call_id="a",
            estimated_input_tokens=1_000,
            output_bound=500,
            breaker_open=True,
        )

        ledger.true_up("a", 900)

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.share_remaining_tokens == 500_000 - 900
        assert state.granted_remaining_tokens == 0

    def test_abandoned_reservations_are_swept_after_900s(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger, call_id="abandoned", estimated_input_tokens=1_000, output_bound=500)

        swept = ledger.sweep(now=1_001.0 + RESERVATION_MAX_AGE_S)

        assert swept == 1
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 100_000
        assert state.reservations == {}

    def test_admission_sweeps_the_runs_abandoned_reservations(self) -> None:
        # The sweep rides admission — no timer thread exists in the SDK. The
        # lease outlives the sweep window here so the reservation, not the
        # lease, is what expires.
        ledger = _granted_ledger(lease_length_s=10_000.0)
        _admit(ledger, call_id="abandoned", estimated_input_tokens=1_000, output_bound=500)

        fresh = _admit(
            ledger,
            call_id="fresh",
            estimated_input_tokens=1_000,
            output_bound=500,
            now=1_001.0 + RESERVATION_MAX_AGE_S,
        )

        assert fresh.decision is LeaseDecision.ADMIT_LOCAL
        state = ledger.state_for(RUN)
        assert state is not None
        assert "abandoned" not in state.reservations
        assert state.granted_remaining_tokens == 100_000 - 1_500

    def test_fresh_reservations_are_not_swept(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)

        swept = ledger.sweep(now=1_001.0 + RESERVATION_MAX_AGE_S - 1.0)

        assert swept == 0


@pytest.mark.unit
class TestRenewalBookkeeping:
    """Renew-ahead thresholds, backoff, and the report payloads."""

    def test_renewal_is_not_due_below_75_percent_depletion(self) -> None:
        ledger = _granted_ledger(granted_tokens=10_000)

        # 7_400 of 10_000 == 74%.
        admission = _admit(ledger, estimated_input_tokens=7_000, output_bound=400)

        assert admission.decision is LeaseDecision.ADMIT_LOCAL
        assert admission.renewal_due is False

    def test_renewal_is_due_at_75_percent_depletion(self) -> None:
        ledger = _granted_ledger(granted_tokens=10_000)

        admission = _admit(ledger, estimated_input_tokens=7_100, output_bound=400)

        assert admission.renewal_due is True

    def test_renewal_is_due_once_the_refresh_deadline_passes(self) -> None:
        ledger = _granted_ledger()  # jitter ratio 1.0 → refresh at 1_000 + 17.25

        assert _admit(ledger, call_id="a", now=1_017.0).renewal_due is False
        assert _admit(ledger, call_id="b", now=1_018.0).renewal_due is True

    def test_renewal_is_suppressed_while_one_is_in_flight(self) -> None:
        ledger = _granted_ledger()
        ledger.renewal_sent(RUN)

        assert _admit(ledger, now=1_100.0).renewal_due is False

    def test_renewal_is_suppressed_until_the_backoff_elapses(self) -> None:
        ledger = _granted_ledger()
        ledger.renewal_sent(RUN)
        ledger.renewal_failed(RUN, now=1_020.0)  # ceiling 1s, _FixedRandom → 1.0s

        assert _admit(ledger, call_id="a", now=1_020.5).renewal_due is False
        assert _admit(ledger, call_id="b", now=1_021.0).renewal_due is True

    def test_backoff_schedule_is_exponential_with_a_30s_cap(self) -> None:
        # _FixedRandom(1.0) returns the top of the full-jitter window, so the
        # schedule the ceiling implies is directly observable.
        ledger = _granted_ledger()
        state = ledger.state_for(RUN)
        assert state is not None

        observed = []
        for attempt in range(7):
            ledger.renewal_failed(RUN, now=float(attempt))
            observed.append(round(state.next_attempt_at - attempt, 3))

        assert observed == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
        assert backoff_ceiling(1) == BACKOFF_BASE_S
        assert backoff_ceiling(99) == BACKOFF_CAP_S

    def test_a_successful_grant_clears_the_backoff_and_in_flight_flag(self) -> None:
        ledger = _granted_ledger()
        ledger.renewal_sent(RUN)
        ledger.renewal_failed(RUN, now=1_000.0)

        ledger.apply_grant_response(RUN, _response(generation=2), now=1_030.0)

        state = ledger.state_for(RUN)
        assert state is not None
        assert state.consecutive_failures == 0
        assert state.next_attempt_at == 0.0
        assert state.renewal_in_flight is False

    def test_final_grant_stops_asking_for_renewals(self) -> None:
        ledger = _granted_ledger(final_grant=True)

        assert _admit(ledger, now=1_100.0).renewal_due is False

    def test_renewal_request_reports_spend_reservations_and_tallies(self) -> None:
        # Arrange — one settled call, one still in flight, one uncounted call
        # from an earlier lease death.
        ledger = _granted_ledger()
        state = ledger.state_for(RUN)
        assert state is not None
        state.uncounted_calls = 3
        state.uncounted_tokens = 900
        _admit(ledger, call_id="settled", estimated_input_tokens=1_000, output_bound=500)
        ledger.true_up("settled", 800)
        _admit(ledger, call_id="in-flight", estimated_input_tokens=1_000, output_bound=500)

        # Act
        request = ledger.build_renewal_request(RUN)

        # Assert
        assert request is not None
        assert request.lease_id == "lse_abc"
        assert request.holder_id == HOLDER
        assert request.generation == 1
        assert request.spent_tokens == 800
        assert request.reserved_tokens == 1_500
        assert request.uncounted_calls == 3
        assert request.uncounted_tokens == 900

    def test_renewal_request_can_redeclare_the_model_set(self) -> None:
        ledger = _granted_ledger()

        request = ledger.build_renewal_request(RUN, model="gpt-5.5-mini")

        assert request is not None
        assert request.model == "gpt-5.5-mini"

    def test_renewal_request_is_none_without_a_lease(self) -> None:
        ledger = _ledger()

        assert ledger.build_renewal_request(RUN) is None

    def test_tallies_clear_only_when_the_report_is_acknowledged(self) -> None:
        # Arrange — report 800 spent, then spend 200 more before the response
        # lands. Only the acknowledged 800 may clear.
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)
        ledger.true_up("a", 800)
        state = ledger.state_for(RUN)
        assert state is not None
        state.uncounted_calls = 2
        state.uncounted_tokens = 500

        request = ledger.build_renewal_request(RUN)
        assert request is not None
        _admit(ledger, call_id="b", estimated_input_tokens=1_000, output_bound=500)
        ledger.true_up("b", 200)

        # Act
        ledger.apply_grant_response(RUN, _response(generation=2), now=1_020.0)

        # Assert
        assert state.spent_tokens_since_report == 200
        assert state.uncounted_calls == 0
        assert state.uncounted_tokens == 0

    def test_a_failed_renewal_keeps_the_tallies_owed(self) -> None:
        ledger = _granted_ledger()
        state = ledger.state_for(RUN)
        assert state is not None
        state.uncounted_calls = 2
        state.uncounted_tokens = 500
        ledger.build_renewal_request(RUN)

        ledger.renewal_failed(RUN, now=1_010.0)

        assert state.uncounted_calls == 2
        assert state.uncounted_tokens == 500

    def test_surrender_request_carries_the_final_true_up(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger, call_id="a", estimated_input_tokens=1_000, output_bound=500)
        ledger.true_up("a", 700)

        request = ledger.build_surrender_request(RUN)

        assert request is not None
        assert request.lease_id == "lse_abc"
        assert request.holder_id == HOLDER
        assert request.generation == 1
        assert request.spent_tokens == 700

    def test_surrender_request_is_none_without_a_lease(self) -> None:
        assert _ledger().build_surrender_request(RUN) is None


@pytest.mark.unit
class TestLeaseStateLifecycle:
    """Drop, discard, fork reset."""

    def test_drop_forgets_the_lease_but_keeps_what_is_owed(self) -> None:
        # 404 lease_not_found / 409 generation conflict: re-grant next
        # admission, but the uncounted tally is still owed to the server.
        ledger = _granted_ledger()
        state = ledger.state_for(RUN)
        assert state is not None
        state.uncounted_calls = 4
        state.uncounted_tokens = 1_000

        ledger.drop(RUN)

        assert ledger.lease_id_for(RUN) is None
        assert _admit(ledger).decision is LeaseDecision.NEED_GRANT
        assert state.uncounted_calls == 4
        assert state.uncounted_tokens == 1_000

    def test_discard_forgets_the_run_entirely(self) -> None:
        ledger = _granted_ledger()
        _admit(ledger, call_id="a")

        ledger.discard(RUN)

        assert ledger.state_for(RUN) is None
        assert ledger.active_run_ids() == []
        # A late true-up for the discarded run is a no-op, not a KeyError.
        ledger.true_up("a", 100)

    def test_fork_reset_drops_all_lease_state(self) -> None:
        # A forked child must re-grant: the parent's lease is bound to the
        # parent's holder identity and its float is the parent's.
        ledger = _granted_ledger()
        _admit(ledger, call_id="a")

        ledger.on_fork_reset()

        assert ledger.state_for(RUN) is None
        assert ledger.active_run_ids() == []
        assert _admit(ledger).decision is LeaseDecision.NEED_GRANT


@pytest.mark.unit
class TestConcurrentBurst:
    """DoD 8: concurrent admissions can never jointly overrun the remainder."""

    def test_concurrent_burst_cannot_overrun(self) -> None:
        # Arrange — 15_000 granted, 1_500 per call: at most 10 may pass, no
        # matter how many threads race. The test holds the enforcer-style
        # lock exactly where BudgetEnforcer holds its _state_lock.
        ledger = _granted_ledger(granted_tokens=15_000)
        lock = threading.Lock()
        start = threading.Barrier(40)
        admissions: list[LeaseAdmission] = []
        results_lock = threading.Lock()

        def worker(index: int) -> None:
            start.wait(timeout=5.0)
            with lock:
                admission = ledger.admit(
                    run_id=RUN,
                    call_id=f"call-{index}",
                    estimated_input_tokens=1_000,
                    model="gpt-5.5",
                    output_bound=500,
                    now=1_001.0,
                    breaker_open=False,
                )
            with results_lock:
                admissions.append(admission)

        # Act
        threads = [threading.Thread(target=worker, args=(index,)) for index in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        # Assert
        admitted = [a for a in admissions if a.decision is LeaseDecision.ADMIT_LOCAL]
        assert len(admissions) == 40
        assert len(admitted) == 15_000 // 1_500
        state = ledger.state_for(RUN)
        assert state is not None
        assert state.granted_remaining_tokens == 0
        assert sum(r.tokens for r in state.reservations.values()) == 15_000
