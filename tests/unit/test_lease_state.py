"""Unit tests for the sans-I/O budget-lease ledger (PJ-2, DoD 3 + DoD 8).

The ledger is the pure decision core the budget enforcers drive: it owns the
admission ladder, the atomic reservation math, fencing, and the renewal
bookkeeping. Every test here drives it directly — no HTTP, no threads except
the burst-atomicity test, which holds an enforcer-style lock exactly as the
sync enforcer does.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from solwyn._lease import (
    INELIGIBLE_RETRY_AFTER_S,
    REFRESH_JITTER_MAX,
    REFRESH_JITTER_MIN,
    GrantOutcome,
    LeaseAdmission,
    LeaseDecision,
    LeaseLedger,
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
