"""Sans-I/O budget-lease ledger — the pure decision core of PJ-2.

A lease is server-granted token authority for one ``agent_run_id``, drawn down
in memory so run-scoped calls stop paying a blocking ``/budgets/check`` each.
This module owns ONLY the decisions: the §4 outage ladder, atomic reservation
math, generation fencing, renewal bookkeeping. It performs no I/O, starts no
threads, and takes no locks.

Concurrency contract: the CALLER serializes every method on one ledger — the
sync ``BudgetEnforcer`` holds its ``_state_lock`` around the whole admission
section (so concurrent calls can never jointly pass against the same
remainder), and the async enforcer relies on event-loop serialization. Time
enters only as a caller-supplied monotonic ``now``; wall clock is never
trusted for lease validity (short leases are comparable to real clock skew).
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from solwyn._types import (
    BudgetMode,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    Modality,
    ProviderName,
)

logger = logging.getLogger(__name__)

# Reservation bound for a call that carries no max_tokens-family cap.
DEFAULT_OUTPUT_BOUND = 4096

# Error paths that never settle (abandoned streams, dispatch crashes) must not
# strand lease tokens; 900s is far beyond any real call.
RESERVATION_MAX_AGE_S = 900.0

# 503 lease_unavailable: retry a grant no sooner than this.
INELIGIBLE_RETRY_AFTER_S = 30.0

# Renew ahead of need at 75% depletion (3/4, kept as ints — no float drift).
RENEWAL_DEPLETION_NUM = 3
RENEWAL_DEPLETION_DEN = 4

# Client-side decorrelation on top of the server's own refresh jitter.
REFRESH_JITTER_MIN = 0.85
REFRESH_JITTER_MAX = 1.15

# Renewal backoff: exponential, full jitter. A failed renewal never blocks a
# call — the next admission retries once the backoff has elapsed.
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0

OnUnreachable = Literal["fail_open", "local_enforce"]


class _Pool(StrEnum):
    """Which counter a reservation drew from — never a bare string."""

    GRANTED = "granted"
    SHARE = "share"


class LeaseDecision(StrEnum):
    """What the enforcer must do with a call, decided without further lease logic."""

    LEGACY_CHECK = "legacy_check"
    """Take today's per-call /budgets/check path (kill switch, ineligible call
    or run, or an empty wallet while the plane is believed up)."""

    NEED_GRANT = "need_grant"
    """No usable lease: request one (cold start, or a stale lease dropped)."""

    ADMIT_LOCAL = "admit_local"
    """Admitted against the granted remainder; the reservation is recorded."""

    ADMIT_OUTAGE_METERED = "admit_outage_metered"
    """Admitted against the headroom share during an outage — still metered."""

    ADMIT_UNCOUNTED = "admit_uncounted"
    """Admitted past expiry under fail_open: no counter covers it; tallied."""

    DENY = "deny"
    """The CUSTOMER's own mode verdict at true exhaustion — never an outage."""


_ADMITTING = frozenset(
    {
        LeaseDecision.ADMIT_LOCAL,
        LeaseDecision.ADMIT_OUTAGE_METERED,
        LeaseDecision.ADMIT_UNCOUNTED,
    }
)


class GrantOutcome(StrEnum):
    """What ``apply_grant_response`` did with a grant/renew response."""

    APPLIED = "applied"
    INELIGIBLE = "ineligible"
    DENIED = "denied"
    STALE = "stale"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class LeaseAdmission:
    """A ladder verdict. ``reason`` is a stable slug, never customer content."""

    decision: LeaseDecision
    lease_id: str | None = None
    reserved_tokens: int = 0
    warning: str | None = None
    renewal_due: bool = False
    mode: BudgetMode | None = None
    reason: str | None = None

    @property
    def admitted(self) -> bool:
        """True when the call may proceed on lease authority."""
        return self.decision in _ADMITTING


@dataclass(frozen=True, slots=True)
class LeaseSnapshot:
    """Last-known display numbers from a grant/renew response (never computed)."""

    project_id: str
    mode: BudgetMode
    budget_limit: float
    current_usage: float
    remaining_budget: float


@dataclass(slots=True)
class _Reservation:
    """One in-flight call's atomically reserved allowance."""

    lease_id: str
    tokens: int
    created_at: float
    pool: _Pool


@dataclass(slots=True)
class _PendingReport:
    """Tallies handed to a renewal, subtracted only once it is acknowledged."""

    spent_tokens: int
    uncounted_calls: int
    uncounted_tokens: int


@dataclass(slots=True)
class LeaseState:
    """Per-run lease record. Mutated only under the caller's serialization."""

    run_id: str
    lease_id: str | None = None
    generation: int = 0
    granted_tokens: int = 0
    # MAY GO NEGATIVE: an uncapped call's overshoot is applied in full; the
    # next admission then sees an exhausted lease and follows the normal path.
    granted_remaining_tokens: int = 0
    share_remaining_tokens: int = 0
    refresh_deadline: float = 0.0
    lease_deadline: float = 0.0
    posture_mode: BudgetMode = BudgetMode.ALERT_ONLY
    on_unreachable: OnUnreachable = "fail_open"
    final_grant: bool = False
    declared_models: set[str] = field(default_factory=set)
    reservations: dict[str, _Reservation] = field(default_factory=dict)
    renewal_in_flight: bool = False
    consecutive_failures: int = 0
    next_attempt_at: float = 0.0
    pending_report: _PendingReport | None = None
    spent_tokens_since_report: int = 0
    uncounted_calls: int = 0
    uncounted_tokens: int = 0
    run_ineligible: bool = False
    ineligible_retry_at: float = 0.0
    snapshot: LeaseSnapshot | None = None

    @property
    def has_lease(self) -> bool:
        """True when a lease is installed (expired or not)."""
        return self.lease_id is not None

    @property
    def reserved_tokens(self) -> int:
        """In-flight reservations funded by the CURRENT lease (demand hint).

        A superseded lease's leftovers no-op at true-up; counting them would
        inflate the demand the server sizes the next grant from.
        """
        return sum(
            reservation.tokens
            for reservation in self.reservations.values()
            if reservation.lease_id == self.lease_id
        )

    def covers(self, model: str, fallback_models: Sequence[str]) -> bool:
        """True when every model the call may reach is in the declared set."""
        return {model, *fallback_models} <= self.declared_models


class LeaseLedger:
    """Holder-side lease state for one SDK client, keyed by ``agent_run_id``.

    NOT thread-safe by itself — see the module docstring's concurrency
    contract. ``holder_id`` is the SDK instance id that binds every lease.
    """

    def __init__(
        self,
        *,
        holder_id: str,
        enabled: bool = True,
        output_bound_default: int = DEFAULT_OUTPUT_BOUND,
        rng: random.Random | None = None,
    ) -> None:
        if output_bound_default <= 0:
            raise RuntimeError("output_bound_default must be positive")
        self.holder_id = holder_id
        self.enabled = enabled
        self.output_bound_default = output_bound_default
        self._rng = rng if rng is not None else random.Random()
        self._states: dict[str, LeaseState] = {}
        # call_id -> run_id, so true-up/release find a reservation in O(1).
        self._call_index: dict[str, str] = {}

    # ── accessors ────────────────────────────────────────────────────────

    def state_for(self, run_id: str) -> LeaseState | None:
        """The run's record, or None when the run has never been seen."""
        return self._states.get(run_id)

    def lease_id_for(self, run_id: str) -> str | None:
        """The run's installed lease id, or None."""
        state = self._states.get(run_id)
        return state.lease_id if state is not None else None

    def snapshot_for(self, run_id: str) -> LeaseSnapshot | None:
        """Last-known display numbers for the run (for enforcer results)."""
        state = self._states.get(run_id)
        return state.snapshot if state is not None else None

    def active_run_ids(self) -> list[str]:
        """Runs holding an installed lease (surrender targets)."""
        return [run_id for run_id, state in self._states.items() if state.has_lease]

    # ── admission ladder ─────────────────────────────────────────────────

    def admit(
        self,
        *,
        run_id: str,
        call_id: str,
        estimated_input_tokens: int,
        model: str,
        now: float,
        breaker_open: bool,
        output_bound: int | None = None,
        modality: Modality = "text",
        has_estimated_media: bool = False,
        fallback_models: Sequence[str] = (),
    ) -> LeaseAdmission:
        """Decide one call, per the SDK admission algorithm steps 2-6.

        Step 1 (sticky-deny precedence) and all I/O stay with the enforcer.
        ``breaker_open`` means "the control plane is believed unreachable".
        On an admitting decision that draws down a counter, the reservation is
        recorded under ``call_id`` — the caller MUST later ``true_up`` or
        ``release`` it.
        """
        if not self.enabled:
            return LeaseAdmission(LeaseDecision.LEGACY_CHECK, reason="lease_disabled")

        if modality != "text" or has_estimated_media:
            return LeaseAdmission(LeaseDecision.LEGACY_CHECK, reason="call_lease_ineligible")

        state = self._states.get(run_id)

        if state is not None and state.run_ineligible:
            if now < state.ineligible_retry_at:
                return LeaseAdmission(LeaseDecision.LEGACY_CHECK, reason="run_lease_ineligible")
            state.run_ineligible = False
            state.ineligible_retry_at = 0.0

        if state is not None and state.has_lease and not state.covers(model, fallback_models):
            return LeaseAdmission(LeaseDecision.LEGACY_CHECK, reason="model_outside_declared_set")

        if state is not None:
            # Abandoned reservations are swept on the admission path — the SDK
            # runs no timer thread.
            self._sweep_state(state, now)

        reserve = max(0, estimated_input_tokens) + self._output_bound(output_bound)

        if state is None or not state.has_lease:
            return LeaseAdmission(LeaseDecision.NEED_GRANT, reason="no_lease")

        if now < state.lease_deadline:
            return self._admit_live(state, call_id, reserve, now, breaker_open)
        return self._admit_expired(state, call_id, reserve, now, breaker_open)

    def _admit_live(
        self,
        state: LeaseState,
        call_id: str,
        reserve: int,
        now: float,
        breaker_open: bool,
    ) -> LeaseAdmission:
        """Step 5: the grant is unexpired — granted remainder, then share."""
        if state.granted_remaining_tokens >= reserve:
            state.granted_remaining_tokens -= reserve
            self._reserve(state, call_id, reserve, now, _Pool.GRANTED)
            return LeaseAdmission(
                LeaseDecision.ADMIT_LOCAL,
                lease_id=state.lease_id,
                reserved_tokens=reserve,
                renewal_due=self.renewal_due(state, now),
            )

        if not breaker_open:
            # Empty wallet is not forbidden: the server stays authoritative.
            return LeaseAdmission(
                LeaseDecision.LEGACY_CHECK,
                lease_id=state.lease_id,
                renewal_due=self.renewal_due(state, now),
                reason="granted_exhausted_plane_up",
            )

        if state.share_remaining_tokens >= reserve:
            state.share_remaining_tokens -= reserve
            self._reserve(state, call_id, reserve, now, _Pool.SHARE)
            return LeaseAdmission(
                LeaseDecision.ADMIT_OUTAGE_METERED,
                lease_id=state.lease_id,
                reserved_tokens=reserve,
                warning=(
                    "Solwyn unreachable; lease grant exhausted — drawing down "
                    "this holder's headroom share"
                ),
                reason="share_drawdown",
            )

        # True exhaustion inside a LIVE authority window escalates to the
        # customer's configured mode: their cap, conservatively enforced.
        if state.posture_mode is BudgetMode.HARD_DENY:
            return LeaseAdmission(
                LeaseDecision.DENY,
                lease_id=state.lease_id,
                mode=state.posture_mode,
                warning=(
                    "Budget lease exhausted and Solwyn unreachable; hard_deny mode denies the call"
                ),
                reason="lease_share_exhausted",
            )
        state.share_remaining_tokens -= reserve
        self._reserve(state, call_id, reserve, now, _Pool.SHARE)
        return LeaseAdmission(
            LeaseDecision.ADMIT_OUTAGE_METERED,
            lease_id=state.lease_id,
            reserved_tokens=reserve,
            mode=state.posture_mode,
            warning=(
                "Budget lease exhausted and Solwyn unreachable; "
                "alert_only mode continues past the share"
            ),
            reason="lease_share_exhausted",
        )

    def _admit_expired(
        self,
        state: LeaseState,
        call_id: str,
        reserve: int,
        now: float,
        breaker_open: bool,
    ) -> LeaseAdmission:
        """Step 6: past the monotonic lease deadline.

        Expiry is evidence of STALENESS, never of the customer's cap (R2-3):
        the server has reclaimed that float, so the granted counter is dead —
        it is never drawn down as authority again.
        """
        if not breaker_open:
            self._drop_lease(state)
            return LeaseAdmission(LeaseDecision.NEED_GRANT, reason="lease_expired")

        if state.on_unreachable == "fail_open":
            state.uncounted_calls += 1
            state.uncounted_tokens += reserve
            return LeaseAdmission(
                LeaseDecision.ADMIT_UNCOUNTED,
                # DELIBERATELY no lease_id: nothing settles this call. Its
                # spend is owed through the uncounted tally on the next
                # renewal, and a confirm tagged with the reclaimed lease would
                # be counted twice (settlement excess AND the tally).
                lease_id=None,
                warning=(
                    "Budget lease expired and Solwyn unreachable; proceeding "
                    "UNCOUNTED in fail-open mode"
                ),
                reason="expired_fail_open",
            )

        # local_enforce: meter against the freshest known BOUND (the last
        # share remainder) — a local bound, never granted authority.
        if state.share_remaining_tokens >= reserve:
            state.share_remaining_tokens -= reserve
            self._reserve(state, call_id, reserve, now, _Pool.SHARE)
            return LeaseAdmission(
                LeaseDecision.ADMIT_OUTAGE_METERED,
                lease_id=state.lease_id,
                reserved_tokens=reserve,
                warning=(
                    "Budget lease expired and Solwyn unreachable; metering "
                    "locally against the last known headroom share"
                ),
                reason="expired_local_enforce",
            )

        if state.posture_mode is BudgetMode.HARD_DENY:
            return LeaseAdmission(
                LeaseDecision.DENY,
                lease_id=state.lease_id,
                mode=state.posture_mode,
                warning=(
                    "Budget lease expired, Solwyn unreachable and the last "
                    "known headroom share is exhausted; hard_deny mode denies the call"
                ),
                reason="local_enforce_bound_exceeded",
            )
        state.share_remaining_tokens -= reserve
        self._reserve(state, call_id, reserve, now, _Pool.SHARE)
        return LeaseAdmission(
            LeaseDecision.ADMIT_OUTAGE_METERED,
            lease_id=state.lease_id,
            reserved_tokens=reserve,
            mode=state.posture_mode,
            warning=(
                "Budget lease expired, Solwyn unreachable and the last known "
                "headroom share is exhausted; alert_only mode continues"
            ),
            reason="local_enforce_bound_exceeded",
        )

    # ── grant application ────────────────────────────────────────────────

    def apply_grant_response(
        self,
        run_id: str,
        response: LeaseGrantResponse,
        *,
        now: float,
        declared_models: Iterable[str] = (),
    ) -> GrantOutcome:
        """Install a grant/renew response, or explain why it was not installed.

        Fencing (R2-4): a response installs only when its ``generation`` is
        newer than the one currently held. Timers restart HERE, from the
        response's durations measured on the caller's monotonic clock.
        """
        state = self._states.get(run_id)
        if state is None:
            state = LeaseState(run_id=run_id)
            self._states[run_id] = state

        if not response.eligible:
            self._drop_lease(state)
            self._store_snapshot(state, response)
            state.run_ineligible = True
            state.ineligible_retry_at = math.inf
            logger.info(
                "lease.run_ineligible: reason=%s (legacy per-call path for this run)",
                response.ineligible_reason,
            )
            return GrantOutcome.INELIGIBLE

        if not response.allowed:
            # An authoritative hard deny — the enforcer feeds it to the
            # sticky-deny machinery; nothing is drawn down locally after it.
            self._drop_lease(state)
            self._store_snapshot(state, response)
            return GrantOutcome.DENIED

        generation = response.generation
        if (
            response.lease_id is None
            or generation is None
            or response.granted_tokens is None
            or response.refresh_interval_s is None
            or response.lease_length_s is None
            or response.headroom_share_tokens is None
            or response.posture is None
        ):
            logger.warning("lease.grant_response_malformed: lease fields missing on an allow")
            self._drop_lease(state)
            state.run_ineligible = True
            state.ineligible_retry_at = now + INELIGIBLE_RETRY_AFTER_S
            return GrantOutcome.MALFORMED

        if state.has_lease and generation <= state.generation:
            logger.debug("lease.stale_generation_ignored")
            return GrantOutcome.STALE

        # The declared set belongs to ONE lease: the server folded that
        # lease's worst-case rate over exactly these models. A different
        # lease_id (fresh grant, or a regrant after a drop) REPLACES the set —
        # inheriting a dead lease's models would admit calls the new grant
        # never priced. Only a renewal of the same lease unions.
        is_same_lease = state.lease_id == response.lease_id
        # Both terms are read BEFORE the new lease is installed, and both are
        # confined to a renewal of the SAME lease: across a lease boundary
        # neither is conserved, because true-up no-ops once the funding lease
        # is gone (charging them would leave nothing standing behind them).
        carried = self._carried_drawdown(state) if is_same_lease else 0
        state.lease_id = response.lease_id
        state.generation = generation
        state.granted_tokens = max(0, response.granted_tokens)
        # The new grant is authority NET of what its sizing could not see: a
        # renewal flies off the hot path, admissions keep drawing meanwhile,
        # and the server sized this response from the report snapshot alone.
        # Installing it verbatim would re-grant tokens already committed —
        # an overshoot scaling with renewal latency x call rate.
        # May go negative; the ladder handles that exactly as an overshoot.
        state.granted_remaining_tokens = state.granted_tokens - carried
        state.share_remaining_tokens = max(0, response.headroom_share_tokens)
        state.refresh_deadline = now + response.refresh_interval_s * self._rng.uniform(
            REFRESH_JITTER_MIN, REFRESH_JITTER_MAX
        )
        state.lease_deadline = now + response.lease_length_s
        state.posture_mode = response.posture.mode
        state.on_unreachable = response.posture.on_unreachable
        state.final_grant = bool(response.final_grant)
        if is_same_lease:
            state.declared_models.update(declared_models)
        else:
            state.declared_models = set(declared_models)
        state.renewal_in_flight = False
        state.consecutive_failures = 0
        state.next_attempt_at = 0.0
        state.run_ineligible = False
        state.ineligible_retry_at = 0.0
        self._store_snapshot(state, response)
        self._settle_pending_report(state)

        if state.final_grant:
            logger.warning(
                "lease.final_grant: no further renewal will be granted for this run "
                "— winding down to the per-call path at expiry"
            )
        return GrantOutcome.APPLIED

    def mark_ineligible(self, run_id: str, *, now: float, retry_after: float | None = None) -> None:
        """Route a run to the legacy path; ``retry_after`` None means forever.

        A Solwyn-side refusal must never block a call — the legacy per-call
        path has its own fail-open. Use this for a refusal of a GRANT: a 409
        ``lease_holder_cap_exceeded`` (permanent for this run) or a 503
        ``lease_unavailable`` (with ``retry_after=INELIGIBLE_RETRY_AFTER_S``).
        It DROPS any installed lease, so a 503 answering a RENEWAL must NOT
        come here — that lease stays valid until its deadline and the failure
        belongs in ``renewal_failed`` (backoff); the ladder governs afterwards.
        """
        state = self._states.get(run_id)
        if state is None:
            state = LeaseState(run_id=run_id)
            self._states[run_id] = state
        self._drop_lease(state)
        state.run_ineligible = True
        state.ineligible_retry_at = math.inf if retry_after is None else now + retry_after

    def record_uncounted(self, run_id: str, tokens: int) -> None:
        """Tally one call that no counter covered (admission step 4, cold start).

        The expiry ladder tallies its own uncounted admissions; this is the
        enforcer's entry point for the case with no lease at all — a cold-start
        grant attempt that found the plane down under fail_open. The tally
        survives ``drop``/expiry, so it rides the first successful renewal of
        whatever lease the run gets next.
        """
        state = self._states.get(run_id)
        if state is None:
            state = LeaseState(run_id=run_id)
            self._states[run_id] = state
        state.uncounted_calls += 1
        state.uncounted_tokens += max(0, tokens)

    def drop(self, run_id: str) -> None:
        """Forget the run's lease (404 lease_not_found / 409 generation conflict).

        The uncounted tallies survive — they are owed to the next renewal.
        """
        state = self._states.get(run_id)
        if state is not None:
            self._drop_lease(state)

    def discard(self, run_id: str) -> None:
        """Forget the run entirely (after a surrender)."""
        state = self._states.pop(run_id, None)
        if state is None:
            return
        for call_id in state.reservations:
            self._call_index.pop(call_id, None)

    def on_fork_reset(self) -> None:
        """Drop ALL lease state — a forked child must re-grant under its own id."""
        self._states.clear()
        self._call_index.clear()

    # ── reservation lifecycle ────────────────────────────────────────────

    def true_up(self, call_id: str, actual_tokens: int) -> None:
        """Settle a reservation against the call's ACTUAL token usage.

        Overshoot is applied in full and may drive the remainder negative:
        the next admission then sees an exhausted lease and follows the
        normal path (renew, or the outage ladder).
        """
        state, reservation = self._take_reservation(call_id)
        if state is None or reservation is None:
            return
        if reservation.lease_id != state.lease_id:
            # The lease that funded this call is gone; its counters are dead.
            return
        actual = max(0, actual_tokens)
        delta = actual - reservation.tokens
        if reservation.pool is _Pool.GRANTED:
            state.granted_remaining_tokens -= delta
        else:
            state.share_remaining_tokens -= delta
        state.spent_tokens_since_report += actual

    def release(self, call_id: str) -> None:
        """Give a reservation back untouched (error paths — no spend happened)."""
        state, reservation = self._take_reservation(call_id)
        if state is None or reservation is None:
            return
        if reservation.lease_id != state.lease_id:
            return
        if reservation.pool is _Pool.GRANTED:
            state.granted_remaining_tokens += reservation.tokens
        else:
            state.share_remaining_tokens += reservation.tokens

    def sweep(self, now: float) -> int:
        """Release reservations older than 900s; returns how many were swept."""
        return sum(self._sweep_state(state, now) for state in list(self._states.values()))

    def _sweep_state(self, state: LeaseState, now: float) -> int:
        stale = [
            call_id
            for call_id, reservation in state.reservations.items()
            if now - reservation.created_at >= RESERVATION_MAX_AGE_S
        ]
        for call_id in stale:
            self.release(call_id)
        if stale:
            logger.warning("lease.reservations_swept: count=%d", len(stale))
        return len(stale)

    # ── renewal bookkeeping ──────────────────────────────────────────────

    def renewal_due(self, state: LeaseState, now: float) -> bool:
        """Due at 75% depletion or past the refresh deadline.

        Suppressed while a renewal is in flight or a backoff is pending — a
        failed renewal never turns into a retry storm on the hot path.
        """
        if not state.has_lease:
            return False
        if state.renewal_in_flight or now < state.next_attempt_at:
            return False
        if state.final_grant:
            return False
        if now >= state.refresh_deadline:
            return True
        if state.granted_tokens <= 0:
            # A zero-token grant (alert_only past cap) is depleted by
            # definition: a ratio test would ask for a renewal on every single
            # call, stacking a round-trip on top of the legacy check that
            # branch already pays. Only the refresh deadline may renew it.
            return False
        depleted = state.granted_tokens - state.granted_remaining_tokens
        return depleted * RENEWAL_DEPLETION_DEN >= state.granted_tokens * RENEWAL_DEPLETION_NUM

    def build_renewal_request(
        self,
        run_id: str,
        *,
        model: str | None = None,
        provider: ProviderName | None = None,
        fallback_providers: Sequence[ProviderName] = (),
        fallback_models: Sequence[str] = (),
    ) -> LeaseRenewRequest | None:
        """Payload for ``POST /budgets/lease/renew``; None when there is no lease.

        The reported tallies are held pending — they clear only when the
        renewal response is applied, so a lost response never loses spend.
        """
        state = self._states.get(run_id)
        if state is None or state.lease_id is None:
            return None
        state.pending_report = _PendingReport(
            spent_tokens=state.spent_tokens_since_report,
            uncounted_calls=state.uncounted_calls,
            uncounted_tokens=state.uncounted_tokens,
        )
        return LeaseRenewRequest(
            lease_id=state.lease_id,
            holder_id=self.holder_id,
            generation=state.generation,
            spent_tokens=state.spent_tokens_since_report,
            reserved_tokens=state.reserved_tokens,
            uncounted_calls=state.uncounted_calls,
            uncounted_tokens=state.uncounted_tokens,
            model=model,
            provider=provider,
            fallback_providers=list(fallback_providers),
            fallback_models=list(fallback_models),
        )

    def build_surrender_request(self, run_id: str) -> LeaseSurrenderRequest | None:
        """Payload for ``POST /budgets/lease/surrender``; None without a lease."""
        state = self._states.get(run_id)
        if state is None or state.lease_id is None:
            return None
        return LeaseSurrenderRequest(
            lease_id=state.lease_id,
            holder_id=self.holder_id,
            generation=state.generation,
            spent_tokens=state.spent_tokens_since_report,
        )

    def renewal_sent(self, run_id: str) -> None:
        """Mark a renewal in flight (suppresses further renewal-due signals)."""
        state = self._states.get(run_id)
        if state is not None:
            state.renewal_in_flight = True

    def renewal_failed(self, run_id: str, now: float) -> None:
        """Record a failed renewal: exponential backoff, base 1s, cap 30s, full jitter."""
        state = self._states.get(run_id)
        if state is None:
            return
        state.renewal_in_flight = False
        state.pending_report = None
        state.consecutive_failures += 1
        state.next_attempt_at = now + self._rng.uniform(
            0.0, backoff_ceiling(state.consecutive_failures)
        )

    # ── internals ────────────────────────────────────────────────────────

    def _output_bound(self, output_bound: int | None) -> int:
        """Resolve the call's output bound.

        None (no max_tokens-family cap on the call) uses the configured
        default. A caller-supplied 0 or negative cap is honored as the
        default too, deliberately: trusting it would reserve the input
        estimate alone, leaving an unbounded response able to overrun the
        remainder with nothing standing behind it.
        """
        if output_bound is None or output_bound <= 0:
            return self.output_bound_default
        return output_bound

    def _reserve(
        self, state: LeaseState, call_id: str, tokens: int, now: float, pool: _Pool
    ) -> None:
        if state.lease_id is None:
            raise RuntimeError("cannot reserve against a run with no lease")
        if call_id in state.reservations:
            # No call path re-admits one call_id today; if one ever does, the
            # first drawdown must come back now rather than at the 900s sweep.
            logger.warning("lease.duplicate_reservation_released")
            self.release(call_id)
        state.reservations[call_id] = _Reservation(
            lease_id=state.lease_id, tokens=tokens, created_at=now, pool=pool
        )
        self._call_index[call_id] = state.run_id

    def _take_reservation(self, call_id: str) -> tuple[LeaseState | None, _Reservation | None]:
        run_id = self._call_index.pop(call_id, None)
        if run_id is None:
            return None, None
        state = self._states.get(run_id)
        if state is None:
            return None, None
        return state, state.reservations.pop(call_id, None)

    def _drop_lease(self, state: LeaseState) -> None:
        """Clear lease authority, keeping what is still owed to the server.

        The uncounted tallies and the display snapshot survive a drop: the
        next lease's first renewal still owes the server that report.
        """
        state.lease_id = None
        state.generation = 0
        state.granted_tokens = 0
        state.granted_remaining_tokens = 0
        state.share_remaining_tokens = 0
        state.refresh_deadline = 0.0
        state.lease_deadline = 0.0
        state.final_grant = False
        state.renewal_in_flight = False
        state.pending_report = None

    def _store_snapshot(self, state: LeaseState, response: LeaseGrantResponse) -> None:
        state.snapshot = LeaseSnapshot(
            project_id=response.project_id,
            mode=response.mode,
            budget_limit=response.budget_limit,
            current_usage=response.current_usage,
            remaining_budget=response.remaining_budget,
        )

    def _carried_drawdown(self, state: LeaseState) -> int:
        """Drawdown a replacement grant must arrive net of, in two terms.

        SETTLED: ``build_renewal_request`` pins the reported figure in
        ``pending_report`` while ``spent_tokens_since_report`` keeps
        accumulating every later true-up; the difference is exactly the spend
        the server sized this response without. Zero when no report is
        outstanding (a fresh grant, or a renewal whose failure cleared it) —
        no snapshot, nothing missed.

        RESERVED: in-flight calls hold bounds drawn from the grant being
        retired, and the server folds them nowhere (``reserved_tokens`` on the
        renewal is a demand hint, never subtracted from the sizing). Carrying
        them keeps the ledger conserved across the swap: a crossing call that
        settles unwinds its bound through true-up's delta and leaves the grant
        net of its ACTUAL, and one that releases gives the bound back whole.

        The two never overlap — settling pops the reservation, so a call is in
        exactly one term.
        """
        pending = state.pending_report
        settled = (
            0 if pending is None else max(0, state.spent_tokens_since_report - pending.spent_tokens)
        )
        return settled + state.reserved_tokens

    def _settle_pending_report(self, state: LeaseState) -> None:
        """An acknowledged renewal clears exactly what it reported, no more."""
        pending = state.pending_report
        if pending is None:
            return
        state.spent_tokens_since_report = max(
            0, state.spent_tokens_since_report - pending.spent_tokens
        )
        state.uncounted_calls = max(0, state.uncounted_calls - pending.uncounted_calls)
        state.uncounted_tokens = max(0, state.uncounted_tokens - pending.uncounted_tokens)
        state.pending_report = None


def backoff_ceiling(consecutive_failures: int) -> float:
    """Full-jitter ceiling for the Nth consecutive renewal failure (1-based)."""
    if consecutive_failures <= 0:
        return 0.0
    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** (consecutive_failures - 1))
