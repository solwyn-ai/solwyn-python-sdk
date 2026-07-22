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

from solwyn._types import BudgetMode, LeaseGrantResponse, Modality

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

_POOL_GRANTED = "granted"
_POOL_SHARE = "share"

OnUnreachable = Literal["fail_open", "local_enforce"]


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
    pool: str


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
        """Currently in-flight reservation total (renewal demand hint)."""
        return sum(reservation.tokens for reservation in self.reservations.values())

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
            self._reserve(state, call_id, reserve, now, _POOL_GRANTED)
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
            self._reserve(state, call_id, reserve, now, _POOL_SHARE)
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

        # True exhaustion inside a live authority window: the customer's own
        # mode decides. This is their cap, conservatively enforced offline.
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
        self._reserve(state, call_id, reserve, now, _POOL_SHARE)
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
                lease_id=state.lease_id,
                warning=(
                    "Budget lease expired and Solwyn unreachable; proceeding "
                    "UNCOUNTED in fail-open mode"
                ),
                reason="expired_fail_open",
            )

        # local_enforce: meter against the freshest known bound (the last
        # share remainder) — a local bound, not granted authority.
        if state.share_remaining_tokens >= reserve:
            state.share_remaining_tokens -= reserve
            self._reserve(state, call_id, reserve, now, _POOL_SHARE)
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
                    "known headroom share is exhausted; hard_deny mode denies "
                    "the call"
                ),
                reason="local_enforce_bound_exceeded",
            )
        state.share_remaining_tokens -= reserve
        self._reserve(state, call_id, reserve, now, _POOL_SHARE)
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

        state.lease_id = response.lease_id
        state.generation = generation
        state.granted_tokens = max(0, response.granted_tokens)
        state.granted_remaining_tokens = state.granted_tokens
        state.share_remaining_tokens = max(0, response.headroom_share_tokens)
        state.refresh_deadline = now + response.refresh_interval_s * self._rng.uniform(
            REFRESH_JITTER_MIN, REFRESH_JITTER_MAX
        )
        state.lease_deadline = now + response.lease_length_s
        state.posture_mode = response.posture.mode
        state.on_unreachable = response.posture.on_unreachable
        state.final_grant = bool(response.final_grant)
        state.declared_models.update(declared_models)
        state.renewal_in_flight = False
        state.consecutive_failures = 0
        state.next_attempt_at = 0.0
        state.run_ineligible = False
        state.ineligible_retry_at = 0.0
        self._store_snapshot(state, response)

        if state.final_grant:
            logger.warning(
                "lease.final_grant: no further renewal will be granted for this run "
                "— winding down to the per-call path at expiry"
            )
        return GrantOutcome.APPLIED

    def mark_ineligible(self, run_id: str, *, now: float, retry_after: float | None = None) -> None:
        """Route a run to the legacy path; ``retry_after`` None means forever.

        A Solwyn-side refusal (503 lease_unavailable, holder cap) must never
        block a call — the legacy per-call path has its own fail-open.
        """
        state = self._states.get(run_id)
        if state is None:
            state = LeaseState(run_id=run_id)
            self._states[run_id] = state
        self._drop_lease(state)
        state.run_ineligible = True
        state.ineligible_retry_at = math.inf if retry_after is None else now + retry_after

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
        depleted = state.granted_tokens - state.granted_remaining_tokens
        return depleted * RENEWAL_DEPLETION_DEN >= state.granted_tokens * RENEWAL_DEPLETION_NUM

    # ── internals ────────────────────────────────────────────────────────

    def _output_bound(self, output_bound: int | None) -> int:
        if output_bound is None or output_bound <= 0:
            return self.output_bound_default
        return output_bound

    def _reserve(self, state: LeaseState, call_id: str, tokens: int, now: float, pool: str) -> None:
        if state.lease_id is None:
            raise RuntimeError("cannot reserve against a run with no lease")
        state.reservations[call_id] = _Reservation(
            lease_id=state.lease_id, tokens=tokens, created_at=now, pool=pool
        )
        self._call_index[call_id] = state.run_id

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
