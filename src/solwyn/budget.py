"""Budget enforcement with cloud API check and local fallback.

BudgetEnforcer (sync) and AsyncBudgetEnforcer (async) handle pre-call
budget checks via the Solwyn cloud API, with local enforcement as
fallback when the cloud is unreachable.

Adapted from solwyn-core CostTracker (Redis -> HTTP cloud API).
Local in-process dict used as fallback when cloud is unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, cast, get_args

import httpx
from pydantic import BaseModel, ConfigDict

from solwyn._lease import (
    DEFAULT_OUTPUT_BOUND,
    INELIGIBLE_RETRY_AFTER_S,
    GrantOutcome,
    LeaseAdmission,
    LeaseDecision,
    LeaseLedger,
)
from solwyn._lifecycle import register_fork_reset, register_lease_holder
from solwyn._read_only_key import handle_read_only_key_error
from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    CircuitState,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MediaUsage,
    Modality,
    ProviderName,
    ServiceTier,
)
from solwyn.circuit_breaker import CircuitBreaker

# Fallback per-token cost when cloud API is unreachable.
DEFAULT_COST_PER_TOKEN: float = 0.00003

# Lease endpoints (PJ-2). The grant rides the caller's budget-check timeout;
# renewals and surrenders never sit on a customer call.
_LEASE_PATH = "/api/v1/budgets/lease"
_LEASE_RENEW_PATH = "/api/v1/budgets/lease/renew"
_LEASE_SURRENDER_PATH = "/api/v1/budgets/lease/surrender"

# A surrender is a courtesy (the server reclaims the float at expiry anyway):
# close() must never sit on a down control plane.
_SURRENDER_TIMEOUT_S = 1.0

# Renewals run off the caller's thread; give them the normal client timeout.
_RENEWAL_TIMEOUT_S = 5.0

# What a grant round-trip resolved to, from the admission path's point of view.
# "legacy" means "the plane answered, but this run takes the per-call path".
_GrantVerdict = Literal["applied", "legacy", "denied", "unreachable"]

# Sticky run denials protect brief Cloud outages but must not retain an
# unbounded stream of run UUIDs in a long-lived SDK process.
_MAX_STICKY_RUN_DENIALS = 128

# The contractual confirm tier values (derived from the ServiceTier literal,
# never hand-copied). Adapters echo arbitrary bounded strings; only these may
# ride the value-strict confirm wire.
_SERVICE_TIER_VALUES: frozenset[str] = frozenset(get_args(ServiceTier))

logger = logging.getLogger(__name__)


class BudgetCheckResult(BaseModel):
    """Result of a pre-flight budget check."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    remaining_budget: float
    project_id: str | None = None
    reservation_id: str | None = None
    mode: BudgetMode = BudgetMode.ALERT_ONLY
    warning: str | None = None
    budget_limit: float = 0.0
    current_usage: float = 0.0
    # PJ-2: set when the call drew on LEASE authority instead of a per-call
    # reservation. Exactly one of reservation_id / lease_id ever settles a call.
    lease_id: str | None = None
    # Server-provided RELATIVE price signal per provider for cost routing
    # (CostPolicy). The SDK never computes price. None on the cache path.
    price_hints: dict[str, float] | None = None
    failover_tuning_allowed: bool | None = None


class _BudgetEnforcerBase:
    """Sans-I/O base class for budget enforcement logic.

    Handles local cost tracking, caching, and request construction.
    Subclasses add the HTTP layer (sync or async).
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        budget_mode: BudgetMode = BudgetMode.ALERT_ONLY,
        fail_open: bool = True,
        cache_ttl: int = 5,
        control_plane_breaker: CircuitBreaker | None = None,
        holder_id: str | None = None,
        lease_enabled: bool = True,
        lease_output_bound_default: int = DEFAULT_OUTPUT_BOUND,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.budget_mode = budget_mode
        self.fail_open = fail_open
        self.cache_ttl = cache_ttl
        # Shared with the reporter: a streak of control-plane failures opens
        # this breaker so the check skips the network call and applies the
        # unreachable posture instantly (thread-safe; None = no breaker).
        self._control_plane_breaker = control_plane_breaker

        # Protects all mutable instance state from concurrent access.
        # The async subclass inherits this lock via base-class methods but
        # contention cannot occur — the event loop serializes coroutines.
        self._state_lock = threading.Lock()

        # Local cost tracking (fallback when cloud is unreachable)
        self._local_costs: dict[str, float] = {}

        # Last-known budget limit from cloud (survives cache expiry)
        self._last_known_budget_limit: float | None = None
        self._last_known_current_usage: float = 0.0

        # Cache for allow decisions (never cache deny)
        self._cached_response: BudgetCheckResponse | None = None
        self._cache_expires_at: float = 0.0

        # Sticky hard-deny state from the last authoritative cloud denial.
        # This is not a cache substitute: live cloud checks still run, but an
        # outage after a hard deny must not reopen spending.
        self._last_hard_deny_response: BudgetCheckResponse | None = None
        self._run_hard_deny_responses: OrderedDict[str, BudgetCheckResponse] = OrderedDict()

        # PJ-2 budget leases. The ledger is sans-I/O and takes no locks: every
        # call into it happens under ``_state_lock`` (the async subclass is
        # additionally serialized by the event loop). ``holder_id`` is the SDK
        # instance id — the identity the server binds each lease to.
        self._lease = LeaseLedger(
            holder_id=holder_id if holder_id is not None else str(uuid.uuid4()),
            enabled=lease_enabled,
            output_bound_default=lease_output_bound_default,
        )
        # Runs with a grant round-trip already on the wire. A concurrent burst
        # must not turn a cold start into N grants, and waiting on the in-flight
        # one would block a customer call — the losers take the legacy path.
        self._lease_grants_in_flight: set[str] = set()

    def _reset_after_fork_in_child(self) -> None:
        """Replace the state lock in a forked child (concrete classes also swap
        the inherited HTTP client — a shared socket across processes corrupts)."""
        self._state_lock = threading.Lock()
        # A child inherits no lease authority: the parent's grants are bound to
        # the parent's holder id, and its in-flight renewal flags are lies here.
        self._lease.on_fork_reset()
        self._lease_grants_in_flight = set()

    def _build_check_request(
        self,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] | None = None,
        fallback_models: list[str] | None = None,
        modality: Modality = "text",
        estimated_media: MediaUsage | None = None,
        agent_run_id: str | None = None,
    ) -> BudgetCheckRequest:
        """Build a BudgetCheckRequest for the cloud API.

        ``fallback_providers``/``fallback_models`` describe the configured
        failover chain (aligned element-for-element) as a hint to the API.
        Both default to empty lists; the caller's lists are never mutated.
        ``modality`` is ``"text"`` for chat calls; the media lifecycle passes the
        surface's modality (e.g. ``"embedding"``) so the API prices the pending
        call on the right basis. ``estimated_media`` carries a non-text surface's
        pre-flight non-token quantities (image counts, media seconds, character
        counts) for a precise check-time cost; None for chat/token calls.
        """
        return BudgetCheckRequest(
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            provider=ProviderName(provider),
            modality=modality,
            estimated_media=estimated_media,
            fallback_providers=[ProviderName(p) for p in (fallback_providers or [])],
            fallback_models=list(fallback_models or []),
            agent_run_id=agent_run_id,
            failover_directive_version="1",
        )

    def _should_use_cache(self) -> bool:
        """Return True if we have a valid cached allow response."""
        with self._state_lock:
            return (
                self._cached_response is not None
                and self._cached_response.allowed
                and time.monotonic() < self._cache_expires_at
            )

    def _cache_response(
        self,
        response: BudgetCheckResponse,
        *,
        agent_run_id: str | None = None,
    ) -> None:
        """Cache an allow response. Never cache deny responses.

        Always updates the last-known budget limit (from both allow and deny)
        so that local enforcement can use it when the cloud becomes unreachable.
        Scoped checks never read or populate the global allow cache. Run-specific
        hard denials are sticky only for their raw run id; project-period hard
        denials remain global and invalidate a stale global allow.
        """
        with self._state_lock:
            # Always remember the limit for local enforcement fallback
            self._last_known_budget_limit = response.budget_limit
            self._last_known_current_usage = response.current_usage

            if response.allowed:
                self._last_hard_deny_response = None
                if agent_run_id is not None:
                    self._run_hard_deny_responses.pop(agent_run_id, None)
                else:
                    self._cached_response = response
                    self._cache_expires_at = time.monotonic() + self.cache_ttl
                return

            if response.mode != BudgetMode.HARD_DENY:
                self._last_hard_deny_response = None
                if agent_run_id is not None:
                    self._run_hard_deny_responses.pop(agent_run_id, None)
                self._cached_response = None
                self._cache_expires_at = 0.0
                return

            if agent_run_id is not None and response.denied_by_period == "agent_run":
                # The run limit denied, so every project period passed. Replace
                # stale global state with the denial scoped to this run only.
                self._last_hard_deny_response = None
                self._run_hard_deny_responses.pop(agent_run_id, None)
                self._run_hard_deny_responses[agent_run_id] = response
                if len(self._run_hard_deny_responses) > _MAX_STICKY_RUN_DENIALS:
                    self._run_hard_deny_responses.popitem(last=False)
                return

            self._cached_response = None
            self._cache_expires_at = 0.0
            self._last_hard_deny_response = response
            # Deny responses are never cached as freshness-skipping allows/denies.

    def _build_prior_hard_deny_unavailable_result(
        self,
        agent_run_id: str | None = None,
    ) -> BudgetCheckResult | None:
        """Return a denial if cloud is down after an authoritative hard deny."""
        with self._state_lock:
            response = self._last_hard_deny_response
            if response is None and agent_run_id is not None:
                response = self._run_hard_deny_responses.get(agent_run_id)
                if response is not None:
                    self._run_hard_deny_responses.move_to_end(agent_run_id)
        if response is None:
            return None
        # Surfaced on both the sync and async check_budget paths, which both
        # return the preserved denial through this builder. The `warning` field
        # below carries the same text for programmatic callers.
        logger.warning(
            "Cloud API unreachable; preserving prior hard deny: $%.2f/$%.2f used",
            response.current_usage,
            response.budget_limit,
        )
        return BudgetCheckResult(
            allowed=False,
            remaining_budget=response.remaining_budget,
            project_id=response.project_id,
            mode=response.mode,
            warning=(
                "Cloud API unreachable; preserving prior hard deny: "
                f"${response.current_usage:.2f}/${response.budget_limit:.2f} used"
            ),
            budget_limit=response.budget_limit,
            current_usage=response.current_usage,
        )

    def _build_unreachable_result(
        self, estimated_input_tokens: int, agent_run_id: str | None
    ) -> BudgetCheckResult:
        """Posture when the control plane is unreachable (or breaker-open)."""
        prior_hard_deny = self._build_prior_hard_deny_unavailable_result(agent_run_id)
        if prior_hard_deny is not None:
            return prior_hard_deny
        if self.fail_open:
            return self._build_fail_open_result(estimated_input_tokens)
        return self._build_local_enforcement_result(estimated_input_tokens)

    def _track_local_cost(self, cost: float) -> None:
        """Track a cost in the local fallback dict."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._state_lock:
            self._local_costs[today] = self._local_costs.get(today, 0.0) + cost

    def _get_local_remaining(self, budget_limit: float) -> float:
        """Get remaining budget from local tracking."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._state_lock:
            current = self._local_costs.get(today, 0.0)
        return max(0.0, budget_limit - current)

    def _get_local_current(self) -> float:
        """Get current local spend for today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._state_lock:
            return self._local_costs.get(today, 0.0)

    def _build_result_from_response(self, response: BudgetCheckResponse) -> BudgetCheckResult:
        """Convert a cloud API response into a BudgetCheckResult.

        Applies cloud response mode logic:
        - allowed -> return allowed=True
        - denied + alert_only -> return allowed=True with warning
        - denied + hard_deny -> return allowed=False
        """
        # Server-provided RELATIVE price hints (cost routing); str-keyed for the
        # routing layer. None when the server has not provided them. The SDK
        # never computes price — it only forwards this signal to CostPolicy.
        price_hints = (
            {provider.value: hint for provider, hint in response.price_hints.items()}
            if response.price_hints is not None
            else None
        )
        failover_tuning_allowed = (
            response.failover_directive.failover_tuning_allowed
            if response.failover_directive is not None
            else None
        )

        if response.allowed:
            return BudgetCheckResult(
                allowed=True,
                remaining_budget=response.remaining_budget,
                project_id=response.project_id,
                reservation_id=response.reservation_id,
                mode=response.mode,
                budget_limit=response.budget_limit,
                current_usage=response.current_usage,
                price_hints=price_hints,
                failover_tuning_allowed=failover_tuning_allowed,
            )

        # Denied by cloud
        if response.mode == BudgetMode.ALERT_ONLY:
            logger.warning(
                "Budget limit reached (alert_only mode): limit=$%.2f, usage=$%.2f",
                response.budget_limit,
                response.current_usage,
            )
            return BudgetCheckResult(
                allowed=True,
                remaining_budget=response.remaining_budget,
                project_id=response.project_id,
                reservation_id=response.reservation_id,
                mode=response.mode,
                warning=(
                    f"Budget limit reached: "
                    f"${response.current_usage:.2f}/${response.budget_limit:.2f} used"
                ),
                budget_limit=response.budget_limit,
                current_usage=response.current_usage,
                price_hints=price_hints,
                failover_tuning_allowed=failover_tuning_allowed,
            )

        # hard_deny
        return BudgetCheckResult(
            allowed=False,
            remaining_budget=response.remaining_budget,
            project_id=response.project_id,
            mode=response.mode,
            warning=(
                f"Budget exceeded: ${response.current_usage:.2f}/${response.budget_limit:.2f} used"
            ),
            budget_limit=response.budget_limit,
            current_usage=response.current_usage,
            failover_tuning_allowed=failover_tuning_allowed,
        )

    def _build_fail_open_result(self, estimated_input_tokens: int) -> BudgetCheckResult:
        """Build a fail-open result when the cloud is unreachable."""
        self._track_local_cost(DEFAULT_COST_PER_TOKEN * estimated_input_tokens)
        return BudgetCheckResult(
            allowed=True,
            remaining_budget=0.0,
            mode=self.budget_mode,
            warning="Cloud API unreachable; proceeding in fail-open mode",
        )

    def _build_local_enforcement_result(
        self,
        estimated_input_tokens: int,
    ) -> BudgetCheckResult:
        """Enforce budget locally when cloud is unreachable and fail_open=False.

        Uses the last-known budget limit from the most recent cloud response.
        If the cloud has never been reached, denies the request (fail-closed)
        since we have no limit to enforce against.
        """
        # Use last-known limit from cloud, or deny if we've never heard from cloud
        if self._last_known_budget_limit is None:
            return BudgetCheckResult(
                allowed=False,
                remaining_budget=0.0,
                mode=self.budget_mode,
                warning=(
                    "Cloud unreachable and no prior budget limit known; "
                    "denying request (fail-closed)"
                ),
            )

        limit = self._last_known_budget_limit
        current = self._get_local_current()
        remaining = max(0.0, limit - current)
        estimated_cost = DEFAULT_COST_PER_TOKEN * estimated_input_tokens

        if current + estimated_cost > limit:
            return BudgetCheckResult(
                allowed=False,
                remaining_budget=remaining,
                mode=self.budget_mode,
                warning=(
                    f"Cloud unreachable; local enforcement denies: "
                    f"${current:.2f} + ${estimated_cost:.2f} > ${limit:.2f}"
                ),
                budget_limit=limit,
                current_usage=current,
            )

        # Within local limit
        self._track_local_cost(estimated_cost)
        return BudgetCheckResult(
            allowed=True,
            remaining_budget=max(0.0, limit - current - estimated_cost),
            mode=self.budget_mode,
            warning="Cloud API unreachable; enforcing locally",
            budget_limit=limit,
            current_usage=current + estimated_cost,
        )

    # ── lease admission (sans-I/O halves; the HTTP lives on the subclasses) ──

    def _lease_path_applies(self, agent_run_id: str | None) -> bool:
        """Whether this call may consult the lease ledger at all.

        Step 1 of the admission algorithm: a sticky hard deny keeps the run on
        the authoritative per-call path (a live check re-decides it, an outage
        preserves it) — local lease authority never outranks a server denial.
        """
        if agent_run_id is None or not self._lease.enabled:
            return False
        with self._state_lock:
            if self._last_hard_deny_response is not None:
                return False
            return agent_run_id not in self._run_hard_deny_responses

    def _lease_breaker_open(self) -> bool:
        """Is the control plane BELIEVED unreachable? (inspection, never consumption).

        ``admit()`` would consume a HALF_OPEN probe slot just to answer this, so
        the ladder reads the frozen snapshot instead. An OPEN-but-recovery-
        eligible breaker counts as reachable: the legacy check that follows owns
        the probe, and treating fewer situations as an outage only ever routes a
        call to the server, never blocks it.
        """
        breaker = self._control_plane_breaker
        if breaker is None:
            return False
        snapshot = breaker.get_state()
        return snapshot.state == CircuitState.OPEN and not snapshot.recovery_eligible

    def _lease_reserve_estimate(
        self, estimated_input_tokens: int, estimated_output_bound: int | None
    ) -> int:
        """The bounded token demand the ledger reserves for one call.

        Mirrors ``LeaseLedger._output_bound``: a missing (or non-positive)
        caller cap uses the configured default rather than trusting an
        unbounded response to stay small.
        """
        bound = (
            estimated_output_bound
            if estimated_output_bound is not None and estimated_output_bound > 0
            else self._lease.output_bound_default
        )
        return max(0, estimated_input_tokens) + bound

    def _admit_lease(
        self,
        *,
        agent_run_id: str,
        call_id: str,
        estimated_input_tokens: int,
        model: str,
        estimated_output_bound: int | None,
        modality: Modality,
        estimated_media: MediaUsage | None,
        fallback_models: Sequence[str],
    ) -> LeaseAdmission:
        """Run the ladder for one call, entirely inside one lock section."""
        breaker_open = self._lease_breaker_open()
        with self._state_lock:
            return self._lease.admit(
                run_id=agent_run_id,
                call_id=call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                now=time.monotonic(),
                breaker_open=breaker_open,
                output_bound=estimated_output_bound,
                modality=modality,
                has_estimated_media=estimated_media is not None,
                fallback_models=list(fallback_models),
            )

    def _build_grant_request(
        self,
        *,
        agent_run_id: str,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: Sequence[str],
        fallback_models: Sequence[str],
    ) -> LeaseGrantRequest:
        """Declare the run's model set and this client's unreachable posture."""
        return LeaseGrantRequest(
            agent_run_id=agent_run_id,
            holder_id=self._lease.holder_id,
            model=model,
            provider=ProviderName(provider),
            fallback_providers=[ProviderName(p) for p in fallback_providers],
            fallback_models=list(fallback_models),
            fail_open=self.fail_open,
            estimated_input_tokens=max(0, estimated_input_tokens),
        )

    def _claim_grant_slot(self, agent_run_id: str) -> bool:
        """Claim the run's single in-flight grant slot (False = someone has it)."""
        with self._state_lock:
            if agent_run_id in self._lease_grants_in_flight:
                return False
            self._lease_grants_in_flight.add(agent_run_id)
            return True

    def _release_grant_slot(self, agent_run_id: str) -> None:
        with self._state_lock:
            self._lease_grants_in_flight.discard(agent_run_id)

    def _apply_lease_response(
        self,
        agent_run_id: str,
        response: LeaseGrantResponse,
        declared_models: Sequence[str],
    ) -> GrantOutcome:
        """Install a grant/renew response under the ledger's serialization."""
        with self._state_lock:
            return self._lease.apply_grant_response(
                agent_run_id,
                response,
                now=time.monotonic(),
                declared_models=declared_models,
            )

    def _classify_lease_failure(
        self, exc: Exception, agent_run_id: str, *, renewal: bool
    ) -> _GrantVerdict:
        """Map a failed lease round-trip onto a verdict, feeding the breaker.

        A REFUSAL is not an outage: 409 (holder cap) and 503 (lease_unavailable)
        are deliberate answers, so the plane is credited and the run simply
        drops to the per-call path. Only a transport failure or a non-503 5xx
        counts as unreachable — the posture ladder then owns the call.
        """
        breaker = self._control_plane_breaker
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        responded = handle_read_only_key_error(exc) or (status is not None and 400 <= status < 500)
        if status == 503:
            responded = True
        if responded:
            if breaker is not None:
                breaker.record_success()
        else:
            if breaker is not None:
                breaker.record_failure()
            logger.warning(
                "lease.%s_failed: %s",
                "renew" if renewal else "grant",
                type(exc).__name__,
            )
            if renewal:
                # The lease stays valid until its own deadline — but the
                # in-flight flag must clear, under a backoff, or the run never
                # renews again AND every later admission would retry at once.
                with self._state_lock:
                    self._lease.renewal_failed(agent_run_id, time.monotonic())
            return "unreachable"

        now = time.monotonic()
        with self._state_lock:
            if renewal and status in (404, 409):
                # The server no longer recognizes this lease (or a newer
                # generation exists): drop it and re-grant on the next call.
                self._lease.drop(agent_run_id)
            elif renewal:
                self._lease.renewal_failed(agent_run_id, now)
            elif status == 409:
                # lease_holder_cap_exceeded — permanent for this run.
                self._lease.mark_ineligible(agent_run_id, now=now, retry_after=None)
            else:
                self._lease.mark_ineligible(
                    agent_run_id, now=now, retry_after=INELIGIBLE_RETRY_AFTER_S
                )
        if status is not None:
            logger.debug("lease.%s_refused: status=%d", "renew" if renewal else "grant", status)
        return "legacy"

    def _lease_deny_result(
        self, agent_run_id: str, response: LeaseGrantResponse
    ) -> BudgetCheckResult:
        """Feed an authoritative lease denial to the UNCHANGED sticky machinery."""
        denial = BudgetCheckResponse(
            allowed=False,
            remaining_budget=response.remaining_budget,
            reservation_id=None,
            mode=response.mode,
            budget_limit=response.budget_limit,
            current_usage=response.current_usage,
            denied_by_period=response.denied_by_period,
            project_id=response.project_id,
        )
        self._cache_response(denial, agent_run_id=agent_run_id)
        return self._build_result_from_response(denial)

    def _result_from_admission(
        self, agent_run_id: str, admission: LeaseAdmission
    ) -> BudgetCheckResult:
        """Map an admitting / denying ladder verdict onto a check result.

        The display numbers come from the last grant response — the SDK never
        computes them, and a DENY here is always the CUSTOMER's own mode
        verdict at true exhaustion, never a Solwyn-availability verdict.
        """
        with self._state_lock:
            snapshot = self._lease.snapshot_for(agent_run_id)
        return BudgetCheckResult(
            allowed=admission.admitted,
            remaining_budget=snapshot.remaining_budget if snapshot is not None else 0.0,
            project_id=snapshot.project_id if snapshot is not None else None,
            reservation_id=None,
            lease_id=admission.lease_id if admission.admitted else None,
            mode=(
                admission.mode
                if admission.mode is not None
                else (snapshot.mode if snapshot is not None else self.budget_mode)
            ),
            warning=admission.warning,
            budget_limit=snapshot.budget_limit if snapshot is not None else 0.0,
            current_usage=snapshot.current_usage if snapshot is not None else 0.0,
        )

    def _record_uncounted_cold_start(self, agent_run_id: str, tokens: int) -> None:
        """Tally a fail-open admission made with no lease state at all."""
        with self._state_lock:
            self._lease.record_uncounted(agent_run_id, tokens)

    def release_reservation(self, call_id: str) -> None:
        """Hand a lease reservation back when the call will never settle.

        Safe to call for any call id: a call that never drew on lease authority
        (legacy reservation, non-run traffic, uncounted admit) is a no-op.
        """
        with self._state_lock:
            self._lease.release(call_id)

    def lease_surrender_payloads(self) -> list[LeaseSurrenderRequest]:
        """Drain every held lease into surrender payloads (best-effort release).

        The runs are discarded as they are drained, so a close() followed by
        the interpreter-exit hook cannot surrender the same lease twice.
        """
        payloads: list[LeaseSurrenderRequest] = []
        with self._state_lock:
            for run_id in self._lease.active_run_ids():
                request = self._lease.build_surrender_request(run_id)
                if request is not None:
                    payloads.append(request)
                self._lease.discard(run_id)
        return payloads

    def _build_renewal(
        self,
        agent_run_id: str,
        *,
        model: str,
        provider: str,
        fallback_providers: Sequence[str],
        fallback_models: Sequence[str],
    ) -> LeaseRenewRequest | None:
        """Build + arm a renewal, or None when there is nothing to renew."""
        with self._state_lock:
            request = self._lease.build_renewal_request(
                agent_run_id,
                model=model,
                provider=ProviderName(provider),
                fallback_providers=[ProviderName(p) for p in fallback_providers],
                fallback_models=list(fallback_models),
            )
            if request is None:
                return None
            self._lease.renewal_sent(agent_run_id)
            return request

    def _finish_renewal(
        self,
        agent_run_id: str,
        response: LeaseGrantResponse,
        declared_models: Sequence[str],
    ) -> None:
        """Apply a renewal response; a denial feeds the sticky-deny machinery."""
        outcome = self._apply_lease_response(agent_run_id, response, declared_models)
        if outcome is GrantOutcome.DENIED:
            self._lease_deny_result(agent_run_id, response)
        elif outcome is GrantOutcome.STALE:
            # A stale response leaves the in-flight flag set; clear it (with a
            # backoff) so the lease can still renew before its deadline.
            with self._state_lock:
                self._lease.renewal_failed(agent_run_id, time.monotonic())

    def _install_grant(
        self,
        agent_run_id: str,
        resp: httpx.Response,
        *,
        declared_models: list[str],
    ) -> tuple[_GrantVerdict, LeaseGrantResponse | None]:
        """Validate + apply a grant body. A malformed body is never fatal."""
        try:
            response = LeaseGrantResponse.model_validate(resp.json())
        except Exception as exc:
            logger.warning("lease.grant_response_unreadable: %s", type(exc).__name__)
            with self._state_lock:
                self._lease.mark_ineligible(
                    agent_run_id, now=time.monotonic(), retry_after=INELIGIBLE_RETRY_AFTER_S
                )
            return "legacy", None
        outcome = self._apply_lease_response(agent_run_id, response, declared_models)
        if outcome is GrantOutcome.APPLIED:
            return "applied", response
        if outcome is GrantOutcome.DENIED:
            return "denied", response
        return "legacy", response

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for cloud API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_confirm_request(
        self,
        *,
        model: str,
        token_details: TokenDetails,
        provider: str,
        call_id: str,
        reservation_id: str | None = None,
        lease_id: str | None = None,
        is_provider_fallback: bool = False,
        provider_region: str | None = None,
        service_tier: str | None = None,
        modality: Modality = "text",
        media_usage: MediaUsage | None = None,
    ) -> BudgetConfirmRequest:
        """Build a validated confirm request for fire-and-forget callers.

        Stream completion builds this synchronously (no I/O) and enqueues
        it on the reporter thread, avoiding a blocking httpx.post. ``provider``
        is the provider that actually served the call (required).
        ``lease_id`` settles a call that drew on LEASE authority (PJ-2): the
        settlement key is exclusive, so a lease-settled confirm carries no
        ``reservation_id``, and building one ALSO trues the call's local
        reservation up from its bound to the actual token usage. Lease
        authority wins if a caller somehow supplies both keys — the reservation
        it drew down is the one that must be settled.
        ``call_id`` is the required per-call reconciliation join key.
        ``provider_region`` is the served endpoint's region for per-region
        pricing (Bedrock); None for providers without regional pricing.
        ``service_tier`` is the RAW tier echoed by the provider response —
        recognized values settle the enforcement counter at the tier-repriced
        rate; a novel echo narrows to None (Standard rates) so the strict
        Cloud-API confirm model never 422s and strands the reservation. The
        raw echo still reaches the API on the MetadataEvent. ``modality`` is
        ``"text"`` for chat confirms; the media lifecycle passes the surface's
        modality (e.g. ``"embedding"``) so the API settles on the right basis.
        ``media_usage`` carries a non-text surface's settled non-token quantities
        so the API settles the enforcement counter on the per-unit basis; None
        for chat/token confirms.
        """
        if not call_id:
            raise RuntimeError("call_id is required for budget confirm reconciliation")
        if service_tier is not None and service_tier not in _SERVICE_TIER_VALUES:
            logger.debug("budget.confirm_service_tier_unrecognized: settling at Standard rates")
            service_tier = None
        if lease_id is not None:
            # Settlement of a lease-funded call: the local reservation moves
            # from its pre-call bound to the tokens actually spent, and the
            # exclusive wire key becomes the lease.
            reservation_id = None
            with self._state_lock:
                self._lease.true_up(call_id, token_details.total_tokens)
        return BudgetConfirmRequest(
            reservation_id=reservation_id,
            lease_id=lease_id,
            model=model,
            provider=ProviderName(provider),
            modality=modality,
            is_provider_fallback=is_provider_fallback,
            token_details=token_details,
            media_usage=media_usage,
            call_id=call_id,
            provider_region=provider_region,
            service_tier=cast("ServiceTier | None", service_tier),
        )


class BudgetEnforcer(_BudgetEnforcerBase):
    """Synchronous budget enforcer using httpx.Client.

    Checks the Solwyn cloud API before each LLM call.
    Falls back to local enforcement when the cloud is unreachable.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        budget_mode: BudgetMode = BudgetMode.ALERT_ONLY,
        fail_open: bool = True,
        cache_ttl: int = 5,
        control_plane_breaker: CircuitBreaker | None = None,
        holder_id: str | None = None,
        lease_enabled: bool = True,
        lease_output_bound_default: int = DEFAULT_OUTPUT_BOUND,
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
            control_plane_breaker=control_plane_breaker,
            holder_id=holder_id,
            lease_enabled=lease_enabled,
            lease_output_bound_default=lease_output_bound_default,
        )
        self._http = httpx.Client(timeout=5.0)
        # Daemon renewal workers, tracked so close() can let an in-flight
        # renewal finish before the transport goes away.
        self._renewal_threads: set[threading.Thread] = set()
        register_fork_reset(self)
        register_lease_holder(self)

    def _reset_after_fork_in_child(self) -> None:
        """Fresh state lock AND a fresh sync client for the forked child.

        The inherited ``httpx.Client`` is abandoned, never closed — the parent
        still owns those sockets.
        """
        super()._reset_after_fork_in_child()
        self._http = httpx.Client(timeout=5.0)
        # Only the forking thread survives in the child: the parent's renewal
        # workers do not exist here.
        self._renewal_threads = set()

    def check_budget(
        self,
        *,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] = [],  # noqa: B006 — read-only; never mutated
        fallback_models: list[str] = [],  # noqa: B006 — read-only; never mutated
        timeout: float | None = None,
        modality: Modality = "text",
        estimated_media: MediaUsage | None = None,
        agent_run_id: str | None = None,
        call_id: str | None = None,
        estimated_output_bound: int | None = None,
    ) -> BudgetCheckResult:
        """Check whether a call is within budget.

        ``fallback_providers``/``fallback_models`` describe the configured
        failover chain (aligned element-for-element) as a hint to the API.
        ``modality`` is ``"text"`` for chat; the media lifecycle passes the
        surface's modality (e.g. ``"embedding"``). ``estimated_media`` carries a
        non-text surface's pre-flight non-token quantities for a precise
        check-time cost; None for chat/token calls.

        Run-scoped, token-billed traffic meets the LEASE path first (PJ-2):
        ``call_id`` keys the reservation the call draws down (so settlement can
        true it up) and ``estimated_output_bound`` is the call's own
        ``max_tokens``-family cap. Everything else — non-run traffic, media,
        an ineligible run, the kill switch — falls through to the per-call
        path below, unchanged.

        Behaviour matrix:
        - Cloud reachable + allowed: return allowed=True
        - Cloud reachable + denied + alert_only: return allowed=True + warning
        - Cloud reachable + denied + hard_deny: return allowed=False
        - Cloud unreachable after hard_deny: return allowed=False
        - Cloud unreachable + fail_open=True: return allowed=True + warning
        - Cloud unreachable + fail_open=False: enforce locally
        """
        if self._lease_path_applies(agent_run_id):
            if agent_run_id is None:
                raise RuntimeError("lease path requires an agent_run_id")
            leased = self._check_lease(
                agent_run_id=agent_run_id,
                call_id=call_id if call_id else str(uuid.uuid4()),
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=timeout,
                modality=modality,
                estimated_media=estimated_media,
                estimated_output_bound=estimated_output_bound,
            )
            if leased is not None:
                return leased

        # Use cache if valid (only allow decisions are cached).
        # Snapshot under the lock to avoid a TOCTOU race between the validity
        # check and reading the cached fields.
        if agent_run_id is None:
            with self._state_lock:
                cached = self._cached_response
                if (
                    cached is not None
                    and cached.allowed
                    and time.monotonic() < self._cache_expires_at
                ):
                    return BudgetCheckResult(
                        allowed=True,
                        remaining_budget=cached.remaining_budget,
                        project_id=cached.project_id,
                        reservation_id=None,  # each call needs its own reservation
                        mode=cached.mode,
                        budget_limit=cached.budget_limit,
                        current_usage=cached.current_usage,
                    )

        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Negative cache: a recent streak of control-plane failures means
            # the posture applies instantly instead of paying the timeout.
            logger.debug("budget.check_skipped_breaker_open")
            return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

        request = self._build_check_request(
            estimated_input_tokens,
            model,
            provider,
            fallback_providers,
            fallback_models,
            modality,
            estimated_media,
            agent_run_id,
        )

        try:
            if timeout is not None:
                resp = self._http.post(
                    f"{self.api_url}/api/v1/budgets/check",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=timeout,
                )
            else:
                resp = self._http.post(
                    f"{self.api_url}/api/v1/budgets/check",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                )
            resp.raise_for_status()

            cloud_response = BudgetCheckResponse.model_validate(resp.json())
            self._cache_response(cloud_response, agent_run_id=agent_run_id)
            if breaker is not None:
                breaker.record_success()
            return self._build_result_from_response(cloud_response)

        except Exception as exc:
            # A read-only-key error means the control plane RESPONDED — record
            # success. Anything else is an outage: record failure. Log the
            # exception TYPE only (never interpolate a body that could carry
            # response text).
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
            else:
                if breaker is not None:
                    breaker.record_failure()
                logger.warning("Cloud API budget check failed: %s", type(exc).__name__)

            return self._build_unreachable_result(estimated_input_tokens, agent_run_id)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    # ── lease path ───────────────────────────────────────────────────────

    def _check_lease(
        self,
        *,
        agent_run_id: str,
        call_id: str,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
        timeout: float | None,
        modality: Modality,
        estimated_media: MediaUsage | None,
        estimated_output_bound: int | None,
    ) -> BudgetCheckResult | None:
        """Admit one run-scoped call on lease authority, or None for the legacy path.

        The ladder (steps 2-6) lives in the ledger; this method performs the
        I/O it prescribes — a blocking grant when there is no usable lease, an
        ASYNC renewal when one is due — and maps the verdict onto a result.
        """
        admission = self._admit_lease(
            agent_run_id=agent_run_id,
            call_id=call_id,
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            estimated_output_bound=estimated_output_bound,
            modality=modality,
            estimated_media=estimated_media,
            fallback_models=fallback_models,
        )

        if admission.decision is LeaseDecision.NEED_GRANT:
            verdict, response = self._grant_lease(
                agent_run_id=agent_run_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=timeout,
            )
            if verdict == "denied" and response is not None:
                return self._lease_deny_result(agent_run_id, response)
            if verdict == "unreachable":
                if self.fail_open:
                    # Nothing counts this call anywhere: no lease exists yet, so
                    # the tally is what the next lease's first renewal owes.
                    self._record_uncounted_cold_start(
                        agent_run_id,
                        self._lease_reserve_estimate(
                            estimated_input_tokens, estimated_output_bound
                        ),
                    )
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)
            if verdict != "applied":
                return None
            admission = self._admit_lease(
                agent_run_id=agent_run_id,
                call_id=call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
            )
            if admission.decision is LeaseDecision.NEED_GRANT:
                # One grant per call, always: never loop against the server.
                return None

        if admission.renewal_due:
            self._start_renewal(
                agent_run_id,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )

        if admission.decision is LeaseDecision.LEGACY_CHECK:
            return None
        return self._result_from_admission(agent_run_id, admission)

    def _grant_lease(
        self,
        *,
        agent_run_id: str,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
        timeout: float | None,
    ) -> tuple[_GrantVerdict, LeaseGrantResponse | None]:
        """Blocking grant round-trip on the caller's thread (budget-check timeout)."""
        if not self._claim_grant_slot(agent_run_id):
            # Another caller is already granting for this run; pay one per-call
            # check instead of stacking a second grant on the same cold start.
            return "legacy", None
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.grant_skipped_breaker_open")
                return "unreachable", None
            request = self._build_grant_request(
                agent_run_id=agent_run_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )
            payload = request.model_dump(mode="json")
            try:
                if timeout is not None:
                    resp = self._http.post(
                        f"{self.api_url}{_LEASE_PATH}",
                        json=payload,
                        headers=self._auth_headers(),
                        timeout=timeout,
                    )
                else:
                    resp = self._http.post(
                        f"{self.api_url}{_LEASE_PATH}",
                        json=payload,
                        headers=self._auth_headers(),
                    )
                resp.raise_for_status()
            except Exception as exc:
                return self._classify_lease_failure(exc, agent_run_id, renewal=False), None

            if breaker is not None:
                breaker.record_success()
            return self._install_grant(
                agent_run_id, resp, declared_models=[model, *fallback_models]
            )
        finally:
            if breaker is not None:
                breaker.release_probe(admission)
            self._release_grant_slot(agent_run_id)

    def _start_renewal(
        self,
        agent_run_id: str,
        *,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
    ) -> None:
        """Fire a renewal OFF the caller's thread — admission never waits on it."""
        request = self._build_renewal(
            agent_run_id,
            model=model,
            provider=provider,
            fallback_providers=fallback_providers,
            fallback_models=fallback_models,
        )
        if request is None:
            return
        thread = threading.Thread(
            target=self._renew_lease,
            args=(agent_run_id, request, [model, *fallback_models]),
            name="solwyn-lease-renew",
            daemon=True,
        )
        with self._state_lock:
            self._renewal_threads = {t for t in self._renewal_threads if t.is_alive()}
            self._renewal_threads.add(thread)
        thread.start()

    def _renew_lease(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        declared_models: list[str],
    ) -> None:
        """Renewal worker: breaker-guarded, never raises, never blocks a call."""
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.renew_skipped_breaker_open")
                with self._state_lock:
                    self._lease.renewal_failed(agent_run_id, time.monotonic())
                return
            try:
                resp = self._http.post(
                    f"{self.api_url}{_LEASE_RENEW_PATH}",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=_RENEWAL_TIMEOUT_S,
                )
                resp.raise_for_status()
            except Exception as exc:
                self._classify_lease_failure(exc, agent_run_id, renewal=True)
                return
            if breaker is not None:
                breaker.record_success()
            try:
                response = LeaseGrantResponse.model_validate(resp.json())
            except Exception as exc:
                logger.warning("lease.renew_response_unreadable: %s", type(exc).__name__)
                with self._state_lock:
                    self._lease.renewal_failed(agent_run_id, time.monotonic())
                return
            self._finish_renewal(agent_run_id, response, declared_models)
        except Exception as exc:  # pragma: no cover — a worker must never raise
            logger.warning("lease.renew_worker_failed: %s", type(exc).__name__)
        finally:
            if breaker is not None:
                breaker.release_probe(admission)

    def _surrender_leases(self) -> None:
        """DHCPRELEASE-style: hand every held lease back, best-effort."""
        for request in self.lease_surrender_payloads():
            breaker = self._control_plane_breaker
            admission = breaker.admit() if breaker is not None else None
            try:
                if admission is not None and not admission.allowed:
                    logger.debug("lease.surrender_skipped_breaker_open")
                    continue
                self._http.post(
                    f"{self.api_url}{_LEASE_SURRENDER_PATH}",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=_SURRENDER_TIMEOUT_S,
                ).raise_for_status()
                if breaker is not None:
                    breaker.record_success()
            except Exception as exc:
                # The server reclaims an unsurrendered lease at its deadline;
                # a failed courtesy release must never surface to the caller.
                logger.debug("lease.surrender_failed: %s", type(exc).__name__)
                if breaker is not None and not handle_read_only_key_error(exc):
                    breaker.record_failure()
            finally:
                if breaker is not None:
                    breaker.release_probe(admission)

    def close(self) -> None:
        """Surrender held leases, then close the underlying HTTP client."""
        with self._state_lock:
            threads = list(self._renewal_threads)
        for thread in threads:
            # Bounded: a renewal in flight when the transport closes would only
            # log a failure, but letting it land keeps the server's view fresh.
            thread.join(timeout=_SURRENDER_TIMEOUT_S)
        self._surrender_leases()
        self._http.close()


class AsyncBudgetEnforcer(_BudgetEnforcerBase):
    """Asynchronous budget enforcer using httpx.AsyncClient.

    Same API and behaviour as BudgetEnforcer, but async.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        budget_mode: BudgetMode = BudgetMode.ALERT_ONLY,
        fail_open: bool = True,
        cache_ttl: int = 5,
        control_plane_breaker: CircuitBreaker | None = None,
        holder_id: str | None = None,
        lease_enabled: bool = True,
        lease_output_bound_default: int = DEFAULT_OUTPUT_BOUND,
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
            control_plane_breaker=control_plane_breaker,
            holder_id=holder_id,
            lease_enabled=lease_enabled,
            lease_output_bound_default=lease_output_bound_default,
        )
        self._http = httpx.AsyncClient(timeout=5.0)
        # Renewal tasks, held strongly so the loop cannot collect them mid-flight.
        self._renewal_tasks: set[asyncio.Task[None]] = set()
        register_fork_reset(self)
        register_lease_holder(self)

    def _reset_after_fork_in_child(self) -> None:
        """Fresh state lock AND a fresh async client for the forked child.

        The inherited ``httpx.AsyncClient`` is abandoned, never closed — the
        parent still owns those sockets, and the child's event loop is new.
        """
        super()._reset_after_fork_in_child()
        self._http = httpx.AsyncClient(timeout=5.0)
        # The parent's tasks belong to a loop that does not exist here.
        self._renewal_tasks = set()

    async def check_budget(
        self,
        *,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] = [],  # noqa: B006 — read-only; never mutated
        fallback_models: list[str] = [],  # noqa: B006 — read-only; never mutated
        timeout: float | None = None,
        modality: Modality = "text",
        estimated_media: MediaUsage | None = None,
        agent_run_id: str | None = None,
        call_id: str | None = None,
        estimated_output_bound: int | None = None,
    ) -> BudgetCheckResult:
        """Async version of budget check. See BudgetEnforcer.check_budget."""
        if self._lease_path_applies(agent_run_id):
            if agent_run_id is None:
                raise RuntimeError("lease path requires an agent_run_id")
            leased = await self._check_lease(
                agent_run_id=agent_run_id,
                call_id=call_id if call_id else str(uuid.uuid4()),
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=timeout,
                modality=modality,
                estimated_media=estimated_media,
                estimated_output_bound=estimated_output_bound,
            )
            if leased is not None:
                return leased

        if agent_run_id is None and self._should_use_cache():
            cached = self._cached_response
            if cached is None:
                raise RuntimeError("_should_use_cache returned True but cache is None")
            return BudgetCheckResult(
                allowed=True,
                remaining_budget=cached.remaining_budget,
                project_id=cached.project_id,
                reservation_id=None,  # Don't reuse — each call needs its own reservation
                mode=cached.mode,
                budget_limit=cached.budget_limit,
                current_usage=cached.current_usage,
            )

        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Negative cache: a recent streak of control-plane failures means
            # the posture applies instantly instead of paying the timeout.
            logger.debug("budget.check_skipped_breaker_open")
            return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

        request = self._build_check_request(
            estimated_input_tokens,
            model,
            provider,
            fallback_providers,
            fallback_models,
            modality,
            estimated_media,
            agent_run_id,
        )

        try:
            if timeout is not None:
                resp = await self._http.post(
                    f"{self.api_url}/api/v1/budgets/check",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=timeout,
                )
            else:
                resp = await self._http.post(
                    f"{self.api_url}/api/v1/budgets/check",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                )
            resp.raise_for_status()

            cloud_response = BudgetCheckResponse.model_validate(resp.json())
            self._cache_response(cloud_response, agent_run_id=agent_run_id)
            if breaker is not None:
                breaker.record_success()
            return self._build_result_from_response(cloud_response)

        except Exception as exc:
            # A read-only-key error means the control plane RESPONDED — record
            # success. Anything else is an outage: record failure. Log the
            # exception TYPE only (never interpolate a body that could carry
            # response text).
            if handle_read_only_key_error(exc):
                if breaker is not None:
                    breaker.record_success()
            else:
                if breaker is not None:
                    breaker.record_failure()
                logger.warning("Cloud API budget check failed: %s", type(exc).__name__)

            return self._build_unreachable_result(estimated_input_tokens, agent_run_id)
        finally:
            # Cancellation (or any BaseException) bypasses the handler above; a
            # consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a success/failure verdict
            # has already released the slot.
            if breaker is not None:
                breaker.release_probe(admission)

    # ── lease path ───────────────────────────────────────────────────────

    async def _check_lease(
        self,
        *,
        agent_run_id: str,
        call_id: str,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
        timeout: float | None,
        modality: Modality,
        estimated_media: MediaUsage | None,
        estimated_output_bound: int | None,
    ) -> BudgetCheckResult | None:
        """Async twin of ``BudgetEnforcer._check_lease``."""
        admission = self._admit_lease(
            agent_run_id=agent_run_id,
            call_id=call_id,
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            estimated_output_bound=estimated_output_bound,
            modality=modality,
            estimated_media=estimated_media,
            fallback_models=fallback_models,
        )

        if admission.decision is LeaseDecision.NEED_GRANT:
            verdict, response = await self._grant_lease(
                agent_run_id=agent_run_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
                timeout=timeout,
            )
            if verdict == "denied" and response is not None:
                return self._lease_deny_result(agent_run_id, response)
            if verdict == "unreachable":
                if self.fail_open:
                    self._record_uncounted_cold_start(
                        agent_run_id,
                        self._lease_reserve_estimate(
                            estimated_input_tokens, estimated_output_bound
                        ),
                    )
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)
            if verdict != "applied":
                return None
            admission = self._admit_lease(
                agent_run_id=agent_run_id,
                call_id=call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
            )
            if admission.decision is LeaseDecision.NEED_GRANT:
                return None

        if admission.renewal_due:
            self._start_renewal(
                agent_run_id,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )

        if admission.decision is LeaseDecision.LEGACY_CHECK:
            return None
        return self._result_from_admission(agent_run_id, admission)

    async def _grant_lease(
        self,
        *,
        agent_run_id: str,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
        timeout: float | None,
    ) -> tuple[_GrantVerdict, LeaseGrantResponse | None]:
        """Awaited grant round-trip (the caller's coroutine, check timeout)."""
        if not self._claim_grant_slot(agent_run_id):
            return "legacy", None
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.grant_skipped_breaker_open")
                return "unreachable", None
            request = self._build_grant_request(
                agent_run_id=agent_run_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )
            payload = request.model_dump(mode="json")
            try:
                if timeout is not None:
                    resp = await self._http.post(
                        f"{self.api_url}{_LEASE_PATH}",
                        json=payload,
                        headers=self._auth_headers(),
                        timeout=timeout,
                    )
                else:
                    resp = await self._http.post(
                        f"{self.api_url}{_LEASE_PATH}",
                        json=payload,
                        headers=self._auth_headers(),
                    )
                resp.raise_for_status()
            except Exception as exc:
                return self._classify_lease_failure(exc, agent_run_id, renewal=False), None

            if breaker is not None:
                breaker.record_success()
            return self._install_grant(
                agent_run_id, resp, declared_models=[model, *fallback_models]
            )
        finally:
            if breaker is not None:
                breaker.release_probe(admission)
            self._release_grant_slot(agent_run_id)

    def _start_renewal(
        self,
        agent_run_id: str,
        *,
        model: str,
        provider: str,
        fallback_providers: list[str],
        fallback_models: list[str],
    ) -> None:
        """Schedule a renewal TASK — the admission returns without awaiting it."""
        request = self._build_renewal(
            agent_run_id,
            model=model,
            provider=provider,
            fallback_providers=fallback_providers,
            fallback_models=fallback_models,
        )
        if request is None:
            return
        task = asyncio.create_task(
            self._renew_lease(agent_run_id, request, [model, *fallback_models])
        )
        self._renewal_tasks.add(task)
        task.add_done_callback(self._renewal_tasks.discard)

    async def _renew_lease(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        declared_models: list[str],
    ) -> None:
        """Renewal task: breaker-guarded, never raises into the loop."""
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.renew_skipped_breaker_open")
                with self._state_lock:
                    self._lease.renewal_failed(agent_run_id, time.monotonic())
                return
            try:
                resp = await self._http.post(
                    f"{self.api_url}{_LEASE_RENEW_PATH}",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=_RENEWAL_TIMEOUT_S,
                )
                resp.raise_for_status()
            except Exception as exc:
                self._classify_lease_failure(exc, agent_run_id, renewal=True)
                return
            if breaker is not None:
                breaker.record_success()
            try:
                response = LeaseGrantResponse.model_validate(resp.json())
            except Exception as exc:
                logger.warning("lease.renew_response_unreadable: %s", type(exc).__name__)
                with self._state_lock:
                    self._lease.renewal_failed(agent_run_id, time.monotonic())
                return
            self._finish_renewal(agent_run_id, response, declared_models)
        except asyncio.CancelledError:
            with self._state_lock:
                self._lease.renewal_failed(agent_run_id, time.monotonic())
            raise
        except Exception as exc:  # pragma: no cover — a task must never raise
            logger.warning("lease.renew_worker_failed: %s", type(exc).__name__)
        finally:
            if breaker is not None:
                breaker.release_probe(admission)

    async def _surrender_leases(self) -> None:
        """DHCPRELEASE-style: hand every held lease back, best-effort."""
        for request in self.lease_surrender_payloads():
            breaker = self._control_plane_breaker
            admission = breaker.admit() if breaker is not None else None
            try:
                if admission is not None and not admission.allowed:
                    logger.debug("lease.surrender_skipped_breaker_open")
                    continue
                resp = await self._http.post(
                    f"{self.api_url}{_LEASE_SURRENDER_PATH}",
                    json=request.model_dump(mode="json"),
                    headers=self._auth_headers(),
                    timeout=_SURRENDER_TIMEOUT_S,
                )
                resp.raise_for_status()
                if breaker is not None:
                    breaker.record_success()
            except Exception as exc:
                logger.debug("lease.surrender_failed: %s", type(exc).__name__)
                if breaker is not None and not handle_read_only_key_error(exc):
                    breaker.record_failure()
            finally:
                if breaker is not None:
                    breaker.release_probe(admission)

    async def close(self) -> None:
        """Surrender held leases, then close the underlying async HTTP client."""
        tasks = list(self._renewal_tasks)
        if tasks:
            # Bounded: an in-flight renewal may land, but close never waits on
            # a hung control plane.
            done, pending = await asyncio.wait(tasks, timeout=_SURRENDER_TIMEOUT_S)
            for task in pending:
                task.cancel()
        await self._surrender_leases()
        await self._http.aclose()
