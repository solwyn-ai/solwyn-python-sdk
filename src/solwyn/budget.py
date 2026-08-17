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
from typing import Annotated, Literal, cast, get_args

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from solwyn import _run_control
from solwyn._constants import CALL_ID_MAX_LENGTH, CALL_ID_PATTERN
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
from solwyn._run_control import clear_termination_if
from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    CircuitState,
    DenySource,
    LeaseGrantRequest,
    LeaseGrantResponse,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MediaUsage,
    Modality,
    ProviderName,
    RunControlDirective,
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

# A burst can make many distinct runs renewal-due at once. Renewal I/O is
# background work, but background must still be bounded: one client owns at
# most this many renewal threads/tasks regardless of run cardinality.
_MAX_RENEWAL_WORKERS = 4

# What a grant round-trip resolved to, from the admission path's point of view.
# "legacy" means "the plane answered, but this run takes the per-call path".
_GrantVerdict = Literal["applied", "legacy", "denied", "unreachable"]

# Sticky run denials protect brief Cloud outages but must not retain an
# unbounded stream of run UUIDs in a long-lived SDK process.
_MAX_STICKY_RUN_DENIALS = 128

# Both the run budget cap and an operator stop are scoped to one raw run id.
# Keep the classification shared by per-call and lease denials through
# ``_cache_response`` so neither can poison unrelated runs during an outage.
_RUN_SCOPED_DENIAL_PERIODS = frozenset({"agent_run", "run_stopped"})

# Uncounted-mode telemetry (§8): loud on ENTRY to a fail-open uncounted
# episode, then at most one line per this interval while it persists. An
# hour-long outage must stay visible without one warning per call.
_UNCOUNTED_WARN_INTERVAL_S = 30.0

# Same footgun as the sticky-deny map: a long-lived process must not retain an
# episode clock per run id forever. Evicting one only costs an extra ENTRY line.
_MAX_UNCOUNTED_EPISODES = 128

# The contractual confirm tier values (derived from the ServiceTier literal,
# never hand-copied). Adapters echo arbitrary bounded strings; only these may
# ride the value-strict confirm wire.
_SERVICE_TIER_VALUES: frozenset[str] = frozenset(get_args(ServiceTier))

# Same validator used by the confirm wire models, applied before lease
# admission so an invalid caller id can never mutate local authority first.
_CallId = Annotated[
    str,
    Field(max_length=CALL_ID_MAX_LENGTH, pattern=CALL_ID_PATTERN),
]
_CALL_ID_ADAPTER = TypeAdapter(_CallId)

logger = logging.getLogger(__name__)


class _MisroutedControlDenial(Exception):
    """Internal semantic failure for a run-stop verdict addressed elsewhere."""


def _validated_call_id(call_id: str | None) -> str:
    """Return a canonical reconciliation id, generating one only for None."""
    candidate = str(uuid.uuid4()) if call_id is None else call_id
    return _CALL_ID_ADAPTER.validate_python(candidate)


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
    denied_by_period: str | None = None
    deny_source: DenySource | None = None
    deny_reason: str | None = None
    # PJ-2: set when the call drew on LEASE authority instead of a per-call
    # reservation. Exactly one of reservation_id / lease_id ever settles a call.
    lease_id: str | None = None
    # Exact local claim capability for the lease reservation. Excluded from
    # serialization because it is process-local and never part of the Cloud API
    # contract; client error/settlement paths must echo it back to the ledger.
    lease_claim_token: int | None = Field(default=None, exclude=True, repr=False)
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
        self._run_hard_deny_observed_at: OrderedDict[str, float] = OrderedDict()

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

        # Uncounted-mode telemetry (§8). run_id -> monotonic time this run's
        # CURRENT lease-death episode last warned. An absent run is not in an
        # episode, so its next uncounted admission is an ENTRY; installing a
        # grant ends the episode. The ledger cannot own this — it is sans-I/O
        # and must never log.
        self._uncounted_episodes: OrderedDict[str, float] = OrderedDict()
        # Renewal workers capture the epoch at dispatch. close() increments it
        # before draining state, so a response that arrives afterwards cannot
        # recreate authority; an observed late successor is surrendered.
        self._close_epoch = 0
        self._closed = False
        self._late_renewal_spend: dict[tuple[str, str, int], int] = {}

    def _reset_after_fork_in_child(self) -> None:
        """Replace the state lock in a forked child (concrete classes also swap
        the inherited HTTP client — a shared socket across processes corrupts)."""
        self._state_lock = threading.Lock()
        # A child inherits no lease authority: the parent's leases live in the
        # parent's process, and its in-flight renewal flags are lies here. The
        # child must also become a DIFFERENT holder — the client's
        # ``_sdk_instance_id`` survives a fork unchanged, and the server
        # releases a same-(project, run, holder) active lease as stale when a
        # grant lands. A child re-granting under the parent's id would kill the
        # parent's live lease, the parent's regrant would kill the child's, and
        # a forked same-run workload would churn one blocking grant per refresh
        # interval each. A fresh id makes the child a legitimate second holder.
        self._lease.holder_id = str(uuid.uuid4())
        self._lease.on_fork_reset()
        self._lease_grants_in_flight = set()
        self._uncounted_episodes = OrderedDict()
        self._close_epoch = 0
        self._closed = False
        self._late_renewal_spend = {}

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
        tags: dict[str, str] | None = None,
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
            tags=dict(tags) if tags is not None else None,
            failover_directive_version="1",
            run_directive_version="1",
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
        cache_allowed_response: bool = True,
        request_epoch: float | None = None,
        observed_at: float | None = None,
    ) -> BudgetCheckResponse:
        """Cache an allow response. Never cache deny responses.

        Always updates the last-known budget limit (from both allow and deny)
        so that local enforcement can use it when the cloud becomes unreachable.
        Scoped checks never read or populate the global allow cache. Run-specific
        hard denials are sticky only for their raw run id; project-period hard
        denials remain global and invalidate a stale global allow. Tag-period
        denials stay selector-local: they clear stale project state without
        creating or erasing run sticky state.
        """
        if response.allowed and agent_run_id is not None and request_epoch is not None:
            with _run_control._locked_registry():
                newer_server_stop = _run_control._clear_server_termination_before_request_locked(
                    agent_run_id,
                    request_epoch=request_epoch,
                )
                with self._state_lock:
                    return self._resolve_ordered_allow_locked(
                        response,
                        agent_run_id=agent_run_id,
                        request_epoch=request_epoch,
                        newer_server_stop=newer_server_stop,
                        cache_allowed_response=cache_allowed_response,
                    )
        if response.allowed and agent_run_id is not None:
            clear_termination_if(agent_run_id, source="server")
        with self._state_lock:
            self._cache_response_locked(
                response,
                agent_run_id=agent_run_id,
                cache_allowed_response=cache_allowed_response,
                observed_at=observed_at,
            )
        return response

    def _resolve_ordered_allow_locked(
        self,
        response: BudgetCheckResponse,
        *,
        agent_run_id: str,
        request_epoch: float,
        newer_server_stop: _run_control.RunTermination | None,
        cache_allowed_response: bool,
    ) -> BudgetCheckResponse:
        """Resolve one live ALLOW while registry and enforcer locks are held."""
        sticky = self._run_hard_deny_responses.get(agent_run_id)
        sticky_observed_at = self._run_hard_deny_observed_at.get(agent_run_id)
        if (
            sticky is not None
            and sticky_observed_at is not None
            and sticky_observed_at >= request_epoch
        ):
            self._run_hard_deny_responses.move_to_end(agent_run_id)
            self._run_hard_deny_observed_at.move_to_end(agent_run_id)
            return sticky
        if newer_server_stop is not None:
            effective = response.model_copy(
                update={
                    "allowed": False,
                    "reservation_id": None,
                    "mode": BudgetMode.HARD_DENY,
                    "denied_by_period": "run_stopped",
                    "run_control": RunControlDirective(
                        version="1",
                        action="terminate",
                        agent_run_id=agent_run_id,
                        reason=newer_server_stop.reason,
                    ),
                }
            )
            self._cache_response_locked(
                effective,
                agent_run_id=agent_run_id,
                observed_at=newer_server_stop.at_monotonic,
            )
            return effective
        self._cache_response_locked(
            response,
            agent_run_id=agent_run_id,
            cache_allowed_response=cache_allowed_response,
        )
        return response

    def _cache_response_locked(
        self,
        response: BudgetCheckResponse,
        *,
        agent_run_id: str | None,
        cache_allowed_response: bool = True,
        observed_at: float | None = None,
    ) -> None:
        """Mutate cache state while the caller holds ``_state_lock``."""
        # Always remember the limit for local enforcement fallback
        self._last_known_budget_limit = response.budget_limit
        self._last_known_current_usage = response.current_usage

        if response.allowed:
            self._last_hard_deny_response = None
            if agent_run_id is not None:
                self._run_hard_deny_responses.pop(agent_run_id, None)
                self._run_hard_deny_observed_at.pop(agent_run_id, None)
            elif cache_allowed_response:
                self._cached_response = response
                self._cache_expires_at = time.monotonic() + self.cache_ttl
            return

        if response.mode != BudgetMode.HARD_DENY:
            self._last_hard_deny_response = None
            if agent_run_id is not None:
                self._run_hard_deny_responses.pop(agent_run_id, None)
                self._run_hard_deny_observed_at.pop(agent_run_id, None)
            self._cached_response = None
            self._cache_expires_at = 0.0
            return

        if response.denied_by_period == "tag":
            self._last_hard_deny_response = None
            return

        if response.denied_by_period in _RUN_SCOPED_DENIAL_PERIODS:
            # A run-cap denial proves every project period passed. A dashboard
            # stop does not, so preserve an older project-period sticky denial.
            if response.denied_by_period == "agent_run":
                self._last_hard_deny_response = None
            if agent_run_id is None:
                # Contract drift left no run identity to scope by. Fall back to
                # the safe global posture used before run-scoped labels existed.
                self._cached_response = None
                self._cache_expires_at = 0.0
                if self._last_hard_deny_response is None:
                    self._last_hard_deny_response = response
                return
            self._run_hard_deny_responses.pop(agent_run_id, None)
            self._run_hard_deny_observed_at.pop(agent_run_id, None)
            self._run_hard_deny_responses[agent_run_id] = response
            self._run_hard_deny_observed_at[agent_run_id] = (
                time.monotonic() if observed_at is None else observed_at
            )
            if len(self._run_hard_deny_responses) > _MAX_STICKY_RUN_DENIALS:
                evicted_run_id, _ = self._run_hard_deny_responses.popitem(last=False)
                self._run_hard_deny_observed_at.pop(evicted_run_id, None)
            return

        self._cached_response = None
        self._cache_expires_at = 0.0
        self._last_hard_deny_response = response
        # Deny responses are never cached as freshness-skipping allows/denies.

    def _directive_matches_run(
        self,
        directive_run_id: str,
        agent_run_id: str | None,
    ) -> bool:
        """Validate an echoed structural run id without applying side effects."""
        if directive_run_id == agent_run_id:
            return True
        logger.warning(
            "run_control.directive_run_mismatch: request_agent_run_id=%s directive_agent_run_id=%s",
            agent_run_id,
            directive_run_id,
        )
        return False

    @staticmethod
    def _run_stopped_lease_response(
        response: LeaseGrantResponse,
    ) -> LeaseGrantResponse:
        """Normalize any directive-bearing lease shape into a hard denial."""
        return response.model_copy(
            update={
                "allowed": False,
                "denied_by_period": "run_stopped",
                "lease_id": None,
                "generation": None,
                "granted_tokens": None,
                "refresh_interval_s": None,
                "lease_length_s": None,
                "headroom_share_tokens": None,
                "posture": None,
                "final_grant": None,
                "mode": BudgetMode.HARD_DENY,
            }
        )

    def _lease_response_for_run(
        self,
        response: LeaseGrantResponse,
        agent_run_id: str,
        *,
        warn_on_mismatch: bool,
    ) -> LeaseGrantResponse:
        """Strip a misrouted directive while preserving the budget verdict."""
        directive = response.run_control
        if directive is None or directive.agent_run_id == agent_run_id:
            return response
        if warn_on_mismatch:
            self._directive_matches_run(directive.agent_run_id, agent_run_id)
        return response.model_copy(update={"run_control": None})

    @staticmethod
    def _is_misrouted_control_denial(
        response: BudgetCheckResponse | LeaseGrantResponse,
        agent_run_id: str | None,
    ) -> bool:
        """Return whether a run-stop verdict belongs to a different run."""
        directive = response.run_control
        return (
            directive is not None
            and directive.agent_run_id != agent_run_id
            and response.denied_by_period == "run_stopped"
        )

    @staticmethod
    def _first_writer_directive(
        directive: RunControlDirective,
        termination: _run_control.RunTermination,
    ) -> RunControlDirective:
        """Keep a server directive aligned with the immutable registry winner."""
        if termination.source != "server" or directive.reason == termination.reason:
            return directive
        return directive.model_copy(update={"reason": termination.reason})

    def _apply_check_run_control(
        self,
        response: BudgetCheckResponse,
        agent_run_id: str | None,
    ) -> tuple[BudgetCheckResponse, bool]:
        """Return the safe verdict and whether ordinary cache mutation may run."""
        directive = response.run_control
        if directive is None:
            return response, True
        if not self._directive_matches_run(
            directive.agent_run_id,
            agent_run_id,
        ):
            if response.denied_by_period == "run_stopped":
                raise _MisroutedControlDenial
            # Enforce the budget verdict, but a structurally mismatched echo is
            # forbidden from mutating any registry/cache/lease state. Strip it
            # so its reason cannot leak into result attribution either.
            return response.model_copy(update={"run_control": None}), False
        agent_run_id = cast(str, agent_run_id)

        # A terminate directive is authoritative even if a drifted server body
        # says ALLOW (or retains alert_only). Normalize it before caching so it
        # can neither clear the registry entry nor leave spend authority live.
        effective = response.model_copy(
            update={
                "allowed": False,
                "reservation_id": None,
                "mode": BudgetMode.HARD_DENY,
                "denied_by_period": "run_stopped",
            }
        )
        # Publish the sticky denial before dropping local authority. Once the
        # drop is visible, another caller cannot race through NEED_GRANT and
        # reinstall authority in the gap before the sticky response is filed.
        with _run_control._locked_registry():
            termination = _run_control._mark_terminated_locked(
                agent_run_id,
                reason=directive.reason,
                source="server",
            )
            effective.run_control = self._first_writer_directive(
                directive,
                termination,
            )
            observed_at = time.monotonic()
            with self._state_lock:
                self._cache_response_locked(
                    effective,
                    agent_run_id=agent_run_id,
                    observed_at=observed_at,
                )
                state = self._lease.state_for(agent_run_id)
                if state is not None and state.lease_id is not None:
                    self._lease.drop_if_current(
                        agent_run_id,
                        lease_id=state.lease_id,
                        generation=state.generation,
                    )
        return effective, False

    def _build_prior_hard_deny_unavailable_result(
        self,
        agent_run_id: str | None = None,
    ) -> BudgetCheckResult | None:
        """Return a denial if cloud is down after an authoritative hard deny."""
        with _run_control._locked_registry():
            retained_termination = (
                _run_control._outage_termination_locked(agent_run_id)
                if agent_run_id is not None
                else None
            )
            with self._state_lock:
                response = self._last_hard_deny_response
                if response is None and agent_run_id is not None:
                    response = self._run_hard_deny_responses.get(agent_run_id)
                    if response is not None:
                        self._run_hard_deny_responses.move_to_end(agent_run_id)
                        if agent_run_id in self._run_hard_deny_observed_at:
                            self._run_hard_deny_observed_at.move_to_end(agent_run_id)
                budget_limit = self._last_known_budget_limit or 0.0
                current_usage = self._last_known_current_usage
        if response is None:
            if retained_termination is None:
                return None
            logger.warning("Cloud API unreachable; preserving retained run stop")
            return BudgetCheckResult(
                allowed=False,
                remaining_budget=max(0.0, budget_limit - current_usage),
                mode=BudgetMode.HARD_DENY,
                warning="Cloud API unreachable; preserving retained run stop",
                budget_limit=budget_limit,
                current_usage=current_usage,
                denied_by_period="run_stopped",
                deny_source="sticky_replay",
                deny_reason=retained_termination.reason,
            )
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
            denied_by_period=response.denied_by_period,
            deny_source="sticky_replay",
            deny_reason=(
                response.run_control.reason
                if response.run_control is not None
                and agent_run_id is not None
                and response.run_control.agent_run_id == agent_run_id
                else response.denied_by_period
            ),
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
                denied_by_period=response.denied_by_period,
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
                denied_by_period=response.denied_by_period,
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
            denied_by_period=response.denied_by_period,
            deny_source="server",
            deny_reason=(
                response.run_control.reason
                if response.run_control is not None
                else response.denied_by_period
            ),
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
                deny_source="local_enforcement",
                deny_reason="no_prior_budget_limit",
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
                deny_source="local_enforcement",
                deny_reason="local_budget_exceeded",
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
        with _run_control._locked_registry():
            if _run_control._outage_termination_locked(agent_run_id) is not None:
                return False
            with self._state_lock:
                if self._closed:
                    return False
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
        breaker_open_override: bool | None = None,
        claim_token: int | None = None,
    ) -> LeaseAdmission:
        """Run the ladder for one call, entirely inside one lock section."""
        breaker_open = (
            self._lease_breaker_open() if breaker_open_override is None else breaker_open_override
        )
        with self._state_lock:
            if self._closed:
                return LeaseAdmission(
                    LeaseDecision.LEGACY_CHECK,
                    reason="enforcer_closed",
                    claim_token=claim_token,
                )
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
                claim_token=claim_token,
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
            run_directive_version="1",
        )

    def _claim_grant_work(self, agent_run_id: str) -> int | None:
        """Claim one grant and return its lifecycle epoch, or None."""
        with self._state_lock:
            if self._closed or agent_run_id in self._lease_grants_in_flight:
                return None
            self._lease_grants_in_flight.add(agent_run_id)
            return self._close_epoch

    def _release_grant_slot(self, agent_run_id: str) -> None:
        with self._state_lock:
            self._lease_grants_in_flight.discard(agent_run_id)

    def _apply_lease_response(
        self,
        agent_run_id: str,
        response: LeaseGrantResponse,
        declared_models: Sequence[str],
        *,
        renewal_request: LeaseRenewRequest | None = None,
        close_epoch: int | None = None,
    ) -> tuple[GrantOutcome, LeaseSurrenderRequest | None]:
        """Install a grant/renew response under the ledger's serialization."""
        if response.run_control is not None:
            with _run_control._locked_registry(), self._state_lock:
                if close_epoch is not None and (self._closed or close_epoch != self._close_epoch):
                    return GrantOutcome.STALE, self._late_lease_surrender_request(response)
                if self._is_misrouted_control_denial(response, agent_run_id):
                    self._directive_matches_run(
                        response.run_control.agent_run_id,
                        agent_run_id,
                    )
                    return GrantOutcome.MALFORMED, None
                effective = self._lease_response_for_run(
                    response,
                    agent_run_id,
                    warn_on_mismatch=True,
                )
                directive = effective.run_control
                if directive is not None:
                    termination = _run_control._mark_terminated_locked(
                        agent_run_id,
                        reason=directive.reason,
                        source="server",
                    )
                    effective.run_control = self._first_writer_directive(
                        directive,
                        termination,
                    )
                    directed_denial = self._run_stopped_lease_response(effective)
                    self._cache_response_locked(
                        self._lease_denial_response(directed_denial),
                        agent_run_id=agent_run_id,
                        observed_at=time.monotonic(),
                    )
                    state = self._lease.state_for(agent_run_id)
                    if state is not None and state.lease_id is not None:
                        self._lease.drop_if_current(
                            agent_run_id,
                            lease_id=state.lease_id,
                            generation=state.generation,
                        )
                    return GrantOutcome.DENIED, None
                return self._apply_ordinary_lease_response_locked(
                    agent_run_id,
                    effective,
                    declared_models,
                    renewal_request=renewal_request,
                ), None

        with self._state_lock:
            if close_epoch is not None and (self._closed or close_epoch != self._close_epoch):
                return GrantOutcome.STALE, self._late_lease_surrender_request(response)
            outcome = self._apply_ordinary_lease_response_locked(
                agent_run_id,
                response,
                declared_models,
                renewal_request=renewal_request,
            )
        return outcome, None

    def _apply_ordinary_lease_response_locked(
        self,
        agent_run_id: str,
        response: LeaseGrantResponse,
        declared_models: Sequence[str],
        *,
        renewal_request: LeaseRenewRequest | None,
    ) -> GrantOutcome:
        """Apply a directive-free response while ``_state_lock`` is held."""
        outcome = self._lease.apply_grant_response(
            agent_run_id,
            response,
            now=time.monotonic(),
            declared_models=declared_models,
            expected_lease_id=(renewal_request.lease_id if renewal_request is not None else None),
            expected_generation=(
                renewal_request.generation if renewal_request is not None else None
            ),
        )
        if outcome is GrantOutcome.APPLIED:
            # Live authority again: the uncounted episode is over, so the next
            # lease death warns on entry as loudly as this one did.
            self._uncounted_episodes.pop(agent_run_id, None)
        return outcome

    def _classify_lease_failure(
        self,
        exc: Exception,
        agent_run_id: str,
        *,
        renewal_request: LeaseRenewRequest | None = None,
    ) -> _GrantVerdict:
        """Map a failed lease round-trip onto a verdict, feeding the breaker.

        A REFUSAL is not an outage: 409 (holder cap) and 503 (lease_unavailable)
        are deliberate answers, so the plane is credited and the run simply
        drops to the per-call path. Only a transport failure or a non-503 5xx
        counts as unreachable — the posture ladder then owns the call.
        """
        renewal = renewal_request is not None
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
            if renewal_request is not None:
                # The lease stays valid until its own deadline — but the
                # in-flight flag must clear, under a backoff, or the run never
                # renews again AND every later admission would retry at once.
                with self._state_lock:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=renewal_request.lease_id,
                        expected_generation=renewal_request.generation,
                    )
            return "unreachable"

        now = time.monotonic()
        with self._state_lock:
            if renewal_request is not None and status in (404, 409):
                # The server no longer recognizes this lease (or a newer
                # generation exists): drop THAT lease and re-grant on the next
                # call. A late answer for A cannot delete replacement B.
                self._lease.drop_if_current(
                    agent_run_id,
                    lease_id=renewal_request.lease_id,
                    generation=renewal_request.generation,
                )
            elif renewal_request is not None:
                self._lease.renewal_failed(
                    agent_run_id,
                    now,
                    expected_lease_id=renewal_request.lease_id,
                    expected_generation=renewal_request.generation,
                )
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
        """Feed a lease denial through the shared run/global sticky classification."""
        denial = self._lease_denial_response(response)
        self._cache_response(denial, agent_run_id=agent_run_id)
        return self._build_result_from_response(denial)

    @staticmethod
    def _lease_denial_response(response: LeaseGrantResponse) -> BudgetCheckResponse:
        """Project a lease denial onto the shared check-response cache shape."""
        return BudgetCheckResponse(
            allowed=False,
            remaining_budget=response.remaining_budget,
            reservation_id=None,
            mode=response.mode,
            budget_limit=response.budget_limit,
            current_usage=response.current_usage,
            denied_by_period=response.denied_by_period,
            project_id=response.project_id,
            run_control=response.run_control,
        )

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
            lease_claim_token=admission.claim_token,
            mode=(
                admission.mode
                if admission.mode is not None
                else (snapshot.mode if snapshot is not None else self.budget_mode)
            ),
            warning=admission.warning,
            budget_limit=snapshot.budget_limit if snapshot is not None else 0.0,
            current_usage=snapshot.current_usage if snapshot is not None else 0.0,
            denied_by_period=("agent_run" if admission.decision is LeaseDecision.DENY else None),
            deny_source=("lease_exhausted" if admission.decision is LeaseDecision.DENY else None),
            deny_reason=(admission.reason if admission.decision is LeaseDecision.DENY else None),
        )

    def _lease_result_when_breaker_refuses(
        self,
        *,
        agent_run_id: str | None,
        call_id: str | None,
        estimated_input_tokens: int,
        model: str,
        estimated_output_bound: int | None,
        modality: Modality,
        estimated_media: MediaUsage | None,
        fallback_models: Sequence[str],
        claim_token: int | None,
    ) -> BudgetCheckResult | None:
        """Re-run lease authority after a HALF_OPEN follower is refused.

        The first lease pass deliberately treats recovery-eligible OPEN and
        HALF_OPEN as reachable so one caller can probe the control plane. If
        that probe slot is already occupied, a follower learns the plane is
        unavailable only when the generic breaker refuses it. Re-running the
        ladder with that fact preserves share drawdown and hard-deny authority
        instead of bypassing both through generic fail-open.
        """
        if agent_run_id is None or call_id is None or not self._lease_path_applies(agent_run_id):
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
            breaker_open_override=True,
            claim_token=claim_token,
        )
        if admission.decision is LeaseDecision.LEGACY_CHECK:
            return None
        if admission.decision is LeaseDecision.NEED_GRANT:
            if self.fail_open:
                self._record_uncounted_cold_start(
                    agent_run_id,
                    self._lease_reserve_estimate(
                        estimated_input_tokens,
                        estimated_output_bound,
                    ),
                    call_id=call_id,
                    claim_token=admission.claim_token,
                )
                self._note_uncounted_admission(
                    agent_run_id,
                    reason="no_lease_breaker_refused",
                )
            return self._build_unreachable_result(
                estimated_input_tokens,
                agent_run_id,
            ).model_copy(update={"lease_claim_token": admission.claim_token})
        if admission.decision is LeaseDecision.ADMIT_UNCOUNTED:
            self._note_uncounted_admission(
                agent_run_id, reason=admission.reason or "expired_fail_open"
            )
        return self._result_from_admission(agent_run_id, admission)

    def _record_uncounted_cold_start(
        self,
        agent_run_id: str,
        tokens: int,
        *,
        call_id: str,
        claim_token: int | None,
    ) -> None:
        """Tally a fail-open admission made with no lease state at all."""
        with self._state_lock:
            self._lease.record_uncounted(
                agent_run_id,
                tokens,
                call_id=call_id,
                claim_token=claim_token,
            )

    def _note_uncounted_admission(self, agent_run_id: str, *, reason: str) -> None:
        """Emit the §8 uncounted-mode telemetry for one fail-open admission.

        Uncounted means no counter anywhere covers the call until the run's
        next successful renewal reports the tally — the customer must be able
        to see that from their logs, so this is loud on ENTRY to the episode
        (lease death or absence) and rate-limited to 1/30s while it persists.
        The episode ends when a grant installs (``_apply_lease_response``).
        """
        now = time.monotonic()
        with self._state_lock:
            last_warned = self._uncounted_episodes.get(agent_run_id)
            if last_warned is not None and now - last_warned < _UNCOUNTED_WARN_INTERVAL_S:
                return
            entry = last_warned is None
            self._uncounted_episodes[agent_run_id] = now
            self._uncounted_episodes.move_to_end(agent_run_id)
            while len(self._uncounted_episodes) > _MAX_UNCOUNTED_EPISODES:
                self._uncounted_episodes.popitem(last=False)

        if entry:
            logger.warning(
                "lease.uncounted_entry: Solwyn is unreachable and this run holds no live "
                "lease; calls proceed UNCOUNTED under fail_open and are tallied for the "
                "next successful renewal to report (reason=%s)",
                reason,
            )
        else:
            logger.warning(
                "lease.uncounted_continuing: still admitting UNCOUNTED under fail_open (reason=%s)",
                reason,
            )

    def release_reservation(
        self,
        call_id: str,
        lease_claim_token: int | None = None,
    ) -> None:
        """Hand a lease reservation back when the call will never settle.

        Safe to call for any call id: a call that never drew on lease authority
        (legacy reservation, non-run traffic, uncounted admit) is a no-op.
        """
        with self._state_lock:
            self._lease.release(call_id, claim_token=lease_claim_token)

    def lease_surrender_payloads(self) -> list[LeaseSurrenderRequest]:
        """Drain every held lease into surrender payloads (best-effort release).

        Every run record is evicted, including inactive/uncounted-only state,
        so a close() followed by the interpreter-exit hook cannot retain or
        surrender anything twice.
        """
        with self._state_lock:
            return self._lease.drain_surrender_requests()

    def _begin_close(self) -> list[LeaseSurrenderRequest] | None:
        """Fence and atomically drain state before any shutdown wait."""
        with self._state_lock:
            if self._closed:
                return None
            # Capture the unacknowledged part of every renewal before the
            # ledger is drained. A late response may prove the server already
            # advanced past the old generation, in which case this delta must
            # follow the successor surrender.
            self._late_renewal_spend.update(self._lease.pending_renewal_spend_deltas())
            self._closed = True
            self._close_epoch += 1
            return self._lease.drain_surrender_requests()

    def _late_lease_surrender_request(
        self,
        response: LeaseGrantResponse,
        *,
        spent_tokens: int = 0,
    ) -> LeaseSurrenderRequest | None:
        """Build a release for authority returned after its lifecycle fence."""
        if (
            not response.eligible
            or not response.allowed
            or response.lease_id is None
            or response.generation is None
        ):
            return None
        return LeaseSurrenderRequest(
            lease_id=response.lease_id,
            holder_id=self._lease.holder_id,
            generation=response.generation,
            spent_tokens=max(0, spent_tokens),
        )

    def _claim_renewal_work(
        self,
        agent_run_id: str,
        *,
        model: str,
        provider: str,
        fallback_providers: Sequence[str],
        fallback_models: Sequence[str],
    ) -> tuple[LeaseRenewRequest, int] | None:
        """Atomically claim renewal identity plus the current lifecycle epoch."""
        with self._state_lock:
            if self._closed:
                return None
            request = self._lease.claim_renewal_request(
                agent_run_id,
                model=model,
                provider=ProviderName(provider),
                fallback_providers=[ProviderName(p) for p in fallback_providers],
                fallback_models=list(fallback_models),
            )
            if request is None:
                return None
            return request, self._close_epoch

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
        work = self._claim_renewal_work(
            agent_run_id,
            model=model,
            provider=provider,
            fallback_providers=fallback_providers,
            fallback_models=fallback_models,
        )
        return work[0] if work is not None else None

    def _finish_renewal(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        response: LeaseGrantResponse,
        declared_models: Sequence[str],
        *,
        close_epoch: int | None = None,
    ) -> LeaseSurrenderRequest | None:
        """Apply a fenced renewal or return a successor that close must release."""
        if response.run_control is not None:
            with _run_control._locked_registry(), self._state_lock:
                if close_epoch is not None and (self._closed or close_epoch != self._close_epoch):
                    spent_tokens = self._late_renewal_spend.pop(
                        (agent_run_id, request.lease_id, request.generation),
                        0,
                    )
                    return self._late_lease_surrender_request(
                        response,
                        spent_tokens=spent_tokens,
                    )

                state = self._lease.state_for(agent_run_id)
                if (
                    state is None
                    or state.lease_id != request.lease_id
                    or state.generation != request.generation
                ):
                    return None

                if self._is_misrouted_control_denial(response, agent_run_id):
                    self._directive_matches_run(
                        response.run_control.agent_run_id,
                        agent_run_id,
                    )
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
                    breaker = self._control_plane_breaker
                    if breaker is not None:
                        breaker.record_failure()
                    return None

                effective = self._lease_response_for_run(
                    response,
                    agent_run_id,
                    warn_on_mismatch=True,
                )
                directive = effective.run_control
                if directive is not None:
                    termination = _run_control._mark_terminated_locked(
                        agent_run_id,
                        reason=directive.reason,
                        source="server",
                    )
                    effective.run_control = self._first_writer_directive(
                        directive,
                        termination,
                    )
                    directed_denial = self._run_stopped_lease_response(effective)
                    self._cache_response_locked(
                        self._lease_denial_response(directed_denial),
                        agent_run_id=agent_run_id,
                        observed_at=time.monotonic(),
                    )
                    self._lease.drop_if_current(
                        agent_run_id,
                        lease_id=request.lease_id,
                        generation=request.generation,
                    )
                    return None

                outcome = self._apply_ordinary_lease_response_locked(
                    agent_run_id,
                    effective,
                    declared_models,
                    renewal_request=request,
                )
                if outcome is GrantOutcome.STALE:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
                elif outcome is GrantOutcome.DENIED:
                    self._cache_response_locked(
                        self._lease_denial_response(effective),
                        agent_run_id=agent_run_id,
                        observed_at=time.monotonic(),
                    )
                return None

        with self._state_lock:
            if close_epoch is not None and (self._closed or close_epoch != self._close_epoch):
                spent_tokens = self._late_renewal_spend.pop(
                    (agent_run_id, request.lease_id, request.generation),
                    0,
                )
                return self._late_lease_surrender_request(
                    response,
                    spent_tokens=spent_tokens,
                )

            # Fence the response BEFORE observing its directive. A late reply
            # from lease A must not be able to terminate a run now operating
            # under replacement B.
            state = self._lease.state_for(agent_run_id)
            if (
                state is None
                or state.lease_id != request.lease_id
                or state.generation != request.generation
            ):
                return None

            outcome = self._apply_ordinary_lease_response_locked(
                agent_run_id,
                response,
                declared_models,
                renewal_request=request,
            )
            if outcome is GrantOutcome.STALE:
                # A duplicate-generation response for the still-current lease
                # needs a backoff; an origin-stale response no-ops by fence.
                self._lease.renewal_failed(
                    agent_run_id,
                    time.monotonic(),
                    expected_lease_id=request.lease_id,
                    expected_generation=request.generation,
                )

        if outcome is GrantOutcome.DENIED:
            self._lease_deny_result(agent_run_id, response)
        return None

    def _install_grant(
        self,
        agent_run_id: str,
        resp: httpx.Response,
        *,
        declared_models: list[str],
        close_epoch: int,
    ) -> tuple[
        _GrantVerdict,
        LeaseGrantResponse | None,
        LeaseSurrenderRequest | None,
    ]:
        """Validate + apply a grant body. A malformed body is never fatal."""
        try:
            response = LeaseGrantResponse.model_validate(resp.json())
        except Exception as exc:
            logger.warning("lease.grant_response_unreadable: %s", type(exc).__name__)
            with self._state_lock:
                if not self._closed and close_epoch == self._close_epoch:
                    self._lease.mark_ineligible(
                        agent_run_id,
                        now=time.monotonic(),
                        retry_after=INELIGIBLE_RETRY_AFTER_S,
                    )
            return "legacy", None, None
        outcome, late_surrender = self._apply_lease_response(
            agent_run_id,
            response,
            declared_models,
            close_epoch=close_epoch,
        )
        effective = self._lease_response_for_run(
            response,
            agent_run_id,
            warn_on_mismatch=False,
        )
        if late_surrender is not None:
            return "legacy", effective, late_surrender
        if outcome is GrantOutcome.APPLIED:
            return "applied", effective, None
        if outcome is GrantOutcome.DENIED:
            if effective.run_control is not None:
                effective = self._run_stopped_lease_response(effective)
            return "denied", effective, None
        if outcome is GrantOutcome.MALFORMED and self._is_misrouted_control_denial(
            response, agent_run_id
        ):
            return "unreachable", None, None
        return "legacy", effective, None

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
        lease_claim_token: int | None = None,
        is_provider_fallback: bool = False,
        provider_region: str | None = None,
        service_tier: str | None = None,
        modality: Modality = "text",
        media_usage: MediaUsage | None = None,
        usage_unmeasured: bool = False,
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
        ``usage_unmeasured`` marks the fail-soft tier where the provider did
        not report usable usage (absent/zero usage, or an unreadable usage
        shape). ``token_details`` may carry a retained adapter estimate or the
        synthetic pre-flight input estimate; either is explicitly marked
        ``is_estimated``. The wire confirm is unchanged, but the LOCAL lease
        reservation settles at its reserved bound instead of being trued up to
        that under-measure: crediting the unspent-looking output back would
        re-lend authority a paid response already consumed, weakening the
        run's hard token cap.
        """
        if not call_id:
            raise RuntimeError("call_id is required for budget confirm reconciliation")
        if service_tier is not None and service_tier not in _SERVICE_TIER_VALUES:
            logger.debug("budget.confirm_service_tier_unrecognized: settling at Standard rates")
            service_tier = None
        if lease_id is not None:
            reservation_id = None
        confirm = BudgetConfirmRequest(
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
        if lease_id is not None:
            if lease_claim_token is None:
                raise RuntimeError("lease_claim_token is required for local lease settlement")
            # Validate the complete wire request BEFORE mutating authority.
            # Settlement of a lease-funded call then moves the local
            # reservation from its bound to the actual spend — or holds it AT
            # the bound when the usage was never measurable.
            with self._state_lock:
                self._lease.true_up(
                    call_id,
                    token_details.total_tokens,
                    claim_token=lease_claim_token,
                    floor_at_reservation=usage_unmeasured,
                )
        return confirm


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
        self._renewal_slots = threading.BoundedSemaphore(_MAX_RENEWAL_WORKERS)
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
        self._renewal_slots = threading.BoundedSemaphore(_MAX_RENEWAL_WORKERS)

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
        tags: dict[str, str] | None = None,
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
        path below, unchanged. A caller that omits ``call_id`` still admits,
        under a synthetic key: its reservation simply cannot be trued up or
        released and comes back on the 900s abandoned-reservation sweep.

        Behaviour matrix:
        - Cloud reachable + allowed: return allowed=True
        - Cloud reachable + denied + alert_only: return allowed=True + warning
        - Cloud reachable + denied + hard_deny: return allowed=False
        - Cloud unreachable after hard_deny: return allowed=False
        - Cloud unreachable + fail_open=True: return allowed=True + warning
        - Cloud unreachable + fail_open=False: enforce locally
        """
        lease_call_id: str | None = None
        lease_claim_token: int | None = None
        if (
            modality == "text"
            and estimated_media is None
            and tags is None
            and self._lease_path_applies(agent_run_id)
        ):
            if agent_run_id is None:
                raise RuntimeError("lease path requires an agent_run_id")
            lease_call_id = (
                _validated_call_id(call_id) if call_id is not None else _validated_call_id(None)
            )
            leased, lease_claim_token = self._check_lease(
                agent_run_id=agent_run_id,
                call_id=lease_call_id,
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
        if agent_run_id is None and tags is None:
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
                        denied_by_period=cached.denied_by_period,
                    )

        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Negative cache: a recent streak of control-plane failures means
            # the posture applies instantly instead of paying the timeout.
            logger.debug("budget.check_skipped_breaker_open")
            leased = self._lease_result_when_breaker_refuses(
                agent_run_id=agent_run_id,
                call_id=lease_call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
                claim_token=lease_claim_token,
            )
            if leased is not None:
                return leased
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
            tags,
        )

        try:
            # ── Phase 1: transport + HTTP status. Failures here are OUTAGE
            # semantics — unchanged from before the split.
            try:
                request_epoch = time.monotonic()
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
            except Exception as exc:
                # A read-only-key error means the control plane RESPONDED —
                # record success. Anything else is an outage: record failure.
                # Log the exception TYPE only (never a body).
                if handle_read_only_key_error(exc):
                    if breaker is not None:
                        breaker.record_success()
                else:
                    if breaker is not None:
                        breaker.record_failure()
                    logger.warning("Cloud API budget check failed: %s", type(exc).__name__)
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

            # ── Phase 2: response processing (R6). The plane RESPONDED 2xx;
            # a body we cannot parse is server contract drift, NOT an outage:
            # credit the breaker (drift must not open it), signal at ERROR
            # with a distinct event, and degrade to the same posture ladder.
            try:
                cloud_response = BudgetCheckResponse.model_validate(resp.json())
            except Exception as exc:
                if breaker is not None:
                    breaker.record_success()
                logger.error(
                    "budget.check_response_unreadable: %s — possible server "
                    "contract drift; enforcement degraded (fail_open=%s)",
                    type(exc).__name__,
                    self.fail_open,
                )
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

            try:
                cloud_response, cache_response = self._apply_check_run_control(
                    cloud_response,
                    agent_run_id,
                )
            except _MisroutedControlDenial:
                if breaker is not None:
                    breaker.record_failure()
                return self._build_unreachable_result(
                    estimated_input_tokens,
                    agent_run_id,
                )
            if cache_response:
                cloud_response = self._cache_response(
                    cloud_response,
                    agent_run_id=agent_run_id,
                    cache_allowed_response=tags is None,
                    request_epoch=request_epoch,
                )
            if breaker is not None:
                breaker.record_success()
            return self._build_result_from_response(cloud_response)
        finally:
            # Cancellation (or any BaseException) bypasses the handlers above;
            # a consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a verdict released it.
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
    ) -> tuple[BudgetCheckResult | None, int | None]:
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
                return self._lease_deny_result(agent_run_id, response), admission.claim_token
            if verdict == "unreachable":
                if self.fail_open:
                    # Nothing counts this call anywhere: no lease exists yet, so
                    # the tally is what the next lease's first renewal owes.
                    self._record_uncounted_cold_start(
                        agent_run_id,
                        self._lease_reserve_estimate(
                            estimated_input_tokens, estimated_output_bound
                        ),
                        call_id=call_id,
                        claim_token=admission.claim_token,
                    )
                    self._note_uncounted_admission(agent_run_id, reason="no_lease_unreachable")
                result = self._build_unreachable_result(
                    estimated_input_tokens,
                    agent_run_id,
                ).model_copy(update={"lease_claim_token": admission.claim_token})
                return result, admission.claim_token
            if verdict != "applied":
                return None, admission.claim_token
            admission = self._admit_lease(
                agent_run_id=agent_run_id,
                call_id=call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
                claim_token=admission.claim_token,
            )
            if admission.decision is LeaseDecision.NEED_GRANT:
                # One grant per call, always: never loop against the server.
                return None, admission.claim_token

        if admission.renewal_due:
            self._start_renewal(
                agent_run_id,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )

        if admission.decision is LeaseDecision.LEGACY_CHECK:
            return None, admission.claim_token
        if admission.decision is LeaseDecision.ADMIT_UNCOUNTED:
            self._note_uncounted_admission(
                agent_run_id, reason=admission.reason or "expired_fail_open"
            )
        return self._result_from_admission(agent_run_id, admission), admission.claim_token

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
        close_epoch = self._claim_grant_work(agent_run_id)
        if close_epoch is None:
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
                return self._classify_lease_failure(exc, agent_run_id), None

            verdict, response, late_surrender = self._install_grant(
                agent_run_id,
                resp,
                declared_models=[model, *fallback_models],
                close_epoch=close_epoch,
            )
            if late_surrender is not None:
                self._surrender_late_renewal(late_surrender)
            if breaker is not None:
                if verdict == "unreachable":
                    breaker.record_failure()
                else:
                    breaker.record_success()
            return verdict, response
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
        if not self._renewal_slots.acquire(blocking=False):
            logger.debug("lease.renew_worker_limit")
            return
        work = self._claim_renewal_work(
            agent_run_id,
            model=model,
            provider=provider,
            fallback_providers=fallback_providers,
            fallback_models=fallback_models,
        )
        if work is None:
            self._renewal_slots.release()
            return
        request, close_epoch = work
        thread = threading.Thread(
            target=self._run_renewal_worker,
            args=(agent_run_id, request, [model, *fallback_models], close_epoch),
            name="solwyn-lease-renew",
            daemon=True,
        )
        with self._state_lock:
            self._renewal_threads = {t for t in self._renewal_threads if t.is_alive()}
            self._renewal_threads.add(thread)
        try:
            thread.start()
        except Exception as exc:
            # Thread exhaustion (or an interpreter shutting down) must not
            # leave the lease believing a renewal is in flight: that flag is
            # what suppresses the next attempt, so without this the run coasts
            # to expiry without ever renewing again.
            logger.warning("lease.renew_dispatch_failed: %s", type(exc).__name__)
            with self._state_lock:
                self._renewal_threads.discard(thread)
                self._lease.renewal_failed(
                    agent_run_id,
                    time.monotonic(),
                    expected_lease_id=request.lease_id,
                    expected_generation=request.generation,
                )
            self._renewal_slots.release()

    def _run_renewal_worker(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        declared_models: list[str],
        close_epoch: int,
    ) -> None:
        """Run one renewal and always return its bounded worker slot."""
        try:
            self._renew_lease(
                agent_run_id,
                request,
                declared_models,
                close_epoch=close_epoch,
            )
        finally:
            self._renewal_slots.release()

    def _renew_lease(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        declared_models: list[str],
        *,
        close_epoch: int | None = None,
    ) -> None:
        """Renewal worker: breaker-guarded, never raises, never blocks a call."""
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.renew_skipped_breaker_open")
                with self._state_lock:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
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
                self._classify_lease_failure(exc, agent_run_id, renewal_request=request)
                return
            if breaker is not None:
                breaker.record_success()
            try:
                response = LeaseGrantResponse.model_validate(resp.json())
            except Exception as exc:
                logger.warning("lease.renew_response_unreadable: %s", type(exc).__name__)
                with self._state_lock:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
                return
            late_surrender = self._finish_renewal(
                agent_run_id,
                request,
                response,
                declared_models,
                close_epoch=close_epoch,
            )
            if late_surrender is not None:
                self._surrender_late_renewal(late_surrender)
        except Exception as exc:  # pragma: no cover — a worker must never raise
            logger.warning("lease.renew_worker_failed: %s", type(exc).__name__)
        finally:
            if breaker is not None:
                breaker.release_probe(admission)

    def _surrender_payloads(
        self,
        payloads: Sequence[LeaseSurrenderRequest],
        deadline: float,
    ) -> None:
        """Send releases over a temporary client until one global deadline."""
        client = httpx.Client(timeout=_SURRENDER_TIMEOUT_S)
        try:
            for request in payloads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                breaker = self._control_plane_breaker
                admission = breaker.admit() if breaker is not None else None
                try:
                    if admission is not None and not admission.allowed:
                        logger.debug("lease.surrender_skipped_breaker_open")
                        continue
                    client.post(
                        f"{self.api_url}{_LEASE_SURRENDER_PATH}",
                        json=request.model_dump(mode="json"),
                        headers=self._auth_headers(),
                        timeout=max(0.001, min(_SURRENDER_TIMEOUT_S, remaining)),
                    ).raise_for_status()
                    if breaker is not None:
                        breaker.record_success()
                except Exception as exc:
                    # The server reclaims an unsurrendered lease at its
                    # deadline; a failed courtesy release never surfaces.
                    logger.debug("lease.surrender_failed: %s", type(exc).__name__)
                    if breaker is not None and not handle_read_only_key_error(exc):
                        breaker.record_failure()
                finally:
                    if breaker is not None:
                        breaker.release_probe(admission)
        finally:
            client.close()

    def _surrender_late_renewal(self, request: LeaseSurrenderRequest) -> None:
        """Release a successor observed after close fenced its installation."""
        self._surrender_payloads(
            [request],
            time.monotonic() + _SURRENDER_TIMEOUT_S,
        )

    def _surrender_leases(self) -> None:
        """Compatibility helper for explicit best-effort drains."""
        self._surrender_payloads(
            self.lease_surrender_payloads(),
            time.monotonic() + _SURRENDER_TIMEOUT_S,
        )

    def close(self) -> None:
        """Fence renewals, drain state, and surrender within one deadline."""
        payloads = self._begin_close()
        if payloads is None:
            self._http.close()
            return
        drain_deadline = time.monotonic() + _SURRENDER_TIMEOUT_S
        with self._state_lock:
            threads = list(self._renewal_threads)

        surrender_worker: threading.Thread | None = None
        if payloads:
            surrender_worker = threading.Thread(
                target=self._surrender_payloads,
                args=(payloads, drain_deadline),
                name="solwyn-lease-surrender",
                daemon=True,
            )
            try:
                surrender_worker.start()
            except RuntimeError:
                logger.debug("lease.surrender_dispatch_failed")
                surrender_worker = None

        for thread in threads:
            thread.join(timeout=max(0.0, drain_deadline - time.monotonic()))
        if surrender_worker is not None:
            surrender_worker.join(timeout=max(0.0, drain_deadline - time.monotonic()))
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
        self._renewal_slots_in_use = 0
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
        self._renewal_slots_in_use = 0

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
        tags: dict[str, str] | None = None,
        call_id: str | None = None,
        estimated_output_bound: int | None = None,
    ) -> BudgetCheckResult:
        """Async version of budget check. See BudgetEnforcer.check_budget."""
        lease_call_id: str | None = None
        lease_claim_token: int | None = None
        if (
            modality == "text"
            and estimated_media is None
            and tags is None
            and self._lease_path_applies(agent_run_id)
        ):
            if agent_run_id is None:
                raise RuntimeError("lease path requires an agent_run_id")
            lease_call_id = (
                _validated_call_id(call_id) if call_id is not None else _validated_call_id(None)
            )
            leased, lease_claim_token = await self._check_lease(
                agent_run_id=agent_run_id,
                call_id=lease_call_id,
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

        if agent_run_id is None and tags is None and self._should_use_cache():
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
                denied_by_period=cached.denied_by_period,
            )

        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        if admission is not None and not admission.allowed:
            # Negative cache: a recent streak of control-plane failures means
            # the posture applies instantly instead of paying the timeout.
            logger.debug("budget.check_skipped_breaker_open")
            leased = self._lease_result_when_breaker_refuses(
                agent_run_id=agent_run_id,
                call_id=lease_call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
                claim_token=lease_claim_token,
            )
            if leased is not None:
                return leased
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
            tags,
        )

        try:
            # ── Phase 1: transport + HTTP status. Failures here are OUTAGE
            # semantics — unchanged from before the split.
            try:
                request_epoch = time.monotonic()
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
            except Exception as exc:
                # A read-only-key error means the control plane RESPONDED —
                # record success. Anything else is an outage: record failure.
                # Log the exception TYPE only (never a body).
                if handle_read_only_key_error(exc):
                    if breaker is not None:
                        breaker.record_success()
                else:
                    if breaker is not None:
                        breaker.record_failure()
                    logger.warning("Cloud API budget check failed: %s", type(exc).__name__)
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

            # ── Phase 2: response processing (R6). The plane RESPONDED 2xx;
            # a body we cannot parse is server contract drift, NOT an outage:
            # credit the breaker (drift must not open it), signal at ERROR
            # with a distinct event, and degrade to the same posture ladder.
            try:
                cloud_response = BudgetCheckResponse.model_validate(resp.json())
            except Exception as exc:
                if breaker is not None:
                    breaker.record_success()
                logger.error(
                    "budget.check_response_unreadable: %s — possible server "
                    "contract drift; enforcement degraded (fail_open=%s)",
                    type(exc).__name__,
                    self.fail_open,
                )
                return self._build_unreachable_result(estimated_input_tokens, agent_run_id)

            try:
                cloud_response, cache_response = self._apply_check_run_control(
                    cloud_response,
                    agent_run_id,
                )
            except _MisroutedControlDenial:
                if breaker is not None:
                    breaker.record_failure()
                return self._build_unreachable_result(
                    estimated_input_tokens,
                    agent_run_id,
                )
            if cache_response:
                cloud_response = self._cache_response(
                    cloud_response,
                    agent_run_id=agent_run_id,
                    cache_allowed_response=tags is None,
                    request_epoch=request_epoch,
                )
            if breaker is not None:
                breaker.record_success()
            return self._build_result_from_response(cloud_response)
        finally:
            # Cancellation (or any BaseException) bypasses the handlers above;
            # a consumed HALF_OPEN probe slot must be freed or every later
            # recovery probe is refused. No-op once a verdict released it.
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
    ) -> tuple[BudgetCheckResult | None, int | None]:
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
                return self._lease_deny_result(agent_run_id, response), admission.claim_token
            if verdict == "unreachable":
                if self.fail_open:
                    self._record_uncounted_cold_start(
                        agent_run_id,
                        self._lease_reserve_estimate(
                            estimated_input_tokens, estimated_output_bound
                        ),
                        call_id=call_id,
                        claim_token=admission.claim_token,
                    )
                    self._note_uncounted_admission(agent_run_id, reason="no_lease_unreachable")
                result = self._build_unreachable_result(
                    estimated_input_tokens,
                    agent_run_id,
                ).model_copy(update={"lease_claim_token": admission.claim_token})
                return result, admission.claim_token
            if verdict != "applied":
                return None, admission.claim_token
            admission = self._admit_lease(
                agent_run_id=agent_run_id,
                call_id=call_id,
                estimated_input_tokens=estimated_input_tokens,
                model=model,
                estimated_output_bound=estimated_output_bound,
                modality=modality,
                estimated_media=estimated_media,
                fallback_models=fallback_models,
                claim_token=admission.claim_token,
            )
            if admission.decision is LeaseDecision.NEED_GRANT:
                return None, admission.claim_token

        if admission.renewal_due:
            self._start_renewal(
                agent_run_id,
                model=model,
                provider=provider,
                fallback_providers=fallback_providers,
                fallback_models=fallback_models,
            )

        if admission.decision is LeaseDecision.LEGACY_CHECK:
            return None, admission.claim_token
        if admission.decision is LeaseDecision.ADMIT_UNCOUNTED:
            self._note_uncounted_admission(
                agent_run_id, reason=admission.reason or "expired_fail_open"
            )
        return self._result_from_admission(agent_run_id, admission), admission.claim_token

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
        close_epoch = self._claim_grant_work(agent_run_id)
        if close_epoch is None:
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
                return self._classify_lease_failure(exc, agent_run_id), None

            verdict, response, late_surrender = self._install_grant(
                agent_run_id,
                resp,
                declared_models=[model, *fallback_models],
                close_epoch=close_epoch,
            )
            if late_surrender is not None:
                await self._surrender_late_renewal(late_surrender)
            if breaker is not None:
                if verdict == "unreachable":
                    breaker.record_failure()
                else:
                    breaker.record_success()
            return verdict, response
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("lease.renew_dispatch_failed: RuntimeError")
            return
        with self._state_lock:
            if self._closed:
                return
            if self._renewal_slots_in_use >= _MAX_RENEWAL_WORKERS:
                logger.debug("lease.renew_worker_limit")
                return
            request = self._lease.claim_renewal_request(
                agent_run_id,
                model=model,
                provider=ProviderName(provider),
                fallback_providers=[ProviderName(p) for p in fallback_providers],
                fallback_models=list(fallback_models),
            )
            if request is None:
                return
            self._renewal_slots_in_use += 1
            close_epoch = self._close_epoch
        coroutine = self._renew_lease(
            agent_run_id,
            request,
            [model, *fallback_models],
            close_epoch=close_epoch,
        )
        try:
            task = loop.create_task(coroutine)
        except Exception as exc:
            coroutine.close()
            logger.warning("lease.renew_dispatch_failed: %s", type(exc).__name__)
            with self._state_lock:
                self._renewal_slots_in_use -= 1
                self._lease.renewal_failed(
                    agent_run_id,
                    time.monotonic(),
                    expected_lease_id=request.lease_id,
                    expected_generation=request.generation,
                )
            return
        with self._state_lock:
            self._renewal_tasks.add(task)
        task.add_done_callback(self._renewal_task_done)

    def _renewal_task_done(self, task: asyncio.Task[None]) -> None:
        """Release one async worker slot after any task outcome."""
        with self._state_lock:
            self._renewal_tasks.discard(task)
            self._renewal_slots_in_use = max(0, self._renewal_slots_in_use - 1)

    async def _renew_lease(
        self,
        agent_run_id: str,
        request: LeaseRenewRequest,
        declared_models: list[str],
        *,
        close_epoch: int | None = None,
    ) -> None:
        """Renewal task: breaker-guarded, never raises into the loop."""
        breaker = self._control_plane_breaker
        admission = breaker.admit() if breaker is not None else None
        try:
            if admission is not None and not admission.allowed:
                logger.debug("lease.renew_skipped_breaker_open")
                with self._state_lock:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
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
                self._classify_lease_failure(exc, agent_run_id, renewal_request=request)
                return
            if breaker is not None:
                breaker.record_success()
            try:
                response = LeaseGrantResponse.model_validate(resp.json())
            except Exception as exc:
                logger.warning("lease.renew_response_unreadable: %s", type(exc).__name__)
                with self._state_lock:
                    self._lease.renewal_failed(
                        agent_run_id,
                        time.monotonic(),
                        expected_lease_id=request.lease_id,
                        expected_generation=request.generation,
                    )
                return
            late_surrender = self._finish_renewal(
                agent_run_id,
                request,
                response,
                declared_models,
                close_epoch=close_epoch,
            )
            if late_surrender is not None:
                await self._surrender_late_renewal(late_surrender)
        except asyncio.CancelledError:
            with self._state_lock:
                self._lease.renewal_failed(
                    agent_run_id,
                    time.monotonic(),
                    expected_lease_id=request.lease_id,
                    expected_generation=request.generation,
                )
            raise
        except Exception as exc:  # pragma: no cover — a task must never raise
            logger.warning("lease.renew_worker_failed: %s", type(exc).__name__)
        finally:
            if breaker is not None:
                breaker.release_probe(admission)

    async def _surrender_payloads(
        self,
        payloads: Sequence[LeaseSurrenderRequest],
        deadline: float,
    ) -> None:
        """Async release drain bounded by one monotonic deadline."""
        async with httpx.AsyncClient(timeout=_SURRENDER_TIMEOUT_S) as client:
            for request in payloads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                breaker = self._control_plane_breaker
                admission = breaker.admit() if breaker is not None else None
                try:
                    if admission is not None and not admission.allowed:
                        logger.debug("lease.surrender_skipped_breaker_open")
                        continue
                    resp = await client.post(
                        f"{self.api_url}{_LEASE_SURRENDER_PATH}",
                        json=request.model_dump(mode="json"),
                        headers=self._auth_headers(),
                        timeout=max(0.001, min(_SURRENDER_TIMEOUT_S, remaining)),
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

    async def _surrender_late_renewal(self, request: LeaseSurrenderRequest) -> None:
        """Release a renewal successor observed after the close epoch changed."""
        await self._surrender_payloads(
            [request],
            time.monotonic() + _SURRENDER_TIMEOUT_S,
        )

    async def _surrender_leases(self) -> None:
        """Compatibility helper for explicit best-effort drains."""
        await self._surrender_payloads(
            self.lease_surrender_payloads(),
            time.monotonic() + _SURRENDER_TIMEOUT_S,
        )

    async def close(self) -> None:
        """Await renewals and release drained leases within one deadline."""
        payloads = self._begin_close()
        if payloads is None:
            await self._http.aclose()
            return
        drain_deadline = time.monotonic() + _SURRENDER_TIMEOUT_S
        with self._state_lock:
            renewal_tasks = list(self._renewal_tasks)

        waitables: list[asyncio.Task[None]] = renewal_tasks
        if payloads:
            waitables.append(
                asyncio.create_task(self._surrender_payloads(payloads, drain_deadline))
            )
        if waitables:
            _, pending = await asyncio.wait(
                waitables,
                timeout=max(0.0, drain_deadline - time.monotonic()),
            )
            for task in pending:
                task.cancel()
            # Let cooperative cancellation run without adding another
            # unbounded shutdown wait.
            await asyncio.sleep(0)
        await self._http.aclose()
