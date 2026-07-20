"""Budget enforcement with cloud API check and local fallback.

BudgetEnforcer (sync) and AsyncBudgetEnforcer (async) handle pre-call
budget checks via the Solwyn cloud API, with local enforcement as
fallback when the cloud is unreachable.

Adapted from solwyn-core CostTracker (Redis -> HTTP cloud API).
Local in-process dict used as fallback when cloud is unreachable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import cast, get_args

import httpx
from pydantic import BaseModel, ConfigDict

from solwyn._read_only_key import handle_read_only_key_error
from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    MediaUsage,
    Modality,
    ProviderName,
    ServiceTier,
)
from solwyn.circuit_breaker import CircuitBreaker

# Fallback per-token cost when cloud API is unreachable.
DEFAULT_COST_PER_TOKEN: float = 0.00003

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

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for cloud API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_confirm_request(
        self,
        reservation_id: str,
        model: str,
        token_details: TokenDetails,
        *,
        provider: str,
        is_provider_fallback: bool = False,
        call_id: str,
        provider_region: str | None = None,
        service_tier: str | None = None,
        modality: Modality = "text",
        media_usage: MediaUsage | None = None,
    ) -> BudgetConfirmRequest:
        """Build a validated confirm request for fire-and-forget callers.

        Stream completion builds this synchronously (no I/O) and enqueues
        it on the reporter thread, avoiding a blocking httpx.post. ``provider``
        is the provider that actually served the call (required).
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
        return BudgetConfirmRequest(
            reservation_id=reservation_id,
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
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
            control_plane_breaker=control_plane_breaker,
        )
        self._http = httpx.Client(timeout=5.0)

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
    ) -> BudgetCheckResult:
        """Check whether a call is within budget.

        ``fallback_providers``/``fallback_models`` describe the configured
        failover chain (aligned element-for-element) as a hint to the API.
        ``modality`` is ``"text"`` for chat; the media lifecycle passes the
        surface's modality (e.g. ``"embedding"``). ``estimated_media`` carries a
        non-text surface's pre-flight non-token quantities for a precise
        check-time cost; None for chat/token calls.

        Behaviour matrix:
        - Cloud reachable + allowed: return allowed=True
        - Cloud reachable + denied + alert_only: return allowed=True + warning
        - Cloud reachable + denied + hard_deny: return allowed=False
        - Cloud unreachable after hard_deny: return allowed=False
        - Cloud unreachable + fail_open=True: return allowed=True + warning
        - Cloud unreachable + fail_open=False: enforce locally
        """
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

    def close(self) -> None:
        """Close the underlying HTTP client."""
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
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
            control_plane_breaker=control_plane_breaker,
        )
        self._http = httpx.AsyncClient(timeout=5.0)

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
    ) -> BudgetCheckResult:
        """Async version of budget check. See BudgetEnforcer.check_budget."""
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

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()
