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
from datetime import UTC, datetime
from typing import cast, get_args

import httpx
from pydantic import BaseModel, ConfigDict

from solwyn._token_details import TokenDetails
from solwyn._types import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    ProviderName,
    ServiceTier,
)

# Fallback per-token cost when cloud API is unreachable.
DEFAULT_COST_PER_TOKEN: float = 0.00003

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
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.budget_mode = budget_mode
        self.fail_open = fail_open
        self.cache_ttl = cache_ttl

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

    def _build_check_request(
        self,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] | None = None,
        fallback_models: list[str] | None = None,
    ) -> BudgetCheckRequest:
        """Build a BudgetCheckRequest for the cloud API.

        ``fallback_providers``/``fallback_models`` describe the configured
        failover chain (aligned element-for-element) as a hint to the API.
        Both default to empty lists; the caller's lists are never mutated.
        """
        return BudgetCheckRequest(
            estimated_input_tokens=estimated_input_tokens,
            model=model,
            provider=ProviderName(provider),
            fallback_providers=[ProviderName(p) for p in (fallback_providers or [])],
            fallback_models=list(fallback_models or []),
        )

    def _should_use_cache(self) -> bool:
        """Return True if we have a valid cached allow response."""
        with self._state_lock:
            return (
                self._cached_response is not None
                and self._cached_response.allowed
                and time.monotonic() < self._cache_expires_at
            )

    def _cache_response(self, response: BudgetCheckResponse) -> None:
        """Cache an allow response. Never cache deny responses.

        Always updates the last-known budget limit (from both allow and deny)
        so that local enforcement can use it when the cloud becomes unreachable.
        """
        with self._state_lock:
            # Always remember the limit for local enforcement fallback
            self._last_known_budget_limit = response.budget_limit
            self._last_known_current_usage = response.current_usage

            if response.allowed:
                self._last_hard_deny_response = None
                self._cached_response = response
                self._cache_expires_at = time.monotonic() + self.cache_ttl
            else:
                self._cached_response = None
                self._cache_expires_at = 0.0
                self._last_hard_deny_response = (
                    response if self.budget_mode == BudgetMode.HARD_DENY else None
                )
            # Deny responses are never cached as freshness-skipping allows/denies.

    def _build_prior_hard_deny_unavailable_result(self) -> BudgetCheckResult | None:
        """Return a denial if cloud is down after an authoritative hard deny."""
        with self._state_lock:
            response = self._last_hard_deny_response
        if response is None:
            return None
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

        Applies budget_mode logic:
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
            )

        # Denied by cloud
        if self.budget_mode == BudgetMode.ALERT_ONLY:
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
        raw echo still reaches the API on the MetadataEvent.
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
            is_provider_fallback=is_provider_fallback,
            token_details=token_details,
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
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
        )
        self._http = httpx.Client(timeout=5.0)
        self._consecutive_confirm_failures = 0
        self._confirm_failure_threshold = 10

    def check_budget(
        self,
        *,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] = [],  # noqa: B006 — read-only; never mutated
        fallback_models: list[str] = [],  # noqa: B006 — read-only; never mutated
        timeout: float | None = None,
    ) -> BudgetCheckResult:
        """Check whether a call is within budget.

        ``fallback_providers``/``fallback_models`` describe the configured
        failover chain (aligned element-for-element) as a hint to the API.

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
        with self._state_lock:
            cached = self._cached_response
            if cached is not None and cached.allowed and time.monotonic() < self._cache_expires_at:
                return BudgetCheckResult(
                    allowed=True,
                    remaining_budget=cached.remaining_budget,
                    project_id=cached.project_id,
                    reservation_id=None,  # Don't reuse — each call needs its own reservation
                    mode=cached.mode,
                    budget_limit=cached.budget_limit,
                    current_usage=cached.current_usage,
                )

        request = self._build_check_request(
            estimated_input_tokens, model, provider, fallback_providers, fallback_models
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
            self._cache_response(cloud_response)
            return self._build_result_from_response(cloud_response)

        except Exception as exc:
            # Log the exception TYPE only — symmetric with confirm_cost and
            # defense-in-depth: never interpolate the full exception (which could
            # carry response/body text) into the log.
            logger.warning("Cloud API budget check failed: %s", type(exc).__name__)

            prior_hard_deny = self._build_prior_hard_deny_unavailable_result()
            if prior_hard_deny is not None:
                return prior_hard_deny

            if self.fail_open:
                return self._build_fail_open_result(estimated_input_tokens)
            else:
                return self._build_local_enforcement_result(estimated_input_tokens)

    def confirm_cost(
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
    ) -> None:
        """Confirm actual token usage for a budget reservation.

        ``provider`` is the provider that actually served the call (required).
        ``call_id`` is the required per-call reconciliation join key.

        Best-effort: failures are logged but do not raise.
        Tracks consecutive failures; after _confirm_failure_threshold consecutive
        failures, logs at ERROR level so operators can see a persistent problem.
        """
        try:
            request = self.build_confirm_request(
                reservation_id,
                model,
                token_details,
                provider=provider,
                is_provider_fallback=is_provider_fallback,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
            )
            resp = self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=request.model_dump(mode="json"),
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            with self._state_lock:
                self._consecutive_confirm_failures = 0
        except Exception as exc:
            with self._state_lock:
                self._consecutive_confirm_failures += 1
                count = self._consecutive_confirm_failures
            if count >= self._confirm_failure_threshold:
                logger.error(
                    "budget.confirm_cost_persistent_failure: exc_type=%s consecutive_failures=%d",
                    type(exc).__name__,
                    count,
                )
            else:
                logger.warning(
                    "budget.confirm_cost_failed: exc_type=%s",
                    type(exc).__name__,
                )

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
    ) -> None:
        super().__init__(
            api_url=api_url,
            api_key=api_key,
            budget_mode=budget_mode,
            fail_open=fail_open,
            cache_ttl=cache_ttl,
        )
        self._http = httpx.AsyncClient(timeout=5.0)
        self._consecutive_confirm_failures = 0
        self._confirm_failure_threshold = 10

    async def check_budget(
        self,
        *,
        estimated_input_tokens: int,
        model: str,
        provider: str,
        fallback_providers: list[str] = [],  # noqa: B006 — read-only; never mutated
        fallback_models: list[str] = [],  # noqa: B006 — read-only; never mutated
        timeout: float | None = None,
    ) -> BudgetCheckResult:
        """Async version of budget check. See BudgetEnforcer.check_budget."""
        if self._should_use_cache():
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

        request = self._build_check_request(
            estimated_input_tokens, model, provider, fallback_providers, fallback_models
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
            self._cache_response(cloud_response)
            return self._build_result_from_response(cloud_response)

        except Exception as exc:
            # Log the exception TYPE only — symmetric with confirm_cost and
            # defense-in-depth: never interpolate the full exception (which could
            # carry response/body text) into the log.
            logger.warning("Cloud API budget check failed: %s", type(exc).__name__)

            prior_hard_deny = self._build_prior_hard_deny_unavailable_result()
            if prior_hard_deny is not None:
                return prior_hard_deny

            if self.fail_open:
                return self._build_fail_open_result(estimated_input_tokens)
            else:
                return self._build_local_enforcement_result(estimated_input_tokens)

    async def confirm_cost(
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
    ) -> None:
        """Async version of cost confirmation. See BudgetEnforcer.confirm_cost."""
        try:
            request = self.build_confirm_request(
                reservation_id,
                model,
                token_details,
                provider=provider,
                is_provider_fallback=is_provider_fallback,
                call_id=call_id,
                provider_region=provider_region,
                service_tier=service_tier,
            )
            resp = await self._http.post(
                f"{self.api_url}/api/v1/budgets/confirm",
                json=request.model_dump(mode="json"),
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            self._consecutive_confirm_failures = 0
        except Exception as exc:
            self._consecutive_confirm_failures += 1
            count = self._consecutive_confirm_failures
            if count >= self._confirm_failure_threshold:
                logger.error(
                    "budget.confirm_cost_persistent_failure: exc_type=%s consecutive_failures=%d",
                    type(exc).__name__,
                    count,
                )
            else:
                logger.warning(
                    "budget.confirm_cost_failed: exc_type=%s",
                    type(exc).__name__,
                )

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()
