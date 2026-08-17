"""Sans-I/O state machine for the in-process Solwyn control-plane double."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from solwyn._types import (
    BreakerStateReport,
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetConfirmRequest,
    BudgetMode,
    FailoverDirective,
    LeaseGrantRequest,
    LeaseRenewRequest,
    LeaseSurrenderRequest,
    MetadataEvent,
    ProviderName,
    UntrackedSurfaceReport,
)
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.testing._transport import FakeControlPlaneTransport
from solwyn.testing._wire import (
    PlaneResponse,
    PreparedPlaneRequest,
    parse_model,
    parse_model_list,
)

if TYPE_CHECKING:
    from solwyn._base import MediaSurfaceSpec

MAGIC_MODELS = frozenset(
    {
        "solwyn-test/deny",
        "solwyn-test/deny-alert",
        "solwyn-test/deny-tag",
        "solwyn-test/deny-stopped",
        "solwyn-test/runaway",
        "solwyn-test/lease-ineligible",
    }
)

_DENIAL_PERIODS = frozenset({"monthly", "agent_run", "run_stopped", "tag"})
_LEASE_PATHS = frozenset(
    {
        "/api/v1/budgets/lease",
        "/api/v1/budgets/lease/renew",
        "/api/v1/budgets/lease/surrender",
    }
)


def _validate_magic_model(model: object) -> None:
    if isinstance(model, str) and model.startswith("solwyn-test/") and model not in MAGIC_MODELS:
        raise RuntimeError(f"unknown solwyn testing magic model: {model!r}")


class _TestingSolwyn(Solwyn):
    def _intercepted_call(
        self,
        *,
        _force_stream: bool = False,
        _surface: str = "chat",
        _responses_leaf: str = "create",
        **kwargs: object,
    ) -> Any:
        _validate_magic_model(kwargs.get("model"))
        for runtime in self._runtimes[1:]:
            _validate_magic_model(runtime.entry.model)
        return super()._intercepted_call(
            _force_stream=_force_stream,
            _surface=_surface,
            _responses_leaf=_responses_leaf,
            **kwargs,
        )

    def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        _validate_magic_model(kwargs.get("model"))
        return super()._media_call(spec, **kwargs)


class _TestingAsyncSolwyn(AsyncSolwyn):
    async def _intercepted_call(
        self,
        *,
        _force_stream: bool = False,
        _surface: str = "chat",
        _responses_leaf: str = "create",
        **kwargs: object,
    ) -> Any:
        _validate_magic_model(kwargs.get("model"))
        for runtime in self._runtimes[1:]:
            _validate_magic_model(runtime.entry.model)
        return await super()._intercepted_call(
            _force_stream=_force_stream,
            _surface=_surface,
            _responses_leaf=_responses_leaf,
            **kwargs,
        )

    async def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        _validate_magic_model(kwargs.get("model"))
        return await super()._media_call(spec, **kwargs)


@dataclass(slots=True)
class _ScenarioWindow:
    kind: str
    remaining: int | None
    path: str | None = None
    seconds: float = 0.0
    status: int = 0
    retry_after: int | None = None

    def consume_if_matching(self, path: str) -> bool:
        if self.path is not None and self.path != path:
            return False
        if self.remaining == 0:
            return False
        if self.remaining is not None:
            self.remaining -= 1
        return True


class FakeControlPlane:
    """In-process Solwyn control-plane double. Deterministic, zero-network.

    Simulates wire behavior only: denials are scripted, never priced —
    the real API owns all pricing. Thread-safe because reporter flush threads
    and lease renewal workers may hit it concurrently.
    """

    def __init__(
        self,
        *,
        mode: BudgetMode | str = BudgetMode.HARD_DENY,
        budget_limit: float = 100.0,
        current_usage: float = 0.0,
        remaining_budget: float | None = None,
        project_id: str = "proj_fake",
        failover_tuning_allowed: bool = True,
        price_hints: dict[str, float] | None = None,
        lease_eligible: bool = True,
        granted_tokens: int = 200_000,
        refresh_interval_s: float = 30.0,
        lease_length_s: float = 90.0,
    ) -> None:
        self.mode = BudgetMode(mode)
        self.budget_limit = budget_limit
        self.current_usage = current_usage
        self.remaining_budget = (
            budget_limit - current_usage if remaining_budget is None else remaining_budget
        )
        self.project_id = project_id
        self.failover_tuning_allowed = failover_tuning_allowed
        self.price_hints = price_hints
        self.lease_eligible = lease_eligible
        self.granted_tokens = granted_tokens
        self.refresh_interval_s = refresh_interval_s
        self.lease_length_s = lease_length_s

        self.api_key = "sk_proj_" + "0" * 64
        self.api_url = "http://control-plane.invalid"
        self.transport = FakeControlPlaneTransport(self)

        self.checks: list[BudgetCheckRequest] = []
        self.confirms: list[BudgetConfirmRequest] = []
        self.ingested: list[MetadataEvent] = []
        self.lease_grants: list[LeaseGrantRequest] = []
        self.lease_renewals: list[LeaseRenewRequest] = []
        self.lease_surrenders: list[LeaseSurrenderRequest] = []
        self.untracked_reports: list[UntrackedSurfaceReport] = []
        self.breaker_reports: list[dict[str, Any]] = []
        self.unmatched_requests: list[tuple[str, str]] = []
        self._pending_denials: deque[str] = deque()
        self._denied_runs: set[str] = set()
        self._runaway_seen: set[str | None] = set()
        self._scenario_windows: list[_ScenarioWindow] = []
        self._active_reservations: set[str] = set()
        self._settled_reservations: set[str] = set()
        self._seen_call_ids: set[str] = set()
        self._reservation_counter = 0
        self._lock = Lock()

    def deny_next(self, n: int = 1, *, period: str = "monthly") -> None:
        """Deny the next ``n`` checks for the selected budget period."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if period not in _DENIAL_PERIODS:
            raise ValueError("period must be monthly, agent_run, run_stopped, or tag")
        with self._lock:
            self._pending_denials.extend(period for _ in range(n))

    def deny_run(self, agent_run_id: str) -> None:
        """Deny checks made in one agent run."""
        with self._lock:
            self._denied_runs.add(agent_run_id)

    def clear_denials(self) -> None:
        """Clear pending and run-scoped scripted denials."""
        with self._lock:
            self._pending_denials.clear()
            self._denied_runs.clear()

    def expire_reservations(self) -> None:
        """Expire all currently outstanding reservation ids."""
        with self._lock:
            self._active_reservations.clear()

    def reset_recording(self) -> None:
        """Clear recorded requests without changing scripted scenario state."""
        with self._lock:
            self.checks.clear()
            self.confirms.clear()
            self.ingested.clear()
            self.lease_grants.clear()
            self.lease_renewals.clear()
            self.lease_surrenders.clear()
            self.untracked_reports.clear()
            self.breaker_reports.clear()
            self.unmatched_requests.clear()

    def wrap(self, provider_client: object, **solwyn_kwargs: Any) -> Solwyn:
        """Wrap a synchronous provider client on this in-process plane."""
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "control_plane_transport": self.transport,
            "budget_check_cache_ttl": 0,
            "lease_enabled": False,
        }
        options.update(solwyn_kwargs)
        return _TestingSolwyn(provider_client, **options)

    def wrap_async(self, provider_client: object, **solwyn_kwargs: Any) -> AsyncSolwyn:
        """Wrap an asynchronous provider client on this in-process plane."""
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "control_plane_transport": self.transport,
            "budget_check_cache_ttl": 0,
            "lease_enabled": False,
        }
        options.update(solwyn_kwargs)
        return _TestingAsyncSolwyn(provider_client, **options)

    @contextmanager
    def outage(self, *, requests: int | None = None) -> Iterator[None]:
        """Raise a connection error for matching requests in this window."""
        with self._scenario(_ScenarioWindow("outage", self._request_count(requests))):
            yield

    @contextmanager
    def slow(
        self,
        seconds: float,
        *,
        path: str = "/api/v1/budgets/confirm",
        requests: int | None = None,
    ) -> Iterator[None]:
        """Delay matched requests without embedding wall-clock logic in verdicts."""
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        window = _ScenarioWindow(
            "slow",
            self._request_count(requests),
            path=path,
            seconds=seconds,
        )
        with self._scenario(window):
            yield

    @contextmanager
    def read_only(self, *, requests: int | None = None) -> Iterator[None]:
        """Refuse write requests as a core read-only project key would."""
        with self._scenario(_ScenarioWindow("read_only", self._request_count(requests))):
            yield

    @contextmanager
    def refuse_checks(
        self,
        *,
        status: int = 503,
        requests: int | None = None,
        retry_after: int | None = None,
    ) -> Iterator[None]:
        """Return a scripted HTTP refusal from the budget-check endpoint."""
        if status not in {422, 429, 503}:
            raise ValueError("status must be 422, 429, or 503")
        if retry_after is not None and retry_after < 0:
            raise ValueError("retry_after must be non-negative")
        window = _ScenarioWindow(
            "refuse_checks",
            self._request_count(requests),
            path="/api/v1/budgets/check",
            status=status,
            retry_after=retry_after,
        )
        with self._scenario(window):
            yield

    @staticmethod
    def _request_count(requests: int | None) -> int | None:
        if requests is not None and requests < 0:
            raise ValueError("requests must be non-negative")
        return requests

    @contextmanager
    def _scenario(self, window: _ScenarioWindow) -> Iterator[None]:
        with self._lock:
            self._scenario_windows.append(window)
        try:
            yield
        finally:
            with self._lock:
                if window in self._scenario_windows:
                    self._scenario_windows.remove(window)

    def _prepare_request(
        self,
        method: str,
        path: str,
        body: object,
        parse_error: PlaneResponse | None,
    ) -> PreparedPlaneRequest:
        """Freeze all request effects before the transport performs I/O."""
        with self._lock:
            delay, outage = self._consume_transport_effects(path)
            if outage:
                response = None
            elif parse_error is not None:
                response = parse_error
            else:
                response = self._handle_locked(method, path, body)
        return PreparedPlaneRequest(delay, outage, response)

    def _consume_transport_effects(self, path: str) -> tuple[float, bool]:
        """Consume matching delay/outage windows while the caller holds the lock."""
        delay = 0.0
        outage = False
        for window in self._scenario_windows:
            if window.kind not in {"outage", "slow"}:
                continue
            if not window.consume_if_matching(path):
                continue
            if window.kind == "outage":
                outage = True
            else:
                delay += window.seconds
        return delay, outage

    def handle(self, method: str, path: str, body: object) -> PlaneResponse:
        """Handle one control-plane request without performing I/O."""
        with self._lock:
            return self._handle_locked(method, path, body)

    def _handle_locked(self, method: str, path: str, body: object) -> PlaneResponse:
        refusal = self._endpoint_refusal_locked(method, path, body)
        if refusal is not None:
            return refusal
        if method == "POST" and path == "/api/v1/budgets/check":
            return self._handle_check(body)
        if method == "POST" and path == "/api/v1/budgets/confirm":
            return self._handle_confirm(body)
        if method == "POST" and path == "/api/v1/metadata/ingest":
            return self._handle_ingest(body)
        if method == "POST" and path == "/api/v1/untracked-surfaces":
            return self._handle_untracked(body)
        if (
            method == "POST"
            and path.startswith("/api/v1/projects/")
            and path.endswith("/providers/breaker-reports")
        ):
            return self._handle_breaker_report(body)
        if method == "GET" and path == "/health":
            return PlaneResponse(200, {"status": "ok"})
        if path in _LEASE_PATHS:
            return PlaneResponse(
                501,
                {"detail": "solwyn.testing lease endpoints arrive in Task 3"},
            )
        self.unmatched_requests.append((method, path))
        return PlaneResponse(404, {"detail": "not found"})

    def _endpoint_refusal_locked(
        self,
        method: str,
        path: str,
        body: object,
    ) -> PlaneResponse | None:
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        for window in self._scenario_windows:
            if window.kind == "refuse_checks" and window.consume_if_matching(path):
                if window.status == 422:
                    raw_model = body.get("model") if isinstance(body, dict) else None
                    raw_provider = body.get("provider") if isinstance(body, dict) else None
                    model = raw_model[:50] if isinstance(raw_model, str) else "unknown"
                    provider = raw_provider if isinstance(raw_provider, str) else "unknown"
                    detail: object = {
                        "code": "unknown_model",
                        "model": model,
                        "provider": provider,
                        "message": (
                            f"Solwyn does not have pricing for model '{model}'. "
                            "File an issue at "
                            "https://github.com/solwyn-ai/solwyn-python-sdk/issues "
                            "or contact support — we typically add new models within 24h."
                        ),
                    }
                    headers = None
                elif window.status == 429:
                    retry_after = window.retry_after if window.retry_after is not None else 60
                    return PlaneResponse(
                        429,
                        {
                            "detail": "Rate limit exceeded",
                            "retry_after": retry_after,
                        },
                        headers={"Retry-After": str(retry_after)},
                    )
                else:
                    detail = "Budget backend temporarily unavailable; retry"
                    headers = None
                return PlaneResponse(window.status, {"detail": detail}, headers=headers)
        for window in self._scenario_windows:
            if window.kind == "read_only" and window.consume_if_matching(path):
                return PlaneResponse(
                    403,
                    {
                        "detail": {
                            "code": "read_only_key",
                            "message": "read-only project key cannot write",
                        }
                    },
                )
        return None

    def _handle_check(self, body: object) -> PlaneResponse:
        parsed = parse_model(BudgetCheckRequest, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        _validate_magic_model(parsed.model)
        for fallback_model in parsed.fallback_models:
            _validate_magic_model(fallback_model)

        self.checks.append(parsed)
        denied_by_period = self._check_denial_period(parsed)
        response_mode = (
            BudgetMode.ALERT_ONLY if parsed.model == "solwyn-test/deny-alert" else self.mode
        )
        self._reservation_counter += 1
        reservation_id = f"res_fake_{self._reservation_counter:08d}"
        if denied_by_period is None:
            self._active_reservations.add(reservation_id)

        opted_in = parsed.failover_directive_version == "1"
        response = BudgetCheckResponse(
            allowed=denied_by_period is None,
            remaining_budget=self.remaining_budget,
            reservation_id=reservation_id if denied_by_period is None else None,
            mode=response_mode,
            budget_limit=self.budget_limit,
            current_usage=self.current_usage,
            denied_by_period=denied_by_period,
            project_id=self.project_id,
            price_hints=(
                {ProviderName(provider): hint for provider, hint in self.price_hints.items()}
                if self.price_hints is not None
                else None
            ),
            failover_directive=(
                FailoverDirective(
                    version="1",
                    failover_tuning_allowed=self.failover_tuning_allowed,
                )
                if opted_in
                else None
            ),
        )
        return PlaneResponse(
            200,
            response,
            exclude_none=opted_in,
            model_only=True,
        )

    def _handle_confirm(self, body: object) -> PlaneResponse:
        parsed = parse_model(BudgetConfirmRequest, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        if parsed.call_id in self._seen_call_ids:
            return PlaneResponse(204)
        if parsed.reservation_id is not None:
            reservation_id = parsed.reservation_id
            if reservation_id in self._settled_reservations:
                return PlaneResponse(204)
            if reservation_id not in self._active_reservations:
                return PlaneResponse(
                    404,
                    {"detail": "Reservation not found or expired"},
                )
            self._active_reservations.remove(reservation_id)
            self._settled_reservations.add(reservation_id)
        self._seen_call_ids.add(parsed.call_id)
        self.confirms.append(parsed)
        return PlaneResponse(204)

    def _handle_ingest(self, body: object) -> PlaneResponse:
        parsed = parse_model_list(MetadataEvent, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        self.ingested.extend(parsed)
        return PlaneResponse(202, {"ingested": len(parsed), "rejected": []})

    def _handle_untracked(self, body: object) -> PlaneResponse:
        if isinstance(body, list) and len(body) > 100:
            return PlaneResponse(
                400,
                {"detail": "untracked surface batches may contain at most 100 reports"},
            )
        parsed = parse_model_list(UntrackedSurfaceReport, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        self.untracked_reports.extend(parsed)
        return PlaneResponse(202, {"accepted": len(parsed)})

    def _handle_breaker_report(self, body: object) -> PlaneResponse:
        parsed = parse_model(BreakerStateReport, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        self.breaker_reports.append(parsed.model_dump(mode="json"))
        return PlaneResponse(204)

    def _check_denial_period(self, request: BudgetCheckRequest) -> str | None:
        if self._pending_denials:
            return self._pending_denials.popleft()
        if request.agent_run_id in self._denied_runs:
            return "agent_run"
        magic_periods = {
            "solwyn-test/deny": "monthly",
            "solwyn-test/deny-alert": "monthly",
            "solwyn-test/deny-tag": "tag",
            "solwyn-test/deny-stopped": "run_stopped",
        }
        magic_period = magic_periods.get(request.model)
        if magic_period is not None:
            return magic_period
        if request.model == "solwyn-test/runaway":
            run_id = request.agent_run_id
            if run_id in self._runaway_seen:
                return "agent_run"
            self._runaway_seen.add(run_id)
        return None
