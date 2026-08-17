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
    LeaseGrantResponse,
    LeasePosture,
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
_LIVE_CONTRACT_UNPRICED_LEASE_MODEL = "no-such-model-for-leases"
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
    code: str = ""
    retry_after: int | None = None

    def consume_if_matching(self, path: str) -> bool:
        if self.path is not None and self.path != path:
            return False
        if self.remaining == 0:
            return False
        if self.remaining is not None:
            self.remaining -= 1
        return True


@dataclass(slots=True)
class _LeaseRecord:
    agent_run_id: str
    holder_id: str
    lease_id: str
    generation: int
    granted_tokens: int
    declared_pairs: tuple[tuple[ProviderName, str], ...]
    fail_open: bool
    last_response_json: str = ""
    last_renewal_request: LeaseRenewRequest | None = None
    terminal_successor: bool = False


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
        if not granted_tokens > 0:
            raise ValueError("granted_tokens must be greater than zero")
        if not lease_length_s > 0:
            raise ValueError("lease_length_s must be greater than zero")
        if not refresh_interval_s > 0:
            raise ValueError("refresh_interval_s must be greater than zero")
        if not refresh_interval_s < lease_length_s:
            raise ValueError("refresh_interval_s must be less than lease_length_s")

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
        self._leases: dict[tuple[str, str], _LeaseRecord] = {}
        self._lease_records: dict[str, _LeaseRecord] = {}
        self._expired_leases: set[str] = set()
        self._released_leases: set[str] = set()
        self._reservation_counter = 0
        self._lease_counter = 0
        self._lock = Lock()

    def deny_next(self, n: int = 1, *, period: str = "monthly") -> None:
        """Script the next ``n`` verdicts as denials for one budget period.

        The wrapped SDK raises in ``hard_deny`` mode and warns before provider
        dispatch in ``alert_only`` mode, matching the ``budget_mode`` knob.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if period not in _DENIAL_PERIODS:
            raise ValueError("period must be monthly, agent_run, run_stopped, or tag")
        with self._lock:
            self._pending_denials.extend(period for _ in range(n))

    def deny_run(self, agent_run_id: str) -> None:
        """Make budget checks for ``agent_run_id`` return an agent-run denial.

        This exercises the SDK's run-scoped sticky-denial behavior without
        affecting unrelated run ids.
        """
        with self._lock:
            self._denied_runs.add(agent_run_id)

    def clear_denials(self) -> None:
        """Clear queued and run-scoped scripts so later checks can recover."""
        with self._lock:
            self._pending_denials.clear()
            self._denied_runs.clear()

    def expire_reservations(self) -> None:
        """Invalidate open reservation ids to exercise late settlement handling."""
        with self._lock:
            self._active_reservations.clear()

    def expire_leases(self) -> None:
        """Expire held leases so the SDK follows its renewal and outage ladder.

        This is relevant only while ``lease_enabled`` is true on the wrapper.
        """
        with self._lock:
            self._expired_leases.update(
                record.lease_id
                for record in self._leases.values()
                if record.lease_id not in self._released_leases
            )

    def reset_recording(self) -> None:
        """Clear every public request recording while preserving scenario state.

        Use this between phases when assertions need to distinguish new control-
        plane traffic without discarding denials, outages, or held leases.
        """
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
        """Return a synchronous ``Solwyn`` using this plane's zero-network transport.

        Keyword arguments are normal ``Solwyn`` configuration overrides;
        provider calls remain the caller's responsibility to mock or serve.
        """
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "control_plane_transport": self.transport,
            "budget_check_cache_ttl": 0,
        }
        options.update(solwyn_kwargs)
        return _TestingSolwyn(provider_client, **options)

    def wrap_async(self, provider_client: object, **solwyn_kwargs: Any) -> AsyncSolwyn:
        """Return an ``AsyncSolwyn`` using this plane's zero-network transport.

        Keyword arguments are normal ``AsyncSolwyn`` configuration overrides;
        provider calls remain the caller's responsibility to mock or serve.
        """
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "control_plane_transport": self.transport,
            "budget_check_cache_ttl": 0,
        }
        options.update(solwyn_kwargs)
        return _TestingAsyncSolwyn(provider_client, **options)

    @contextmanager
    def outage(self, *, requests: int | None = None) -> Iterator[None]:
        """Make matching control-plane requests fail at the transport boundary.

        This drives the SDK's ``fail_open`` posture and the
        ``control_plane_failure_threshold`` / ``control_plane_recovery_timeout``
        breaker knobs. ``requests`` limits how many matching calls fail.
        """
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
        """Delay matched requests to exercise an SDK timeout boundary.

        Budget preflights are bounded by ``budget_check_timeout``; reporter paths
        are bounded at shutdown by ``reporter_shutdown_deadline``. ``path`` keeps
        the delay scoped to the endpoint under test.
        """
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
        """Refuse writes with the real read-only-key response shape.

        This proves the SDK treats an authorization refusal as a reachable
        control plane rather than opening its outage circuit breaker.
        """
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
        """Return a reachable 422, 429, or 503 budget-check refusal.

        This exercises endpoint-refusal posture separately from ``fail_open``
        transport failures; ``retry_after`` scripts the public 429 header.
        """
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

    @contextmanager
    def refuse_leases(
        self,
        *,
        status: int = 503,
        code: str = "lease_unavailable",
        requests: int | None = None,
    ) -> Iterator[None]:
        """Return a public lease-unavailable or holder-cap refusal.

        With ``lease_enabled`` true, the SDK falls back to per-call checks rather
        than treating this reachable lease response as a transport outage.
        """
        valid_pairs = {
            (503, "lease_unavailable"),
            (409, "lease_holder_cap_exceeded"),
        }
        if (status, code) not in valid_pairs:
            raise ValueError(
                "status/code must be 503/lease_unavailable or 409/lease_holder_cap_exceeded"
            )
        window = _ScenarioWindow(
            "refuse_leases",
            self._request_count(requests),
            status=status,
            code=code,
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
        """Evaluate one real control-plane wire request without network I/O.

        Most callers should use :meth:`wrap`; this lower-level seam supports
        contract tests that need the raw status, headers, and response model.
        """
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
        if method == "POST" and path == "/api/v1/budgets/lease":
            return self._handle_lease_grant(body)
        if method == "POST" and path == "/api/v1/budgets/lease/renew":
            return self._handle_lease_renew(body)
        if method == "POST" and path == "/api/v1/budgets/lease/surrender":
            return self._handle_lease_surrender(body)
        if (
            method == "POST"
            and path.startswith("/api/v1/projects/")
            and path.endswith("/providers/breaker-reports")
        ):
            return self._handle_breaker_report(body)
        if method == "GET" and path == "/health":
            return PlaneResponse(200, {"status": "ok"})
        self.unmatched_requests.append((method, path))
        return PlaneResponse(404, {"detail": "not found"})

    def _handle_lease_grant(self, body: object) -> PlaneResponse:
        parsed = parse_model(LeaseGrantRequest, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        _validate_magic_model(parsed.model)
        for fallback_model in parsed.fallback_models:
            _validate_magic_model(fallback_model)

        self.lease_grants.append(parsed)
        key = (parsed.agent_run_id, parsed.holder_id)
        current = self._leases.get(key)
        declared_pairs = self._grant_declared_pairs(parsed)
        if current is not None and self._lease_is_active(current):
            if current.declared_pairs == declared_pairs and current.fail_open == parsed.fail_open:
                return self._replay_lease_response(current)
            return self._lease_error(
                409,
                "lease_holder_cap_exceeded",
                "Active lease holder limit exceeded",
            )

        denied_by_period = self._denial_period(parsed.agent_run_id, parsed.model)
        if denied_by_period is not None:
            return PlaneResponse(
                200,
                self._lease_verdict_response(
                    eligible=True,
                    allowed=False,
                    denied_by_period=denied_by_period,
                    mode=(
                        BudgetMode.ALERT_ONLY
                        if parsed.model == "solwyn-test/deny-alert"
                        else self.mode
                    ),
                ),
                model_only=True,
            )
        if not self.lease_eligible or parsed.model in {
            "solwyn-test/lease-ineligible",
            _LIVE_CONTRACT_UNPRICED_LEASE_MODEL,
        }:
            return PlaneResponse(
                200,
                self._lease_verdict_response(
                    eligible=False,
                    allowed=True,
                    ineligible_reason="zero_rate_model",
                ),
                model_only=True,
            )
        self._lease_counter += 1
        record = _LeaseRecord(
            agent_run_id=parsed.agent_run_id,
            holder_id=parsed.holder_id,
            lease_id=f"lse_fake{self._lease_counter}",
            generation=1,
            granted_tokens=self.granted_tokens,
            declared_pairs=declared_pairs,
            fail_open=parsed.fail_open,
        )
        self._leases[key] = record
        self._lease_records[record.lease_id] = record
        return self._store_lease_response(record, self._lease_response(record))

    def _handle_lease_renew(self, body: object) -> PlaneResponse:
        parsed = parse_model(LeaseRenewRequest, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        renewal_pairs = self._renewal_declared_pairs(parsed)
        for _, fallback_model in renewal_pairs:
            _validate_magic_model(fallback_model)

        self.lease_renewals.append(parsed)
        record = self._find_lease(parsed.lease_id, parsed.holder_id)
        if (
            record is None
            or record.lease_id in self._expired_leases
            or record.lease_id in self._released_leases
        ):
            return self._lease_error(404, "lease_not_found", "Budget lease not found")
        if record.last_renewal_request == parsed:
            return self._replay_lease_response(record)
        if record.terminal_successor:
            return self._lease_error(
                409,
                "lease_generation_conflict",
                "Budget lease generation conflict",
            )
        if parsed.generation != record.generation:
            return self._lease_error(
                409,
                "lease_generation_conflict",
                "Budget lease generation conflict",
            )

        effective_model = renewal_pairs[0][1] if renewal_pairs else record.declared_pairs[0][1]
        denied_by_period = self._denial_period(record.agent_run_id, effective_model)
        if denied_by_period is not None:
            response = self._lease_verdict_response(
                eligible=True,
                allowed=False,
                denied_by_period=denied_by_period,
                mode=(
                    BudgetMode.ALERT_ONLY
                    if effective_model == "solwyn-test/deny-alert"
                    else self.mode
                ),
            )
            record.generation += 1
            return self._store_lease_response(
                record,
                response,
                renewal_request=parsed,
                terminal_successor=True,
            )
        if not self.lease_eligible or effective_model in {
            "solwyn-test/lease-ineligible",
            _LIVE_CONTRACT_UNPRICED_LEASE_MODEL,
        }:
            response = self._lease_verdict_response(
                eligible=False,
                allowed=True,
                ineligible_reason="zero_rate_model",
            )
            record.generation += 1
            return self._store_lease_response(
                record,
                response,
                renewal_request=parsed,
                terminal_successor=True,
            )

        record.declared_pairs = self._union_declared_pairs(
            record.declared_pairs,
            renewal_pairs,
        )
        record.generation += 1
        return self._store_lease_response(
            record,
            self._lease_response(record),
            renewal_request=parsed,
        )

    def _handle_lease_surrender(self, body: object) -> PlaneResponse:
        parsed = parse_model(LeaseSurrenderRequest, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        self.lease_surrenders.append(parsed)
        record = self._find_lease(parsed.lease_id, parsed.holder_id)
        if record is None:
            return self._lease_error(404, "lease_not_found", "Budget lease not found")
        if parsed.generation != record.generation:
            return self._lease_error(
                409,
                "lease_generation_conflict",
                "Budget lease generation conflict",
            )
        if record.lease_id in self._expired_leases or record.lease_id in self._released_leases:
            return PlaneResponse(200, {"released_tokens": 0})
        self._released_leases.add(record.lease_id)
        return PlaneResponse(200, {"released_tokens": record.granted_tokens})

    def _find_lease(self, lease_id: str, holder_id: str) -> _LeaseRecord | None:
        record = self._lease_records.get(lease_id)
        if record is None or record.holder_id != holder_id:
            return None
        return record

    def _lease_is_active(self, record: _LeaseRecord) -> bool:
        return (
            record.lease_id not in self._expired_leases
            and record.lease_id not in self._released_leases
        )

    @staticmethod
    def _grant_declared_pairs(
        request: LeaseGrantRequest,
    ) -> tuple[tuple[ProviderName, str], ...]:
        return (
            (request.provider, request.model),
            *zip(request.fallback_providers, request.fallback_models, strict=True),
        )

    @staticmethod
    def _renewal_declared_pairs(
        request: LeaseRenewRequest,
    ) -> tuple[tuple[ProviderName, str], ...]:
        if request.model is None or request.provider is None:
            return ()
        return (
            (request.provider, request.model),
            *zip(request.fallback_providers, request.fallback_models, strict=True),
        )

    @staticmethod
    def _union_declared_pairs(
        current: tuple[tuple[ProviderName, str], ...],
        additions: tuple[tuple[ProviderName, str], ...],
    ) -> tuple[tuple[ProviderName, str], ...]:
        widened = list(current)
        for pair in additions:
            if pair not in widened:
                widened.append(pair)
        return tuple(widened)

    @staticmethod
    def _lease_error(status: int, code: str, message: str) -> PlaneResponse:
        return PlaneResponse(
            status,
            {"detail": {"code": code, "message": message}},
        )

    def _lease_response(self, record: _LeaseRecord) -> LeaseGrantResponse:
        return LeaseGrantResponse(
            eligible=True,
            allowed=True,
            lease_id=record.lease_id,
            generation=record.generation,
            granted_tokens=record.granted_tokens,
            refresh_interval_s=self.refresh_interval_s,
            lease_length_s=self.lease_length_s,
            headroom_share_tokens=record.granted_tokens,
            posture=LeasePosture(
                mode=self.mode,
                on_unreachable=("fail_open" if record.fail_open else "local_enforce"),
            ),
            final_grant=False,
            project_id=self.project_id,
            mode=self.mode,
            budget_limit=self.budget_limit,
            current_usage=self.current_usage,
            remaining_budget=self.remaining_budget,
        )

    def _store_lease_response(
        self,
        record: _LeaseRecord,
        response: LeaseGrantResponse,
        *,
        renewal_request: LeaseRenewRequest | None = None,
        terminal_successor: bool = False,
    ) -> PlaneResponse:
        record.last_response_json = response.model_dump_json(exclude_none=True)
        record.last_renewal_request = (
            renewal_request.model_copy(deep=True) if renewal_request is not None else None
        )
        record.terminal_successor = terminal_successor
        return PlaneResponse(200, response, model_only=True)

    @staticmethod
    def _replay_lease_response(record: _LeaseRecord) -> PlaneResponse:
        if not record.last_response_json:
            raise RuntimeError("solwyn.testing lease record has no frozen response")
        response = LeaseGrantResponse.model_validate_json(record.last_response_json)
        return PlaneResponse(200, response, model_only=True)

    def _lease_verdict_response(
        self,
        *,
        eligible: bool,
        allowed: bool,
        ineligible_reason: str | None = None,
        denied_by_period: str | None = None,
        mode: BudgetMode | None = None,
    ) -> LeaseGrantResponse:
        return LeaseGrantResponse(
            eligible=eligible,
            ineligible_reason=ineligible_reason,
            allowed=allowed,
            denied_by_period=denied_by_period,
            project_id=self.project_id,
            mode=self.mode if mode is None else mode,
            budget_limit=self.budget_limit,
            current_usage=self.current_usage,
            remaining_budget=self.remaining_budget,
        )

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
            if (
                window.kind == "refuse_leases"
                and path in _LEASE_PATHS
                and window.consume_if_matching(path)
            ):
                messages = {
                    "lease_unavailable": ("Budget lease service temporarily unavailable; retry"),
                    "lease_holder_cap_exceeded": "Active lease holder limit exceeded",
                }
                return self._lease_error(window.status, window.code, messages[window.code])
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
        denied_by_period = self._denial_period(parsed.agent_run_id, parsed.model)
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

    def _denial_period(self, agent_run_id: str | None, model: str) -> str | None:
        if self._pending_denials:
            return self._pending_denials.popleft()
        if agent_run_id in self._denied_runs:
            return "agent_run"
        magic_periods = {
            "solwyn-test/deny": "monthly",
            "solwyn-test/deny-alert": "monthly",
            "solwyn-test/deny-tag": "tag",
            "solwyn-test/deny-stopped": "run_stopped",
        }
        magic_period = magic_periods.get(model)
        if magic_period is not None:
            return magic_period
        if model == "solwyn-test/runaway":
            run_id = agent_run_id
            if run_id in self._runaway_seen:
                return "agent_run"
            self._runaway_seen.add(run_id)
        return None
