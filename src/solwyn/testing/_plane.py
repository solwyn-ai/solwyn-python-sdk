"""Sans-I/O state machine for the in-process Solwyn control-plane double."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any

from pydantic_core import PydanticUndefined

from solwyn._run import current_run
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
from solwyn.config import SolwynConfig
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
_RUN_SCOPED_PERIODS = frozenset({"agent_run", "run_stopped"})
_DENIAL_SCOPES = frozenset({"check", "lease"})
_RUN_SCOPED_MAGIC_PERIODS = {
    "solwyn-test/deny-stopped": "run_stopped",
    "solwyn-test/runaway": "agent_run",
}
_SCOPED_RULES_INELIGIBLE_REASON = "scoped_rules_present"
_LIVE_CONTRACT_UNPRICED_LEASE_MODEL = "no-such-model-for-leases"
_LEASE_PATHS = frozenset(
    {
        "/api/v1/budgets/lease",
        "/api/v1/budgets/lease/renew",
        "/api/v1/budgets/lease/surrender",
    }
)
_RESERVED_WRAP_KWARGS = ("api_key", "api_url", "control_plane_transport")
_ENV_PREFIX = "SOLWYN_"
# api_key/api_url are pinned by the wrapper itself; providers/default_params are
# derived by the client constructor from its own arguments.
_WRAPPER_OWNED_CONFIG_FIELDS = frozenset({"api_key", "api_url", "providers", "default_params"})
_GRANT_ONLY_LEASE_PATH = "/api/v1/budgets/lease"


def _validate_magic_model(model: object) -> None:
    if isinstance(model, str) and model.startswith("solwyn-test/") and model not in MAGIC_MODELS:
        raise RuntimeError(f"unknown solwyn testing magic model: {model!r}")


def _run_scope_error(period: str) -> RuntimeError:
    return RuntimeError(
        f"solwyn.testing: run-scoped denial {period!r} requires an agent_run_id on the request"
    )


def _require_run_scope(period: str, agent_run_id: str | None) -> None:
    if agent_run_id is None:
        raise _run_scope_error(period)


def _validate_run_scoped_model(model: object) -> None:
    """Reject a run-scoped magic model used outside a ``solwyn.run(...)`` scope."""
    if not isinstance(model, str):
        return
    period = _RUN_SCOPED_MAGIC_PERIODS.get(model)
    if period is None:
        return
    agent_run_id, _agent_run_name = current_run()
    _require_run_scope(period, agent_run_id)


def _validate_testing_model(model: object) -> None:
    """Validate one caller-supplied model against the testing magic-model rules."""
    _validate_magic_model(model)
    _validate_run_scoped_model(model)


def _hermetic_config_options() -> dict[str, Any]:
    """Pin every ambient ``SOLWYN_*`` setting back to its declared default.

    ``SolwynConfig`` fills each unset field from ``SOLWYN_<FIELD>``, so an
    exported variable would silently reconfigure a double-backed test — the
    exact process-global input this tool exists to exclude. Passing the declared
    default explicitly keeps the environment loader out of the wrapped client;
    the caller's own keyword arguments are applied afterwards and always win.

    A field whose default is required, or is ``None`` (the loader reads the
    environment for a ``None`` value too), cannot be pinned this way and is left
    alone.
    """
    options: dict[str, Any] = {}
    for name, field in SolwynConfig.model_fields.items():
        if name in _WRAPPER_OWNED_CONFIG_FIELDS:
            continue
        if f"{_ENV_PREFIX}{name.upper()}" not in os.environ:
            continue
        default = field.get_default(call_default_factory=True)
        if default is PydanticUndefined or default is None:
            continue
        options[name] = default
    return options


def _reject_reserved_wrap_kwargs(kwargs: dict[str, Any]) -> None:
    supplied = [key for key in _RESERVED_WRAP_KWARGS if key in kwargs]
    if supplied:
        joined = ", ".join(supplied)
        raise TypeError(f"reserved control-plane wiring cannot be overridden: {joined}")


def _metadata_legacy_identity(event: MetadataEvent) -> tuple[datetime, str]:
    timestamp = event.timestamp
    normalized = (
        timestamp.replace(tzinfo=UTC)
        if timestamp.utcoffset() is None
        else timestamp.astimezone(UTC)
    )
    return normalized, event.sdk_instance_id


class _TestingSolwyn(Solwyn):
    def _intercepted_call(
        self,
        *,
        _force_stream: bool = False,
        _surface: str = "chat",
        _responses_leaf: str = "create",
        **kwargs: object,
    ) -> Any:
        _validate_testing_model(kwargs.get("model"))
        for runtime in self._runtimes[1:]:
            _validate_testing_model(runtime.entry.model)
        return super()._intercepted_call(
            _force_stream=_force_stream,
            _surface=_surface,
            _responses_leaf=_responses_leaf,
            **kwargs,
        )

    def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        _validate_testing_model(kwargs.get("model"))
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
        _validate_testing_model(kwargs.get("model"))
        for runtime in self._runtimes[1:]:
            _validate_testing_model(runtime.entry.model)
        return await super()._intercepted_call(
            _force_stream=_force_stream,
            _surface=_surface,
            _responses_leaf=_responses_leaf,
            **kwargs,
        )

    async def _media_call(self, spec: MediaSurfaceSpec, **kwargs: object) -> Any:
        _validate_testing_model(kwargs.get("model"))
        return await super()._media_call(spec, **kwargs)


@dataclass(slots=True, eq=False)
class _ScenarioWindow:
    """One scripted transport or endpoint window.

    Identity-compared (``eq=False``) so a scenario's exit-time removal can never
    drop a different window that happens to hold the same counter values.
    """

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
    last_renewal_from_generation: int | None = None


class FakeControlPlane:
    """In-process Solwyn control-plane double. Deterministic, zero-network.

    Simulates wire behavior only: denials are scripted, never priced —
    the real API owns all pricing. Thread-safe because reporter flush threads
    and lease renewal workers may hit it concurrently.

    Plane state is per-process: a forked child works on a copy, and the parent's
    recordings never see the child's traffic. Stateful use across a fork is
    unsupported.

    Lease population knobs mirror real server populations: ``granted_tokens=0``
    is the zero grant an ``alert_only`` project past its cap receives, so
    admission falls straight back to the per-call check path;
    ``headroom_share_tokens`` sizes the holder's share of run headroom
    independently of the grant, which is what the outage ladder spends;
    ``final_grant=True`` marks every grant and renewal terminal, driving the
    ledger's wind-down (the SDK suppresses further renewals).

    Scenario errors raised inside the transport (an unknown magic model, a
    run-scoped verdict on a request carrying no run) are guaranteed loud only
    through :meth:`wrap`, :meth:`wrap_async`, or a raw ``httpx`` client. A
    directly constructed ``BudgetEnforcer`` converts transport exceptions into
    its fail-open outage posture instead.
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
        headroom_share_tokens: int | None = None,
        final_grant: bool = False,
        refresh_interval_s: float = 30.0,
        lease_length_s: float = 90.0,
    ) -> None:
        if granted_tokens < 0:
            raise ValueError("granted_tokens must not be negative")
        if headroom_share_tokens is not None and headroom_share_tokens < 0:
            raise ValueError("headroom_share_tokens must not be negative")
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
        self.headroom_share_tokens = headroom_share_tokens
        self.final_grant = final_grant
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
        self._pending_lease_denials: deque[str] = deque()
        self._denied_runs: set[str] = set()
        self._runaway_seen: set[str] = set()
        self._scenario_windows: list[_ScenarioWindow] = []
        self._active_reservations: set[str] = set()
        self._settled_reservations: set[str] = set()
        self._seen_call_ids: set[str] = set()
        self._seen_ingest_identities: set[tuple[str, int]] = set()
        self._seen_ingest_legacy_identities: set[tuple[datetime, str]] = set()
        self._last_untracked_report_ids: dict[tuple[str, str, str, str], str] = {}
        self._leases: dict[tuple[str, str], _LeaseRecord] = {}
        self._lease_records: dict[str, _LeaseRecord] = {}
        self._expired_leases: set[str] = set()
        self._released_leases: set[str] = set()
        self._reservation_counter = 0
        self._lease_counter = 0
        self._lock = Lock()

    def deny_next(self, n: int = 1, *, period: str = "monthly", scope: str = "check") -> None:
        """Script the next ``n`` check verdicts as denials for one budget period.

        The wrapped SDK raises in ``hard_deny`` mode and warns before provider
        dispatch in ``alert_only`` mode, matching the ``budget_mode`` knob.
        ``scope="lease"`` queues a separate script consumed by lease grants and
        renewals instead, so background lease traffic cannot eat a denial meant
        for a per-call check. A lease denial cannot use the ``tag`` period: a
        project with tag-scoped rules is lease-INELIGIBLE, so the live plane
        never emits one.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if period not in _DENIAL_PERIODS:
            raise ValueError("period must be monthly, agent_run, run_stopped, or tag")
        if scope not in _DENIAL_SCOPES:
            raise ValueError("scope must be check or lease")
        if scope == "lease" and period == "tag":
            raise ValueError("lease denials cannot use the tag period")
        with self._lock:
            queue = self._pending_lease_denials if scope == "lease" else self._pending_denials
            queue.extend(period for _ in range(n))

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
            self._pending_lease_denials.clear()
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

    def _wrap_options(self, solwyn_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the wrapper options: hermetic defaults, wiring, caller kwargs."""
        _reject_reserved_wrap_kwargs(solwyn_kwargs)
        options: dict[str, Any] = _hermetic_config_options()
        options.update(
            {
                "api_key": self.api_key,
                "api_url": self.api_url,
                "control_plane_transport": self.transport,
                "budget_check_cache_ttl": 0,
            }
        )
        options.update(solwyn_kwargs)
        return options

    def wrap(self, provider_client: object, **solwyn_kwargs: Any) -> Solwyn:
        """Return a synchronous ``Solwyn`` using this plane's zero-network transport.

        Ordinary ``Solwyn`` configuration keyword arguments override testing
        defaults. ``api_key``, ``api_url``, and ``control_plane_transport`` are
        reserved so the returned client cannot bypass this plane.

        Ambient ``SOLWYN_*`` configuration is neutralized: every environment-
        mapped setting with a declared default falls back to that default
        instead of the process environment, so an exported variable cannot flip
        what a double-backed test asserts. Configure the wrapper through
        explicit keyword arguments.
        """
        return _TestingSolwyn(provider_client, **self._wrap_options(solwyn_kwargs))

    def wrap_async(self, provider_client: object, **solwyn_kwargs: Any) -> AsyncSolwyn:
        """Return an ``AsyncSolwyn`` using this plane's zero-network transport.

        Ordinary ``AsyncSolwyn`` configuration keyword arguments override
        testing defaults. ``api_key``, ``api_url``, and
        ``control_plane_transport`` are reserved so the returned client cannot
        bypass this plane.

        Ambient ``SOLWYN_*`` configuration is neutralized: every environment-
        mapped setting with a declared default falls back to that default
        instead of the process environment, so an exported variable cannot flip
        what a double-backed test asserts. Configure the wrapper through
        explicit keyword arguments.
        """
        return _TestingAsyncSolwyn(provider_client, **self._wrap_options(solwyn_kwargs))

    @contextmanager
    def outage(self, *, requests: int | None = None, path: str | None = None) -> Iterator[None]:
        """Make matching control-plane requests fail at the transport boundary.

        This drives the SDK's ``fail_open`` posture and the
        ``control_plane_failure_threshold`` / ``control_plane_recovery_timeout``
        breaker knobs. ``requests`` limits how many matching calls fail, and
        ``path`` scopes the window to one endpoint (the default matches every
        endpoint, like a real outage). Pin ``path`` whenever a count-bounded
        window runs alongside a live reporter or lease renewals, so background
        traffic cannot consume the scripted budget.
        """
        window = _ScenarioWindow("outage", self._request_count(requests), path=path)
        with self._scenario(window):
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
    def read_only(self, *, requests: int | None = None, path: str | None = None) -> Iterator[None]:
        """Refuse writes with the real read-only-key response shape.

        This proves the SDK treats an authorization refusal as a reachable
        control plane rather than opening its outage circuit breaker. ``path``
        scopes the window to one endpoint (the default refuses every write, like
        a real read-only key). Pin ``path`` whenever a count-bounded window runs
        alongside a live reporter or lease renewals, so background traffic
        cannot consume the scripted budget.
        """
        window = _ScenarioWindow("read_only", self._request_count(requests), path=path)
        with self._scenario(window):
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
        than treating this reachable lease response as a transport outage. The
        503 ``lease_unavailable`` form refuses grants, renewals, and surrenders
        alike; the 409 ``lease_holder_cap_exceeded`` form refuses ONLY grants,
        because the live plane raises the holder cap inside grant alone —
        renewal and surrender can never emit that code. A request the window
        does not match consumes none of its scripted budget.
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
        """Consume matching delay/outage windows while the caller holds the lock.

        An outage wins outright and short-circuits: the scripted connection
        error never reaches the server, so it must neither wait out nor consume
        a server-latency window's scripted budget.
        """
        for window in self._scenario_windows:
            if window.kind == "outage" and window.consume_if_matching(path):
                return 0.0, True
        delay = 0.0
        for window in self._scenario_windows:
            if window.kind == "slow" and window.consume_if_matching(path):
                delay += window.seconds
        return delay, False

    def handle(self, method: str, path: str, body: object) -> PlaneResponse:
        """Evaluate one real control-plane wire request without network I/O.

        Most callers should use :meth:`wrap`; this lower-level seam supports
        contract tests that need the raw status, headers, and response model.
        """
        with self._lock:
            return self._handle_locked(method, path, body)

    def _handle_locked(self, method: str, path: str, body: object) -> PlaneResponse:
        """Resolve the route first so a refusal window never hides a 404."""
        handler = self._resolve_handler(method, path)
        if handler is None:
            self.unmatched_requests.append((method, path))
            return PlaneResponse(404, {"detail": "not found"})
        refusal = self._endpoint_refusal_locked(method, path, body)
        if refusal is not None:
            return refusal
        return handler(body)

    def _resolve_handler(
        self,
        method: str,
        path: str,
    ) -> Callable[[object], PlaneResponse] | None:
        if method == "GET" and path == "/health":
            return self._handle_health
        if method != "POST":
            return None
        if path == "/api/v1/budgets/check":
            return self._handle_check
        if path == "/api/v1/budgets/confirm":
            return self._handle_confirm
        if path == "/api/v1/metadata/ingest":
            return self._handle_ingest
        if path == "/api/v1/untracked-surfaces":
            return self._handle_untracked
        if path == "/api/v1/budgets/lease":
            return self._handle_lease_grant
        if path == "/api/v1/budgets/lease/renew":
            return self._handle_lease_renew
        if path == "/api/v1/budgets/lease/surrender":
            return self._handle_lease_surrender
        if path.startswith("/api/v1/projects/") and path.endswith("/providers/breaker-reports"):
            return self._handle_breaker_report
        return None

    @staticmethod
    def _handle_health(_body: object) -> PlaneResponse:
        return PlaneResponse(200, {"status": "ok"})

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
                return self._replay_lease_grant(current)
            return self._lease_error(
                409,
                "lease_holder_cap_exceeded",
                "Active lease holder limit exceeded",
            )

        verdict = self._lease_chain_verdict(
            parsed.agent_run_id,
            (parsed.model, *parsed.fallback_models),
        )
        if verdict is not None:
            return PlaneResponse(200, verdict, model_only=True)
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
        for _, renewal_model in renewal_pairs:
            _validate_magic_model(renewal_model)

        self.lease_renewals.append(parsed)
        record = self._find_lease(parsed.lease_id, parsed.holder_id)
        if (
            record is None
            or record.lease_id in self._expired_leases
            or record.lease_id in self._released_leases
        ):
            return self._lease_error(404, "lease_not_found", "Budget lease not found")
        if (
            record.last_renewal_from_generation == parsed.generation
            and record.generation == parsed.generation + 1
            and record.last_response_json
        ):
            return self._replay_lease_response(record)
        if parsed.generation != record.generation or (
            record.last_renewal_from_generation is not None
            and record.last_renewal_from_generation >= record.generation
        ):
            return self._lease_error(
                409,
                "lease_generation_conflict",
                "Budget lease generation conflict",
            )

        # The live plane evaluates a renewal over the UNION of the lease's
        # declared pairs and this request's re-declaration — never the
        # re-declared pairs alone — so a trigger already in the declared set
        # keeps denying after a narrower re-declaration.
        effective_pairs = self._union_declared_pairs(record.declared_pairs, renewal_pairs)
        verdict = self._lease_chain_verdict(
            record.agent_run_id,
            tuple(model for _, model in effective_pairs),
        )
        if verdict is not None:
            record.last_renewal_from_generation = parsed.generation
            record.generation += 1
            return self._store_lease_response(record, verdict)

        record.declared_pairs = self._union_declared_pairs(
            record.declared_pairs,
            renewal_pairs,
        )
        record.last_renewal_from_generation = parsed.generation
        record.generation += 1
        return self._store_lease_response(record, self._lease_response(record))

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
        return FakeControlPlane._normalize_declared_pairs(
            (
                (request.provider, request.model),
                *zip(request.fallback_providers, request.fallback_models, strict=True),
            )
        )

    @staticmethod
    def _renewal_declared_pairs(
        request: LeaseRenewRequest,
    ) -> tuple[tuple[ProviderName, str], ...]:
        if request.model is None or request.provider is None:
            return ()
        return FakeControlPlane._normalize_declared_pairs(
            (
                (request.provider, request.model),
                *zip(request.fallback_providers, request.fallback_models, strict=True),
            )
        )

    @staticmethod
    def _normalize_declared_pairs(
        pairs: tuple[tuple[ProviderName, str], ...],
    ) -> tuple[tuple[ProviderName, str], ...]:
        return tuple(dict.fromkeys(pairs))

    @staticmethod
    def _union_declared_pairs(
        current: tuple[tuple[ProviderName, str], ...],
        additions: tuple[tuple[ProviderName, str], ...],
    ) -> tuple[tuple[ProviderName, str], ...]:
        return FakeControlPlane._normalize_declared_pairs((*current, *additions))

    @staticmethod
    def _lease_error(status: int, code: str, message: str) -> PlaneResponse:
        return PlaneResponse(
            status,
            {"detail": {"code": code, "message": message}},
        )

    def _lease_response(self, record: _LeaseRecord) -> LeaseGrantResponse:
        """Build the grant/renew success body from the plane's populations."""
        return LeaseGrantResponse(
            eligible=True,
            allowed=True,
            lease_id=record.lease_id,
            generation=record.generation,
            granted_tokens=record.granted_tokens,
            refresh_interval_s=self.refresh_interval_s,
            lease_length_s=self.lease_length_s,
            headroom_share_tokens=(
                record.granted_tokens
                if self.headroom_share_tokens is None
                else self.headroom_share_tokens
            ),
            posture=LeasePosture(
                mode=self.mode,
                on_unreachable=("fail_open" if record.fail_open else "local_enforce"),
            ),
            final_grant=self.final_grant,
            project_id=self.project_id,
            mode=self.mode,
            budget_limit=self.budget_limit,
            current_usage=self.current_usage,
            remaining_budget=self.remaining_budget,
        )

    @staticmethod
    def _store_lease_response(
        record: _LeaseRecord,
        response: LeaseGrantResponse,
    ) -> PlaneResponse:
        record.last_response_json = response.model_dump_json(exclude_none=True)
        return PlaneResponse(200, response, model_only=True)

    @staticmethod
    def _frozen_lease_response(record: _LeaseRecord) -> LeaseGrantResponse:
        if not record.last_response_json:
            raise RuntimeError("solwyn.testing lease record has no frozen response")
        return LeaseGrantResponse.model_validate_json(record.last_response_json)

    @staticmethod
    def _replay_lease_response(record: _LeaseRecord) -> PlaneResponse:
        return PlaneResponse(
            200,
            FakeControlPlane._frozen_lease_response(record),
            model_only=True,
        )

    @staticmethod
    def _replay_lease_grant(record: _LeaseRecord) -> PlaneResponse:
        """Replay a held grant unless the stored successor carries no authority.

        A terminal deny or ineligibility response has no relative timers to
        extend, so a re-grant against it is a generation conflict rather than a
        replay.
        """
        response = FakeControlPlane._frozen_lease_response(record)
        if response.lease_length_s is None or response.refresh_interval_s is None:
            return FakeControlPlane._lease_error(
                409,
                "lease_generation_conflict",
                "Budget lease generation conflict",
            )
        return PlaneResponse(200, response, model_only=True)

    def _lease_chain_verdict(
        self,
        agent_run_id: str,
        models: tuple[str, ...],
    ) -> LeaseGrantResponse | None:
        """Return the terminal lease verdict for one chain, or None to proceed.

        Every lease-path denial is a hard deny, a stopped run reports no
        remaining budget, and a tag-scoped trigger surfaces as lease
        ineligibility because scoped rules keep the whole project off leases.
        """
        denied_by_period, _trigger_model, ineligible = self._evaluate_chain(
            agent_run_id,
            models,
            lease=True,
        )
        if denied_by_period == "tag":
            return self._lease_verdict_response(
                eligible=False,
                allowed=True,
                ineligible_reason=_SCOPED_RULES_INELIGIBLE_REASON,
            )
        if denied_by_period is not None:
            return self._lease_verdict_response(
                eligible=True,
                allowed=False,
                denied_by_period=denied_by_period,
                mode=BudgetMode.HARD_DENY,
                remaining_budget=self._denied_remaining_budget(denied_by_period),
            )
        if ineligible:
            return self._lease_verdict_response(
                eligible=False,
                allowed=True,
                ineligible_reason="zero_rate_model",
            )
        return None

    def _denied_remaining_budget(self, denied_by_period: str) -> float:
        """A stopped run reports zero; every other denial floors at zero."""
        if denied_by_period == "run_stopped":
            return 0.0
        return max(0.0, self.remaining_budget)

    def _lease_verdict_response(
        self,
        *,
        eligible: bool,
        allowed: bool,
        ineligible_reason: str | None = None,
        denied_by_period: str | None = None,
        mode: BudgetMode | None = None,
        remaining_budget: float | None = None,
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
            remaining_budget=(
                self.remaining_budget if remaining_budget is None else remaining_budget
            ),
        )

    @staticmethod
    def _refusable_lease_paths(code: str) -> frozenset[str]:
        """Scope a lease refusal to the paths the live plane can emit it on."""
        if code == "lease_holder_cap_exceeded":
            return frozenset({_GRANT_ONLY_LEASE_PATH})
        return _LEASE_PATHS

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
                and path in self._refusable_lease_paths(window.code)
                and window.consume_if_matching(path)
            ):
                refusal_messages = {
                    "lease_unavailable": ("Budget lease service temporarily unavailable; retry"),
                    "lease_holder_cap_exceeded": "Active lease holder limit exceeded",
                }
                return self._lease_error(
                    window.status,
                    window.code,
                    refusal_messages[window.code],
                )
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
        denied_by_period, trigger_model, _ = self._evaluate_chain(
            parsed.agent_run_id,
            (parsed.model, *parsed.fallback_models),
            lease=False,
        )
        response_mode = (
            BudgetMode.ALERT_ONLY if trigger_model == "solwyn-test/deny-alert" else self.mode
        )
        if denied_by_period == "run_stopped":
            response_mode = BudgetMode.HARD_DENY
        remaining_budget = (
            self.remaining_budget
            if denied_by_period is None
            else self._denied_remaining_budget(denied_by_period)
        )
        self._reservation_counter += 1
        reservation_id = f"res_fake_{self._reservation_counter:08d}"
        if denied_by_period is None:
            self._active_reservations.add(reservation_id)

        opted_in = parsed.failover_directive_version == "1"
        response = BudgetCheckResponse(
            allowed=denied_by_period is None,
            remaining_budget=remaining_budget,
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
        newly_ingested: list[MetadataEvent] = []
        for event in parsed:
            identity = (event.call_id, event.attempt_index)
            legacy_identity = _metadata_legacy_identity(event)
            if (
                identity in self._seen_ingest_identities
                or legacy_identity in self._seen_ingest_legacy_identities
            ):
                continue
            self._seen_ingest_identities.add(identity)
            self._seen_ingest_legacy_identities.add(legacy_identity)
            newly_ingested.append(event)
        self.ingested.extend(newly_ingested)
        return PlaneResponse(202, {"ingested": len(newly_ingested), "rejected": []})

    def _handle_untracked(self, body: object) -> PlaneResponse:
        if isinstance(body, list) and len(body) > 100:
            return PlaneResponse(
                400,
                {"detail": "untracked surface batches may contain at most 100 reports"},
            )
        parsed = parse_model_list(UntrackedSurfaceReport, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        for report in parsed:
            observation_key = (
                report.provider.value,
                report.client_shape,
                report.mode,
                report.surface,
            )
            if self._last_untracked_report_ids.get(observation_key) == report.report_id:
                continue
            self._last_untracked_report_ids[observation_key] = report.report_id
            self.untracked_reports.append(report)
        return PlaneResponse(202, {"accepted": len(parsed)})

    def _handle_breaker_report(self, body: object) -> PlaneResponse:
        parsed = parse_model(BreakerStateReport, body)
        if isinstance(parsed, PlaneResponse):
            return parsed
        self.breaker_reports.append(parsed.model_dump(mode="json"))
        return PlaneResponse(204)

    def _evaluate_chain(
        self,
        agent_run_id: str | None,
        models: tuple[str, ...],
        *,
        lease: bool,
    ) -> tuple[str | None, str | None, bool]:
        """Evaluate one request's ordered model chain exactly once.

        Programmatic request/run denials take precedence, and each scope keeps
        its own scripted queue. After that, the first applicable magic trigger
        wins; lease-only ineligibility triggers are transparent to legacy
        per-call checks. A run-scoped verdict on a request that carries no
        ``agent_run_id`` fails the scenario loudly instead of silently drifting
        from the live plane.
        """
        pending = self._pending_lease_denials if lease else self._pending_denials
        if pending:
            period = pending.popleft()
            if period in _RUN_SCOPED_PERIODS:
                _require_run_scope(period, agent_run_id)
            return period, None, False
        if agent_run_id in self._denied_runs:
            return "agent_run", None, False
        magic_periods = {
            "solwyn-test/deny": "monthly",
            "solwyn-test/deny-alert": "monthly",
            "solwyn-test/deny-tag": "tag",
            "solwyn-test/deny-stopped": "run_stopped",
        }
        for model in models:
            magic_period = magic_periods.get(model)
            if magic_period is not None:
                if magic_period in _RUN_SCOPED_PERIODS:
                    _require_run_scope(magic_period, agent_run_id)
                return magic_period, model, False
            if model == "solwyn-test/runaway":
                if agent_run_id is None:
                    raise _run_scope_error(_RUN_SCOPED_MAGIC_PERIODS[model])
                if agent_run_id in self._runaway_seen:
                    return "agent_run", model, False
                self._runaway_seen.add(agent_run_id)
                return None, model, False
            if lease and model in {
                "solwyn-test/lease-ineligible",
                _LIVE_CONTRACT_UNPRICED_LEASE_MODEL,
            }:
                return None, model, True
        return None, None, lease and not self.lease_eligible
