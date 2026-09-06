"""Control-plane HTTP construction honors an injected transport everywhere."""

from __future__ import annotations

import ast
import gc
import inspect
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, get_args, get_type_hints

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID, call_uuid

from solwyn import run
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, ProviderName
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter
from solwyn.testing import FakeControlPlane

_API_URL = "http://control-plane.test"


class _Recorder:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/v1/budgets/check":
            return httpx.Response(200, json=ALLOW_BUDGET_RESPONSE)
        if request.url.path == "/api/v1/budgets/lease":
            return httpx.Response(
                200,
                json={
                    "eligible": True,
                    "allowed": True,
                    "lease_id": "lease_transport_seam",
                    "generation": 1,
                    "granted_tokens": 100_000,
                    "refresh_interval_s": 300.0,
                    "lease_length_s": 600.0,
                    "headroom_share_tokens": 50_000,
                    "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
                    "final_grant": False,
                    "project_id": VALID_PROJECT_ID,
                    "mode": "alert_only",
                    "budget_limit": 100.0,
                    "current_usage": 20.0,
                    "remaining_budget": 80.0,
                },
            )
        if request.url.path == "/api/v1/budgets/confirm":
            return httpx.Response(204)
        return httpx.Response(202, json={"ingested": 1, "rejected": []})


class _SyncCompletionsStub:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class _SyncChatStub:
    def __init__(self) -> None:
        self.completions = _SyncCompletionsStub()


class _SyncOpenAIClientStub:
    def __init__(self) -> None:
        self.chat = _SyncChatStub()

    def with_options(self, **_kwargs: object) -> _SyncOpenAIClientStub:
        return self


_SyncOpenAIClientStub.__module__ = "openai._client"
_SyncOpenAIClientStub.__name__ = "OpenAI"


class _AsyncCompletionsStub:
    async def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class _AsyncChatStub:
    def __init__(self) -> None:
        self.completions = _AsyncCompletionsStub()


class _AsyncOpenAIClientStub:
    def __init__(self) -> None:
        self.chat = _AsyncChatStub()

    def with_options(self, **_kwargs: object) -> _AsyncOpenAIClientStub:
        return self


_AsyncOpenAIClientStub.__module__ = "openai._client"
_AsyncOpenAIClientStub.__name__ = "AsyncOpenAI"


class _SyncOnlyTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=ALLOW_BUDGET_RESPONSE)

    async def aclose(self) -> None:
        return None


class _AsyncOnlyTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=ALLOW_BUDGET_RESPONSE)

    def close(self) -> None:
        return None


class _CoroutineHandleRequestTransport:
    """A transport whose ``handle_request`` is itself a coroutine function."""

    async def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=ALLOW_BUDGET_RESPONSE)


class _DualWithoutSyncHandler(httpx.BaseTransport, httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=ALLOW_BUDGET_RESPONSE)


class _DualWithoutAsyncHandler(httpx.BaseTransport, httpx.AsyncBaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=ALLOW_BUDGET_RESPONSE)


class _StructuralDualTransport:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._recorder.handler(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._recorder.handler(request)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _SyncAsyncHandlerTransport(_StructuralDualTransport):
    """``handle_async_request`` has the WRONG (sync) shape — a real rejection:
    the SDK calls it with ``await`` and a non-coroutine breaks that call."""

    def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        return self._recorder.handler(request)


class _SyncAsyncCloseTransport(_StructuralDualTransport):
    """``aclose`` has the wrong (sync) shape — irrelevant: the SDK never calls
    close/aclose on a caller-owned transport, so this must be ACCEPTED."""

    def aclose(self) -> None:  # type: ignore[override]
        return None


class _AsyncSyncHandlerTransport(_StructuralDualTransport):
    """``handle_request`` has the WRONG (async) shape — a real rejection: the
    SDK calls it synchronously and a coroutine function breaks that call."""

    async def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        return self._recorder.handler(request)


class _AsyncSyncCloseTransport(_StructuralDualTransport):
    """``close`` has the wrong (async) shape — irrelevant: the SDK never calls
    close/aclose on a caller-owned transport, so this must be ACCEPTED."""

    async def close(self) -> None:  # type: ignore[override]
        return None


class _MinimalDualTransport:
    """The two methods the SDK actually calls — no close, no aclose at all.

    Functionally sufficient: the non-closing wrappers never forward close or
    aclose to a caller-owned transport, so a transport that never implements
    them must still be accepted."""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._recorder.handler(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._recorder.handler(request)


class _CloseSensitiveDualTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.closed = False
        self.close_calls = 0
        self.aclose_calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.closed:
            raise RuntimeError("transport already closed")
        return self._recorder.handler(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.closed:
            raise RuntimeError("transport already closed")
        return self._recorder.handler(request)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed = True


class _AsyncClosableComponent(Protocol):
    async def close(self) -> None: ...


class _SyncClosableComponent(Protocol):
    def close(self) -> None: ...


def _new_async_enforcer_with_transport(transport: object) -> AsyncBudgetEnforcer:
    return AsyncBudgetEnforcer(_API_URL, VALID_API_KEY, transport=transport)  # type: ignore[arg-type]


def _new_async_reporter_with_transport(transport: object) -> AsyncMetadataReporter:
    return AsyncMetadataReporter(_API_URL, VALID_API_KEY, transport=transport)  # type: ignore[arg-type]


def _new_async_solwyn_with_transport(transport: object) -> AsyncSolwyn:
    return AsyncSolwyn(
        _AsyncOpenAIClientStub(),
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,  # type: ignore[arg-type]
    )


def _new_sync_enforcer_with_transport(transport: object) -> BudgetEnforcer:
    return BudgetEnforcer(_API_URL, VALID_API_KEY, transport=transport)  # type: ignore[arg-type]


def _new_sync_reporter_with_transport(transport: object) -> MetadataReporter:
    return MetadataReporter(_API_URL, VALID_API_KEY, transport=transport)  # type: ignore[arg-type]


def _new_sync_solwyn_with_transport(transport: object) -> Solwyn:
    return Solwyn(
        _SyncOpenAIClientStub(),
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,  # type: ignore[arg-type]
    )


def _check(enforcer: BudgetEnforcer) -> None:
    result = enforcer.check_budget(
        estimated_input_tokens=10,
        model="gpt-5.5",
        provider="openai",
    )
    assert result.allowed is True


async def _acheck(enforcer: AsyncBudgetEnforcer) -> None:
    result = await enforcer.check_budget(
        estimated_input_tokens=10,
        model="gpt-5.5",
        provider="openai",
    )
    assert result.allowed is True


@pytest.mark.unit
def test_control_plane_component_transport_is_keyword_only() -> None:
    for component in (
        BudgetEnforcer,
        AsyncBudgetEnforcer,
        MetadataReporter,
        AsyncMetadataReporter,
    ):
        parameter = inspect.signature(component).parameters["transport"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    for component in (BudgetEnforcer, AsyncBudgetEnforcer):
        parameter = inspect.signature(component).parameters["control_plane_breaker"]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.unit
def test_async_control_plane_parameters_require_one_combined_transport_protocol() -> None:
    parameters = (
        (AsyncBudgetEnforcer.__init__, "transport"),
        (AsyncMetadataReporter.__init__, "transport"),
        (AsyncSolwyn.__init__, "control_plane_transport"),
    )

    for constructor, parameter_name in parameters:
        annotation = get_type_hints(constructor)[parameter_name]
        members = set(get_args(annotation)) - {type(None)}
        assert len(members) == 1
        protocol = members.pop()
        assert getattr(protocol, "_is_protocol", False) is True


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_async_enforcer_with_transport,
        _new_async_reporter_with_transport,
        _new_async_solwyn_with_transport,
    ],
)
@pytest.mark.parametrize(
    "transport_factory",
    [
        _SyncOnlyTransport,
        _AsyncOnlyTransport,
        _DualWithoutSyncHandler,
        _DualWithoutAsyncHandler,
    ],
)
async def test_async_components_reject_one_sided_transports(
    component_factory: Callable[[object], _AsyncClosableComponent],
    transport_factory: Callable[[], object],
) -> None:
    component: _AsyncClosableComponent | None = None
    try:
        with pytest.raises(TypeError, match="both sync and async"):
            component = component_factory(transport_factory())
    finally:
        if component is not None:
            with suppress(Exception):
                await component.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_async_enforcer_with_transport,
        _new_async_reporter_with_transport,
        _new_async_solwyn_with_transport,
    ],
)
@pytest.mark.parametrize(
    "transport_factory",
    [
        _SyncAsyncHandlerTransport,
        _AsyncSyncHandlerTransport,
    ],
)
async def test_async_components_reject_wrong_sync_async_method_shapes_at_construction(
    component_factory: Callable[[object], _AsyncClosableComponent],
    transport_factory: Callable[[_Recorder], object],
) -> None:
    recorder = _Recorder()

    with pytest.raises(TypeError, match="sync and async method shapes"):
        component_factory(transport_factory(recorder))

    assert recorder.paths == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_async_enforcer_with_transport,
        _new_async_reporter_with_transport,
        _new_async_solwyn_with_transport,
    ],
)
@pytest.mark.parametrize(
    "transport_factory",
    [
        _MinimalDualTransport,
        _SyncAsyncCloseTransport,
        _AsyncSyncCloseTransport,
    ],
)
async def test_async_components_accept_transports_regardless_of_close_aclose_shape(
    component_factory: Callable[[object], _AsyncClosableComponent],
    transport_factory: Callable[[_Recorder], object],
) -> None:
    """A dual transport is accepted based on ``handle_request``/
    ``handle_async_request`` alone. A missing, wrongly-shaped, or altogether
    absent ``close``/``aclose`` must NOT be rejected: the non-closing wrappers
    never forward either to a caller-owned transport, so requiring them would
    reject a transport the SDK can already serve every request through."""
    recorder = _Recorder()

    component = component_factory(transport_factory(recorder))

    await component.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_sync_enforcer_with_transport,
        _new_sync_reporter_with_transport,
        _new_sync_solwyn_with_transport,
    ],
)
@pytest.mark.parametrize(
    "transport_factory",
    [
        _AsyncOnlyTransport,
        object,
        _CoroutineHandleRequestTransport,
        httpx.BaseTransport,
    ],
)
def test_sync_components_reject_one_sided_transports(
    component_factory: Callable[[object], _SyncClosableComponent],
    transport_factory: Callable[[], object],
) -> None:
    with pytest.raises(TypeError, match="httpx sync transport interface"):
        component_factory(transport_factory())


@pytest.mark.unit
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_sync_enforcer_with_transport,
        _new_sync_reporter_with_transport,
        _new_sync_solwyn_with_transport,
    ],
)
def test_sync_components_reject_fake_control_plane_instance(
    component_factory: Callable[[object], _SyncClosableComponent],
) -> None:
    # The most likely real-world mistake: passing the FakeControlPlane
    # instance itself instead of its `.transport` attribute.
    plane = FakeControlPlane()

    with pytest.raises(TypeError, match="httpx sync transport interface"):
        component_factory(plane)


@pytest.mark.unit
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_sync_enforcer_with_transport,
        _new_sync_reporter_with_transport,
        _new_sync_solwyn_with_transport,
    ],
)
def test_sync_components_accept_mock_transport(
    component_factory: Callable[[object], _SyncClosableComponent],
) -> None:
    recorder = _Recorder()
    component = component_factory(httpx.MockTransport(recorder.handler))
    component.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "component_factory",
    [
        _new_sync_enforcer_with_transport,
        _new_sync_reporter_with_transport,
        _new_sync_solwyn_with_transport,
    ],
)
def test_sync_components_accept_fake_control_plane_transport(
    component_factory: Callable[[object], _SyncClosableComponent],
) -> None:
    plane = FakeControlPlane()

    component = component_factory(plane.transport)
    component.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_accepts_a_structural_dual_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=_StructuralDualTransport(recorder),
    )

    try:
        await _acheck(enforcer)
    finally:
        await enforcer.close()

    assert recorder.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
def test_sync_enforcer_constructor_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    try:
        _check(enforcer)
    finally:
        enforcer.close()

    assert recorder.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
def test_sync_enforcer_fork_reset_reuses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    pre_reset_http = enforcer._http
    enforcer._reset_after_fork_in_child()
    try:
        assert enforcer._http is not pre_reset_http
        _check(enforcer)
    finally:
        enforcer.close()

    assert recorder.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
def test_sync_reporter_exit_client_uses_injected_transport() -> None:
    recorder = _Recorder()
    reporter = MetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    client = reporter._new_exit_http_client()
    try:
        response = client.post(f"{_API_URL}/api/v1/budgets/confirm")
    finally:
        client.close()
        reporter.close()

    assert response.status_code == 204
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_constructor_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    try:
        await _acheck(enforcer)
    finally:
        await enforcer.close()

    assert recorder.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_fork_reset_reuses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    pre_reset_http = enforcer._http
    enforcer._reset_after_fork_in_child()
    try:
        assert enforcer._http is not pre_reset_http
        await _acheck(enforcer)
    finally:
        await enforcer.close()

    assert recorder.paths == ["/api/v1/budgets/check"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_reporter_exit_client_uses_injected_transport() -> None:
    recorder = _Recorder()
    reporter = AsyncMetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=httpx.MockTransport(recorder.handler),
    )

    client = reporter._new_exit_http_client()
    try:
        response = client.post(f"{_API_URL}/api/v1/budgets/confirm")
    finally:
        client.close()
        await reporter.close()

    assert response.status_code == 204
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1


@pytest.mark.unit
def test_async_reporter_gc_exit_flush_retains_transport_without_retaining_reporter() -> None:
    recorder = _Recorder()
    transport = _CloseSensitiveDualTransport(recorder)
    reporter = AsyncMetadataReporter(
        _API_URL,
        VALID_API_KEY,
        transport=transport,
    )
    reporter.report_confirm(
        BudgetConfirmRequest(
            reservation_id="res_transport_seam",
            model="gpt-5.5",
            provider=ProviderName.OPENAI,
            call_id=call_uuid("transport-seam-finalizer"),
            token_details=TokenDetails(input_tokens=10, output_tokens=5),
        )
    )

    del reporter
    gc.collect()

    assert recorder.paths.count("/api/v1/budgets/confirm") == 1
    assert transport.close_calls == 0
    assert transport.aclose_calls == 0
    assert transport.closed is False

    response = transport.handle_request(httpx.Request("POST", f"{_API_URL}/api/v1/budgets/check"))
    assert response.status_code == 200

    transport.close()
    assert transport.close_calls == 1
    assert transport.closed is True


@pytest.mark.unit
def test_sync_enforcer_surrender_drain_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = BudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        holder_id="holder_transport_seam",
        transport=httpx.MockTransport(recorder.handler),
    )

    result = enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=10,
        model="gpt-5.5",
        provider="openai",
        agent_run_id="run_transport_seam",
        call_id=call_uuid("transport-seam-surrender"),
    )
    assert result.lease_id == "lease_transport_seam"

    enforcer.close()

    assert "/api/v1/budgets/lease/surrender" in recorder.paths


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_enforcer_surrender_drain_uses_injected_transport() -> None:
    recorder = _Recorder()
    enforcer = AsyncBudgetEnforcer(
        _API_URL,
        VALID_API_KEY,
        holder_id="holder_transport_seam",
        transport=httpx.MockTransport(recorder.handler),
    )

    result = await enforcer.check_budget(
        estimated_input_tokens=10,
        estimated_output_bound=10,
        model="gpt-5.5",
        provider="openai",
        agent_run_id="run_transport_seam",
        call_id=call_uuid("async-transport-seam-surrender"),
    )
    assert result.lease_id == "lease_transport_seam"

    await enforcer.close()

    assert "/api/v1/budgets/lease/surrender" in recorder.paths


@pytest.mark.unit
def test_solwyn_threads_transport_to_budget_and_reporter() -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(recorder.handler)
    solwyn = Solwyn(
        _SyncOpenAIClientStub(),
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,
        reporter_flush_interval=3600.0,
    )

    solwyn.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
    )
    solwyn.close()

    assert recorder.paths.count("/api/v1/budgets/check") == 1
    assert recorder.paths.count("/api/v1/budgets/confirm") == 1
    assert recorder.paths.count("/api/v1/metadata/ingest") == 1


@pytest.mark.unit
def test_sync_injected_transport_is_caller_owned_through_settlement_and_surrender() -> None:
    recorder = _Recorder()
    transport = _CloseSensitiveDualTransport(recorder)
    solwyn = Solwyn(
        _SyncOpenAIClientStub(),
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,
        reporter_flush_interval=3600.0,
    )

    try:
        with run("sync-transport-ownership"):
            solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )
        solwyn.close()

        # The run scope releases its lease as it exits (S2), so the surrender
        # precedes the confirm this deliberately-slow reporter still holds.
        # Ordering is not an accounting risk: a confirm landing after the
        # release settles as excess against a zero baseline, so the durable
        # total is the same either way. What this test owns is that BOTH still
        # ride the caller's injected transport.
        confirm_index = recorder.paths.index("/api/v1/budgets/confirm")
        surrender_index = recorder.paths.index("/api/v1/budgets/lease/surrender")
        assert surrender_index < confirm_index
        assert transport.close_calls == 0
        assert transport.aclose_calls == 0
        assert transport.closed is False
        response = transport.handle_request(
            httpx.Request("POST", f"{_API_URL}/api/v1/budgets/check")
        )
        assert response.status_code == 200
    finally:
        if not transport.closed:
            transport.close()

    assert transport.close_calls == 1
    assert transport.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_injected_transport_is_caller_owned_through_settlement_and_surrender() -> None:
    recorder = _Recorder()
    transport = _CloseSensitiveDualTransport(recorder)
    solwyn = AsyncSolwyn(
        _AsyncOpenAIClientStub(),
        api_key=VALID_API_KEY,
        api_url=_API_URL,
        control_plane_transport=transport,
        reporter_flush_interval=3600.0,
    )

    try:
        async with run("async-transport-ownership"):
            await solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )
        await solwyn.close()

        # The run scope releases its lease as it exits (S2), so the surrender
        # precedes the confirm this deliberately-slow reporter still holds.
        # Ordering is not an accounting risk: a confirm landing after the
        # release settles as excess against a zero baseline, so the durable
        # total is the same either way. What this test owns is that BOTH still
        # ride the caller's injected transport.
        confirm_index = recorder.paths.index("/api/v1/budgets/confirm")
        surrender_index = recorder.paths.index("/api/v1/budgets/lease/surrender")
        assert surrender_index < confirm_index
        assert transport.close_calls == 0
        assert transport.aclose_calls == 0
        assert transport.closed is False
        response = await transport.handle_async_request(
            httpx.Request("POST", f"{_API_URL}/api/v1/budgets/check")
        )
        assert response.status_code == 200
    finally:
        if not transport.closed:
            await transport.aclose()

    assert transport.aclose_calls == 1
    assert transport.closed is True


@pytest.mark.unit
def test_http_clients_are_only_constructed_in_component_factories() -> None:
    """Direct ``httpx.Client``/``AsyncClient`` construction is confined to the
    three component factories. Also flags two evasions of the plain
    ``httpx.Client(...)`` attribute-call pattern the walk below looks for:
    ``from httpx import Client``/``AsyncClient`` (an unqualified name a caller
    could construct without ever writing ``httpx.Client``, so it is flagged
    outright rather than allowlisted) and ``import httpx as <alias>`` followed
    by ``<alias>.Client(...)``/``<alias>.AsyncClient(...)`` (checked through
    the SAME allowlist as the direct form, since it's the same call shape
    under a different name).
    """
    source_root = Path(__file__).parents[2] / "src" / "solwyn"
    hits: list[str] = []
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        httpx_aliases = {"httpx"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "httpx" and alias.asname:
                        httpx_aliases.add(alias.asname)
            elif isinstance(node, ast.ImportFrom) and node.module == "httpx":
                for alias in node.names:
                    if alias.name in {"Client", "AsyncClient"}:
                        violations.append(
                            f"{path.relative_to(source_root)}:{node.lineno}:"
                            f"<import {alias.name} from httpx>"
                        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in httpx_aliases:
                continue
            if node.func.attr not in {"Client", "AsyncClient"}:
                continue
            owner: ast.AST | None = node
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            owner_name = (
                owner.name if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
            )
            location = f"{path.relative_to(source_root)}:{node.lineno}:{owner_name}"
            hits.append(location)
            if path.name not in {"budget.py", "reporter.py"} or owner_name not in {
                "_new_http_client",
                "_new_async_http_client",
                "_new_exit_http_client",
            }:
                violations.append(location)

    assert not violations, f"direct httpx client construction bypasses factories: {violations}"
    assert len(hits) == 6, f"expected six component-factory construction sites, got {hits}"
