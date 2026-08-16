"""Offline smoke coverage against the real OpenAI Agents SDK."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, cast

import pytest
import respx

pytest.importorskip("agents", reason="install the frameworks dependency group")

from agents import (
    Agent,
    Model,
    ModelProvider,
    ModelRetrySettings,
    ModelSettings,
    MultiProvider,
    RunConfig,
    RunHooks,
    Runner,
    function_tool,
    retry_policies,
    set_default_openai_api,
    set_default_openai_client,
)
from openai import AsyncOpenAI

import solwyn
from solwyn import AsyncSolwyn

from . import (
    CONTROL_PLANE_URL,
    MODEL,
    SOLWYN_API_KEY,
    FrameworkSmokeHarness,
    make_offline_openai_client,
)


class RetrySafeAsyncSolwyn(AsyncSolwyn):
    """Keep the one retry option Agents injects on Solwyn's metered path."""

    def with_options(self, **kwargs: object) -> RetrySafeAsyncSolwyn:
        if kwargs != {"max_retries": 0}:
            raise RuntimeError(
                "This Agents recipe only supports with_options(max_retries=0); "
                "configure every other provider option on AsyncOpenAI before wrapping it"
            )
        # Solwyn already forces max_retries=0 on the raw SDK client for every
        # provider hop. Returning this metered wrapper preserves that behavior
        # without exposing the raw client's unmetered with_options copy.
        return self


class SolwynModel(Model):
    """Activate the pending logical agent run only while provider work executes."""

    def __init__(
        self,
        delegate: Model,
        pending_handle: ContextVar[solwyn.RunHandle | None],
    ) -> None:
        self._delegate = delegate
        self._pending_handle = pending_handle
        self._handle: solwyn.RunHandle | None = None
        self._attempt_active = False

    def _consume_handle(self) -> solwyn.RunHandle:
        if self._handle is not None:
            if self._pending_handle.get() is not None:
                raise RuntimeError("Agents reused a bound Solwyn model for a new model turn")
            return self._handle
        handle = self._pending_handle.get()
        self._pending_handle.set(None)
        if handle is None:
            raise RuntimeError("Agents model call has no pending Solwyn agent run")
        self._handle = handle
        return handle

    def _start_attempt(self) -> solwyn.RunHandle:
        if self._attempt_active:
            raise RuntimeError("Concurrent calls through one Solwyn Agents model are unsupported")
        handle = self._consume_handle()
        self._attempt_active = True
        return handle

    def _finish_attempt(self) -> None:
        if not self._attempt_active:
            raise RuntimeError("Solwyn Agents model attempt state is inconsistent")
        self._attempt_active = False

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        handle = self._start_attempt()
        try:
            with handle.activate():
                return await self._delegate.get_response(*args, **kwargs)
        finally:
            self._finish_attempt()

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        handle = self._start_attempt()
        try:
            with handle.activate():
                stream = self._delegate.stream_response(*args, **kwargs)
        except BaseException:
            self._finish_attempt()
            raise
        return SolwynModelStream(stream, handle, self._finish_attempt)

    def get_retry_advice(self, request: Any) -> Any:
        return self._delegate.get_retry_advice(request)

    async def close(self) -> None:
        await self._delegate.close()

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self._delegate._cleanup_on_run_end(owner)


class SolwynModelStream(AsyncIterator[Any]):
    """Activate one logical run around each provider stream operation."""

    def __init__(
        self,
        delegate: AsyncIterator[Any],
        handle: solwyn.RunHandle,
        finish_attempt: Callable[[], None],
    ) -> None:
        self._delegate = delegate
        self._handle = handle
        self._finish_attempt = finish_attempt
        self._closed = False
        self._pulling = False

    def __aiter__(self) -> SolwynModelStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        if self._pulling:
            raise RuntimeError("Concurrent iteration of one Solwyn Agents stream is unsupported")
        self._pulling = True
        try:
            with self._handle.activate():
                event = await anext(self._delegate)
        except StopAsyncIteration:
            self._pulling = False
            await self.aclose()
            raise
        except BaseException:
            self._pulling = False
            await self.aclose()
            raise
        self._pulling = False
        return event

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._pulling:
            raise RuntimeError("Cannot close a Solwyn Agents stream while iteration is active")
        try:
            aclose = getattr(self._delegate, "aclose", None)
            if callable(aclose):
                with self._handle.activate():
                    await aclose()
        finally:
            self._closed = True
            self._finish_attempt()


class SolwynModelProvider(ModelProvider):
    """Wrap every model resolved by an Agents provider without changing arguments."""

    def __init__(
        self,
        delegate: ModelProvider,
        pending_handle: ContextVar[solwyn.RunHandle | None],
    ) -> None:
        self._delegate = delegate
        self._pending_handle = pending_handle

    def get_model(self, model_name: str | None) -> Model:
        return SolwynModel(self._delegate.get_model(model_name), self._pending_handle)

    async def aclose(self) -> None:
        await self._delegate.aclose()


class SolwynRunHooks(RunHooks[None]):
    """Create stable logical agent runs and nominate one for each model turn."""

    def __init__(self, model_provider: ModelProvider) -> None:
        self.handles: dict[int, solwyn.RunHandle] = {}
        self._pending_handle: ContextVar[solwyn.RunHandle | None] = ContextVar(
            "solwyn_agents_pending_handle",
            default=None,
        )
        self.model_provider = SolwynModelProvider(model_provider, self._pending_handle)

    @staticmethod
    def _completed() -> asyncio.Future[None]:
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(None)
        return completed

    def on_agent_start(self, context: Any, agent: Agent[None]) -> asyncio.Future[None]:
        agent_key = id(agent)
        if agent_key in self.handles:
            raise RuntimeError("Agents started an already-active agent instance")
        self.handles[agent_key] = solwyn.create_run(f"agent:{agent.name}")
        return self._completed()

    def on_llm_start(
        self,
        context: Any,
        agent: Agent[None],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> asyncio.Future[None]:
        handle = self.handles.get(id(agent))
        if handle is None:
            raise RuntimeError("Agents model turn has no active Solwyn agent run")
        self._pending_handle.set(handle)
        return self._completed()

    def _finish_agent(self, agent: Agent[None]) -> None:
        agent_key = id(agent)
        handle = self.handles.get(agent_key)
        if handle is not None:
            handle.finish()
            self.handles.pop(agent_key)

    def on_agent_end(self, context: Any, agent: Agent[None], output: Any) -> asyncio.Future[None]:
        self._finish_agent(agent)
        return self._completed()

    def on_handoff(
        self, context: Any, from_agent: Agent[None], to_agent: Agent[None]
    ) -> asyncio.Future[None]:
        self._finish_agent(from_agent)
        return self._completed()

    def discard_abandoned(self) -> None:
        """Finish inactive logical runs whose runner ended on an exception."""
        self._pending_handle.set(None)
        for agent_key, handle in list(self.handles.items()):
            handle.finish()
            self.handles.pop(agent_key)

    @property
    def pending_handle(self) -> solwyn.RunHandle | None:
        return self._pending_handle.get()


def _make_client(router: respx.MockRouter) -> RetrySafeAsyncSolwyn:
    raw_client = make_offline_openai_client(router)
    return RetrySafeAsyncSolwyn(
        raw_client,
        api_key=SOLWYN_API_KEY,
        api_url=CONTROL_PLANE_URL,
        provider="openai",
        budget_mode="hard_deny",
        budget_check_cache_ttl=0,
        lease_enabled=False,
        breaker_reporting_enabled=False,
        reporter_flush_interval=60.0,
    )


def _configure_agents(client: AsyncSolwyn) -> None:
    # The real framework setter is the admission gate this integration needs.
    assert isinstance(client, AsyncOpenAI)
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")


def _make_hooks_and_run_config(
    *, retry_model_call: bool = False
) -> tuple[SolwynRunHooks, RunConfig]:
    hooks = SolwynRunHooks(MultiProvider())
    model_settings = None
    if retry_model_call:
        model_settings = ModelSettings(
            retry=ModelRetrySettings(
                max_retries=1,
                policy=retry_policies.network_error(),
                backoff={
                    "initial_delay": 0,
                    "max_delay": 0,
                    "multiplier": 0,
                    "jitter": False,
                },
            )
        )
    run_config = RunConfig(
        model_provider=hooks.model_provider,
        model_settings=model_settings,
        tracing_disabled=True,
        trace_include_sensitive_data=False,
    )
    return hooks, run_config


@function_tool
def lookup_case() -> str:
    """Return an offline fixture result for the real Agents tool loop."""
    return "case found"


@pytest.mark.framework_smoke
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run-streamed"])
async def test_openai_agents_uses_wrapped_client_and_preserves_run_hierarchy(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    *,
    streaming: bool,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, streaming=streaming)
    client = _make_client(respx_mock)
    _configure_agents(client)

    agent = Agent[None](name="Triage", instructions="Return a short result.", model=MODEL)
    hooks, run_config = _make_hooks_and_run_config()

    async with client:
        with solwyn.run("triage-workflow", tags={"team": "support"}) as workflow_run_id:
            if streaming:
                result = Runner.run_streamed(
                    agent,
                    "Route this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )
                async for _event in result.stream_events():
                    pass
            else:
                result = await Runner.run(
                    agent,
                    "Route this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )

    assert result.final_output == "This is a fake response."
    assert hooks.handles == {}
    assert hooks.pending_handle is None
    assert harness.model_call_count == 1
    assert len(harness.budget_checks) == 1
    assert len(harness.confirms) == 1

    success_events = [event for event in harness.events if event["status"] == "success"]
    assert len(success_events) == 1
    event = success_events[0]
    assert event["agent_run_id"] == harness.budget_checks[0]["agent_run_id"]
    assert event["agent_run_name"] == "agent:Triage"
    assert event["parent_agent_run_id"] == workflow_run_id
    assert event["tags"] == {"team": "support"}


@pytest.mark.framework_smoke
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run-streamed"])
async def test_openai_agents_handoff_keeps_agent_runs_as_workflow_siblings(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    *,
    streaming: bool,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, streaming=streaming, handoff=True)
    client = _make_client(respx_mock)
    _configure_agents(client)

    specialist = Agent[None](
        name="Specialist",
        instructions="Resolve the routed support request.",
        model=MODEL,
    )
    triage = Agent[None](
        name="Triage",
        instructions="Route this request to the specialist.",
        model=MODEL,
        handoffs=[specialist],
    )
    hooks, run_config = _make_hooks_and_run_config()

    async with client:
        with solwyn.run("triage-workflow", tags={"team": "support"}) as workflow_run_id:
            if streaming:
                result = Runner.run_streamed(
                    triage,
                    "Route this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )
                async for _event in result.stream_events():
                    pass
            else:
                result = await Runner.run(
                    triage,
                    "Route this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )

    assert result.final_output == "The specialist resolved this request."
    assert result.last_agent.name == "Specialist"
    assert hooks.handles == {}
    assert hooks.pending_handle is None
    assert harness.model_call_count == 2
    assert len(harness.budget_checks) == 2
    assert len(harness.confirms) == 2

    success_events = [event for event in harness.events if event["status"] == "success"]
    assert len(success_events) == 2
    events_by_name = {event["agent_run_name"]: event for event in success_events}
    assert set(events_by_name) == {"agent:Triage", "agent:Specialist"}

    triage_event = events_by_name["agent:Triage"]
    specialist_event = events_by_name["agent:Specialist"]
    assert triage_event["agent_run_id"] != specialist_event["agent_run_id"]
    for event in (triage_event, specialist_event):
        assert event["parent_agent_run_id"] == workflow_run_id
        assert event["tags"] == {"team": "support"}

    budget_run_ids = {check["agent_run_id"] for check in harness.budget_checks}
    assert budget_run_ids == {
        triage_event["agent_run_id"],
        specialist_event["agent_run_id"],
    }


@pytest.mark.framework_smoke
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run-streamed"])
async def test_openai_agents_same_agent_tool_turn_reuses_one_logical_run(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    *,
    streaming: bool,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, streaming=streaming, function_tool=True)
    client = _make_client(respx_mock)
    _configure_agents(client)

    agent = Agent[None](
        name="Worker",
        instructions="Look up the case, then return a short result.",
        model=MODEL,
        tools=[lookup_case],
    )
    hooks, run_config = _make_hooks_and_run_config()

    async with client:
        with solwyn.run("triage-workflow", tags={"team": "support"}) as workflow_run_id:
            if streaming:
                result = Runner.run_streamed(
                    agent,
                    "Resolve this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )
                async for _event in result.stream_events():
                    pass
            else:
                result = await Runner.run(
                    agent,
                    "Resolve this support request.",
                    hooks=hooks,
                    run_config=run_config,
                )

    assert result.final_output == "The worker completed both turns."
    assert hooks.handles == {}
    assert hooks.pending_handle is None
    assert harness.model_call_count == 2
    assert len(harness.budget_checks) == 2
    assert len(harness.confirms) == 2

    success_events = [event for event in harness.events if event["status"] == "success"]
    assert len(success_events) == 2
    run_ids = {event["agent_run_id"] for event in success_events}
    assert len(run_ids) == 1
    assert {check["agent_run_id"] for check in harness.budget_checks} == run_ids
    for event in success_events:
        assert event["agent_run_name"] == "agent:Worker"
        assert event["parent_agent_run_id"] == workflow_run_id
        assert event["tags"] == {"team": "support"}


@pytest.mark.framework_smoke
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run-streamed"])
async def test_openai_agents_runner_retry_reuses_stable_logical_run(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    caplog: pytest.LogCaptureFixture,
    *,
    streaming: bool,
) -> None:
    harness = FrameworkSmokeHarness(
        respx_mock,
        streaming=streaming,
        transient_failure=True,
    )
    client = _make_client(respx_mock)
    _configure_agents(client)

    agent = Agent[None](name="RetryWorker", instructions="Return a short result.", model=MODEL)
    hooks, run_config = _make_hooks_and_run_config(retry_model_call=True)

    try:
        async with client:
            with solwyn.run("triage-workflow", tags={"team": "support"}) as workflow_run_id:
                if streaming:
                    result = Runner.run_streamed(
                        agent,
                        "Resolve this support request.",
                        hooks=hooks,
                        run_config=run_config,
                    )
                    async for _event in result.stream_events():
                        pass
                else:
                    result = await Runner.run(
                        agent,
                        "Resolve this support request.",
                        hooks=hooks,
                        run_config=run_config,
                    )
    finally:
        if hooks.handles:
            hooks.discard_abandoned()

    assert result.final_output == "This is a fake response."
    assert hooks.handles == {}
    assert hooks.pending_handle is None
    assert harness.model_call_count == 2
    assert len(harness.budget_checks) == 2
    # The failed connection releases its reservation and emits ERROR metadata;
    # only the successful attempt has provider usage to confirm.
    assert len(harness.confirms) == 1

    settled_events = [event for event in harness.events if event["status"] in {"error", "success"}]
    assert [event["status"] for event in settled_events] == ["error", "success"]
    run_ids = {event["agent_run_id"] for event in settled_events}
    assert len(run_ids) == 1
    assert {check["agent_run_id"] for check in harness.budget_checks} == run_ids
    for event in settled_events:
        assert event["agent_run_name"] == "agent:RetryWorker"
        assert event["parent_agent_run_id"] == workflow_run_id
        assert event["tags"] == {"team": "support"}
    call_ids = {event["call_id"] for event in settled_events}
    assert len(call_ids) == 2
    assert harness.confirms[0]["call_id"] == settled_events[1]["call_id"]
    assert all(
        "untracked surface 'with_options'" not in record.getMessage() for record in caplog.records
    )


@pytest.mark.framework_smoke
@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 1.0},
        {"max_retries": 0, "timeout": 1.0},
        {"max_retries": 1},
    ],
)
async def test_openai_agents_retry_safe_client_rejects_other_options(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
    options: dict[str, object],
) -> None:
    client = _make_client(respx_mock)

    try:
        with pytest.raises(RuntimeError, match=r"only supports with_options\(max_retries=0\)"):
            client.with_options(**options)
    finally:
        await client.close()


@pytest.mark.framework_smoke
async def test_solwyn_model_stream_early_close_restores_attempt_state(
    fake_provider_harness_only: None,
) -> None:
    async def events() -> AsyncIterator[object]:
        yield object()

    handle = solwyn.create_run("agent:EarlyClose")
    pending_handle: ContextVar[solwyn.RunHandle | None] = ContextVar(
        "solwyn_agents_early_close_handle",
        default=handle,
    )
    delegate = cast(Model, SimpleNamespace(stream_response=lambda *_args, **_kwargs: events()))
    model = SolwynModel(delegate, pending_handle)

    stream = model.stream_response()
    await anext(stream)
    await stream.aclose()

    assert pending_handle.get() is None
    assert model._attempt_active is False
    assert solwyn.current_run_context().id is None
    handle.finish()


@pytest.mark.framework_smoke
async def test_solwyn_model_stream_cancellation_restores_attempt_state(
    fake_provider_harness_only: None,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def events() -> AsyncIterator[object]:
        started.set()
        await blocked.wait()
        yield object()

    handle = solwyn.create_run("agent:CancelledStream")
    pending_handle: ContextVar[solwyn.RunHandle | None] = ContextVar(
        "solwyn_agents_cancelled_stream_handle",
        default=handle,
    )
    delegate = cast(Model, SimpleNamespace(stream_response=lambda *_args, **_kwargs: events()))
    model = SolwynModel(delegate, pending_handle)
    stream = model.stream_response()

    pull = asyncio.create_task(anext(stream))
    await started.wait()
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull

    assert pending_handle.get() is None
    assert model._attempt_active is False
    assert solwyn.current_run_context().id is None
    handle.finish()


@pytest.mark.framework_smoke
async def test_openai_agents_propagates_budget_denial_before_model_call(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, allow_budget=False)
    client = _make_client(respx_mock)
    _configure_agents(client)

    agent = Agent[None](name="Triage", instructions="Return a short result.", model=MODEL)
    hooks, run_config = _make_hooks_and_run_config()

    async with client:
        with solwyn.run("triage-workflow", tags={"team": "support"}) as workflow_run_id:
            try:
                with pytest.raises(solwyn.BudgetExceededError):
                    await Runner.run(
                        agent,
                        "Route this support request.",
                        hooks=hooks,
                        run_config=run_config,
                    )
            finally:
                hooks.discard_abandoned()

    assert harness.model_call_count == 0
    assert len(harness.budget_checks) == 1
    assert harness.confirms == []
    assert hooks.handles == {}
    assert hooks.pending_handle is None
    assert solwyn.current_run_context().id is None
    with solwyn.run("post-denial-stack-check") as probe_run_id:
        assert solwyn.current_run_context().id == probe_run_id
    assert solwyn.current_run_context().id is None
    denied_events = [event for event in harness.events if event["status"] == "budget_denied"]
    assert len(denied_events) == 1
    event = denied_events[0]
    assert event["agent_run_name"] == "agent:Triage"
    assert event["parent_agent_run_id"] == workflow_run_id
