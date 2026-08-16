"""Offline smoke coverage against the real CrewAI package."""

from __future__ import annotations

import os
from threading import Event
from typing import Any

import pytest
import respx

# CrewAI reads these flags while its package is imported. The smoke deliberately
# leaves no path for its optional telemetry to transmit task or model content.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

try:
    from crewai.events.event_bus import crewai_event_bus
except ImportError:  # CrewAI 0.157 compatibility floor
    from crewai.utilities.events.crewai_event_bus import crewai_event_bus

import solwyn
from solwyn import Solwyn
from solwyn._run import _capture_run_context
from solwyn.integrations.crewai import SolwynEventListener

from . import (
    CONTROL_PLANE_URL,
    MODEL,
    OPENAI_BASE_URL,
    SOLWYN_API_KEY,
    FrameworkSmokeHarness,
    make_offline_openai_sync_client,
)


def _client_options() -> dict[str, Any]:
    return {
        "api_key": SOLWYN_API_KEY,
        "api_url": CONTROL_PLANE_URL,
        "provider": "openai",
        "budget_mode": "hard_deny",
        "budget_check_cache_ttl": 0,
        "lease_enabled": False,
        "breaker_reporting_enabled": False,
        "reporter_flush_interval": 60.0,
    }


class SolwynCrewAILLM(BaseLLM):
    """Test-only enforcement adapter; the shipped listener remains content-free."""

    def __init__(
        self,
        *,
        model: str,
        client: Solwyn,
        listener: SolwynEventListener,
    ) -> None:
        super().__init__(model=model)
        self._client = client
        self._listener = listener

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        **opaque_kwargs: Any,
    ) -> str:
        if from_task is None:
            raise RuntimeError("CrewAI did not provide the structural task source")
        provider_messages = (
            [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": provider_messages,
        }
        if tools:
            request["tools"] = tools
        with self._listener.activate(from_task):
            response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class _RecordingListener(SolwynEventListener):
    """Capture only detached structural identities for smoke assertions."""

    def __init__(self, **kwargs: Any) -> None:
        self.structural_runs: list[
            tuple[str | None, str | None, dict[str, str] | None, str | None]
        ] = []
        super().__init__(**kwargs)

    def _start_crew(self, source: Any) -> None:
        super()._start_crew(source)
        with self.activate(source):
            self.structural_runs.append(_capture_run_context())

    def _start_task(self, source: Any) -> None:
        super()._start_task(source)
        with self.activate(source):
            self.structural_runs.append(_capture_run_context())


class _DelayedFirstCrewTerminalListener(SolwynEventListener):
    """Hold the first structural crew terminal callback across a reuse start."""

    def __init__(self, **kwargs: Any) -> None:
        self.terminal_entered = Event()
        self.release_terminal = Event()
        self._delayed_terminal = False
        self._task_starts = 0
        super().__init__(**kwargs)

    def _finish_crew(self, source: Any) -> None:
        if not self._delayed_terminal:
            self._delayed_terminal = True
            self.terminal_entered.set()
            flush = getattr(crewai_event_bus, "flush", None)
            if flush is not None and not self.release_terminal.wait(timeout=10.0):
                raise RuntimeError("timed out waiting to release structural terminal")
        super()._finish_crew(source)

    def _start_task(self, source: Any) -> None:
        self._task_starts += 1
        if self._task_starts == 2:
            self.release_terminal.set()
        super()._start_task(source)


def _crew(llm: BaseLLM, *, name: str) -> Crew:
    agent = Agent(
        role="Offline analyst",
        goal="Return the offline fixture result",
        backstory="A deterministic test-only agent",
        llm=llm,
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        verbose=False,
    )
    task = Task(
        description="Process the offline fixture without external access",
        expected_output="The canned offline result",
        agent=agent,
    )
    return Crew(
        name=name,
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        cache=False,
        memory=False,
        planning=False,
        verbose=False,
    )


def _flush_crewai_events() -> None:
    flush = getattr(crewai_event_bus, "flush", None)
    if flush is not None and not flush(timeout=10.0):
        raise RuntimeError("CrewAI event handlers did not finish before the smoke deadline")


def _assert_listener_clean(listener: SolwynEventListener) -> None:
    assert listener._crews == {}
    assert listener._task_registrations == {}
    assert listener._task_handles == {}
    assert listener._started_tasks == set()
    assert listener._stale_crew_terminals == {}
    assert listener._blocked_sources == set()
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)


@pytest.mark.framework_smoke
def test_crewai_native_litellm_is_attributed_but_not_budget_checked(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock)
    native_llm = LLM(
        model=f"openai/{MODEL}",
        api_key="sk-provider-test",
        base_url=OPENAI_BASE_URL,
        is_litellm=True,
        max_retries=0,
        num_retries=0,
    )

    with crewai_event_bus.scoped_handlers():
        listener = _RecordingListener(tags={"framework": "crewai"})
        _crew(native_llm, name="NativeCrew").kickoff()
        _flush_crewai_events()

    assert harness.model_call_count == 1
    assert harness.budget_checks == []
    assert harness.confirms == []
    assert harness.events == []
    runs_by_name = {snapshot[1]: snapshot for snapshot in listener.structural_runs}
    assert set(runs_by_name) == {"crew:NativeCrew", "task:0"}
    assert runs_by_name["task:0"][3] == runs_by_name["crew:NativeCrew"][0]
    assert all(snapshot[2] == {"framework": "crewai"} for snapshot in runs_by_name.values())
    _assert_listener_clean(listener)


@pytest.mark.framework_smoke
def test_crewai_custom_basellm_enforces_and_attributes_crew_task_hierarchy(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock)
    client = Solwyn(make_offline_openai_sync_client(respx_mock), **_client_options())

    with crewai_event_bus.scoped_handlers():
        listener = SolwynEventListener(tags={"framework": "crewai"})
        model = SolwynCrewAILLM(model=MODEL, client=client, listener=listener)
        with client:
            _crew(model, name="EnforcedCrew").kickoff()
        _flush_crewai_events()

    success_events = [event for event in harness.events if event["status"] == "success"]
    assert harness.model_call_count == 1
    assert len(harness.budget_checks) == len(harness.confirms) == 1
    assert len(success_events) == 1
    event = success_events[0]
    assert event["agent_run_name"] == "task:0"
    assert event["agent_run_id"] == harness.budget_checks[0]["agent_run_id"]
    assert event["call_id"] == harness.confirms[0]["call_id"]
    assert event["parent_agent_run_id"] is not None
    assert event["tags"] == {"framework": "crewai"}
    _assert_listener_clean(listener)


@pytest.mark.framework_smoke
def test_crewai_reused_crew_gets_fresh_hierarchy_while_prior_terminal_is_late(
    respx_mock: respx.MockRouter,
    fake_provider_harness_only: None,
) -> None:
    harness = FrameworkSmokeHarness(respx_mock, model_calls=2)
    client = Solwyn(make_offline_openai_sync_client(respx_mock), **_client_options())

    with crewai_event_bus.scoped_handlers():
        listener = _DelayedFirstCrewTerminalListener(tags={"framework": "crewai"})
        model = SolwynCrewAILLM(model=MODEL, client=client, listener=listener)
        crew = _crew(model, name="ReusedCrew")
        with client:
            crew.kickoff()
            assert listener.terminal_entered.wait(timeout=10.0)
            try:
                crew.kickoff()
            finally:
                listener.release_terminal.set()
        _flush_crewai_events()

    success_events = [event for event in harness.events if event["status"] == "success"]
    assert harness.model_call_count == 2
    assert len(harness.budget_checks) == len(harness.confirms) == 2
    assert len(success_events) == 2
    assert len({event["agent_run_id"] for event in success_events}) == 2
    assert len({event["parent_agent_run_id"] for event in success_events}) == 2
    assert all(event["agent_run_name"] == "task:0" for event in success_events)
    assert all(event["parent_agent_run_id"] is not None for event in success_events)
    _assert_listener_clean(listener)
