"""Protocol-double tests for the CrewAI run-attribution listener."""

from __future__ import annotations

import importlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from types import ModuleType
from typing import Any
from uuid import uuid4

import pytest

import solwyn
from solwyn._run import _capture_run_context

_STRUCTURAL_EVENTS = {
    "CrewKickoffStartedEvent",
    "CrewKickoffCompletedEvent",
    "CrewKickoffFailedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
}
_CONTENT_EVENT_DENYLIST = {
    "LLMCallStartedEvent",
    "LLMCallCompletedEvent",
    "LLMCallFailedEvent",
    "LLMStreamChunkEvent",
    "ToolUsageStartedEvent",
    "ToolUsageFinishedEvent",
    "ToolUsageErrorEvent",
}


class _UnreadableEvent:
    """Raise if the listener inspects, formats, or logs an event payload."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"content-bearing event was accessed through {name}")

    def __repr__(self) -> str:
        raise AssertionError("content-bearing event was formatted")

    def __str__(self) -> str:
        raise AssertionError("content-bearing event was formatted")


class _TaskSource:
    """Expose only CrewAI's structural task identifier."""

    def __init__(self) -> None:
        object.__setattr__(self, "id", uuid4())

    def __getattribute__(self, name: str) -> Any:
        if name == "id":
            return object.__getattribute__(self, name)
        raise AssertionError(f"non-structural task source field was accessed through {name}")

    def __repr__(self) -> str:
        raise AssertionError("task source was formatted")

    def __str__(self) -> str:
        raise AssertionError("task source was formatted")


class _CrewSource:
    """Expose only CrewAI's structural crew identifier, name, and task order."""

    def __init__(self, tasks: tuple[Any, ...], name: str) -> None:
        object.__setattr__(self, "id", uuid4())
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tasks", list(tasks))

    def __getattribute__(self, name: str) -> Any:
        if name in {"id", "name", "tasks"}:
            return object.__getattribute__(self, name)
        raise AssertionError(f"non-structural crew source field was accessed through {name}")

    def __repr__(self) -> str:
        raise AssertionError("crew source was formatted")

    def __str__(self) -> str:
        raise AssertionError("crew source was formatted")


class _FakeEventBus:
    """Minimal implementation of CrewAI's ``.on(EventType)`` protocol."""

    def __init__(self) -> None:
        self.registry: dict[type[Any], list[Any]] = {}

    def on(self, event_type: type[Any]) -> Any:
        def register(handler: Any) -> Any:
            self.registry.setdefault(event_type, []).append(handler)
            return handler

        return register

    def emit(self, event_type: type[Any], source: Any) -> None:
        event = event_type()
        for handler in self.registry.get(event_type, []):
            handler(source, event)


def _load_listener(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_location: str = "current",
) -> tuple[type[Any], _FakeEventBus, dict[str, type[Any]]]:
    bus = _FakeEventBus()
    event_types = {
        name: type(name, (_UnreadableEvent,), {})
        for name in _STRUCTURAL_EVENTS | _CONTENT_EVENT_DENYLIST
    }

    class _FakeBaseEventListener:
        def __init__(self) -> None:
            self.setup_listeners(bus)

    events = ModuleType("crewai.events" if api_location == "current" else "crewai.utilities.events")
    events.__path__ = []  # type: ignore[attr-defined]
    events.BaseEventListener = _FakeBaseEventListener  # type: ignore[attr-defined]
    for name, event_type in event_types.items():
        setattr(events, name, event_type)

    crewai = ModuleType("crewai")
    crewai.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    if api_location == "current":
        crewai.events = events  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "crewai.events", events)
        sys.modules.pop("crewai.utilities", None)
        sys.modules.pop("crewai.utilities.events", None)
        sys.modules.pop("crewai.utilities.events.base_event_listener", None)
    else:
        utilities = ModuleType("crewai.utilities")
        utilities.__path__ = []  # type: ignore[attr-defined]
        base_event_listener = ModuleType("crewai.utilities.events.base_event_listener")
        base_event_listener.BaseEventListener = _FakeBaseEventListener  # type: ignore[attr-defined]
        utilities.events = events  # type: ignore[attr-defined]
        crewai.utilities = utilities  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "crewai.utilities", utilities)
        monkeypatch.setitem(sys.modules, "crewai.utilities.events", events)
        monkeypatch.setitem(
            sys.modules,
            "crewai.utilities.events.base_event_listener",
            base_event_listener,
        )
        sys.modules.pop("crewai.events", None)

    sys.modules.pop("solwyn.integrations.crewai", None)
    module = importlib.import_module("solwyn.integrations.crewai")
    return module.SolwynEventListener, bus, event_types


@pytest.fixture
def listener_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[Any], _FakeEventBus, dict[str, type[Any]]]:
    return _load_listener(monkeypatch)


def _task_source() -> _TaskSource:
    return _TaskSource()


def _crew_source(*tasks: Any, name: str = "SupportCrew") -> _CrewSource:
    return _CrewSource(tasks, name)


def _assert_no_active_context() -> None:
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)


def _assert_inactive(listener: Any, *sources: Any) -> None:
    for source in sources:
        with (
            pytest.raises(RuntimeError, match="inactive CrewAI source"),
            listener.activate(source),
        ):
            pass
    _assert_no_active_context()


@pytest.mark.unit
def test_crewai_integration_module_is_shipped() -> None:
    integration_path = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "src"
        / "solwyn"
        / "integrations"
        / "crewai.py"
    )

    assert integration_path.is_file(), "CrewAI integration module is missing"


@pytest.mark.unit
def test_registers_exactly_the_six_structural_lifecycle_events(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, _ = listener_protocol

    listener_type()

    registered_names = {event_type.__name__ for event_type in bus.registry}
    assert registered_names == _STRUCTURAL_EVENTS
    assert registered_names.isdisjoint(_CONTENT_EVENT_DENYLIST)
    assert all(len(handlers) == 1 for handlers in bus.registry.values())


@pytest.mark.unit
def test_legacy_crewai_event_import_path_registers_the_same_safe_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener_type, bus, _ = _load_listener(monkeypatch, api_location="legacy")

    listener_type()

    assert {event_type.__name__ for event_type in bus.registry} == _STRUCTURAL_EVENTS


@pytest.mark.unit
def test_success_lifecycle_preserves_crew_task_hierarchy_and_never_leaks_context(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type(tags={"framework": "crewai"})

    bus.emit(events["CrewKickoffStartedEvent"], crew)
    _assert_no_active_context()
    with listener.activate(crew):
        crew_snapshot = _capture_run_context()
    _assert_no_active_context()

    bus.emit(events["TaskStartedEvent"], task)
    _assert_no_active_context()
    with listener.activate(task):
        task_snapshot = _capture_run_context()
    _assert_no_active_context()

    assert crew_snapshot[1:] == ("crew:SupportCrew", {"framework": "crewai"}, None)
    assert task_snapshot[1:] == (
        "task:0",
        {"framework": "crewai"},
        crew_snapshot[0],
    )

    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)

    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_crew_start_preregisters_task_scope_before_async_task_callback(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()

    bus.emit(events["CrewKickoffStartedEvent"], crew)

    with listener.activate(crew):
        crew_run_id = _capture_run_context()[0]
    with listener.activate(task):
        task_snapshot = _capture_run_context()
    assert task_snapshot[1:] == ("task:0", None, crew_run_id)

    with pytest.raises(RuntimeError, match="ended before its start callback"):
        bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_task_completed_callback_winning_threadpool_race_is_cleaned_at_crew_terminal(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], crew)

    with pytest.raises(RuntimeError, match="ended before its start callback"):
        bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["TaskStartedEvent"], task)
    with caplog.at_level(logging.WARNING):
        bus.emit(events["CrewKickoffCompletedEvent"], crew)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0]
        .getMessage()
        .startswith("solwyn.integrations.crewai: forced structural scope cleanup for run_")
    )
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_task_and_crew_failure_events_finish_their_detached_handles(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()

    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events["TaskFailedEvent"], task)
    bus.emit(events["CrewKickoffFailedEvent"], crew)

    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_crew_failure_forces_unfinished_task_cleanup_with_structural_warning(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()

    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)
    with caplog.at_level(logging.WARNING):
        bus.emit(events["CrewKickoffFailedEvent"], crew)

    assert [record.getMessage() for record in caplog.records] == [
        next(
            message
            for message in (record.getMessage() for record in caplog.records)
            if message.startswith(
                "solwyn.integrations.crewai: forced structural scope cleanup for run_"
            )
        )
    ]
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_duplicate_crew_start_fails_closed_and_poisons_ambiguous_scope(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], crew)

    with pytest.raises(
        RuntimeError,
        match="duplicate crew source id.*overlaps an active task lifecycle",
    ):
        bus.emit(events["CrewKickoffStartedEvent"], crew)

    with (
        pytest.raises(RuntimeError, match="blocked CrewAI source"),
        listener.activate(crew),
    ):
        pass
    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_sequential_same_crew_reuse_gets_fresh_generation_before_old_terminal(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()

    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)
    with listener.activate(crew):
        first_crew_run_id = _capture_run_context()[0]
    with listener.activate(task):
        first_task_snapshot = _capture_run_context()
    bus.emit(events["TaskCompletedEvent"], task)

    bus.emit(events["CrewKickoffStartedEvent"], crew)
    with listener.activate(crew):
        second_crew_run_id = _capture_run_context()[0]
    with listener.activate(task):
        second_task_snapshot = _capture_run_context()
    bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events["TaskCompletedEvent"], task)

    assert first_crew_run_id != second_crew_run_id
    assert first_task_snapshot[0] != second_task_snapshot[0]
    assert first_task_snapshot[3] == first_crew_run_id
    assert second_task_snapshot[3] == second_crew_run_id

    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    with listener.activate(crew):
        assert _capture_run_context()[0] == second_crew_run_id
    bus.emit(events["CrewKickoffCompletedEvent"], crew)

    assert listener._stale_crew_terminals == {}
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_reuse_with_delayed_task_start_is_poisoned_until_old_crew_cleans_up(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], crew)

    with pytest.raises(RuntimeError, match="overlaps an active task lifecycle"):
        bus.emit(events["CrewKickoffStartedEvent"], crew)
    with (
        pytest.raises(RuntimeError, match="blocked CrewAI source"),
        listener.activate(task),
    ):
        pass

    bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)

    assert listener._blocked_sources == set()
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("task_terminal", "crew_terminal"),
    [
        ("TaskCompletedEvent", "CrewKickoffCompletedEvent"),
        ("TaskFailedEvent", "CrewKickoffFailedEvent"),
    ],
)
def test_concurrent_same_crew_reuse_cannot_activate_old_task_generation(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
    task_terminal: str,
    crew_terminal: str,
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)

    first_activation = listener.activate(task)
    first_run_id = first_activation.__enter__()
    try:
        with pytest.raises(RuntimeError, match="overlaps an active task lifecycle"):
            bus.emit(events["CrewKickoffStartedEvent"], crew)
        assert solwyn.current_run_context().id == first_run_id
        with (
            pytest.raises(RuntimeError, match="blocked CrewAI source"),
            listener.activate(task),
        ):
            pass
    finally:
        first_activation.__exit__(None, None, None)

    with pytest.raises(RuntimeError, match="duplicate task source id"):
        bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events[task_terminal], task)
    with pytest.raises(RuntimeError, match="overlaps an active task lifecycle"):
        bus.emit(events["CrewKickoffStartedEvent"], crew)
    with (
        pytest.raises(RuntimeError, match="blocked CrewAI source"),
        listener.activate(task),
    ):
        pass
    bus.emit(events[crew_terminal], crew)

    assert listener._blocked_sources == set()
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_duplicate_task_start_fails_closed(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)

    with pytest.raises(RuntimeError, match="duplicate task source id"):
        bus.emit(events["TaskStartedEvent"], task)

    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_task_start_with_unknown_crew_parent_fails_closed(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    listener_type()

    with pytest.raises(RuntimeError, match="unknown crew parent"):
        bus.emit(events["TaskStartedEvent"], _task_source())

    _assert_no_active_context()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event_name", "source_factory", "match"),
    [
        ("TaskCompletedEvent", _task_source, "unknown task source id"),
        ("TaskFailedEvent", _task_source, "unknown task source id"),
        ("CrewKickoffCompletedEvent", _crew_source, "unknown crew source id"),
        ("CrewKickoffFailedEvent", _crew_source, "unknown crew source id"),
    ],
)
def test_unknown_terminal_event_fails_closed(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
    event_name: str,
    source_factory: Any,
    match: str,
) -> None:
    listener_type, bus, events = listener_protocol
    listener_type()

    with pytest.raises(RuntimeError, match=match):
        bus.emit(events[event_name], source_factory())

    _assert_no_active_context()


@pytest.mark.unit
def test_task_source_id_cannot_be_owned_by_two_active_crews(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    first_crew = _crew_source(task, name="FirstCrew")
    second_crew = _crew_source(task, name="SecondCrew")
    listener = listener_type()
    bus.emit(events["CrewKickoffStartedEvent"], first_crew)

    with pytest.raises(RuntimeError, match="task source id is already owned"):
        bus.emit(events["CrewKickoffStartedEvent"], second_crew)

    with listener.activate(first_crew):
        assert solwyn.current_run_context().name == "crew:FirstCrew"
    with (
        pytest.raises(RuntimeError, match="blocked CrewAI source"),
        listener.activate(task),
    ):
        pass
    with (
        pytest.raises(RuntimeError, match="inactive CrewAI source"),
        listener.activate(second_crew),
    ):
        pass
    bus.emit(events["CrewKickoffCompletedEvent"], first_crew)
    _assert_inactive(listener, task, first_crew, second_crew)


@pytest.mark.unit
def test_task_collision_poison_clears_when_original_crew_generation_retires(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    first_crew = _crew_source(task, name="FirstCrew")
    colliding_crew = _crew_source(task, name="CollidingCrew")
    task_id = str(task.id)
    first_crew_id = str(first_crew.id)
    colliding_crew_id = str(colliding_crew.id)
    listener = listener_type()

    bus.emit(events["CrewKickoffStartedEvent"], first_crew)
    bus.emit(events["TaskStartedEvent"], task)
    with listener.activate(task):
        first_task_run_id = _capture_run_context()[0]
    bus.emit(events["TaskCompletedEvent"], task)

    with pytest.raises(RuntimeError, match="task source id is already owned"):
        bus.emit(events["CrewKickoffStartedEvent"], colliding_crew)

    assert listener._task_registrations[task_id].crew_id == first_crew_id
    assert colliding_crew_id not in listener._crews
    with (
        pytest.raises(RuntimeError, match="blocked CrewAI source"),
        listener.activate(task),
    ):
        pass
    with (
        pytest.raises(RuntimeError, match="inactive CrewAI source"),
        listener.activate(colliding_crew),
    ):
        pass

    bus.emit(events["CrewKickoffStartedEvent"], first_crew)

    assert task_id in listener._task_handles
    assert set(listener._task_handles).isdisjoint(listener._blocked_sources)
    assert listener._task_registrations[task_id].crew_id == first_crew_id
    with listener.activate(task):
        fresh_task_run_id = _capture_run_context()[0]
    assert fresh_task_run_id != first_task_run_id

    bus.emit(events["TaskStartedEvent"], task)
    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], first_crew)
    bus.emit(events["CrewKickoffCompletedEvent"], first_crew)
    _assert_inactive(listener, task, first_crew, colliding_crew)


@pytest.mark.unit
def test_terminal_callbacks_defer_cleanup_until_cross_context_activation_exits(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()
    Context().run(bus.emit, events["CrewKickoffStartedEvent"], crew)
    Context().run(bus.emit, events["TaskStartedEvent"], task)

    activation = listener.activate(task)
    activation.__enter__()
    try:
        with caplog.at_level(logging.WARNING):
            Context().run(bus.emit, events["TaskCompletedEvent"], task)
            Context().run(bus.emit, events["CrewKickoffCompletedEvent"], crew)
        assert solwyn.current_run_context().name == "task:0"
    finally:
        activation.__exit__(None, None, None)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith(
        "solwyn.integrations.crewai: deferred structural scope close for run_"
    )
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_lifecycle_can_cross_fresh_contexts_without_token_ownership_or_leakage(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    task = _task_source()
    crew = _crew_source(task)
    listener = listener_type()

    Context().run(bus.emit, events["CrewKickoffStartedEvent"], crew)
    Context().run(bus.emit, events["TaskStartedEvent"], task)

    def capture() -> tuple[str | None, str | None, dict[str, str] | None, str | None]:
        with listener.activate(task):
            return _capture_run_context()

    snapshot = Context().run(capture)
    assert snapshot[1] == "task:0"
    assert snapshot[3] is not None

    Context().run(bus.emit, events["TaskCompletedEvent"], task)
    Context().run(bus.emit, events["CrewKickoffCompletedEvent"], crew)
    _assert_inactive(listener, task, crew)


@pytest.mark.unit
def test_concurrent_distinct_crews_keep_task_parentage_isolated(
    listener_protocol: tuple[type[Any], _FakeEventBus, dict[str, type[Any]]],
) -> None:
    listener_type, bus, events = listener_protocol
    listener = listener_type()
    first_task = _task_source()
    second_task = _task_source()
    first_crew = _crew_source(first_task, name="FirstCrew")
    second_crew = _crew_source(second_task, name="SecondCrew")

    def execute(crew: Any, task: Any) -> tuple[str, str]:
        bus.emit(events["CrewKickoffStartedEvent"], crew)
        with listener.activate(crew):
            crew_run_id = _capture_run_context()[0]
        bus.emit(events["TaskStartedEvent"], task)
        with listener.activate(task):
            task_snapshot = _capture_run_context()
        bus.emit(events["TaskCompletedEvent"], task)
        bus.emit(events["CrewKickoffCompletedEvent"], crew)
        assert solwyn.current_run_context().id is None
        if crew_run_id is None or task_snapshot[3] is None:
            raise AssertionError("expected structural run hierarchy")
        return crew_run_id, task_snapshot[3]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(execute, first_crew, first_task),
            pool.submit(execute, second_crew, second_task),
        ]
        associations = [future.result() for future in futures]

    assert all(crew_run_id == parent_run_id for crew_run_id, parent_run_id in associations)
    assert len({crew_run_id for crew_run_id, _ in associations}) == 2
    _assert_inactive(listener, first_task, first_crew, second_task, second_crew)
