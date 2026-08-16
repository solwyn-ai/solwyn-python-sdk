"""CrewAI crew/task run attribution without observing model or tool content.

Attribution only: native CrewAI model calls do not pass through Solwyn. Use
``SolwynEventListener.activate(...)`` from a custom ``BaseLLM`` around calls
made through a Solwyn-wrapped provider client to add enforcement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from uuid import UUID

import solwyn

_HandlerT = TypeVar("_HandlerT", bound=Callable[..., object])


class _EventBus(Protocol):
    def on(self, event_type: type[Any]) -> Callable[[_HandlerT], _HandlerT]: ...


if TYPE_CHECKING:

    class BaseEventListener:
        def __init__(self) -> None: ...

    class CrewKickoffCompletedEvent: ...

    class CrewKickoffFailedEvent: ...

    class CrewKickoffStartedEvent: ...

    class TaskCompletedEvent: ...

    class TaskFailedEvent: ...

    class TaskStartedEvent: ...

else:
    try:
        from crewai.events import (
            BaseEventListener,
            CrewKickoffCompletedEvent,
            CrewKickoffFailedEvent,
            CrewKickoffStartedEvent,
            TaskCompletedEvent,
            TaskFailedEvent,
            TaskStartedEvent,
        )
    except ImportError:
        try:
            from crewai.utilities.events import (
                CrewKickoffCompletedEvent,
                CrewKickoffFailedEvent,
                CrewKickoffStartedEvent,
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )
            from crewai.utilities.events.base_event_listener import BaseEventListener
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "solwyn.integrations.crewai requires CrewAI; "
                "install with: pip install 'solwyn[crewai]'"
            ) from exc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CrewState:
    handle: solwyn.RunHandle
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TaskRegistration:
    crew_id: str
    index: int


class SolwynEventListener(BaseEventListener):
    """Map structural CrewAI lifecycle sources to detached Solwyn runs.

    CrewAI may dispatch start and terminal callbacks in different threads.
    Detached handles keep lifecycle ownership independent of ``ContextVar``
    tokens; :meth:`activate` binds one identity only for the provider call that
    needs attribution.
    """

    def __init__(self, *, tags: dict[str, str] | None = None) -> None:
        self._tags = dict(tags) if tags is not None else None
        self._crews: dict[str, _CrewState] = {}
        self._task_registrations: dict[str, _TaskRegistration] = {}
        self._task_handles: dict[str, solwyn.RunHandle] = {}
        self._started_tasks: set[str] = set()
        self._terminal_crews: set[str] = set()
        self._terminal_tasks: set[str] = set()
        self._stale_crew_terminals: dict[str, int] = {}
        self._blocked_sources: set[str] = set()
        self._warned_cleanup: set[tuple[str, str]] = set()
        self._state_lock = Lock()
        super().__init__()

    def setup_listeners(self, crewai_event_bus: _EventBus) -> None:
        """Register only crew/task structural lifecycle events."""

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Any, event: CrewKickoffStartedEvent) -> None:
            self._start_crew(source)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source: Any, event: CrewKickoffCompletedEvent) -> None:
            self._finish_crew(source)

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source: Any, event: CrewKickoffFailedEvent) -> None:
            self._finish_crew(source)

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source: Any, event: TaskStartedEvent) -> None:
            self._start_task(source)

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Any, event: TaskCompletedEvent) -> None:
            self._finish_task(source)

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
            self._finish_task(source)

    @contextmanager
    def activate(self, source: Any) -> Iterator[str]:
        """Temporarily bind the active run identity for one CrewAI source."""
        source_id = self._source_id(source)
        with self._state_lock:
            if source_id in self._blocked_sources:
                raise RuntimeError("cannot activate a blocked CrewAI source")
            task_handle = self._task_handles.get(source_id)
            crew_state = self._crews.get(source_id)
            handle = (
                task_handle
                if task_handle is not None
                else (crew_state.handle if crew_state is not None else None)
            )
        if handle is None:
            raise RuntimeError("cannot activate an inactive CrewAI source")

        try:
            with handle.activate() as run_id:
                yield run_id
        finally:
            self._retry_terminal_cleanup(source_id)

    @staticmethod
    def _source_id(source: Any) -> str:
        raw_id = getattr(source, "id", None)
        if isinstance(raw_id, UUID):
            return str(raw_id)
        if isinstance(raw_id, str) and raw_id:
            return raw_id
        raise RuntimeError("CrewAI lifecycle sources require a structural UUID or string id")

    def _start_crew(self, source: Any) -> None:
        crew_id = self._source_id(source)
        crew_name = getattr(source, "name", None)
        if crew_name is None:
            crew_name = "crew"
        if not isinstance(crew_name, str) or not crew_name.strip():
            raise RuntimeError("CrewAI crew sources require a structural name")
        source_tasks = getattr(source, "tasks", None)
        if not isinstance(source_tasks, (list, tuple)):
            raise RuntimeError("CrewAI crew sources require a structural task sequence")
        task_ids = tuple(self._source_id(task) for task in source_tasks)
        if len(task_ids) != len(set(task_ids)):
            raise RuntimeError("CrewAI crew has duplicate task source ids")

        with self._state_lock:
            existing_crew = self._crews.get(crew_id)
            if existing_crew is not None:
                self._retire_reusable_crew_locked(crew_id, existing_crew)
            if crew_id in self._task_registrations or crew_id in task_ids:
                raise RuntimeError("CrewAI source id collides across crew and task scopes")
            owned_task_ids = tuple(
                task_id in self._task_registrations or task_id in self._crews
                for task_id in task_ids
            )
            if any(owned_task_ids):
                self._blocked_sources.update(
                    task_id
                    for task_id, is_owned in zip(task_ids, owned_task_ids, strict=True)
                    if is_owned
                )
                raise RuntimeError("CrewAI task source id is already owned by an active crew")
            if crew_id in self._blocked_sources or any(
                task_id in self._blocked_sources for task_id in task_ids
            ):
                raise RuntimeError("CrewAI cannot start a blocked source id")

            crew_handle = solwyn.create_run(f"crew:{crew_name}", tags=self._tags)
            task_handles: dict[str, solwyn.RunHandle] = {}
            try:
                with crew_handle.activate():
                    for index, task_id in enumerate(task_ids):
                        task_handles[task_id] = solwyn.create_run(
                            f"task:{index}",
                            tags=self._tags,
                        )
            except BaseException:
                for task_handle in task_handles.values():
                    task_handle.finish()
                crew_handle.finish()
                raise

            self._crews[crew_id] = _CrewState(handle=crew_handle, task_ids=task_ids)
            self._task_handles.update(task_handles)
            for index, task_id in enumerate(task_ids):
                self._task_registrations[task_id] = _TaskRegistration(crew_id, index)

    def _start_task(self, source: Any) -> None:
        task_id = self._source_id(source)
        with self._state_lock:
            registration = self._task_registrations.get(task_id)
            if registration is None:
                raise RuntimeError("CrewAI task callback has an unknown crew parent")
            if task_id in self._started_tasks:
                raise RuntimeError("CrewAI started a duplicate task source id")
            crew_state = self._crews.get(registration.crew_id)
            if crew_state is None or registration.crew_id in self._terminal_crews:
                raise RuntimeError("CrewAI task callback has an inactive crew parent")
            if task_id not in self._task_handles:
                raise RuntimeError("CrewAI task scope registration is inconsistent")
            self._started_tasks.add(task_id)

    def _finish_task(self, source: Any) -> None:
        task_id = self._source_id(source)
        with self._state_lock:
            if task_id not in self._task_registrations or task_id not in self._task_handles:
                raise RuntimeError("CrewAI ended an unknown task source id")
            if task_id not in self._started_tasks:
                raise RuntimeError("CrewAI task ended before its start callback")
            if task_id in self._terminal_tasks:
                raise RuntimeError("CrewAI ended a duplicate task source id")
            self._terminal_tasks.add(task_id)
            self._drain_task_locked(task_id)

    def _finish_crew(self, source: Any) -> None:
        crew_id = self._source_id(source)
        with self._state_lock:
            stale_terminals = self._stale_crew_terminals.get(crew_id, 0)
            if stale_terminals:
                if stale_terminals == 1:
                    self._stale_crew_terminals.pop(crew_id)
                else:
                    self._stale_crew_terminals[crew_id] = stale_terminals - 1
                return
            crew_state = self._crews.get(crew_id)
            if crew_state is None:
                raise RuntimeError("CrewAI ended an unknown crew source id")
            if crew_id in self._terminal_crews:
                raise RuntimeError("CrewAI ended a duplicate crew source id")
            self._terminal_crews.add(crew_id)

            for task_id in crew_state.task_ids:
                task_handle = self._task_handles.get(task_id)
                if task_handle is None:
                    continue
                if task_id in self._started_tasks and task_id not in self._terminal_tasks:
                    self._warn_forced(task_id, task_handle)
                if task_id not in self._terminal_tasks:
                    self._terminal_tasks.add(task_id)
                self._drain_task_locked(task_id)
            self._drain_crew_locked(crew_id)

    def _retire_reusable_crew_locked(
        self,
        crew_id: str,
        crew_state: _CrewState,
    ) -> None:
        has_live_task = not crew_state.task_ids or any(
            task_id in self._task_handles for task_id in crew_state.task_ids
        )
        if has_live_task or crew_id in self._terminal_crews or crew_id in self._blocked_sources:
            self._block_crew_locked(crew_id, crew_state)
            raise RuntimeError(
                "CrewAI started a duplicate crew source id that overlaps an active task lifecycle"
            )

        try:
            crew_state.handle.finish()
        except RuntimeError:
            self._block_crew_locked(crew_id, crew_state)
            raise RuntimeError(
                "CrewAI started a duplicate crew source id that overlaps an active task lifecycle"
            ) from None

        self._crews.pop(crew_id)
        self._blocked_sources.discard(crew_id)
        self._warned_cleanup.discard(("crew", crew_id))
        for task_id in crew_state.task_ids:
            self._task_registrations.pop(task_id, None)
            self._started_tasks.discard(task_id)
            self._terminal_tasks.discard(task_id)
            self._blocked_sources.discard(task_id)
            self._warned_cleanup.discard(("task", task_id))
        self._stale_crew_terminals[crew_id] = self._stale_crew_terminals.get(crew_id, 0) + 1

    def _block_crew_locked(self, crew_id: str, crew_state: _CrewState) -> None:
        self._blocked_sources.add(crew_id)
        self._blocked_sources.update(crew_state.task_ids)

    def _retry_terminal_cleanup(self, source_id: str) -> None:
        with self._state_lock:
            registration = self._task_registrations.get(source_id)
            if source_id in self._terminal_tasks:
                self._drain_task_locked(source_id)
            if registration is not None and registration.crew_id in self._terminal_crews:
                self._drain_crew_locked(registration.crew_id)
            if source_id in self._terminal_crews:
                self._drain_crew_locked(source_id)

    def _drain_task_locked(self, task_id: str) -> None:
        if task_id not in self._terminal_tasks:
            return
        handle = self._task_handles.get(task_id)
        if handle is None:
            return
        try:
            handle.finish()
        except RuntimeError:
            self._warn_deferred("task", task_id, handle)
            return
        self._task_handles.pop(task_id)
        self._started_tasks.discard(task_id)
        self._terminal_tasks.discard(task_id)
        self._warned_cleanup.discard(("task", task_id))

    def _drain_crew_locked(self, crew_id: str) -> None:
        if crew_id not in self._terminal_crews:
            return
        crew_state = self._crews.get(crew_id)
        if crew_state is None:
            return
        if any(task_id in self._task_handles for task_id in crew_state.task_ids):
            return
        try:
            crew_state.handle.finish()
        except RuntimeError:
            self._warn_deferred("crew", crew_id, crew_state.handle)
            return

        self._crews.pop(crew_id)
        self._terminal_crews.discard(crew_id)
        self._blocked_sources.discard(crew_id)
        self._warned_cleanup.discard(("crew", crew_id))
        for task_id in crew_state.task_ids:
            self._task_registrations.pop(task_id, None)
            self._started_tasks.discard(task_id)
            self._terminal_tasks.discard(task_id)
            self._blocked_sources.discard(task_id)
            self._warned_cleanup.discard(("task", task_id))

    def _warn_deferred(self, kind: str, source_id: str, handle: solwyn.RunHandle) -> None:
        warning_key = (kind, source_id)
        if warning_key in self._warned_cleanup:
            return
        self._warned_cleanup.add(warning_key)
        run_id = handle.run_id
        logger.warning(
            "solwyn.integrations.crewai: deferred structural scope close for %s",
            run_id,
        )

    def _warn_forced(self, task_id: str, handle: solwyn.RunHandle) -> None:
        warning_key = ("task", task_id)
        if warning_key in self._warned_cleanup:
            return
        self._warned_cleanup.add(warning_key)
        run_id = handle.run_id
        logger.warning(
            "solwyn.integrations.crewai: forced structural scope cleanup for %s",
            run_id,
        )


__all__ = ["SolwynEventListener"]
