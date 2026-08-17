"""Bounded process-wide termination state for agent runs.

The registry stores structural run identifiers, bounded reasons supplied by
callers, a source label, and monotonic timestamps only. It performs no I/O and
never handles prompt or response content.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from solwyn._lifecycle import register_fork_reset
from solwyn._run import current_run

TerminationSource: TypeAlias = Literal["server", "local_velocity"]

_MAX_TERMINATED_RUNS = 256


@dataclass(frozen=True)
class RunTermination:
    """Immutable reason that a run may not dispatch further provider calls."""

    reason: str
    source: TerminationSource
    at_monotonic: float


@dataclass(eq=False)
class _TerminationHandle:
    """Stable one-stream cell latched before bounded registry eviction."""

    run_id: str
    generation: int
    termination: RunTermination | None = None
    released: bool = False

    def release(self) -> None:
        """Drop this active watcher once its stream has a final disposition."""
        with _STATE.lock:
            if self.released:
                return
            self.released = True
            group = _STATE.active_handles.get(self.run_id)
            if group is None:
                return
            group.handles.discard(self)
            if not group.handles:
                del _STATE.active_handles[self.run_id]


@dataclass
class _ActiveHandleGroup:
    """One bounded active-run watcher group with an ordered clear epoch."""

    generation: int = 0
    observed_at: float | None = None
    handles: set[_TerminationHandle] = field(default_factory=set)

    def __iter__(self) -> Iterator[_TerminationHandle]:
        return iter(self.handles)

    def __len__(self) -> int:
        return len(self.handles)


class _RunControlState:
    """Holder whose inherited lock can be replaced without losing entries."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.terminations: OrderedDict[str, RunTermination] = OrderedDict()
        # Last control-plane observation for each first-writer record. This is
        # separate so repeated directives can order responses without changing
        # the public immutable termination or its first-writer timestamp.
        self.observed_at: OrderedDict[str, float] = OrderedDict()
        self.active_handles: dict[str, _ActiveHandleGroup] = {}

    def _reset_after_fork_in_child(self) -> None:
        # Termination is authoritative process state and survives fork. Only
        # the inherited lock is unsafe to retain in the child.
        self.lock = threading.Lock()
        # Provider streams inherited across fork are not safe to continue.
        # Detach their parent-owned watcher cells in the child; already-latched
        # immutable values remain on the wrapper itself.
        self.active_handles = {}

    def _clear_for_test_locked(self) -> None:
        """Reset all process state while tests already hold ``lock``."""
        self.terminations.clear()
        self.observed_at.clear()
        self.active_handles.clear()


_STATE = _RunControlState()
register_fork_reset(_STATE)


def _active_group_termination_locked(
    run_id: str,
    *,
    source: TerminationSource | None = None,
) -> RunTermination | None:
    """Return a winner only from the active group's current generation."""
    group = _STATE.active_handles.get(run_id)
    if group is None:
        return None
    return next(
        (
            handle.termination
            for handle in group.handles
            if handle.generation == group.generation
            and handle.termination is not None
            and (source is None or handle.termination.source == source)
        ),
        None,
    )


def _advance_active_generation_locked(run_id: str) -> None:
    """Fence obsolete sibling winners without retaining cleared run ids."""
    group = _STATE.active_handles.get(run_id)
    if group is not None:
        group.generation += 1
        group.observed_at = None


def _trim_registry_locked() -> None:
    """Keep termination and observation LRUs aligned at the fixed cap."""
    while len(_STATE.terminations) > _MAX_TERMINATED_RUNS:
        evicted_run_id, _ = _STATE.terminations.popitem(last=False)
        _STATE.observed_at.pop(evicted_run_id, None)


@contextmanager
def _locked_registry() -> Iterator[None]:
    """Serialize a registry-to-enforcer transaction in canonical lock order."""
    with _STATE.lock:
        yield


def _mark_terminated_locked(
    run_id: str,
    *,
    reason: str,
    source: TerminationSource,
) -> RunTermination:
    """First-writer mark while ``_STATE.lock`` is already held."""
    observed_at = time.monotonic()
    group = _STATE.active_handles.get(run_id)
    termination = _STATE.terminations.get(run_id)
    if termination is not None:
        _STATE.observed_at[run_id] = observed_at
    else:
        termination = _active_group_termination_locked(run_id)
        if termination is None:
            termination = RunTermination(
                reason=reason,
                source=source,
                at_monotonic=observed_at,
            )
            _STATE.terminations[run_id] = termination
            _STATE.observed_at[run_id] = observed_at
        else:
            # A fresh stop observation after global LRU eviction must restore
            # ordering authority without changing the live stream's immutable
            # first winner. Both maps remain bounded by the ordinary LRU cap.
            _STATE.terminations[run_id] = termination
            _STATE.observed_at[run_id] = observed_at
    # A bounded global entry may already be gone while one sibling still owns
    # the immutable first winner. Always latch every unlatched watcher before
    # returning; iteration order must never decide which sibling learns it.
    if group is not None:
        group.observed_at = observed_at
        for handle in tuple(group.handles):
            if handle.generation == group.generation and handle.termination is None:
                handle.termination = termination
    _trim_registry_locked()
    return termination


def _clear_server_termination_before_request_locked(
    run_id: str,
    *,
    request_epoch: float,
) -> RunTermination | None:
    """Clear a strictly older server mark or return the ambiguous/newer winner."""
    termination = _STATE.terminations.get(run_id)
    active_termination = _active_group_termination_locked(run_id, source="server")
    if termination is None:
        termination = active_termination
    if termination is None or termination.source != "server":
        return None
    group = _STATE.active_handles.get(run_id)
    observed_at = (
        group.observed_at
        if active_termination is not None and group is not None and group.observed_at is not None
        else _STATE.observed_at.get(run_id, termination.at_monotonic)
    )
    if observed_at >= request_epoch:
        # The bounded registry may have evicted this live run after its latest
        # stop. Restore the active-only winner before refreshing LRU recency;
        # moving an absent key is both incorrect and a control-path exception.
        _STATE.terminations[run_id] = termination
        _STATE.observed_at[run_id] = observed_at
        _STATE.terminations.move_to_end(run_id)
        _STATE.observed_at.move_to_end(run_id)
        _trim_registry_locked()
        return RunTermination(
            reason=termination.reason,
            source=termination.source,
            at_monotonic=observed_at,
        )
    _STATE.terminations.pop(run_id, None)
    _STATE.observed_at.pop(run_id, None)
    _advance_active_generation_locked(run_id)
    return None


def mark_terminated(
    run_id: str,
    *,
    reason: str,
    source: TerminationSource,
) -> None:
    """Preserve the first winner while recording the latest stop observation."""
    with _STATE.lock:
        _mark_terminated_locked(run_id, reason=reason, source=source)


def _acquire_termination_handle(run_id: str) -> _TerminationHandle:
    """Register one active stream and seed it from any existing stop."""
    with _STATE.lock:
        group = _STATE.active_handles.setdefault(run_id, _ActiveHandleGroup())
        termination = _STATE.terminations.get(run_id)
        if termination is None:
            termination = _active_group_termination_locked(run_id)
        if termination is not None and group.observed_at is None:
            group.observed_at = _STATE.observed_at.get(
                run_id,
                termination.at_monotonic,
            )
        handle = _TerminationHandle(
            run_id=run_id,
            generation=group.generation,
            termination=termination,
        )
        group.handles.add(handle)
        return handle


def run_termination(run_id: str) -> RunTermination | None:
    """Return the run's exact termination and refresh its LRU recency."""
    with _STATE.lock:
        termination = _STATE.terminations.get(run_id)
        if termination is not None:
            _STATE.terminations.move_to_end(run_id)
            if run_id in _STATE.observed_at:
                _STATE.observed_at.move_to_end(run_id)
        return termination


def _outage_termination_locked(run_id: str) -> RunTermination | None:
    """Return any exact or active stop while ``_STATE.lock`` is held."""
    termination = _STATE.terminations.get(run_id)
    if termination is None:
        termination = _active_group_termination_locked(run_id)
    return termination


def _postcheck_termination(run_id: str) -> RunTermination | None:
    """Return any stop that became authoritative during a live budget check."""
    with _STATE.lock:
        return _outage_termination_locked(run_id)


def clear_termination_if(run_id: str, *, source: TerminationSource) -> None:
    """Clear the run only when its first-writer source matches ``source``."""
    with _STATE.lock:
        termination = _STATE.terminations.get(run_id)
        sibling_termination = _active_group_termination_locked(run_id, source=source)
        clears_current = (termination is not None and termination.source == source) or (
            termination is None and sibling_termination is not None
        )
        if clears_current:
            _STATE.terminations.pop(run_id, None)
            _STATE.observed_at.pop(run_id, None)
            _advance_active_generation_locked(run_id)


def clear_run_termination(run_id: str) -> None:
    """Clear any termination source for ``run_id``."""
    with _STATE.lock:
        termination = _STATE.terminations.pop(run_id, None)
        sibling_termination = _active_group_termination_locked(run_id)
        _STATE.observed_at.pop(run_id, None)
        if (
            termination is not None
            or sibling_termination is not None
            or run_id in _STATE.active_handles
        ):
            _advance_active_generation_locked(run_id)


def current_run_terminated() -> bool:
    """Return whether the ambient agent-run scope is terminated."""
    run_id, _ = current_run()
    return run_id is not None and run_termination(run_id) is not None
