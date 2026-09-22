"""Bounded process-wide termination state for agent runs.

The registry stores structural run identifiers, bounded reasons supplied by
callers, a source label, and monotonic timestamps only. It performs no I/O and
never handles prompt or response content.

Active streams share termination authority through per-generation epochs
rather than per-handle cells: every stream acquired since the run's last clear
reads the same ``_Epoch``, so a stop is one write and every current watcher
sees it, while a clear installs a fresh epoch and leaves the old one frozen for
the streams still holding it.
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
class _Epoch:
    """Termination authority shared by every watcher of one clear generation.

    ``termination`` is the epoch's first winner; ``owners`` counts the live
    handles that hold this epoch. Only a group's CURRENT epoch is ever written:
    a stop latches the winner while it has owners, and the last owner's release
    retires it. A clear replaces the group's epoch instead of mutating it, so a
    superseded epoch is frozen by construction — a handle that latched a stop
    keeps it, and one that never latched is never latched by a later stop.
    """

    termination: RunTermination | None = None
    owners: int = 0


@dataclass(eq=False)
class _TerminationHandle:
    """One active stream's view of its generation's shared termination cell."""

    run_id: str
    _group: _ActiveHandleGroup
    _epoch: _Epoch
    released: bool = False

    @property
    def termination(self) -> RunTermination | None:
        """Return the stop latched for this handle's generation, if any."""
        return self._epoch.termination

    def release(self) -> None:
        """Drop this active watcher once its stream has a final disposition."""
        with _STATE.lock:
            if self.released:
                return
            self.released = True
            group = _STATE.active_handles.get(self.run_id)
            # Identity is the fork fence: a pre-fork handle can outlive the
            # detached parent group, even after a child creates a replacement
            # group for the same run id. Only the group that registered this
            # handle may account for its release.
            if group is not self._group:
                return
            if group.members <= 0:
                raise RuntimeError("active handle group released more handles than it holds")
            group.members -= 1
            epoch = self._epoch
            if epoch is group.epoch:
                if epoch.owners <= 0:
                    raise RuntimeError("current epoch released more owners than it holds")
                epoch.owners -= 1
                if epoch.owners == 0:
                    # Authority retires with the last current owner; a later
                    # stream must re-seed from the bounded registry, never
                    # inherit a winner no live stream still owns.
                    epoch.termination = None
            if group.members == 0:
                del _STATE.active_handles[self.run_id]


@dataclass
class _ActiveHandleGroup:
    """Live watcher ownership for one run id, keyed on its current epoch.

    ``epoch`` is the only cell a stop or acquisition writes; ``members`` counts
    every live handle of the group across all generations so the group is
    dropped exactly when its last handle releases. Old-generation handles keep
    their superseded epochs but never extend the current winner's lifetime
    after global LRU eviction: that is owned by ``epoch.owners`` alone.
    ``observed_at`` retains the group's last observation until clear or total
    group cleanup.
    """

    epoch: _Epoch = field(default_factory=_Epoch)
    observed_at: float | None = None
    members: int = 0


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
        # Detach their parent-owned groups in the child; each inherited handle
        # keeps its parent epoch, so an already-latched stop stays readable.
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
    """Return a winner only from the active group's current clear epoch."""
    group = _STATE.active_handles.get(run_id)
    if group is None:
        return None
    termination = group.epoch.termination
    if termination is not None and (source is None or termination.source == source):
        return termination
    return None


def _advance_active_generation_locked(run_id: str) -> None:
    """Fence obsolete sibling winners without retaining cleared run ids.

    The old epoch is left with the handles that hold it: nothing writes a
    non-current epoch, so their latched stop (or absence of one) is final.
    """
    group = _STATE.active_handles.get(run_id)
    if group is not None:
        group.epoch = _Epoch()
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
) -> tuple[RunTermination, float]:
    """First-writer mark while ``_STATE.lock`` is already held.

    Returns the preserved winner AND the stamp this mark recorded. Callers
    that also file an enforcer-side sticky denial must reuse THIS value rather
    than read the clock again: the registry and the sticky are ordered against
    the same request epochs, so two stamps would let one live ALLOW clear the
    registry while the sticky survives and denies.
    """
    observed_at = time.monotonic()
    group = _STATE.active_handles.get(run_id)
    termination = _STATE.terminations.get(run_id)
    if termination is not None:
        _STATE.observed_at[run_id] = observed_at
        # A repeated stop is also a fresh reference to a live run; both maps
        # must reach the MRU end so the bounded cap evicts an older run first.
        _STATE.terminations.move_to_end(run_id)
        _STATE.observed_at.move_to_end(run_id)
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
    if group is not None:
        group.observed_at = observed_at
        # One write latches every current-generation watcher at once; the
        # shared epoch is what they read. A winner is owned only while a
        # current stream is live, so an ownerless epoch never gains one.
        epoch = group.epoch
        if epoch.owners > 0 and epoch.termination is None:
            epoch.termination = termination
    _trim_registry_locked()
    return termination, observed_at


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
) -> RunTermination:
    """Return the preserved first winner while recording the latest stop."""
    with _STATE.lock:
        termination, _observed_at = _mark_terminated_locked(
            run_id,
            reason=reason,
            source=source,
        )
        return termination


def _acquire_termination_handle(run_id: str) -> _TerminationHandle:
    """Register one active stream on the run's current epoch, seeded from any stop."""
    with _STATE.lock:
        group = _STATE.active_handles.get(run_id)
        if group is None:
            group = _ActiveHandleGroup()
            _STATE.active_handles[run_id] = group
        termination = _STATE.terminations.get(run_id)
        if termination is None:
            termination = _active_group_termination_locked(run_id)
        if termination is not None and group.observed_at is None:
            group.observed_at = _STATE.observed_at.get(
                run_id,
                termination.at_monotonic,
            )
        epoch = group.epoch
        if epoch.termination is None:
            epoch.termination = termination
        epoch.owners += 1
        group.members += 1
        return _TerminationHandle(run_id=run_id, _group=group, _epoch=epoch)


def run_termination(run_id: str) -> RunTermination | None:
    """Return the run's exact termination and refresh its LRU recency."""
    with _STATE.lock:
        termination = _STATE.terminations.get(run_id)
        if termination is not None:
            _STATE.terminations.move_to_end(run_id)
            if run_id in _STATE.observed_at:
                _STATE.observed_at.move_to_end(run_id)
        return termination


def run_observed_at(run_id: str) -> float | None:
    """Return the stamp the registry last ordered this run's stop against.

    The enforcer files its sticky denial after the registry mark returns, so a
    re-cache on that path must adopt THIS stamp instead of reading the clock a
    second time — see ``_mark_terminated_locked``.
    """
    with _STATE.lock:
        observed_at = _STATE.observed_at.get(run_id)
        if observed_at is not None:
            return observed_at
        group = _STATE.active_handles.get(run_id)
        return group.observed_at if group is not None else None


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
    """Clear any termination source for ``run_id``.

    Clearing is forward-looking only. It drops the registry entry and installs
    a fresh epoch on the active group, so every LATER call and every stream
    that has not yet latched a stop sees a live run again. A stream whose
    watcher already latched the termination keeps aborting: its handle still
    reads the superseded epoch, which no writer touches again
    (``_TerminationHandle.termination``), and ``client._stream_abort_exception``
    reads that cell, not the registry. That is deliberate — an in-flight
    stream was admitted under an authority that has since said stop, and
    re-admitting it mid-body would need spend authority no one has re-granted.
    Restart the stream to run under the cleared state.
    """
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
