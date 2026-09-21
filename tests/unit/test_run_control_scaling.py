"""Run-stop authority lifetimes and deterministic healthy-path work bounds."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import call_uuid

import solwyn
from solwyn import _run_control as control
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane

pytestmark = pytest.mark.unit


def _evict(run_id: str) -> None:
    for index in range(control._MAX_TERMINATED_RUNS):
        control.mark_terminated(f"eviction-{index}", reason="other", source="server")
    assert control.run_termination(run_id) is None


@pytest.mark.parametrize("source", ["server", "local_velocity"])
@pytest.mark.parametrize("release_old_first", [False, True])
def test_authority_expires_with_last_current_owner_not_last_group_owner(
    source: control.TerminationSource, release_old_first: bool
) -> None:
    old = control._acquire_termination_handle("run")
    old_stop = control.mark_terminated("run", reason="old", source="server")
    control.clear_run_termination("run")
    current = control._acquire_termination_handle("run")
    winner = control.mark_terminated("run", reason="current", source=source)
    _evict("run")
    sibling = control._acquire_termination_handle("run")
    assert sibling.termination is winner
    assert control._postcheck_termination("run") is winner
    assert old.termination is old_stop

    if release_old_first:
        old.release()
    current.release()
    current.release()  # Double cleanup cannot retire the remaining sibling.
    assert control._postcheck_termination("run") is winner
    sibling.release()
    assert control._postcheck_termination("run") is None
    fresh = control._acquire_termination_handle("run")
    assert fresh.termination is None
    assert old.termination is old_stop
    fresh.release()
    old.release()
    assert control._STATE.active_handles == {}


@pytest.mark.parametrize("source", ["server", "local_velocity"])
def test_stop_without_current_owners_does_not_gain_active_only_authority(
    source: control.TerminationSource,
) -> None:
    old = control._acquire_termination_handle("run")
    control.mark_terminated("run", reason="old", source="server")
    control.clear_run_termination("run")
    control.mark_terminated("run", reason="no_current_owner", source=source)
    _evict("run")
    assert control._postcheck_termination("run") is None
    new = control._acquire_termination_handle("run")
    assert new.termination is None
    new.release()
    old.release()
    assert control._STATE.active_handles == {}


def test_last_current_release_does_not_clear_global_authority_or_group_observation() -> None:
    old = control._acquire_termination_handle("run")
    control.clear_run_termination("run")
    current = control._acquire_termination_handle("run")
    winner = control.mark_terminated("run", reason="current", source="server")
    stamp = control.run_observed_at("run")
    current.release()
    assert control.run_termination("run") is winner
    assert control._postcheck_termination("run") is winner
    _evict("run")
    assert control._postcheck_termination("run") is None
    assert control.run_observed_at("run") == stamp
    old.release()
    assert control.run_observed_at("run") is None


def test_public_registry_reads_do_not_promote_active_only_authority() -> None:
    with solwyn.run("public-visibility") as run_id:
        handle = control._acquire_termination_handle(run_id)
        winner = control.mark_terminated(run_id, reason="stop", source="local_velocity")
        _evict(run_id)
        assert not solwyn.current_run_terminated()
        assert solwyn.run_termination(run_id) is None
        assert control._postcheck_termination(run_id) is winner
        handle.release()
    assert control._STATE.active_handles == {}


@pytest.mark.parametrize("source", ["server", "local_velocity"])
def test_source_filter_and_first_writer_after_clear_and_eviction(
    source: control.TerminationSource,
) -> None:
    other: control.TerminationSource = "local_velocity" if source == "server" else "server"
    old = control._acquire_termination_handle("run")
    control.mark_terminated("run", reason="obsolete", source=other)
    control.clear_run_termination("run")
    current = control._acquire_termination_handle("run")
    winner = control.mark_terminated("run", reason="winner", source=source)
    _evict("run")
    control.clear_termination_if("run", source=other)
    assert control._postcheck_termination("run") is winner
    assert control.mark_terminated("run", reason="loser", source=other) is winner
    _evict("run")
    control.clear_termination_if("run", source=source)
    assert control._postcheck_termination("run") is None
    assert current.termination is winner
    current.release()
    old.release()


def test_inherited_handle_release_cannot_retire_child_owner_of_same_generation() -> None:
    parent = control._acquire_termination_handle("run")
    control._STATE._reset_after_fork_in_child()
    child = control._acquire_termination_handle("run")
    winner = control.mark_terminated("run", reason="child", source="server")
    _evict("run")
    parent.release()
    assert control._postcheck_termination("run") is winner
    sibling = control._acquire_termination_handle("run")
    assert sibling.termination is winner
    child.release()
    assert control._postcheck_termination("run") is winner
    sibling.release()
    assert control._STATE.active_handles == {}


class _CountedHandles(set[control._TerminationHandle]):
    """Count handle visits, including scans moved into acquire/clear/release."""

    visits = 0

    def __iter__(self) -> Iterator[control._TerminationHandle]:
        for handle in super().__iter__():
            self.visits += 1
            yield handle


@pytest.mark.parametrize("size", [1, 32, 512])
@pytest.mark.parametrize("independent", [False, True])
def test_healthy_acquire_admission_postcheck_and_release_do_not_visit_handles(
    size: int, independent: bool
) -> None:
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
    handles = []
    counted = []
    try:
        for index in range(size):
            run_id = f"run-{index}" if independent else "run-0"
            handle = control._acquire_termination_handle(run_id)
            handles.append(handle)
            group = control._STATE.active_handles[run_id]
            if not isinstance(group.handles, _CountedHandles):
                group.handles = _CountedHandles(group.handles)
                counted.append(group.handles)
        for index in range(20):
            identity = call_uuid(f"healthy-{index}")
            result = enforcer.check_budget(
                provider="openai",
                model="gpt-5.5",
                agent_run_id="run-0",
                call_id=identity,
                estimated_input_tokens=1,
                estimated_output_bound=10,
            )
            assert result.allowed and result.lease_id is not None
            enforcer.release_reservation(identity, result.lease_claim_token)
            assert control._postcheck_termination("run-0") is None
        assert len(plane.lease_grants) == 1
        assert plane.checks == []
        for handle in handles:
            control.clear_run_termination(handle.run_id)
            handle.release()
            handle.release()
        assert control._STATE.active_handles == {}
        assert sum(group.visits for group in counted) == 0
    finally:
        for handle in handles:
            handle.release()
        enforcer.close()


def test_generation_churn_retains_only_live_handles_without_release_scans() -> None:
    old = control._acquire_termination_handle("run")
    group = control._STATE.active_handles["run"]
    counted = _CountedHandles(group.handles)
    group.handles = counted
    for _ in range(512):
        control.clear_run_termination("run")
        new = control._acquire_termination_handle("run")
        assert new.termination is None
        new.release()
        assert len(group) == 1
    old.release()
    assert control._STATE.active_handles == {}
    assert counted.visits == 0


def test_other_runs_progress_while_large_group_uses_the_production_lock() -> None:
    handles = [control._acquire_termination_handle("large") for _ in range(4096)]
    rounds = threading.Barrier(3, timeout=10)

    def exercise(run_id: str) -> None:
        for _ in range(50):
            rounds.wait()
            handle = control._acquire_termination_handle(run_id)
            assert control._postcheck_termination(run_id) is None
            handle.release()
            rounds.wait()

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(exercise, run_id) for run_id in ("large", "small-a", "small-b")]
            for task in tasks:
                task.result(timeout=20)
        assert set(control._STATE.active_handles) == {"large"}
        assert len(control._STATE.active_handles["large"]) == len(handles)
    finally:
        for handle in handles:
            handle.release()
    assert control._STATE.active_handles == {}


@pytest.mark.parametrize("size", [1, 32, 512])
def test_stopped_cohort_release_does_not_scan_or_retain_current_authority(size: int) -> None:
    old = control._acquire_termination_handle("run")
    control.clear_run_termination("run")
    handles = [control._acquire_termination_handle("run") for _ in range(size)]
    group = control._STATE.active_handles["run"]
    counted = _CountedHandles(group.handles)
    group.handles = counted
    winner = control.mark_terminated("run", reason="current", source="local_velocity")
    assert all(handle.termination is winner for handle in handles)
    assert old.termination is None
    assert counted.visits == size + 1  # One real stop latches the live cohort.
    _evict("run")
    counted.visits = 0
    for index, handle in enumerate(handles):
        assert control._postcheck_termination("run") is winner
        handle.release()
        handle.release()
        assert len(group) == size - index
    assert control._postcheck_termination("run") is None
    assert counted.visits == 0
    old.release()
    assert control._STATE.active_handles == {}
