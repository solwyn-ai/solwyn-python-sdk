"""Process-wide agent-run termination registry tests."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

import solwyn

_PROJECT_ROOT = Path(__file__).parents[2]


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    from solwyn import _run_control

    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()
    yield
    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()


@pytest.mark.unit
def test_mark_read_is_frozen_and_first_writer_wins() -> None:
    from solwyn._run_control import mark_terminated, run_termination

    with patch("solwyn._run_control.time.monotonic", return_value=12.5):
        mark_terminated("run_a", reason="velocity:repeat_size", source="local_velocity")
    mark_terminated("run_a", reason="run_stopped", source="server")

    termination = run_termination("run_a")

    assert termination is not None
    assert termination.reason == "velocity:repeat_size"
    assert termination.source == "local_velocity"
    assert termination.at_monotonic == 12.5
    with pytest.raises(FrozenInstanceError):
        termination.reason = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_mark_returns_first_winner_for_the_initial_and_repeated_stop() -> None:
    from solwyn._run_control import mark_terminated, run_termination

    with patch("solwyn._run_control.time.monotonic", return_value=7.5):
        first = mark_terminated("run_a", reason="velocity:repeat_size", source="local_velocity")

    assert first.reason == "velocity:repeat_size"
    assert first.source == "local_velocity"
    assert first.at_monotonic == 7.5
    assert run_termination("run_a") is first

    repeated = mark_terminated("run_a", reason="run_stopped", source="server")

    assert repeated is first


@pytest.mark.unit
def test_mark_after_eviction_returns_the_live_sibling_first_winner() -> None:
    from solwyn._run_control import _acquire_termination_handle, mark_terminated, run_termination

    handle = _acquire_termination_handle("run_stream")
    first = mark_terminated("run_stream", reason="first_winner", source="server")
    for index in range(257):
        mark_terminated(f"other_{index}", reason="later", source="server")
    assert run_termination("run_stream") is None

    restored = mark_terminated("run_stream", reason="later_loser", source="local_velocity")

    assert restored is first
    assert handle.termination is first
    handle.release()


@pytest.mark.unit
def test_clear_termination_if_matches_source_and_public_clear_removes_any_source() -> None:
    from solwyn._run_control import (
        clear_run_termination,
        clear_termination_if,
        mark_terminated,
        run_termination,
    )

    mark_terminated("run_local", reason="velocity:repeat_size", source="local_velocity")
    mark_terminated("run_server", reason="run_stopped", source="server")

    clear_termination_if("run_local", source="server")
    clear_termination_if("run_server", source="server")

    assert run_termination("run_local") is not None
    assert run_termination("run_server") is None

    clear_run_termination("run_local")

    assert run_termination("run_local") is None


@pytest.mark.unit
def test_registry_evicts_least_recently_read_entry() -> None:
    from solwyn._run_control import mark_terminated, run_termination

    for index in range(256):
        mark_terminated(f"run_{index}", reason="run_stopped", source="server")

    assert run_termination("run_0") is not None
    mark_terminated("run_256", reason="run_stopped", source="server")

    assert run_termination("run_0") is not None
    assert run_termination("run_1") is None
    assert run_termination("run_256") is not None


@pytest.mark.unit
def test_repeated_mark_refreshes_recency_so_an_older_run_is_evicted_first() -> None:
    from solwyn import _run_control
    from solwyn._run_control import mark_terminated

    for index in range(256):
        mark_terminated(f"run_{index}", reason="run_stopped", source="server")

    mark_terminated("run_0", reason="run_stopped", source="server")
    mark_terminated("run_256", reason="run_stopped", source="server")

    assert "run_0" in _run_control._STATE.terminations
    assert "run_0" in _run_control._STATE.observed_at
    assert "run_1" not in _run_control._STATE.terminations
    assert "run_1" not in _run_control._STATE.observed_at


@pytest.mark.unit
def test_evicted_local_stop_is_forgotten_without_stopping_a_new_lifecycle() -> None:
    from solwyn import _run_control
    from solwyn._run_control import mark_terminated, run_termination

    mark_terminated("forgotten-local", reason="velocity:repeat_size", source="local_velocity")
    for index in range(257):
        mark_terminated(
            f"local-churn-{index}",
            reason="velocity:repeat_size",
            source="local_velocity",
        )

    assert "forgotten-local" not in _run_control._STATE.terminations
    assert run_termination("forgotten-local") is None
    assert len(_run_control._STATE.terminations) == 256

    mark_terminated(
        "forgotten-local",
        reason="velocity:monotonic_growth",
        source="local_velocity",
    )
    restarted = run_termination("forgotten-local")
    assert restarted is not None
    assert restarted.reason == "velocity:monotonic_growth"


@pytest.mark.unit
def test_evicted_server_stop_obeys_ordered_allow_without_an_active_handle() -> None:
    from solwyn import _run_control
    from solwyn._run_control import mark_terminated

    with patch("solwyn._run_control.time.monotonic", return_value=1.0):
        mark_terminated("forgotten-server", reason="manual_kill", source="server")
        for index in range(257):
            mark_terminated(f"server-churn-{index}", reason="later", source="server")

    with _run_control._locked_registry():
        newer_stop = _run_control._clear_server_termination_before_request_locked(
            "forgotten-server",
            request_epoch=2.0,
        )
        outage_stop = _run_control._outage_termination_locked("forgotten-server")

    assert newer_stop is None
    assert outage_stop is None

    mark_terminated("forgotten-server", reason="new_lifecycle_stop", source="server")
    restarted = _run_control.run_termination("forgotten-server")
    assert restarted is not None
    assert restarted.reason == "new_lifecycle_stop"


@pytest.mark.unit
def test_evicted_server_stop_has_no_stale_match_without_active_handle() -> None:
    from solwyn import _run_control
    from solwyn._run_control import mark_terminated

    with patch("solwyn._run_control.time.monotonic", return_value=1.0):
        mark_terminated("forgotten-server", reason="first_stop", source="server")
    with patch("solwyn._run_control.time.monotonic", return_value=3.0):
        mark_terminated("forgotten-server", reason="repeated_stop", source="server")
        for index in range(257):
            mark_terminated(f"server-churn-{index}", reason="later", source="server")

    with _run_control._locked_registry():
        newer_stop = _run_control._clear_server_termination_before_request_locked(
            "forgotten-server",
            request_epoch=2.0,
        )

    assert newer_stop is None


@pytest.mark.unit
def test_ten_thousand_stops_keep_exact_state_fixed_fast_and_false_stop_free() -> None:
    from solwyn import _run_control
    from solwyn._run_control import mark_terminated, run_termination

    victim = "private-structural-run-id"
    mark_terminated(victim, reason="velocity:repeat_size", source="local_velocity")
    started = time.perf_counter()
    for index in range(10_000):
        mark_terminated(
            f"bounded-stop-{index}",
            reason="velocity:repeat_size",
            source="local_velocity",
        )
    elapsed = time.perf_counter() - started

    assert len(_run_control._STATE.terminations) == 256
    assert len(_run_control._STATE.observed_at) == 256
    assert not hasattr(_run_control._STATE, "tombstone_bits")
    assert run_termination(victim) is None
    assert all(run_termination(f"never-stopped-{index}") is None for index in range(10_000))
    assert elapsed < 2.0


@pytest.mark.unit
def test_clearing_an_evicted_id_never_creates_future_stop_state() -> None:
    from solwyn._run_control import clear_run_termination, mark_terminated, run_termination

    for index in range(600):
        mark_terminated(
            f"clear-churn-{index}",
            reason="velocity:repeat_size",
            source="local_velocity",
        )
    clear_run_termination("clear-churn-0")

    assert run_termination("clear-churn-0") is None
    assert run_termination("never-stopped-after-clear") is None


@pytest.mark.unit
def test_registry_churn_never_stops_an_unseen_run_id() -> None:
    from solwyn._run_control import mark_terminated, run_termination

    for index in range(10_000):
        mark_terminated(
            f"stopped-local-{index}",
            reason="velocity:repeat_size",
            source="local_velocity",
        )

    false_stops = [
        run_id
        for index in range(10_000)
        if run_termination(run_id := f"never-stopped-{index}") is not None
    ]

    assert false_stops == []


@pytest.mark.unit
def test_handle_acquired_after_registry_eviction_inherits_live_sibling_winner() -> None:
    from solwyn._run_control import (
        _acquire_termination_handle,
        mark_terminated,
        run_termination,
    )

    first = _acquire_termination_handle("run_stream")
    mark_terminated("run_stream", reason="first_winner", source="server")
    assert first.termination is not None
    for index in range(257):
        mark_terminated(f"other_{index}", reason="later", source="server")
    assert run_termination("run_stream") is None

    second = _acquire_termination_handle("run_stream")

    assert second.termination is first.termination
    second.release()
    first.release()


@pytest.mark.unit
def test_repeated_mark_latches_every_unlatched_sibling_to_first_winner() -> None:
    from solwyn._run_control import (
        RunTermination,
        _acquire_termination_handle,
        mark_terminated,
    )

    first = _acquire_termination_handle("run_stream")
    second = _acquire_termination_handle("run_stream")
    winner = RunTermination(reason="first_winner", source="server", at_monotonic=4.5)
    first.termination = winner

    mark_terminated("run_stream", reason="later_loser", source="local_velocity")

    assert first.termination is winner
    assert second.termination is winner
    second.release()
    first.release()


@pytest.mark.unit
def test_public_clear_starts_a_clean_active_handle_generation() -> None:
    from solwyn._run_control import (
        _acquire_termination_handle,
        clear_run_termination,
        mark_terminated,
    )

    old = _acquire_termination_handle("run_stream")
    mark_terminated("run_stream", reason="old_stop", source="server")
    old_winner = old.termination

    clear_run_termination("run_stream")
    new = _acquire_termination_handle("run_stream")

    assert old.termination is old_winner
    assert new.termination is None

    mark_terminated("run_stream", reason="new_stop", source="server")

    assert old.termination is old_winner
    assert new.termination is not None
    assert new.termination.reason == "new_stop"
    new.release()
    old.release()


@pytest.mark.unit
def test_matching_source_clear_excludes_obsolete_active_siblings() -> None:
    from solwyn._run_control import (
        _acquire_termination_handle,
        clear_termination_if,
        mark_terminated,
        run_termination,
    )

    old = _acquire_termination_handle("run_stream")
    mark_terminated("run_stream", reason="old_stop", source="server")

    clear_termination_if("run_stream", source="local_velocity")
    still_stopped = _acquire_termination_handle("run_stream")
    assert still_stopped.termination is old.termination
    still_stopped.release()
    for index in range(257):
        mark_terminated(f"other_{index}", reason="later", source="server")
    assert run_termination("run_stream") is None

    clear_termination_if("run_stream", source="server")
    clean = _acquire_termination_handle("run_stream")

    assert clean.termination is None
    clean.release()
    old.release()


@pytest.mark.unit
def test_current_run_terminated_uses_ambient_run_scope() -> None:
    from solwyn._run_control import current_run_terminated, mark_terminated

    assert current_run_terminated() is False

    with solwyn.run("registry-test") as run_id:
        assert current_run_terminated() is False
        mark_terminated(run_id, reason="run_stopped", source="server")
        assert current_run_terminated() is True

    assert current_run_terminated() is False


@pytest.mark.unit
def test_fork_reset_preserves_entries_and_replaces_lock() -> None:
    from solwyn import _lifecycle, _run_control

    _run_control.mark_terminated("run_a", reason="run_stopped", source="server")
    handle = _run_control._acquire_termination_handle("run_a")
    old_lock = _run_control._STATE.lock

    assert _run_control._STATE in _lifecycle._FORK_RESETTABLE
    _run_control._STATE._reset_after_fork_in_child()

    assert _run_control.run_termination("run_a") is not None
    assert _run_control._STATE.lock is not old_lock
    assert _run_control._STATE.active_handles == {}
    assert handle.termination is not None
    assert handle.termination.reason == "run_stopped"
    handle.release()


@pytest.mark.unit
def test_fork_reset_does_not_restore_evicted_exact_entries() -> None:
    from solwyn import _run_control

    _run_control.mark_terminated(
        "fork-retained-local",
        reason="velocity:repeat_size",
        source="local_velocity",
    )
    for index in range(257):
        _run_control.mark_terminated(
            f"fork-churn-{index}",
            reason="velocity:repeat_size",
            source="local_velocity",
        )
    old_lock = _run_control._STATE.lock

    _run_control._STATE._reset_after_fork_in_child()

    assert _run_control._STATE.lock is not old_lock
    assert _run_control.run_termination("fork-retained-local") is None
    assert _run_control.run_termination("fork-churn-256") is not None


@pytest.mark.unit
def test_run_control_public_exports_and_no_run_terminated_error_alias() -> None:
    from solwyn._run_control import RunTermination

    assert solwyn.RunTermination is RunTermination
    assert solwyn.run_termination is not None
    assert solwyn.clear_run_termination is not None
    assert solwyn.current_run_terminated is not None
    assert not hasattr(solwyn, "RunTerminatedError")
    assert not hasattr(solwyn.exceptions, "RunTerminatedError")


@pytest.mark.unit
def test_public_docs_describe_run_stop_hierarchy_and_cooperative_api() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    error_handling = readme.split("## Error Handling", 1)[1].split("## Data Transparency", 1)[0]
    normalized_error_handling = " ".join(error_handling.split())

    changelog = (_PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    normalized_unreleased = " ".join(unreleased.split())

    for documented_contract in (normalized_error_handling, normalized_unreleased):
        assert "inherits directly from `SolwynError`, not `BudgetExceededError`" in (
            documented_contract
        )
        assert "`agent_run_id`, `reason`, and `source`" in documented_contract
        assert "a `BudgetExceededError` subclass" not in documented_contract
        assert "budget snapshot" not in documented_contract

    for public_api in (
        "`current_run_terminated()`",
        "`run_termination(run_id)`",
        "`clear_run_termination(run_id)`",
        "`RunTermination`",
    ):
        assert public_api in normalized_error_handling

    for bounded_registry_contract in (
        "256-entry LRU",
        "never guesses from fingerprints",
        "may be forgotten after LRU eviction",
    ):
        assert bounded_registry_contract in normalized_error_handling
        assert bounded_registry_contract in normalized_unreleased

    for stream_contract in (
        "Active stream handles retain",
        "next raw provider-chunk boundary",
        "pulled and discarded",
        "partial success",
        "exactly once",
    ):
        assert stream_contract in normalized_error_handling
        assert stream_contract in normalized_unreleased
