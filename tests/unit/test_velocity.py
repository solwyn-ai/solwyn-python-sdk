"""Tests for content-free, run-scoped velocity detection."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import get_args

import pytest

from solwyn._types import VelocityFlag
from solwyn._velocity import (
    DENY_ELIGIBLE_RULES,
    VelocityConfig,
    VelocityMonitor,
)

_RATE_MEMORY_LIMIT = 256
_WARNING_MEMORY_LIMIT = 256
_PROJECT_ROOT = Path(__file__).parents[2]


def _config(**overrides: object) -> VelocityConfig:
    values = {
        "mode": "warn",
        "repeat_count": 5,
        "repeat_window_s": 60.0,
        "growth_streak": 8,
        "growth_factor": 3.0,
        "accel_floor_per_min": 30,
        "accel_factor": 3.0,
    }
    values.update(overrides)
    return VelocityConfig(**values)  # type: ignore[arg-type]


def _observe(
    monitor: VelocityMonitor,
    *,
    run_id: str = "run-1",
    size: int,
    model: str = "gpt-5.5",
    now: float,
) -> tuple[str, ...]:
    return monitor.observe(
        run_id=run_id,
        estimated_input_tokens=size,
        model=model,
        now=now,
    )


@pytest.mark.unit
def test_repeat_size_fires_on_fifth_near_identical_same_model_call() -> None:
    monitor = VelocityMonitor(_config())

    for now in range(4):
        assert _observe(monitor, size=1000, now=float(now)) == ()

    assert "repeat_size" in _observe(monitor, size=1005, now=4.0)


@pytest.mark.unit
def test_repeat_size_ignores_other_models_and_stale_windows() -> None:
    monitor = VelocityMonitor(_config())

    for now in range(4):
        _observe(monitor, size=1000, now=float(now), model="gpt-5.5")

    assert "repeat_size" not in _observe(monitor, size=1000, now=4.0, model="gpt-5.4")
    assert "repeat_size" not in _observe(monitor, size=1000, now=100.0, model="gpt-5.5")


@pytest.mark.unit
def test_monotonic_growth_fires_for_fast_strict_growth() -> None:
    monitor = VelocityMonitor(_config())
    sizes = [100, 220, 500, 900, 1500, 2200, 3000, 4000]

    flags = ()
    for index, size in enumerate(sizes):
        flags = _observe(monitor, size=size, now=float(index * 2))

    assert "monotonic_growth" in flags


@pytest.mark.unit
def test_monotonic_growth_ignores_slow_cadence() -> None:
    monitor = VelocityMonitor(_config())
    sizes = [100, 220, 500, 900, 1500, 2200, 3000, 4000]

    flags = ()
    for index, size in enumerate(sizes):
        flags = _observe(monitor, size=size, now=float(index * 120))

    assert "monotonic_growth" not in flags


@pytest.mark.unit
def test_rate_acceleration_compares_current_and_prior_minutes() -> None:
    monitor = VelocityMonitor(_config())

    for now in range(5):
        _observe(monitor, size=100 + now, now=float(now))

    flags = ()
    for now in range(61, 96):
        flags = _observe(monitor, size=100 + now, now=float(now))

    assert "rate_acceleration" in flags
    assert frozenset({"repeat_size", "monotonic_growth"}) == DENY_ELIGIBLE_RULES


@pytest.mark.unit
def test_rate_acceleration_uses_complete_windows_after_detailed_history_wraps() -> None:
    monitor = VelocityMonitor(_config())

    for _ in range(100):
        _observe(monitor, size=1000, now=0.0)

    flags = ()
    for _ in range(48):
        flags = _observe(monitor, size=1000, now=61.0)

    assert len(monitor._runs["run-1"]) == 64
    assert "rate_acceleration" not in flags


@pytest.mark.unit
def test_rate_acceleration_includes_age_sixty_in_current_window() -> None:
    monitor = VelocityMonitor(_config(accel_floor_per_min=2, accel_factor=2.0))

    _observe(monitor, size=100, now=40.0)

    flags = _observe(monitor, size=200, now=100.0)

    assert "rate_acceleration" in flags


@pytest.mark.unit
def test_rate_acceleration_puts_just_over_sixty_in_prior_window() -> None:
    monitor = VelocityMonitor(_config(accel_floor_per_min=2, accel_factor=2.0))

    for index, now in enumerate((0.0, 39.999, 40.0)):
        _observe(monitor, size=100 + index, now=now)

    flags = _observe(monitor, size=200, now=100.0)

    assert "rate_acceleration" not in flags


@pytest.mark.unit
def test_rate_acceleration_includes_age_one_twenty_in_prior_window() -> None:
    monitor = VelocityMonitor(_config(accel_floor_per_min=2, accel_factor=2.0))

    for index, now in enumerate((0.0, 20.0, 110.0)):
        _observe(monitor, size=100 + index, now=now)

    flags = _observe(monitor, size=200, now=120.0)

    assert "rate_acceleration" not in flags


@pytest.mark.unit
def test_rate_acceleration_excludes_ages_over_one_twenty() -> None:
    monitor = VelocityMonitor(_config(accel_floor_per_min=2, accel_factor=2.0))

    for index, now in enumerate((0.0, 20.0, 110.0)):
        _observe(monitor, size=100 + index, now=now)

    flags = _observe(monitor, size=200, now=120.001)

    assert "rate_acceleration" in flags


@pytest.mark.unit
def test_rate_acceleration_evaluates_when_dropped_history_is_stale() -> None:
    monitor = VelocityMonitor(_config())

    for _ in range(64):
        _observe(monitor, size=1000, now=0.0)

    flags = ()
    for _ in range(30):
        flags = _observe(monitor, size=1000, now=200.0)

    assert len(monitor._runs["run-1"]) == 64
    assert "rate_acceleration" in flags


@pytest.mark.unit
def test_rate_acceleration_suppresses_returning_run_after_lru_eviction() -> None:
    monitor = VelocityMonitor(_config())
    for _ in range(64):
        _observe(monitor, run_id="a", size=1000, now=0.0)
    for index in range(128):
        _observe(monitor, run_id=f"other-{index}", size=index, now=61.0)

    flags = ()
    for _ in range(30):
        flags = _observe(monitor, run_id="a", size=1000, now=61.0)

    assert "rate_acceleration" not in flags


@pytest.mark.unit
def test_evicted_rate_horizon_expires_only_after_age_one_twenty() -> None:
    monitor = VelocityMonitor(_config())
    for _ in range(64):
        _observe(monitor, run_id="a", size=1000, now=0.0)
    for index in range(128):
        _observe(monitor, run_id=f"other-{index}", size=index, now=61.0)

    flags = ()
    for _ in range(30):
        flags = _observe(monitor, run_id="a", size=1000, now=120.0)

    assert "rate_acceleration" not in flags
    assert "rate_acceleration" in _observe(
        monitor,
        run_id="a",
        size=1000,
        now=120.001,
    )


@pytest.mark.unit
def test_tombstone_overflow_conservatively_suppresses_never_seen_run() -> None:
    monitor = VelocityMonitor(_config())
    for index in range(128):
        _observe(monitor, run_id=f"initial-{index}", size=index, now=0.0)
    for index in range(256):
        _observe(monitor, run_id=f"churn-{index}", size=index, now=1.0)

    flags = ()
    for _ in range(30):
        flags = _observe(monitor, run_id="never-seen", size=1000, now=1.0)

    assert "rate_acceleration" not in flags
    assert len(monitor._rate_tombstones) <= _RATE_MEMORY_LIMIT
    assert monitor._rate_overflow_suppress_through >= 121.0


@pytest.mark.unit
def test_returning_tombstone_survives_unrelated_churn() -> None:
    monitor = VelocityMonitor(_config())
    for _ in range(64):
        _observe(monitor, run_id="victim", size=1000, now=0.0)
    for index in range(128):
        _observe(monitor, run_id=f"initial-{index}", size=index, now=1.0)
    for index in range(256):
        _observe(monitor, run_id=f"churn-{index}", size=index, now=1.0)

    assert len(monitor._rate_tombstones) <= _RATE_MEMORY_LIMIT

    flags = ()
    for _ in range(30):
        flags = _observe(monitor, run_id="victim", size=1000, now=1.0)

    assert "rate_acceleration" not in flags


@pytest.mark.unit
def test_runs_are_isolated_and_lru_bounded() -> None:
    monitor = VelocityMonitor(_config())

    for now in range(4):
        _observe(monitor, run_id="old-run", size=1000, now=float(now))
    assert "repeat_size" not in _observe(
        monitor,
        run_id="isolated-run",
        size=1000,
        now=4.0,
    )
    for index in range(129):
        _observe(monitor, run_id=f"run-{index}", size=index, now=4.0)

    assert monitor.run_count() <= 128
    # Still inside the repeat window: retained history would make this the fifth
    # matching call and fire. No flag therefore proves the oldest run was evicted.
    assert "repeat_size" not in _observe(monitor, run_id="old-run", size=1000, now=5.0)


@pytest.mark.unit
def test_should_warn_rate_limits_each_run_and_rule_pair() -> None:
    monitor = VelocityMonitor(_config())

    assert monitor.should_warn("run-1", "repeat_size", 0.0) is True
    assert monitor.should_warn("run-1", "repeat_size", 10.0) is False
    assert monitor.should_warn("run-1", "monotonic_growth", 10.0) is True
    assert monitor.should_warn("run-1", "repeat_size", 31.0) is True


@pytest.mark.unit
def test_run_eviction_does_not_clear_an_active_warning_cooldown() -> None:
    monitor = VelocityMonitor(_config())
    _observe(monitor, run_id="a", size=1000, now=0.0)
    assert monitor.should_warn("a", "repeat_size", 0.1) is True

    for index in range(128):
        _observe(monitor, run_id=f"other-{index}", size=index, now=0.2)

    assert monitor.run_count() == 128
    assert monitor.should_warn("a", "repeat_size", 0.4) is False


@pytest.mark.unit
def test_warning_churn_is_fixed_memory_and_fails_closed() -> None:
    monitor = VelocityMonitor(_config())
    for index in range(_WARNING_MEMORY_LIMIT):
        assert monitor.should_warn(f"run-{index}", "repeat_size", 0.0) is True

    assert monitor.should_warn("unrelated", "repeat_size", 0.1) is False
    assert len(monitor._warning_times) <= _WARNING_MEMORY_LIMIT
    assert monitor._warning_overflow_suppress_through == 30.1


@pytest.mark.unit
def test_warning_overflow_suppression_expires_at_exact_boundary() -> None:
    monitor = VelocityMonitor(_config())
    for index in range(_WARNING_MEMORY_LIMIT):
        assert monitor.should_warn(f"run-{index}", "repeat_size", 0.0)
    assert not monitor.should_warn("overflow", "repeat_size", 0.25)

    assert not monitor.should_warn("fresh", "repeat_size", 30.249)
    assert monitor.should_warn("fresh", "repeat_size", 30.25)


@pytest.mark.unit
def test_ten_thousand_unique_runs_keep_all_scalar_state_fixed_and_fast() -> None:
    monitor = VelocityMonitor(_config())

    started = time.perf_counter()
    for index in range(10_000):
        now = float(index) / 1000.0
        _observe(monitor, run_id=f"run-{index}", size=index, now=now)
        monitor.should_warn(f"run-{index}", "repeat_size", now)
    elapsed = time.perf_counter() - started

    assert monitor.run_count() == 128
    assert len(monitor._rate_tombstones) <= _RATE_MEMORY_LIMIT
    assert len(monitor._warning_times) <= _WARNING_MEMORY_LIMIT
    # The detector used to scan ever-growing maps on every observation. This
    # generous ceiling catches that quadratic shape without timing micro-ops.
    assert elapsed < 2.0


@pytest.mark.unit
def test_warning_cooldown_expires_at_exactly_thirty_seconds() -> None:
    monitor = VelocityMonitor(_config())

    assert monitor.should_warn("run-1", "repeat_size", 0.0) is True
    assert monitor.should_warn("run-1", "repeat_size", 29.999) is False
    assert monitor.should_warn("run-1", "repeat_size", 30.0) is True


@pytest.mark.unit
def test_fork_reset_clears_state_and_replaces_lock() -> None:
    monitor = VelocityMonitor(_config())
    _observe(monitor, size=1000, now=0.0)
    assert monitor.should_warn("run-1", "repeat_size", 0.0) is True
    monitor._rate_tombstones["evicted-run"] = 120.0
    old_lock = monitor._lock

    monitor._reset_after_fork_in_child()

    assert monitor.run_count() == 0
    assert monitor._rate_tombstones == {}
    assert monitor._rate_overflow_suppress_through == 0.0
    assert monitor._warning_overflow_suppress_through == 0.0
    assert monitor._lock is not old_lock
    assert monitor.should_warn("run-1", "repeat_size", 0.0) is True


@pytest.mark.unit
def test_public_docs_enumerate_all_velocity_settings_and_privacy_contract() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for name in (
        "velocity_mode",
        "velocity_repeat_count",
        "velocity_repeat_window_s",
        "velocity_growth_streak",
        "velocity_growth_factor",
        "velocity_accel_floor_per_min",
        "velocity_accel_factor",
    ):
        assert f"`{name}`" in readme
        assert f"`SOLWYN_{name.upper()}`" in readme
    assert "`repeat_size` and\n`monotonic_growth` are eligible" in readme
    assert "`rate_acceleration` is advisory only" in readme
    assert "never prompts or responses" in readme


@pytest.mark.unit
def test_emitted_rule_names_stay_inside_the_wire_flag_literal() -> None:
    """Every rule name the monitor can emit must be a `VelocityFlag` member.

    The success settlement path builds `MetadataEvent` with the flags the
    monitor emitted, AFTER the provider call was paid — a name outside the wire
    literal would raise ValidationError there and crash the paid call. The
    emitted vocabulary is pinned as a reviewed literal: extending the monitor
    requires extending `VelocityFlag` in the same change.
    """
    source = (_PROJECT_ROOT / "src" / "solwyn" / "_velocity.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'flags\.append\("([^"]+)"\)', source))
    assert emitted == {"repeat_size", "monotonic_growth", "rate_acceleration"}
    wire = set(get_args(VelocityFlag))
    assert emitted <= wire
    assert set(DENY_ELIGIBLE_RULES) <= wire
