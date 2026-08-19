"""Content-free velocity signals for agent runs.

This module handles scalar token counts, timestamps, and structural run/model
identifiers only. It never handles prompt or response content and performs no
logging.
"""

from __future__ import annotations

import statistics
import threading
from collections import OrderedDict, deque
from typing import Literal, NamedTuple

DENY_ELIGIBLE_RULES = frozenset({"repeat_size", "monotonic_growth"})
VELOCITY_HISTORY_LIMIT = 64
_RUN_LIMIT = 128
_RATE_HORIZON_S = 120.0
_WARNING_INTERVAL_S = 30.0
_RATE_MEMORY_LIMIT = 256
_WARNING_MEMORY_LIMIT = 256


class VelocityConfig(NamedTuple):
    """Immutable detector configuration."""

    mode: Literal["off", "warn", "deny"]
    repeat_count: int
    repeat_window_s: float
    growth_streak: int
    growth_factor: float
    accel_floor_per_min: int
    accel_factor: float


class VelocityMonitor:
    """Thread-safe run velocity state with fixed detail and TTL eviction memory."""

    def __init__(self, config: VelocityConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, deque[tuple[float, int, str]]] = OrderedDict()
        # Exact entries cover the common case. If churn exhausts either fixed
        # table, one scalar deadline temporarily suppresses the corresponding
        # signal for every identity. That false-positive suppression is safe:
        # lost identity can never make us falsely emit a warning/acceleration.
        self._rate_tombstones: OrderedDict[str, float] = OrderedDict()
        self._rate_overflow_suppress_through = 0.0
        self._warning_times: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._warning_overflow_suppress_through = 0.0

    def observe(
        self,
        *,
        run_id: str,
        estimated_input_tokens: int,
        model: str,
        now: float,
    ) -> tuple[str, ...]:
        if self._config.mode == "off":
            return ()

        with self._lock:
            if now > self._rate_overflow_suppress_through:
                self._rate_overflow_suppress_through = 0.0
            history = self._runs.get(run_id)
            if history is None:
                history = deque(maxlen=VELOCITY_HISTORY_LIMIT)
                self._runs[run_id] = history
                if len(self._runs) > _RUN_LIMIT:
                    evicted_run, evicted_history = self._runs.popitem(last=False)
                    self._remember_rate_eviction(
                        evicted_run,
                        evicted_history,
                        now,
                    )
            else:
                self._runs.move_to_end(run_id)

            dropped_at = history[0][0] if len(history) == VELOCITY_HISTORY_LIMIT else None
            history.append((now, estimated_input_tokens, model))
            repeat_matches = 0
            current_minute_count = 0
            prior_minute_count = 0
            growth_count = 0
            growth_strict = True
            growth_first_size = 0
            growth_latest_size = 0
            growth_previous_time = 0.0
            growth_previous_size = 0
            growth_gaps: list[float] = []
            growth_start = max(0, len(history) - self._config.growth_streak)
            tolerance = max(8.0, 0.02 * estimated_input_tokens)

            for index, (seen_at, size, seen_model) in enumerate(history):
                age = now - seen_at
                if 0.0 <= age <= 60.0:
                    current_minute_count += 1
                elif 60.0 < age <= 120.0:
                    prior_minute_count += 1
                if (
                    0.0 <= age <= self._config.repeat_window_s
                    and seen_model == model
                    and abs(size - estimated_input_tokens) <= tolerance
                ):
                    repeat_matches += 1
                if index >= growth_start:
                    if growth_count == 0:
                        growth_first_size = size
                    else:
                        growth_gaps.append(seen_at - growth_previous_time)
                        if size <= growth_previous_size:
                            growth_strict = False
                    growth_count += 1
                    growth_latest_size = size
                    growth_previous_time = seen_at
                    growth_previous_size = size

            tombstone_deadline = self._rate_tombstones.get(run_id)
            if tombstone_deadline is not None and now > tombstone_deadline:
                del self._rate_tombstones[run_id]
                tombstone_deadline = None
            dropped_history_incomplete = (
                dropped_at is not None and 0.0 <= now - dropped_at <= _RATE_HORIZON_S
            )
            evicted_history_incomplete = (
                tombstone_deadline is not None and now <= tombstone_deadline
            )
            overflow_history_incomplete = (
                self._rate_overflow_suppress_through > 0.0
                and now <= self._rate_overflow_suppress_through
            )
            rate_window_incomplete = (
                dropped_history_incomplete
                or evicted_history_incomplete
                or overflow_history_incomplete
            )

            flags: list[str] = []
            if repeat_matches >= self._config.repeat_count:
                flags.append("repeat_size")

            if (
                growth_count == self._config.growth_streak
                and growth_strict
                and growth_latest_size >= self._config.growth_factor * growth_first_size
                and statistics.median(growth_gaps) < 30.0
            ):
                flags.append("monotonic_growth")

            if (
                not rate_window_incomplete
                and current_minute_count >= self._config.accel_floor_per_min
                and current_minute_count >= self._config.accel_factor * prior_minute_count
            ):
                flags.append("rate_acceleration")

            return tuple(flags)

    def should_warn(self, run_id: str, rule: str, now: float) -> bool:
        key = (run_id, rule)
        with self._lock:
            if now < self._warning_overflow_suppress_through:
                return False
            if self._warning_overflow_suppress_through:
                self._warning_overflow_suppress_through = 0.0
            last_warning = self._warning_times.get(key)
            if last_warning is not None:
                if now - last_warning < _WARNING_INTERVAL_S:
                    return False
                del self._warning_times[key]
            if len(self._warning_times) >= _WARNING_MEMORY_LIMIT:
                self._warning_times.clear()
                self._warning_overflow_suppress_through = now + _WARNING_INTERVAL_S
                return False
            self._warning_times[key] = now
            return True

    def run_count(self) -> int:
        with self._lock:
            return len(self._runs)

    def _reset_after_fork_in_child(self) -> None:
        self._runs.clear()
        self._rate_tombstones.clear()
        self._rate_overflow_suppress_through = 0.0
        self._warning_times.clear()
        self._warning_overflow_suppress_through = 0.0
        self._lock = threading.Lock()

    def _remember_rate_eviction(
        self,
        run_id: str,
        history: deque[tuple[float, int, str]],
        now: float,
    ) -> None:
        suppress_through = history[-1][0] + _RATE_HORIZON_S
        if now > suppress_through:
            return

        existing = self._rate_tombstones.get(run_id)
        if existing is None and len(self._rate_tombstones) >= _RATE_MEMORY_LIMIT:
            self._rate_tombstones.clear()
            self._rate_overflow_suppress_through = max(
                self._rate_overflow_suppress_through,
                now + _RATE_HORIZON_S,
                suppress_through,
            )
            return
        self._rate_tombstones[run_id] = (
            suppress_through if existing is None else max(existing, suppress_through)
        )
        self._rate_tombstones.move_to_end(run_id)
