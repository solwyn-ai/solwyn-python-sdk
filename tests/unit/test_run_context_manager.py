"""Tests for ``solwyn.run("name")`` context manager.

The context manager binds an active agent_run_id + agent_run_name to a
contextvar so the metadata-event builder can tag cost events with the
current run. Outside the scope, both fields must be None so the API's
server-side auto-fallback engages.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

import solwyn
from solwyn import _run
from solwyn._constants import AGENT_RUN_ID_MAX_LENGTH
from solwyn._run import current_run


@pytest.mark.unit
class TestRunIdGenerator:
    """The id generator must produce stable, prefixed, unique ids."""

    def test_private_generator_exists(self) -> None:
        assert hasattr(_run, "_new_run_id")

    def test_starts_with_run_prefix(self) -> None:
        assert _run._new_run_id().startswith("run_")

    def test_ids_are_unique(self) -> None:
        ids = {_run._new_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_ids_use_uuid_format(self) -> None:
        run_id = _run._new_run_id()
        assert len(run_id) == len("run_00000000-0000-0000-0000-000000000000")
        assert run_id[12] == "-"

    def test_id_fits_wire_max_length(self) -> None:
        # The id must comfortably fit the wire field cap.
        assert len(_run._new_run_id()) <= AGENT_RUN_ID_MAX_LENGTH


@pytest.mark.unit
class TestRunContextManagerSync:
    """Synchronous ``with solwyn.run("name")`` behavior."""

    def test_yields_stable_run_id(self) -> None:
        with solwyn.run("foo") as run_id:
            # Same id throughout the scope.
            assert run_id.startswith("run_")
            assert current_run() == (run_id, "foo")

    def test_outside_scope_returns_none(self) -> None:
        # Before any scope is entered, no run is active.
        assert current_run() == (None, None)

    def test_exit_clears_active_run(self) -> None:
        with solwyn.run("foo"):
            pass
        assert current_run() == (None, None)

    def test_sequential_scopes_have_different_ids(self) -> None:
        # Each entry generates a fresh id — even when the name repeats.
        with solwyn.run("foo") as a:
            pass
        with solwyn.run("foo") as b:
            pass
        assert a != b

    def test_nested_inner_replaces_outer(self) -> None:
        # Documented behavior: inner replaces outer for its duration.
        # Matches OpenTelemetry span semantics.
        with solwyn.run("outer") as outer_id:
            assert current_run() == (outer_id, "outer")
            with solwyn.run("inner") as inner_id:
                assert inner_id != outer_id
                assert current_run() == (inner_id, "inner")
            # Outer is restored after inner exits.
            assert current_run() == (outer_id, "outer")
        assert current_run() == (None, None)

    def test_scope_tags_use_private_snapshot_without_changing_public_tuple(self) -> None:
        with solwyn.run("tagged", tags={"team": "research"}) as run_id:
            assert current_run() == (run_id, "tagged")
            assert _run._capture_run_context() == (
                run_id,
                "tagged",
                {"team": "research"},
            )

    def test_nested_scope_restores_outer_tags(self) -> None:
        with solwyn.run("outer", tags={"scope": "outer"}) as outer_id:
            with solwyn.run("inner", tags={"scope": "inner"}) as inner_id:
                assert _run._capture_run_context() == (
                    inner_id,
                    "inner",
                    {"scope": "inner"},
                )
            assert _run._capture_run_context() == (
                outer_id,
                "outer",
                {"scope": "outer"},
            )

    def test_scope_copies_caller_mapping_and_private_snapshot(self) -> None:
        tags = {"team": "research"}
        scope = solwyn.run("copied", tags=tags)
        tags["team"] = "mutated-before-entry"

        with scope:
            captured = _run._capture_run_context()
            assert captured[2] == {"team": "research"}
            assert captured[2] is not None
            captured[2]["team"] = "mutated-snapshot"
            assert _run._capture_run_context()[2] == {"team": "research"}

    @pytest.mark.parametrize(
        "tags",
        [
            {f"key-{index}": "value" for index in range(11)},
            {"": "value"},
            {"k" * 65: "value"},
            {"key": "v" * 257},
            {1: "value"},
            {"key": 1},
        ],
    )
    def test_invalid_scope_tags_fail_before_entry(self, tags: dict[object, object]) -> None:
        with pytest.raises((TypeError, ValueError)):
            solwyn.run("invalid", tags=tags)  # type: ignore[arg-type]

    def test_private_snapshot_shallow_merges_per_call_tags(self) -> None:
        with solwyn.run("merged", tags={"team": "platform", "env": "prod"}):
            assert _run._capture_run_context({"env": "stage", "job": "batch"})[2] == {
                "team": "platform",
                "env": "stage",
                "job": "batch",
            }
            assert _run._capture_run_context(None)[2] == {
                "team": "platform",
                "env": "prod",
            }

    @pytest.mark.parametrize(
        ("client_tags", "scope_tags", "call_tags", "expected"),
        [
            (None, None, None, None),
            (
                {"shared": "client", "client": "only"},
                None,
                None,
                {"shared": "client", "client": "only"},
            ),
            (
                {"shared": "client", "client": "only"},
                {"shared": "scope", "scope": "only"},
                None,
                {"shared": "scope", "client": "only", "scope": "only"},
            ),
            (
                {"shared": "client", "client": "only"},
                {"shared": "scope", "scope": "only"},
                {"shared": "call", "call": "only"},
                {
                    "shared": "call",
                    "client": "only",
                    "scope": "only",
                    "call": "only",
                },
            ),
            ({"empty": ""}, None, None, {"empty": ""}),
        ],
    )
    def test_private_snapshot_uses_client_scope_call_precedence(
        self,
        client_tags: dict[str, str] | None,
        scope_tags: dict[str, str] | None,
        call_tags: dict[str, str] | None,
        expected: dict[str, str] | None,
    ) -> None:
        if scope_tags is None:
            captured = _run._capture_run_context(call_tags, default_tags=client_tags)
        else:
            with solwyn.run("precedence", tags=scope_tags):
                captured = _run._capture_run_context(call_tags, default_tags=client_tags)

        assert captured[2] == expected

    def test_empty_merged_mapping_is_absent(self) -> None:
        assert _run._capture_run_context({}) == (None, None, None)

    def test_per_call_only_mapping_is_copied(self) -> None:
        tags = {"job": "batch"}
        captured = _run._capture_run_context(tags)
        tags["job"] = "mutated"

        assert captured == (None, None, {"job": "batch"})

    def test_exception_propagates_and_resets_state(self) -> None:
        with pytest.raises(RuntimeError, match="boom"), solwyn.run("foo"):
            raise RuntimeError("boom")
        # State must be reset even on exception — no leaked run.
        assert current_run() == (None, None)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"), solwyn.run(""):
            pass

    def test_whitespace_only_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"), solwyn.run("   "):
            pass

    def test_name_max_length_enforced(self) -> None:
        # Wire field cap is 255; reject longer names eagerly so callers
        # find out at scope entry rather than via wire validation later.
        with pytest.raises(ValueError, match="max length"), solwyn.run("x" * 256):
            pass

    @pytest.mark.parametrize("name", ["nightly\nbatch", "nightly\x00batch", "nightly\x7fbatch"])
    def test_control_chars_in_name_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="control characters"), solwyn.run(name):
            pass

    @pytest.mark.parametrize(
        "name",
        [
            "nightly\u0085batch",
            "nightly\u2028batch",
            "nightly\u2029batch",
            "nightly\u200bbatch",
            "nightly\u202ebatch",
        ],
    )
    def test_unicode_control_and_format_chars_in_name_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="control characters"):
            solwyn.run(name)

    def test_non_string_name_rejected(self) -> None:
        class DuckName:
            def strip(self) -> str:
                return "ok"

            def __iter__(self):
                return iter("ok")

            def __len__(self) -> int:
                return 2

        with pytest.raises(TypeError, match="requires str"):
            solwyn.run(DuckName())  # type: ignore[arg-type]


@pytest.mark.unit
class TestRunContextManagerAsync:
    """Async ``async with solwyn.run("name")`` behavior.

    The contextvar must isolate concurrent asyncio tasks — each task sees
    only its own run, never the other's.
    """

    @pytest.mark.asyncio
    async def test_async_with_yields_run_id(self) -> None:
        async with solwyn.run("foo") as run_id:
            assert run_id.startswith("run_")
            assert current_run() == (run_id, "foo")
        assert current_run() == (None, None)

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_independent_runs(self) -> None:
        # Two tasks enter their own scopes simultaneously. The contextvar
        # must isolate them — task A must never see task B's run id and
        # vice versa.
        a_seen: list[tuple[str | None, str | None]] = []
        b_seen: list[tuple[str | None, str | None]] = []

        async def task_a() -> str:
            async with solwyn.run("task-a") as run_id:
                a_seen.append(current_run())
                # Yield to the scheduler so task_b runs in between.
                await asyncio.sleep(0)
                a_seen.append(current_run())
                return run_id

        async def task_b() -> str:
            async with solwyn.run("task-b") as run_id:
                b_seen.append(current_run())
                await asyncio.sleep(0)
                b_seen.append(current_run())
                return run_id

        a_id, b_id = await asyncio.gather(task_a(), task_b())

        # Each task observed only its own run across the await boundary.
        assert a_id != b_id
        assert all(seen == (a_id, "task-a") for seen in a_seen)
        assert all(seen == (b_id, "task-b") for seen in b_seen)

    @pytest.mark.asyncio
    async def test_concurrent_tasks_have_independent_tags(self) -> None:
        async def capture(label: str) -> tuple[str, dict[str, str] | None]:
            async with solwyn.run(label, tags={"task": label}) as run_id:
                await asyncio.sleep(0)
                return run_id, _run._capture_run_context()[2]

        first, second = await asyncio.gather(capture("first"), capture("second"))

        assert first[0] != second[0]
        assert first[1] == {"task": "first"}
        assert second[1] == {"task": "second"}


@pytest.mark.unit
class TestRunScopeFailureModes:
    """Adversarial scenarios for scope reuse and abandoned async cleanup."""

    @pytest.mark.asyncio
    async def test_shared_scope_across_tasks_isolated(self) -> None:
        scope = solwyn.run("shared")
        seen: list[tuple[str, tuple[str | None, str | None]]] = []

        async def task(label: str) -> str:
            async with scope as run_id:
                await asyncio.sleep(0)
                seen.append((label, current_run()))
                return run_id

        first_id, second_id = await asyncio.gather(task("first"), task("second"))

        assert first_id != second_id
        assert ("first", (first_id, "shared")) in seen
        assert ("second", (second_id, "shared")) in seen
        assert current_run() == (None, None)

    def test_double_enter_balances_outer_state(self) -> None:
        scope = solwyn.run("x")

        first_id = scope.__enter__()
        second_id = scope.__enter__()
        assert current_run() == (second_id, "x")

        scope.__exit__(None, None, None)
        assert current_run() == (first_id, "x")

        scope.__exit__(None, None, None)
        assert current_run() == (None, None)

    def test_out_of_order_exit_raises_without_corrupting_active_run(self) -> None:
        outer = solwyn.run("outer")
        inner = solwyn.run("inner")

        outer_id = outer.__enter__()
        inner_id = inner.__enter__()

        with pytest.raises(RuntimeError, match="LIFO"):
            outer.__exit__(None, None, None)

        assert current_run() == (inner_id, "inner")
        inner.__exit__(None, None, None)
        assert current_run() == (outer_id, "outer")
        outer.__exit__(None, None, None)
        assert current_run() == (None, None)

    def test_run_scope_does_not_create_per_instance_contextvars(self) -> None:
        for idx in range(100):
            with solwyn.run(f"run-{idx}"):
                pass

        leaked_frame_vars = [
            var.name for var in copy_context() if var.name.startswith("solwyn_run_scope_frames_")
        ]
        assert leaked_frame_vars == []

    def test_run_in_executor_propagates_context(self) -> None:
        with solwyn.run("batch") as run_id, ThreadPoolExecutor(max_workers=1) as executor:
            future = solwyn.run_in_executor(executor, current_run)

        assert future.result() == (run_id, "batch")

    def test_raw_executor_submit_does_not_propagate_context(self) -> None:
        with solwyn.run("batch"), ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(current_run)

        assert future.result() == (None, None)

    @pytest.mark.asyncio
    async def test_run_inside_async_generator_is_rejected_before_contaminating_consumer(
        self,
    ) -> None:
        async def producer():
            async with solwyn.run("gen"):
                yield 1

        async with solwyn.run("outer") as outer_id:
            gen = producer()
            with pytest.raises(TypeError, match="async generators"):
                await gen.__anext__()
            assert current_run() == (outer_id, "outer")


@pytest.mark.unit
class TestRunPublicSurface:
    """The context manager must be exported at the package top level."""

    def test_run_is_exported(self) -> None:
        assert hasattr(solwyn, "run")
        assert callable(solwyn.run)

    def test_run_is_in_dunder_all(self) -> None:
        assert "run" in solwyn.__all__

    def test_run_in_executor_is_exported(self) -> None:
        assert hasattr(solwyn, "run_in_executor")
        assert callable(solwyn.run_in_executor)

    def test_run_in_executor_is_in_dunder_all(self) -> None:
        assert "run_in_executor" in solwyn.__all__
