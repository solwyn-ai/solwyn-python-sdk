"""Tests for begin/end-shaped run scopes used by framework adapters."""

from __future__ import annotations

import asyncio
from contextvars import copy_context

import pytest

import solwyn
from solwyn._run import _capture_run_context


@pytest.mark.unit
def test_start_run_scopes_events_until_finish() -> None:
    handle = solwyn.start_run("adapter-scope", tags={"team": "x"})

    context = solwyn.current_run_context()
    assert context == solwyn.RunContext(handle.run_id, "adapter-scope", {"team": "x"})

    handle.finish()
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)


@pytest.mark.unit
def test_start_run_inherits_outer_tags_by_default() -> None:
    with solwyn.run("outer", tags={"team": "platform", "shared": "outer"}) as outer_id:
        handle = solwyn.start_run("inner", tags={"shared": "inner"})

        assert _capture_run_context() == (
            handle.run_id,
            "inner",
            {"team": "platform", "shared": "inner"},
            outer_id,
        )

        handle.finish()
        assert solwyn.current_run_context() == solwyn.RunContext(
            outer_id,
            "outer",
            {"team": "platform", "shared": "outer"},
        )


@pytest.mark.unit
def test_nested_handles_thread_parent_run_id() -> None:
    outer = solwyn.start_run("outer")
    inner = solwyn.start_run("inner")

    assert _capture_run_context()[3] == outer.run_id

    inner.finish()
    outer.finish()


@pytest.mark.unit
def test_double_finish_raises_runtime_error() -> None:
    handle = solwyn.start_run("once")
    handle.finish()

    with pytest.raises(RuntimeError, match="already finished"):
        handle.finish()


@pytest.mark.unit
def test_non_lifo_finish_leaves_handle_retryable() -> None:
    outer = solwyn.start_run("outer")
    inner = solwyn.start_run("inner")

    with pytest.raises(RuntimeError, match="LIFO"):
        outer.finish()

    assert solwyn.current_run_context().id == inner.run_id
    inner.finish()
    assert solwyn.current_run_context().id == outer.run_id
    outer.finish()
    assert solwyn.current_run_context().id is None


@pytest.mark.unit
def test_finish_from_context_without_frame_raises_and_owner_can_retry() -> None:
    owner_context = copy_context()
    foreign_context = copy_context()

    def finish_in_owner_context() -> None:
        handle = solwyn.start_run("owner-context")

        with pytest.raises(RuntimeError, match="same context"):
            foreign_context.run(handle.finish)

        assert foreign_context.run(solwyn.current_run_context).id is None
        assert solwyn.current_run_context().id == handle.run_id
        handle.finish()
        assert solwyn.current_run_context().id is None

    owner_context.run(finish_in_owner_context)
    assert solwyn.current_run_context().id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finish_from_inherited_child_task_raises_and_owner_can_retry() -> None:
    async def owner() -> None:
        handle = solwyn.start_run("owner-task")

        async def child() -> tuple[solwyn.RunContext, solwyn.RunContext]:
            before = solwyn.current_run_context()
            with pytest.raises(RuntimeError, match="same context"):
                handle.finish()
            return before, solwyn.current_run_context()

        child_contexts = await asyncio.create_task(child())

        expected = solwyn.RunContext(handle.run_id, "owner-task", None)
        assert child_contexts == (expected, expected)
        assert solwyn.current_run_context() == expected
        handle.finish()
        assert solwyn.current_run_context().id is None

    await asyncio.create_task(owner())
    assert solwyn.current_run_context().id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_context_is_task_local_and_finishes_in_creating_task() -> None:
    started = asyncio.Event()
    observed = asyncio.Event()
    owner_contexts: list[solwyn.RunContext] = []
    observer_contexts: list[solwyn.RunContext] = []

    async def owner() -> str:
        handle = solwyn.start_run("owner-task", tags={"task": "owner"})
        owner_contexts.append(solwyn.current_run_context())
        started.set()
        await observed.wait()
        owner_contexts.append(solwyn.current_run_context())
        handle.finish()
        owner_contexts.append(solwyn.current_run_context())
        return handle.run_id

    async def observer() -> None:
        await started.wait()
        observer_contexts.append(solwyn.current_run_context())
        observed.set()
        await asyncio.sleep(0)
        observer_contexts.append(solwyn.current_run_context())

    run_id, _ = await asyncio.gather(owner(), observer())

    assert owner_contexts == [
        solwyn.RunContext(run_id, "owner-task", {"task": "owner"}),
        solwyn.RunContext(run_id, "owner-task", {"task": "owner"}),
        solwyn.RunContext(None, None, None),
    ]
    assert observer_contexts == [
        solwyn.RunContext(None, None, None),
        solwyn.RunContext(None, None, None),
    ]
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)
