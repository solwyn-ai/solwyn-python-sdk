"""Tests for begin/end-shaped run scopes used by framework adapters."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.unit
def test_create_run_snapshots_identity_without_changing_current_context() -> None:
    tags = {"role": "worker"}
    with solwyn.run("workflow", tags={"team": "support"}) as workflow_id:
        before = solwyn.current_run_context()
        handle = solwyn.create_run("agent:Worker", tags=tags)

        assert solwyn.current_run_context() == before
        stable_run_id = handle.run_id
        assert handle.run_id == stable_run_id
        tags["role"] = "mutated-after-create"

    with handle.activate() as activated_id:
        assert activated_id == handle.run_id
        assert _capture_run_context() == (
            handle.run_id,
            "agent:Worker",
            {"team": "support", "role": "worker"},
            workflow_id,
        )

    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)
    handle.finish()


@pytest.mark.unit
def test_create_run_validates_name_and_can_skip_inherited_tags() -> None:
    with pytest.raises(TypeError, match=r"solwyn\.create_run\(name\) requires str"):
        solwyn.create_run(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a non-empty name"):
        solwyn.create_run(" ")

    with solwyn.run("workflow", tags={"team": "support"}) as workflow_id:
        handle = solwyn.create_run(
            "agent:Worker",
            tags={"role": "worker"},
            inherit_tags=False,
        )

    with handle.activate():
        assert _capture_run_context() == (
            handle.run_id,
            "agent:Worker",
            {"role": "worker"},
            workflow_id,
        )
    handle.finish()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detached_run_reuses_identity_across_sequential_tasks() -> None:
    with solwyn.run("workflow", tags={"team": "support"}) as workflow_id:
        handle = solwyn.create_run("agent:Worker", tags={"queue": "priority"})

    async def observe() -> tuple[str | None, str | None, dict[str, str] | None, str | None]:
        with handle.activate():
            return _capture_run_context()

    first = await asyncio.create_task(observe())
    second = await asyncio.create_task(observe())

    expected = (
        handle.run_id,
        "agent:Worker",
        {"team": "support", "queue": "priority"},
        workflow_id,
    )
    assert first == expected
    assert second == expected
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)
    handle.finish()


@pytest.mark.unit
def test_detached_run_parent_and_tags_are_creation_time_snapshots() -> None:
    with solwyn.run("creation-parent", tags={"phase": "creation"}) as creation_parent_id:
        handle = solwyn.create_run("agent:Worker", tags={"agent": "worker"})

    with solwyn.run("activation-parent", tags={"phase": "activation"}) as activation_parent_id:
        with handle.activate():
            assert _capture_run_context() == (
                handle.run_id,
                "agent:Worker",
                {"phase": "creation", "agent": "worker"},
                creation_parent_id,
            )
        assert solwyn.current_run_context().id == activation_parent_id

    handle.finish()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detached_run_allows_concurrent_activations_in_distinct_tasks() -> None:
    handle = solwyn.create_run("agent:Worker", tags={"team": "support"})
    both_active = asyncio.Event()
    release = asyncio.Event()
    active_count = 0
    seen: list[solwyn.RunContext] = []

    async def worker() -> None:
        nonlocal active_count
        with handle.activate():
            seen.append(solwyn.current_run_context())
            active_count += 1
            if active_count == 2:
                both_active.set()
            await release.wait()

    first = asyncio.create_task(worker())
    second = asyncio.create_task(worker())
    await both_active.wait()
    try:
        with pytest.raises(RuntimeError, match="activations are still active"):
            handle.finish()
    finally:
        release.set()
    await asyncio.gather(first, second)

    assert seen == [
        solwyn.RunContext(handle.run_id, "agent:Worker", {"team": "support"}),
        solwyn.RunContext(handle.run_id, "agent:Worker", {"team": "support"}),
    ]
    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)
    handle.finish()


@pytest.mark.unit
def test_detached_run_rejects_nested_activation_in_one_context() -> None:
    handle = solwyn.create_run("agent:Worker")

    with handle.activate():
        with (
            pytest.raises(RuntimeError, match="already active in this context"),
            handle.activate(),
        ):
            pass
        assert solwyn.current_run_context().id == handle.run_id

    handle.finish()


@pytest.mark.unit
def test_detached_run_rejects_activation_after_finish_and_double_finish() -> None:
    handle = solwyn.create_run("agent:Worker")
    handle.finish()

    with pytest.raises(RuntimeError, match="already finished"), handle.activate():
        pass
    with pytest.raises(RuntimeError, match="already finished"):
        handle.finish()


@pytest.mark.unit
def test_detached_finish_while_active_is_retryable_after_exit() -> None:
    handle = solwyn.create_run("agent:Worker")

    with handle.activate(), pytest.raises(RuntimeError, match="activations are still active"):
        handle.finish()

    handle.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        handle.finish()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detached_activation_cancellation_restores_context_and_active_count() -> None:
    handle = solwyn.create_run("agent:Worker")
    entered = asyncio.Event()
    restored: list[solwyn.RunContext] = []

    async def worker() -> None:
        try:
            with handle.activate():
                entered.set()
                await asyncio.Event().wait()
        finally:
            restored.append(solwyn.current_run_context())

    task = asyncio.create_task(worker())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert restored == [solwyn.RunContext(None, None, None)]
    handle.finish()


@pytest.mark.unit
def test_detached_activation_exception_restores_context_and_active_count() -> None:
    handle = solwyn.create_run("agent:Worker")

    with pytest.raises(LookupError, match="provider failed"), handle.activate():
        raise LookupError("provider failed")

    assert solwyn.current_run_context() == solwyn.RunContext(None, None, None)
    handle.finish()


@pytest.mark.unit
def test_detached_activation_non_lifo_exit_is_retryable() -> None:
    outer_handle = solwyn.create_run("outer")
    inner_handle = solwyn.create_run("inner")
    outer_activation = outer_handle.activate()
    inner_activation = inner_handle.activate()
    outer_activation.__enter__()
    inner_activation.__enter__()

    with pytest.raises(RuntimeError, match="LIFO"):
        outer_activation.__exit__(None, None, None)

    assert solwyn.current_run_context().id == inner_handle.run_id
    inner_activation.__exit__(None, None, None)
    assert solwyn.current_run_context().id == outer_handle.run_id
    outer_activation.__exit__(None, None, None)
    assert solwyn.current_run_context().id is None
    inner_handle.finish()
    outer_handle.finish()


@pytest.mark.unit
def test_detached_activation_wrong_context_exit_is_retryable() -> None:
    handle = solwyn.create_run("agent:Worker")
    owner_context = copy_context()

    def activate_in_owner() -> None:
        activation = handle.activate()
        activation.__enter__()
        # Copy after entry so the foreign context contains the frame but not
        # the ContextVar token's owning Context.
        foreign_context = copy_context()

        with pytest.raises(RuntimeError, match="same context"):
            foreign_context.run(activation.__exit__, None, None, None)

        assert solwyn.current_run_context().id == handle.run_id
        activation.__exit__(None, None, None)
        assert solwyn.current_run_context().id is None

    owner_context.run(activate_in_owner)
    handle.finish()


@pytest.mark.unit
def test_detached_run_thread_race_protects_finish_state() -> None:
    handle = solwyn.create_run("agent:Worker")
    entered = threading.Event()
    release = threading.Event()

    def worker() -> solwyn.RunContext:
        with handle.activate():
            entered.set()
            release.wait(timeout=5)
            return solwyn.current_run_context()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="activations are still active"):
                handle.finish()
        finally:
            release.set()
        assert future.result(timeout=5).id == handle.run_id

    handle.finish()
