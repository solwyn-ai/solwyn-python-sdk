"""Run attribution and lease ownership when each generator step copies context."""

from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import sys
import weakref
from contextlib import nullcontext
from contextvars import copy_context
from types import SimpleNamespace

import anyio
import pytest

import solwyn
from solwyn import _run, _run_control
from solwyn._lifecycle import _drain_run_cleanups
from solwyn.testing import FakeControlPlane

pytestmark = pytest.mark.unit


class _OpenAIStub:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))


_OpenAIStub.__module__ = "openai._client"
_OpenAIStub.__name__ = "OpenAI"


def _call(client):
    client.chat.completions.create(model="gpt-5.5", messages=[], max_tokens=8)


async def _wait_for_release(plane, count):
    with anyio.fail_after(3):
        while len(plane.lease_surrenders) < count:
            await anyio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("threaded", [False, True], ids=["context-copy", "anyio-worker"])
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("terminal", ["exhaustion", "exception", "close", "gc"])
async def test_every_step_keeps_run_and_releases_before_client_close(threaded, outer, terminal):
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub())
    scope = solwyn.run("chunks")

    def chunks():
        with scope as run_id:
            for _ in range(3):
                _call(client)
                yield run_id
            if terminal == "exception":
                raise ValueError("stream failed")

    try:
        with solwyn.run("request") if outer else nullcontext() as parent_id:
            base = copy_context()
            entered = base.copy()
            generator = chunks()

            async def step(fn):
                if threaded:
                    return await anyio.to_thread.run_sync(fn)
                return base.copy().run(fn)

            # Keep the entry context to exercise later cleanup of a closed record.
            run_id = await step(lambda: entered.run(next, generator))
            assert await step(lambda: next(generator)) == run_id
            _call(client)  # The suspended generator must not capture driver calls.
            assert await step(lambda: next(generator)) == run_id
            if terminal in {"exhaustion", "exception"}:
                # Avoid propagating StopIteration through an async worker boundary.
                def finish():
                    with pytest.raises(ValueError if terminal == "exception" else StopIteration):
                        next(generator)

                await step(finish)
            elif terminal == "close":
                await step(generator.close)
            else:
                reference = weakref.ref(generator)
                del generator
                gc.collect()
                assert reference() is None
            await _wait_for_release(plane, 1)
            assert [grant.agent_run_id for grant in plane.lease_grants] == (
                [run_id, parent_id] if outer else [run_id]
            )
            assert plane.lease_surrenders[0].lease_id == "lse_fake1"
            entered.run(scope.__exit__, None, None, None)
            assert len(plane.lease_surrenders) == 1
        await _wait_for_release(plane, 2 if outer else 1)
    finally:
        client.close()

    assert [(e.agent_run_id, e.parent_agent_run_id) for e in plane.ingested] == [
        (run_id, parent_id),
        (run_id, parent_id),
        (parent_id, None),
        (run_id, parent_id),
    ]
    assert len(plane.lease_surrenders) == (2 if outer else 1)


@pytest.mark.parametrize("terminal", ["exhaustion", "exception", "close", "gc"])
def test_detached_activation_across_yield_can_finish(terminal):
    handle = solwyn.create_run("detached")

    def chunks():
        with handle.activate():
            yield solwyn.current_run_context().id
            yield solwyn.current_run_context().id
            if terminal == "exception":
                raise ValueError("stream failed")

    base = copy_context()
    generator = chunks()
    assert base.copy().run(next, generator) == handle.run_id
    assert base.copy().run(next, generator) == handle.run_id
    if terminal in {"exhaustion", "exception"}:
        with pytest.raises(ValueError if terminal == "exception" else StopIteration):
            base.copy().run(next, generator)
    elif terminal == "close":
        base.copy().run(generator.close)
    else:
        reference = weakref.ref(generator)
        del generator
        gc.collect()
        assert reference() is None
    handle.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 2])
async def test_shared_scope_generators_keep_separate_runs_on_reused_workers(workers):
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub())
    scope = solwyn.run("shared")
    limiter = anyio.CapacityLimiter(workers)

    def chunks():
        with scope as run_id:
            for _ in range(3):
                _call(client)
                yield run_id

    async def consume():
        generator = chunks()
        run_ids = []
        try:
            for _ in range(3):
                run_ids.append(await anyio.to_thread.run_sync(next, generator, limiter=limiter))
                await anyio.to_thread.run_sync(_call, client, limiter=limiter)
        finally:
            await anyio.to_thread.run_sync(generator.close, limiter=limiter)
        assert len(set(run_ids)) == 1
        return run_ids[0]

    try:
        first, second = await asyncio.gather(consume(), consume())
        assert first != second
        await _wait_for_release(plane, 2)
        assert {grant.agent_run_id for grant in plane.lease_grants} == {first, second}
    finally:
        client.close()
    assert sum(e.agent_run_id == first for e in plane.ingested) == 3
    assert sum(e.agent_run_id == second for e in plane.ingested) == 3
    assert sum(e.agent_run_id is None for e in plane.ingested) == 6
    assert all(e.parent_agent_run_id is None for e in plane.ingested)
    assert len(plane.lease_surrenders) == 2


def test_nested_scopes_and_callback_activations_override_only_during_the_step():
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub(), lease_enabled=False)
    handle = solwyn.create_run("callback")
    base = copy_context()

    def callback():
        with handle.activate():
            _call(client)

    def chunks():
        with solwyn.run("chunks", tags={"stream": "yes"}) as run_id:
            yield run_id
            assert solwyn.current_run_context() == solwyn.RunContext(
                run_id, "chunks", {"stream": "yes"}
            )
            with solwyn.run("child") as child_id:
                _call(client)
                callback()
                _call(client)
            # Same-frame activation must also shadow the fallback.
            with handle.activate():
                _call(client)
            _call(client)
            yield child_id

    generator = chunks()
    try:
        run_id = base.copy().run(next, generator)
        child_id = base.copy().run(next, generator)
        _call(client)
        base.copy().run(generator.close)
    finally:
        handle.finish()
        client.close()
    assert [(e.agent_run_id, e.parent_agent_run_id) for e in plane.ingested] == [
        (child_id, run_id),
        (handle.run_id, None),
        (child_id, run_id),
        (handle.run_id, None),
        (run_id, None),
        (None, None),
    ]
    assert plane.ingested[0].tags == {"stream": "yes"}


@pytest.mark.asyncio
async def test_shared_scope_nested_in_one_generator_surrenders_each_run_once():
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub())
    scope = solwyn.run("shared")

    def chunks():
        with scope as parent:
            _call(client)
            with scope as child:
                _call(client)
                yield parent, child
                _call(client)

    generator = chunks()
    base = copy_context()
    try:
        parent, child = base.copy().run(next, generator)
        with pytest.raises(StopIteration):
            base.copy().run(next, generator)
        await _wait_for_release(plane, 2)
    finally:
        client.close()
    assert [(e.agent_run_id, e.parent_agent_run_id) for e in plane.ingested] == [
        (parent, None),
        (child, parent),
        (child, parent),
    ]
    assert [s.lease_id for s in plane.lease_surrenders] == ["lse_fake2", "lse_fake1"]


def test_stopped_run_is_enforced_in_later_steps():
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub(), lease_enabled=False)

    def chunks():
        with solwyn.run("chunks") as run_id:
            _call(client)
            yield run_id
            with pytest.raises(solwyn.RunStoppedError):
                _call(client)
            assert _run_control.current_run_terminated()
            yield "stopped"
            with pytest.raises(solwyn.RunStoppedError):
                _call(client)

    base = copy_context()
    generator = chunks()
    try:
        run_id = base.copy().run(next, generator)
        plane.stop_run(run_id)
        assert base.copy().run(next, generator) == "stopped"
        with pytest.raises(StopIteration):
            base.copy().run(next, generator)
    finally:
        client.close()
    assert all(check.agent_run_id == run_id for check in plane.checks)
    assert any(
        e.agent_run_id == run_id and e.status.value == "budget_denied" for e in plane.ingested
    )


def test_testing_plane_run_scoped_model_works_in_later_steps():
    plane = FakeControlPlane()
    client = plane.wrap(_OpenAIStub(), lease_enabled=False)

    def chunks():
        with solwyn.run("chunks") as run_id:
            yield run_id
            with pytest.raises(solwyn.RunStoppedError):
                client.chat.completions.create(model="solwyn-test/deny-stopped", messages=[])

    base = copy_context()
    generator = chunks()
    try:
        run_id = base.copy().run(next, generator)
        with pytest.raises(StopIteration):
            base.copy().run(next, generator)
    finally:
        client.close()
    assert plane.checks[0].agent_run_id == run_id


def test_generator_frame_identity_is_stable_across_context_copies():
    """CI runs this assumption on every supported Python version (3.11/3.12)."""
    keys = []

    def chunks():
        with solwyn.run("chunks"):
            for _ in range(3):
                frame = sys._getframe()
                keys.append((id(frame), frame.f_code))
                del frame
                yield solwyn.current_run_context().id

    generator = chunks()
    base = copy_context()
    ids = [base.copy().run(next, generator) for _ in range(3)]
    base.copy().run(generator.close)
    assert len(set(ids)) == 1
    assert all(key[0] == keys[0][0] and key[1] is chunks.__code__ for key in keys)
    _drain_run_cleanups()
    assert keys[0][0] not in _run._generator_runs


def test_plain_for_loop_keeps_existing_consumer_attribution():
    def chunks():
        with solwyn.run("chunks") as run_id:
            yield run_id

    for run_id in chunks():
        assert solwyn.current_run_context().id == run_id
    assert solwyn.current_run_context().id is None


def test_thread_started_in_a_fallback_step_does_not_inherit_the_run():
    from concurrent.futures import ThreadPoolExecutor

    def chunks():
        with solwyn.run("chunks"):
            yield
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(solwyn.current_run_context).result().id is None

    base = copy_context()
    generator = chunks()
    base.copy().run(next, generator)
    with pytest.raises(StopIteration):
        base.copy().run(next, generator)


def test_without_frame_introspection_scopes_still_exit_safely(monkeypatch):
    monkeypatch.setattr(_run, "sys", SimpleNamespace())
    scope = solwyn.run("portable")

    def chunks():
        with scope as run_id:
            assert solwyn.current_run_context().id == run_id
            yield run_id

    generator = chunks()
    base = copy_context()
    base.copy().run(next, generator)
    with pytest.raises(StopIteration):
        base.copy().run(next, generator)


def test_resolver_does_not_walk_frames_without_open_generators(monkeypatch):
    with solwyn.run("plain") as run_id, monkeypatch.context() as patcher:
        patcher.setattr(
            _run, "sys", SimpleNamespace(_getframe=lambda depth: pytest.fail("unexpected walk"))
        )
        assert solwyn.current_run_context().id == run_id


def test_original_activation_context_cleanup_does_not_close_a_new_activation():
    handle = solwyn.create_run("reused")
    activation = handle.activate()

    def chunks():
        with activation:
            yield handle.run_id

    base = copy_context()
    entered = base.copy()
    first = chunks()
    entered.run(next, first)
    base.copy().run(first.close)
    second = chunks()
    base.copy().run(next, second)
    entered.run(activation.__exit__, None, None, None)
    try:
        with pytest.raises(RuntimeError, match="activations are still active"):
            handle.finish()
    finally:
        base.copy().run(second.close)
    handle.finish()


def test_original_generator_context_cleanup_preserves_shared_scope_outer_run():
    scope = solwyn.run("shared")

    def chunks():
        with scope as run_id:
            yield run_id

    with scope as outer:
        base = copy_context()
        entered = base.copy()
        generator = chunks()
        inner = entered.run(next, generator)
        assert inner != outer
        base.copy().run(generator.close)
        entered.run(scope.__exit__, None, None, None)
        assert entered.run(solwyn.current_run_context).id == outer
        assert solwyn.current_run_context().id == outer


def test_without_frame_introspection_shared_scope_does_not_guess_the_exiting_run(monkeypatch):
    monkeypatch.setattr(_run, "sys", SimpleNamespace())
    scope = solwyn.run("portable-shared")
    released = []
    monkeypatch.setattr(_run, "_surrender_run", released.append)

    def chunks():
        with scope as run_id:
            yield run_id

    base = copy_context()
    first, second = chunks(), chunks()
    first_id = base.copy().run(next, first)
    second_id = base.copy().run(next, second)
    base.copy().run(first.close)
    base.copy().run(second.close)
    assert first_id != second_id
    # Without owner introspection both foreign exits are ambiguous. Degrade
    # safely to client.close()/expiry rather than surrender the wrong run.
    assert released == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize("held_lock", ["registry", "scope", "record", "handle", "cleanup"])
def test_forked_child_can_resolve_and_close_run_with_inherited_locks_held(held_lock):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import signal
import sys
from contextvars import copy_context
from threading import Event, Thread
import solwyn
from solwyn import _run
from solwyn import _lifecycle

released = []
_run._surrender_run = released.append
_run._queue_run_surrender = released.append
handle = solwyn.create_run('chunks') if sys.argv[1] == 'handle' else None
scope = handle.activate() if handle else solwyn.run('chunks')
def chunks():
    with scope as rid:
        yield rid

generator = chunks()
base = copy_context()
entered = base.copy()
rid = entered.run(next, generator)
record = entered.run(_run._run_frames.get)[-1].record
lock = {
    'registry': _run._generator_runs_lock,
    'scope': getattr(scope, '_records_lock', None),
    'record': record.lock,
    'handle': handle._state_lock if handle else None,
    'cleanup': _lifecycle._run_cleanup_lock,
}[sys.argv[1]]
held, release = Event(), Event()
def hold():
    with lock:
        held.set()
        release.wait(10)
thread = Thread(target=hold)
thread.start()
assert held.wait(5)
pid = os.fork()
if pid == 0:
    signal.alarm(2)
    assert solwyn.current_run_context().id is None
    base.copy().run(generator.close)
    if handle:
        handle.finish()
    _lifecycle._drain_run_cleanups()
    assert released == [rid]
    os._exit(0)
release.set()
thread.join(5)
_, status = os.waitpid(pid, 0)
base.copy().run(generator.close)
if handle:
    handle.finish()
_lifecycle._drain_run_cleanups()
assert os.waitstatus_to_exitcode(status) == 0, status
""",
            held_lock,
        ],
        check=True,
        timeout=15,
    )
