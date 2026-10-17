"""Generator cleanup must tolerate cyclic GC and preserve ordinary contexts."""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from types import SimpleNamespace

import pytest

import solwyn
from solwyn import _run
from solwyn._lifecycle import _drain_run_cleanups

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("held_lock", ["weakref", "registry", "scope", "record", "handle"])
def test_cyclic_generator_gc_cannot_deadlock_cleanup(held_lock):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
import sys
from contextlib import nullcontext
from contextvars import copy_context
import solwyn
from solwyn import _run
from solwyn._lifecycle import _drain_run_cleanups

gc.disable()
released = []
_run._surrender_run = released.append
_run._queue_run_surrender = released.append
kind = sys.argv[1]
scope = solwyn.run('shared') if kind != 'weakref' else None
handle = solwyn.create_run('detached') if kind == 'handle' else None

class ChatStream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        binding = handle.activate() if handle else scope or solwyn.run('local')
        with binding as rid:
            yield rid
            yield rid

stream = ChatStream()
entered = copy_context()
rid = entered.run(next, stream.chunks)
record = entered.run(_run._run_frames.get)[-1].record
lock = {
    'weakref': _run._generator_runs_lock,
    'registry': _run._generator_runs_lock,
    'scope': scope._records_lock if scope else nullcontext(),
    'record': record.lock,
    'handle': handle._state_lock if handle else nullcontext(),
}[kind]
del entered, record, stream
with lock:
    gc.collect()
_drain_run_cleanups()
assert solwyn.current_run_context().id is None
assert not _run._generator_runs
if handle:
    handle.finish()
assert released == [rid], released
""",
            held_lock,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_natural_cyclic_stream_disconnects_do_not_deadlock_other_requests():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
from contextvars import copy_context
import solwyn
from solwyn import _run

class ChatStream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        with solwyn.run('chat-stream'):
            yield b'a'
            yield b'b'

def deep(depth):
    return solwyn.current_run() if depth == 0 else deep(depth - 1)

for _ in range(20000):
    stream = ChatStream()
    copy_context().run(next, stream.chunks)
    del stream
    assert deep(30) == (None, None)
gc.collect()
assert solwyn.current_run() == (None, None)
assert not _run._generator_runs
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize("detached", [False, True])
def test_lifo_error_releases_generator_record_before_frame_address_reuse(detached, monkeypatch):
    scope = solwyn.run("shared")
    handle = solwyn.create_run("shared") if detached else None
    second_handle = solwyn.create_run("other") if detached else None
    released = []
    monkeypatch.setattr(_run, "_surrender_run", released.append)
    monkeypatch.setattr(_run, "_queue_run_surrender", released.append)

    def chunks(binding):
        if binding is not None:
            with binding as rid:
                yield rid
        else:
            yield solwyn.current_run_context().id

    def request():
        first = chunks(handle.activate() if handle else scope)
        second = chunks(second_handle.activate() if second_handle else scope)
        first_id = next(first)
        second_id = next(second)
        old_key = id(first.gi_frame)
        with pytest.raises(RuntimeError, match="LIFO"):
            first.close()
        second.close()
        return first_id, second_id, old_key

    first_id, second_id, old_key = copy_context().run(request)
    _drain_run_cleanups()
    assert old_key not in _run._generator_runs
    assert not scope._open_runs
    if handle is not None and second_handle is not None:
        handle.finish()
        second_handle.finish()
    assert released == [first_id, second_id]

    # Keep all candidate frames alive so the allocator exercises reused slots.
    # The registry-empty assertion above makes the test deterministic even if
    # a particular interpreter does not reuse this address during the loop.
    candidates = [chunks(None) for _ in range(2000)]
    try:
        assert all(copy_context().run(next, candidate) is None for candidate in candidates)
    finally:
        for candidate in candidates:
            candidate.close()


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("detached", [False, True])
def test_contextmanager_helper_keeps_outliving_copied_context(outer, detached):
    handle = solwyn.create_run("inner") if detached else None

    @contextmanager
    def traced_run():
        with handle.activate() if handle else solwyn.run("inner") as rid:
            yield rid

    with solwyn.run("outer") if outer else nullcontext():
        with traced_run() as rid:
            kept = copy_context()
        assert kept.run(solwyn.current_run_context).id == rid
    assert kept.run(solwyn.current_run_context).id == rid
    if handle is not None:
        handle.finish()


@pytest.mark.asyncio
async def test_contextmanager_helper_keeps_outliving_task_attribution():
    released = asyncio.Event()

    @contextmanager
    def traced_run():
        with solwyn.run("helper") as rid:
            yield rid

    async def work():
        await released.wait()
        return solwyn.current_run_context().id

    with solwyn.run("outer"), traced_run() as rid:
        task = asyncio.create_task(work())
    released.set()
    assert await task == rid


@pytest.mark.parametrize("detached", [False, True])
def test_generator_that_exits_in_entry_context_keeps_outliving_attribution(detached):
    handle = solwyn.create_run("generator") if detached else None
    kept = None

    def chunks():
        nonlocal kept
        with handle.activate() if handle else solwyn.run("generator") as rid:
            kept = copy_context()
            yield rid

    generator = chunks()
    entered = copy_context()
    rid = entered.run(next, generator)
    entered.run(generator.close)
    assert kept.run(solwyn.current_run_context).id == rid
    _drain_run_cleanups()
    if handle is not None:
        handle.finish()


def test_suspended_generator_activation_rejects_a_copied_context_exit():
    handle = solwyn.create_run("detached")
    activation = handle.activate()

    def chunks():
        with activation:
            yield copy_context()

    generator = chunks()
    entered = copy_context()
    copied = entered.run(next, generator)
    try:
        with pytest.raises(RuntimeError, match="same context"):
            copied.run(activation.__exit__, None, None, None)
        with pytest.raises(RuntimeError, match="activations are still active"):
            handle.finish()
        assert entered.run(solwyn.current_run_context).id == handle.run_id
    finally:
        entered.run(generator.close)
        handle.finish()


@pytest.mark.parametrize("detached", [False, True])
def test_contextmanager_helper_does_not_enable_frame_walks(detached, monkeypatch):
    handle = solwyn.create_run("helper") if detached else None

    @contextmanager
    def traced_run():
        with handle.activate() if handle else solwyn.run("helper") as rid:
            yield rid

    with traced_run() as rid, monkeypatch.context() as patcher:
        assert not _run._generator_runs
        patcher.setattr(
            _run, "sys", SimpleNamespace(_getframe=lambda depth: pytest.fail("unexpected walk"))
        )
        assert solwyn.current_run_context().id == rid
        # Another thread has no helper frame, either, and must also avoid a walk.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(solwyn.current_run_context).result(timeout=3).id is None
    if handle is not None:
        handle.finish()


def test_shared_scope_without_introspection_does_not_retain_ambiguous_records(monkeypatch):
    monkeypatch.setattr(_run, "sys", SimpleNamespace())
    scope = solwyn.run("portable")

    def chunks():
        with scope:
            yield

    for _ in range(100):
        first, second = chunks(), chunks()
        copy_context().run(next, first)
        copy_context().run(next, second)
        copy_context().run(first.close)
        copy_context().run(second.close)
    gc.collect()
    assert not scope._open_runs


def test_started_generator_finish_refusal_unregisters_until_owner_retries():
    def chunks():
        handle = solwyn.start_run("started")
        try:
            yield handle
        finally:
            handle.finish()

    generator = chunks()
    entered = copy_context()
    handle = entered.run(next, generator)
    try:
        with pytest.raises(RuntimeError, match="same context"):
            copy_context().run(handle.finish)
        assert id(generator.gi_frame) not in _run._generator_runs
    finally:
        entered.run(generator.close)


def test_started_generator_can_finish_from_its_executing_frame_in_a_later_step():
    def chunks():
        handle = solwyn.start_run("started")
        yield handle.run_id
        assert solwyn.current_run_context().id == handle.run_id
        handle.finish()
        assert solwyn.current_run_context().id is None

    generator = chunks()
    copy_context().run(next, generator)
    with pytest.raises(StopIteration):
        copy_context().run(next, generator)
    assert not _run._generator_runs
