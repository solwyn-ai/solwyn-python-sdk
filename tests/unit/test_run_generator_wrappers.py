"""Streaming generator ownership survives delegating context managers."""

from __future__ import annotations

import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import copy_context
from uuid import uuid4

import pytest
from test_integrations_crewai import _crew_source, _task_source
from test_integrations_crewai import listener_protocol as listener_protocol
from test_integrations_langchain import _FakeCallbackManager
from test_integrations_langchain import handler_type as handler_type

import solwyn
from solwyn import _run
from solwyn._lifecycle import _drain_run_cleanups

pytestmark = pytest.mark.unit


@contextmanager
def _helper(binding):
    with binding as rid:
        yield rid


class _HandwrittenManager:
    def __init__(self, binding):
        self.binding = binding

    def __enter__(self):
        return self.binding.__enter__()

    def __exit__(self, *exc):
        return self.binding.__exit__(*exc)


@contextmanager
def _binding(kind, binding):
    if kind == "helper":
        with _helper(binding) as rid:
            yield rid
    elif kind == "exitstack":
        with ExitStack() as stack:
            yield stack.enter_context(binding)
    elif kind == "handwritten":
        with _HandwrittenManager(binding) as rid:
            yield rid
    else:
        with ExitStack() as stack:
            yield stack.enter_context(_HandwrittenManager(_helper(binding)))


@pytest.mark.parametrize("kind", ["helper", "exitstack", "handwritten", "nested"])
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("outer", [False, True])
def test_wrapped_generator_scope_keeps_every_step_and_can_finish(kind, detached, outer):
    handle = solwyn.create_run("stream") if detached else None

    def chunks():
        with _binding(kind, handle.activate() if handle else solwyn.run("stream")) as rid:
            yield rid, solwyn.current_run_context()
            yield rid, solwyn.current_run_context()

    with solwyn.run("request") if outer else nullcontext() as parent:
        base = copy_context()
        stream = chunks()
        try:
            first, seen_first = base.copy().run(next, stream)
            second, seen_second = base.copy().run(next, stream)
            assert seen_first.id == first == second == seen_second.id
            assert base.run(solwyn.current_run_context).id == parent
            with pytest.raises(StopIteration):
                base.copy().run(next, stream)
        finally:
            base.copy().run(stream.close)
            if handle:
                handle.finish()


@pytest.mark.parametrize("fresh", [False, True])
def test_started_generator_finish_rejects_nested_scope_and_remains_retryable(fresh):
    def chunks():
        handle = solwyn.start_run("started")
        finished = False
        try:
            yield handle.run_id
            with solwyn.run("nested"), pytest.raises(RuntimeError, match="LIFO"):
                handle.finish()
            handle.finish()
            finished = True
        finally:
            if not finished:
                handle.finish()

    stream = chunks()
    entered = copy_context()
    entered.run(next, stream)
    with pytest.raises(StopIteration):
        (copy_context() if fresh else entered).run(next, stream)


def test_langchain_activation_in_a_stream_keeps_attribution_and_finishes(handler_type):
    handler = handler_type()
    manager = _FakeCallbackManager(handler)
    chain, model = uuid4(), uuid4()
    manager.start(run_id=chain, name="stream")
    manager.chat_model_start(run_id=model, parent_run_id=chain)
    rid = handler._handles[chain].run_id

    def chunks():
        with handler.activate_model_call():
            yield solwyn.current_run_context().id
            yield solwyn.current_run_context().id

    stream = chunks()
    assert copy_context().run(next, stream) == rid
    assert copy_context().run(next, stream) == rid
    with pytest.raises(StopIteration):
        copy_context().run(next, stream)
    manager.model_end(run_id=model, parent_run_id=chain)
    manager.end(run_id=chain)
    assert not handler._handles


def test_crewai_activation_in_a_stream_keeps_attribution_and_finishes(listener_protocol):
    listener_type, bus, events = listener_protocol
    listener = listener_type()
    task = _task_source()
    crew = _crew_source(task)
    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)
    rid = listener._task_handles[str(task.id)].run_id

    def chunks():
        with listener.activate(task):
            yield solwyn.current_run_context().id
            yield solwyn.current_run_context().id

    stream = chunks()
    assert copy_context().run(next, stream) == rid
    assert copy_context().run(next, stream) == rid
    with pytest.raises(StopIteration):
        copy_context().run(next, stream)
    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    assert not listener._task_handles
    assert not listener._crews


@pytest.mark.parametrize("rebuild", ["register", "unregister"])
def test_registry_rebuild_prunes_an_already_collected_entry(rebuild):
    class Collected:
        pass

    value = Collected()
    dead = weakref.ref(value)
    del value

    def chunks():
        with solwyn.run("stream"):
            yield

    stream = chunks()
    key = (id(stream.gi_frame), stream.gi_code)
    try:
        if rebuild == "register":
            _run._generator_runs[key[0]] = ((key[1], dead),)
        copy_context().run(next, stream)
        entries = _run._generator_runs[key[0]]
        assert all(ref() is not None for _, ref in entries)
        if rebuild == "unregister":
            # This models an entry resurrected after its weakref callback was
            # drained: no subsequent callback can remove it a second time.
            _run._generator_runs[key[0]] = (*entries, (key[1], dead))
            _run._unregister_generator(key, entries[0][1])
            assert key[0] not in _run._generator_runs
    finally:
        copy_context().run(stream.close)
        _drain_run_cleanups()


def test_started_generator_finish_checks_retired_inner_context_frames_in_lifo_order():
    inner = solwyn.run("inner")

    def chunks():
        handle = solwyn.start_run("outer")
        try:
            with inner:
                yield handle
                with pytest.raises(RuntimeError, match="LIFO"):
                    handle.finish()
        finally:
            handle.finish()

    stream = chunks()
    entered = copy_context()
    entered.run(next, stream)
    copy_context().run(inner.__exit__, None, None, None)
    with pytest.raises(StopIteration):
        entered.run(next, stream)
