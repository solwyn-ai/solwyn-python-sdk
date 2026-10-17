"""Hermetic run-span ownership and activation tests with the optional OTel SDK."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from test_integrations_crewai import _crew_source, _task_source
from test_integrations_crewai import listener_protocol as listener_protocol
from test_integrations_langchain import _FakeCallbackManager
from test_integrations_langchain import handler_type as handler_type
from test_otel_population import _raw_client, _response
from test_real_sdk_otel_spans import spans as spans

import solwyn
from solwyn import _otel
from solwyn.testing import FakeControlPlane

trace = pytest.importorskip("opentelemetry.trace")
context = pytest.importorskip("opentelemetry.context")
sampling = pytest.importorskip("opentelemetry.sdk.trace.sampling")
sdk = pytest.importorskip("opentelemetry.sdk.trace")


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    from solwyn._run import _active_run, _run_frames

    monkeypatch.setenv("SOLWYN_OTEL_SPANS", "true")
    active_token = _active_run.set(None)
    frames_token = _run_frames.set(())
    context_token = context.attach(context.Context())
    yield
    context.detach(context_token)
    _run_frames.reset(frames_token)
    _active_run.reset(active_token)


def _runs(spans):
    return [span for span in spans.exporter.get_finished_spans() if span.name == "solwyn.run"]


def _assert_run(spans, span, run_id, *, parent_id=None, error=False):
    expected = {"solwyn.agent_run_id": run_id}
    if parent_id is not None:
        expected["solwyn.parent_agent_run_id"] = parent_id
    assert dict(span.attributes) == expected
    assert span.kind == trace.SpanKind.INTERNAL
    assert span.status.status_code == (trace.StatusCode.ERROR if error else trace.StatusCode.OK)
    assert span.status.description is None
    assert not span.events
    assert spans.ends[span.context.span_id] == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("terminal", ["normal", "exception", "interrupt"])
async def test_run_scope_lifecycle_and_application_parentage(spans, async_mode, terminal):
    error = None
    if terminal == "exception":
        error = ValueError("PRIVATE exception message")
    elif terminal == "interrupt":
        error = asyncio.CancelledError() if async_mode else KeyboardInterrupt()
    with spans.tracer.start_as_current_span("application") as parent:
        scope = solwyn.run("PRIVATE run name", tags={"PRIVATE": "tag"})
        with pytest.raises(type(error)) if error is not None else nullcontext():
            if async_mode:
                async with scope as run_id:
                    current = trace.get_current_span()
                    with spans.tracer.start_as_current_span("customer.work"):
                        if error is not None:
                            raise error
            else:
                with scope as run_id:
                    current = trace.get_current_span()
                    with spans.tracer.start_as_current_span("customer.work"):
                        if error is not None:
                            raise error
        assert trace.get_current_span() is parent
    assert len(_runs(spans)) == 1
    run = _runs(spans)[0]
    assert current.get_span_context() == run.context
    assert run.parent.span_id == parent.get_span_context().span_id
    customer = next(s for s in spans.exporter.get_finished_spans() if s.name == "customer.work")
    assert customer.parent.span_id == run.context.span_id
    _assert_run(spans, run, run_id, error=error is not None)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("call_spans", [False, True])
async def test_run_call_provider_parentage_and_call_start_correlation(
    spans, async_mode, call_spans
):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=call_spans)

    def dispatch(**kwargs):
        with spans.tracer.start_as_current_span("provider.native"):
            return _response()

    raw.chat.completions.create.side_effect = dispatch
    try:
        with solwyn.run("PRIVATE") as run_id:
            run_context = trace.get_current_span().get_span_context()
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            if async_mode:
                await result
            assert trace.get_current_span().get_span_context() == run_context
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    run = _runs(spans)[0]
    _assert_run(spans, run, run_id)
    native = next(s for s in spans.exporter.get_finished_spans() if s.name == "provider.native")
    call_parent = spans.calls()[0] if call_spans else run
    assert native.parent.span_id == call_parent.context.span_id
    if call_spans:
        assert spans.calls()[0].parent.span_id == run.context.span_id
    assert plane.ingested
    assert {event.trace_id for event in plane.ingested} == {
        trace.format_trace_id(run.context.trace_id)
    }
    assert {event.span_id for event in plane.ingested} == {
        trace.format_span_id(call_parent.context.span_id)
    }


@pytest.mark.unit
def test_nested_and_reused_scopes_have_independent_run_spans(spans):
    scope = solwyn.run("PRIVATE", tags={"private": "tag"})
    with scope as outer_id:
        outer_context = trace.get_current_span().get_span_context()
        with scope as inner_id:
            inner_context = trace.get_current_span().get_span_context()
        assert trace.get_current_span().get_span_context() == outer_context
    with scope as next_id:
        next_context = trace.get_current_span().get_span_context()
    by_id = {s.attributes["solwyn.agent_run_id"]: s for s in _runs(spans)}
    assert len(by_id) == 3
    assert by_id[inner_id].parent.span_id == outer_context.span_id
    assert by_id[inner_id].context == inner_context
    assert by_id[next_id].context == next_context
    assert by_id[next_id].parent is None
    _assert_run(spans, by_id[outer_id], outer_id)
    _assert_run(spans, by_id[inner_id], inner_id, parent_id=outer_id)
    _assert_run(spans, by_id[next_id], next_id)


@pytest.mark.unit
def test_detached_run_created_once_and_attached_per_activation(spans):
    with solwyn.run("PRIVATE parent") as parent_id:
        parent_context = trace.get_current_span().get_span_context()
        handle = solwyn.create_run("PRIVATE framework run", tags={"PRIVATE": "tag"})
        assert trace.get_current_span().get_span_context() == parent_context
    contexts = []
    for _ in range(2):
        with handle.activate():
            contexts.append(trace.get_current_span().get_span_context())
            with spans.tracer.start_as_current_span("framework.work"):
                pass
        assert not trace.get_current_span().get_span_context().is_valid
        assert len(_runs(spans)) == 1
    handle.finish()
    assert len(_runs(spans)) == 2
    run = next(s for s in _runs(spans) if s.attributes["solwyn.agent_run_id"] == handle.run_id)
    assert contexts == [run.context, run.context]
    assert run.parent.span_id == parent_context.span_id
    _assert_run(spans, run, handle.run_id, parent_id=parent_id)
    assert all(
        s.parent.span_id == run.context.span_id
        for s in spans.exporter.get_finished_spans()
        if s.name == "framework.work"
    )
    with pytest.raises(RuntimeError, match="already finished"):
        handle.finish()
    with pytest.raises(RuntimeError, match="already finished"), handle.activate():
        pass
    assert spans.ends[run.context.span_id] == 1


@pytest.mark.unit
@pytest.mark.parametrize("abandon", [False, True])
@pytest.mark.parametrize("started", [False, True])
def test_handle_finish_or_gc_ends_span_exactly_once(spans, abandon, started):
    handle = (solwyn.start_run if started else solwyn.create_run)("PRIVATE")
    run_id = handle.run_id
    if not abandon:
        handle.finish()
    del handle
    gc.collect()
    assert len(_runs(spans)) == 1
    _assert_run(spans, _runs(spans)[0], run_id, error=abandon)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_detached_activation_exception_marks_error_without_ending(spans, async_mode):
    handle = solwyn.create_run("PRIVATE")
    with pytest.raises(ValueError, match="PRIVATE"):
        if async_mode:
            async with handle.activate():
                raise ValueError("PRIVATE")
        else:
            with handle.activate():
                raise ValueError("PRIVATE")
    assert not _runs(spans)
    with handle.activate():
        pass
    handle.finish()
    _assert_run(spans, _runs(spans)[0], handle.run_id, error=True)


@pytest.mark.unit
def test_detached_activations_are_thread_local_and_can_finish_elsewhere(spans):
    handle = solwyn.create_run("PRIVATE")
    barrier = threading.Barrier(2)

    def work():
        with spans.tracer.start_as_current_span("worker") as worker:
            with handle.activate():
                current = trace.get_current_span().get_span_context()
                barrier.wait(timeout=5)
                with pytest.raises(RuntimeError, match="activations are still active"):
                    handle.finish()
            assert trace.get_current_span() is worker
            return current

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(work) for _ in range(2)]
        contexts = [f.result(timeout=10) for f in futures]
        executor.submit(handle.finish).result(timeout=5)
    run = _runs(spans)[0]
    assert contexts == [run.context, run.context]
    assert run.parent is None
    _assert_run(spans, run, handle.run_id)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detached_activations_are_task_local(spans):
    handle = solwyn.create_run("PRIVATE")
    ready = asyncio.Event()
    count = 0

    async def work():
        nonlocal count
        with spans.tracer.start_as_current_span("task") as task:
            async with handle.activate():
                count += 1
                if count == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=5)
                current = trace.get_current_span().get_span_context()
            assert trace.get_current_span() is task
            return current

    contexts = await asyncio.gather(work(), work())
    handle.finish()
    run = _runs(spans)[0]
    assert contexts == [run.context, run.context]
    _assert_run(spans, run, handle.run_id)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt, asyncio.CancelledError])
async def test_detach_failure_is_soft_and_restores_parent(spans, detached, async_mode, failure):
    with spans.tracer.start_as_current_span("parent") as parent:
        handle = solwyn.create_run("PRIVATE") if detached else None
        scope = handle.activate() if detached else solwyn.run("PRIVATE")
        run_id = await scope.__aenter__() if async_mode else scope.__enter__()
        raised = None
        with patch.object(context, "detach", side_effect=failure("different Context")):
            try:
                if async_mode:
                    await scope.__aexit__(None, None, None)
                else:
                    scope.__exit__(None, None, None)
            except BaseException as exc:
                raised = type(exc)
        assert raised is None
        assert trace.get_current_span() is parent
        assert solwyn.current_run_context().id is None
        if handle is not None:
            handle.finish()
    _assert_run(spans, _runs(spans)[0], run_id)


@pytest.mark.unit
def test_scope_exit_in_copied_context_restores_both_contexts(spans):
    with spans.tracer.start_as_current_span("parent") as parent:
        scope = solwyn.run("PRIVATE")
        run_id = scope.__enter__()
        foreign = copy_context()
        foreign.run(scope.__exit__, None, None, None)
        assert foreign.run(trace.get_current_span) is parent
        scope.__exit__(None, None, None)
        assert trace.get_current_span() is parent
    _assert_run(spans, _runs(spans)[0], run_id)


@pytest.mark.unit
@pytest.mark.parametrize("terminal", ["normal", "exception", "close"])
def test_generator_scope_ends_span_when_each_step_has_a_fresh_context(spans, terminal):
    scope = solwyn.run("PRIVATE")

    def chunks():
        with scope as run_id:
            yield run_id
            yield 2
            if terminal == "exception":
                raise ValueError("PRIVATE")

    generator = chunks()
    base = copy_context()
    entered = base.copy()
    run_id = entered.run(next, generator)
    assert base.copy().run(next, generator) == 2
    assert not _runs(spans)
    if terminal == "close":
        base.copy().run(generator.close)
    else:
        with pytest.raises(ValueError if terminal == "exception" else StopIteration):
            base.copy().run(next, generator)
    assert len(_runs(spans)) == 1
    _assert_run(spans, _runs(spans)[0], run_id, error=terminal != "normal")

    # Cleaning up the still-visible original frame must not end the span again.
    entered.run(scope.__exit__, None, None, None)
    _assert_run(spans, _runs(spans)[0], run_id, error=terminal != "normal")


@pytest.mark.unit
def test_scope_without_its_frame_ends_span_and_preserves_unrelated_run(spans):
    scope = solwyn.run("PRIVATE")
    entered = copy_context()
    run_id = entered.run(scope.__enter__)
    with solwyn.run("PRIVATE unrelated") as unrelated_id:
        unrelated_span = trace.get_current_span()
        scope.__exit__(None, None, None)
        assert solwyn.current_run_context().id == unrelated_id
        assert trace.get_current_span() is unrelated_span
        assert len(_runs(spans)) == 1
        _assert_run(spans, _runs(spans)[0], run_id)
    entered.run(scope.__exit__, None, None, None)
    _assert_run(spans, _runs(spans)[0], run_id)


@pytest.mark.unit
@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("terminal", ["exhaustion", "exception", "close", "gc"])
def test_fresh_context_calls_remain_children_of_run_and_end_once(spans, outer, terminal):
    plane = FakeControlPlane()
    raw = _raw_client(False)
    raw.chat.completions.create.return_value = _response()
    wrapper = plane.wrap(raw, otel_spans=True)

    def chunks():
        with solwyn.run("chunks") as run_id:
            for _ in range(3):
                wrapper.chat.completions.create(model="gpt-5.5", messages=[])
                yield run_id
            if terminal == "exception":
                raise ValueError("stream failed")

    try:
        with solwyn.run("request") if outer else nullcontext() as parent_id:
            base = copy_context()
            generator = chunks()
            run_ids = [base.copy().run(next, generator) for _ in range(3)]
            if terminal in {"exhaustion", "exception"}:
                with pytest.raises(ValueError if terminal == "exception" else StopIteration):
                    base.copy().run(next, generator)
            elif terminal == "close":
                base.copy().run(generator.close)
            else:
                ref = weakref.ref(generator)
                del generator
                gc.collect()
                assert ref() is None
            run = next(s for s in _runs(spans) if s.attributes["solwyn.agent_run_id"] == run_ids[0])
            _assert_run(spans, run, run_ids[0], parent_id=parent_id, error=terminal != "exhaustion")
            assert len(spans.calls()) == 3
            assert all(call.parent.span_id == run.context.span_id for call in spans.calls())
            assert all(call.context.trace_id == run.context.trace_id for call in spans.calls())
    finally:
        wrapper.close()


@pytest.mark.unit
def test_shared_generator_scope_closes_its_own_span_and_keeps_other_run_open(spans):
    scope = solwyn.run("shared")

    def chunks():
        with scope as run_id:
            yield run_id
            yield solwyn.current_run_context().id

    base = copy_context()
    first, second = chunks(), chunks()
    first_id = base.copy().run(next, first)
    second_id = base.copy().run(next, second)
    base.copy().run(first.close)
    assert len(_runs(spans)) == 1
    _assert_run(spans, _runs(spans)[0], first_id, error=True)
    assert base.copy().run(next, second) == second_id
    base.copy().run(second.close)
    assert len(_runs(spans)) == 2
    _assert_run(spans, _runs(spans)[1], second_id, error=True)


@pytest.mark.unit
def test_refused_started_handle_finish_keeps_span_open(spans):
    outer = solwyn.start_run("PRIVATE outer")
    outer_context = trace.get_current_span().get_span_context()
    inner = solwyn.start_run("PRIVATE inner")
    with pytest.raises(RuntimeError, match="LIFO"):
        outer.finish()
    assert not _runs(spans)
    inner.finish()
    foreign = copy_context()
    with pytest.raises(RuntimeError, match="same context"):
        foreign.run(outer.finish)
    assert len(_runs(spans)) == 1
    assert trace.get_current_span().get_span_context() == outer_context
    outer.finish()
    assert len(_runs(spans)) == 2
    _assert_run(spans, _runs(spans)[-1], outer.run_id)


@pytest.mark.unit
def test_interrupted_scope_attachment_ends_the_new_span(spans):
    with spans.tracer.start_as_current_span("parent") as parent:
        with (
            patch.object(context, "attach", side_effect=KeyboardInterrupt()),
            pytest.raises(KeyboardInterrupt),
            solwyn.run("PRIVATE"),
        ):
            pass
        assert solwyn.current_run_context().id is None
        assert trace.get_current_span() is parent
    assert len(_runs(spans)) == 1
    run = _runs(spans)[0]
    _assert_run(spans, run, run.attributes["solwyn.agent_run_id"], error=True)


@pytest.mark.unit
def test_interrupted_detached_activation_is_retryable(spans):
    handle = solwyn.create_run("PRIVATE")
    with (
        patch.object(context, "attach", side_effect=KeyboardInterrupt()),
        pytest.raises(KeyboardInterrupt),
        handle.activate(),
    ):
        pass
    assert solwyn.current_run_context().id is None
    assert not trace.get_current_span().get_span_context().is_valid
    with handle.activate():
        pass
    handle.finish()
    _assert_run(spans, _runs(spans)[0], handle.run_id)


@pytest.mark.unit
@pytest.mark.parametrize("detached", [False, True])
def test_interrupted_record_registration_retires_owned_lifetime(spans, detached):
    handle = solwyn.create_run("detached") if detached else None
    scope = handle.activate() if detached else solwyn.run("scoped")
    with (
        patch("solwyn._run.register_fork_reset", side_effect=KeyboardInterrupt()),
        pytest.raises(KeyboardInterrupt),
    ):
        scope.__enter__()
    assert solwyn.current_run_context().id is None
    if handle is not None:
        with handle.activate():
            pass
        handle.finish()
    assert len(_runs(spans)) == 1
    run = _runs(spans)[0]
    _assert_run(spans, run, run.attributes["solwyn.agent_run_id"], error=not detached)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["scope", "started", "detached"])
def test_interruption_after_attachment_restores_context_before_ownership_transfer(spans, kind):
    with spans.tracer.start_as_current_span("parent") as parent:
        handle = solwyn.create_run("PRIVATE") if kind == "detached" else None
        scope = handle.activate() if handle is not None else solwyn.run("PRIVATE")
        with (
            patch.object(_otel, "_RunSpanActivation", autospec=True, side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            if kind == "started":
                solwyn.start_run("PRIVATE")
            else:
                scope.__enter__()
        assert solwyn.current_run_context().id is None
        assert trace.get_current_span() is parent
        if handle is not None:
            with handle.activate():
                pass
            handle.finish()
    assert len(_runs(spans)) == 1
    run = _runs(spans)[0]
    _assert_run(spans, run, run.attributes["solwyn.agent_run_id"], error=kind != "detached")


@pytest.mark.unit
@pytest.mark.parametrize("enabled_by", ["off", "client", "environment"])
def test_run_opt_in_uses_existing_flag(spans, monkeypatch, enabled_by):
    monkeypatch.delenv("SOLWYN_OTEL_SPANS", raising=False)
    plane = FakeControlPlane()
    wrapper = plane.wrap(_raw_client(False), otel_spans=enabled_by == "client")
    if enabled_by == "environment":
        monkeypatch.setenv("SOLWYN_OTEL_SPANS", "true")
    try:
        with solwyn.run("PRIVATE"):
            pass
        handle = solwyn.create_run("PRIVATE detached")
        with handle.activate():
            pass
        handle.finish()
    finally:
        wrapper.close()
    assert len(_runs(spans)) == (0 if enabled_by == "off" else 2)


@pytest.mark.unit
def test_disabled_runs_never_resolve_the_optional_api(monkeypatch):
    monkeypatch.delenv("SOLWYN_OTEL_SPANS", raising=False)
    with patch.object(_otel, "_get_trace_api", side_effect=AssertionError("disabled")):
        with solwyn.run("PRIVATE"):
            pass
        started = solwyn.start_run("PRIVATE started")
        started.finish()
        handle = solwyn.create_run("PRIVATE detached")
        with handle.activate():
            pass
        handle.finish()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("enabled_by", ["client", "environment"])
async def test_run_opt_in_is_sticky_after_wrapper_close_without_gc(
    spans, monkeypatch, async_mode, enabled_by
):
    monkeypatch.delenv("SOLWYN_OTEL_SPANS", raising=False)
    if enabled_by == "environment":
        monkeypatch.setenv("SOLWYN_OTEL_SPANS", "yes")
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        _raw_client(async_mode), otel_spans=enabled_by == "client"
    )
    with solwyn.run("PRIVATE first"):
        pass
    if async_mode:
        await wrapper.close()
    else:
        wrapper.close()
    del wrapper
    monkeypatch.setenv("SOLWYN_OTEL_SPANS", "false")

    # A later opted-out wrapper must not undo the process-wide run-span opt-in.
    wrapper = plane.wrap(_raw_client(False), otel_spans=False)
    try:
        with solwyn.run("PRIVATE next"):
            pass
        handle = solwyn.create_run("PRIVATE detached")
        handle.finish()
    finally:
        wrapper.close()
    assert len(_runs(spans)) == 3


@pytest.mark.unit
def test_run_opt_in_survives_wrapper_collection_without_retaining_it(spans, monkeypatch):
    monkeypatch.delenv("SOLWYN_OTEL_SPANS", raising=False)
    wrapper = FakeControlPlane().wrap(_raw_client(False), otel_spans=True)
    reference = weakref.ref(wrapper)
    wrapper.close()
    del wrapper
    gc.collect()
    assert reference() is None
    with solwyn.run("PRIVATE"):
        pass
    assert len(_runs(spans)) == 1


@pytest.mark.unit
def test_langchain_detached_callback_runs_inherit_spans(spans, handler_type):
    handler = handler_type(tags={"PRIVATE": "tag"})
    manager = _FakeCallbackManager(handler)
    root, child, model = uuid4(), uuid4(), uuid4()
    manager.start(run_id=root, name="PRIVATE root")
    manager.start(run_id=child, parent_run_id=root, name="PRIVATE child")
    manager.chat_model_start(run_id=model, parent_run_id=child)
    with handler.activate_model_call():
        child_id = solwyn.current_run_context().id
        child_context = trace.get_current_span().get_span_context()
    manager.model_end(run_id=model, parent_run_id=child)
    manager.end(run_id=child, parent_run_id=root)
    manager.end(run_id=root)
    assert len(_runs(spans)) == 2
    child_span, root_span = _runs(spans)
    assert child_span.context == child_context
    assert child_span.parent.span_id == root_span.context.span_id
    root_id = root_span.attributes["solwyn.agent_run_id"]
    _assert_run(spans, root_span, root_id)
    _assert_run(spans, child_span, child_id, parent_id=root_id)


@pytest.mark.unit
def test_crewai_detached_callback_runs_inherit_spans(spans, listener_protocol):
    listener_type, bus, events = listener_protocol
    listener = listener_type(tags={"PRIVATE": "tag"})
    task = _task_source()
    crew = _crew_source(task, name="PRIVATE crew")
    bus.emit(events["CrewKickoffStartedEvent"], crew)
    bus.emit(events["TaskStartedEvent"], task)
    with listener.activate(task):
        task_id = solwyn.current_run_context().id
        task_context = trace.get_current_span().get_span_context()
    bus.emit(events["TaskCompletedEvent"], task)
    bus.emit(events["CrewKickoffCompletedEvent"], crew)
    assert len(_runs(spans)) == 2
    task_span, crew_span = _runs(spans)
    assert task_span.context == task_context
    assert task_span.parent.span_id == crew_span.context.span_id
    crew_id = crew_span.attributes["solwyn.agent_run_id"]
    _assert_run(spans, crew_span, crew_id)
    _assert_run(spans, task_span, task_id, parent_id=crew_id)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["absent", "noop", "drop", "record_only"])
def test_nonexportable_run_spans_preserve_caller_context(spans, monkeypatch, mode):
    with spans.tracer.start_as_current_span("parent") as parent:
        if mode == "absent":
            monkeypatch.setattr(_otel, "_trace_api", None)
        else:
            provider = (
                trace.NoOpTracerProvider()
                if mode == "noop"
                else type(spans.provider)(
                    sampler=sampling.StaticSampler(
                        sampling.Decision.DROP if mode == "drop" else sampling.Decision.RECORD_ONLY
                    )
                )
            )
            monkeypatch.setattr(spans.api, "get_tracer", provider.get_tracer)
        with solwyn.run("PRIVATE"):
            assert trace.get_current_span() is parent
        handle = solwyn.create_run("PRIVATE")
        with handle.activate():
            assert trace.get_current_span() is parent
        handle.finish()
    assert not _runs(spans)


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["success", "error", "adoption_error"])
def test_run_status_objects_are_portable_to_older_otel_sdks(spans, monkeypatch, outcome):
    original_set_status = sdk.Span.set_status

    def old_set_status(span, status):
        if not isinstance(status, trace.Status):
            raise TypeError("Status required")
        return original_set_status(span, status)

    monkeypatch.setattr(sdk.Span, "set_status", old_set_status)
    if outcome == "adoption_error":
        span = spans.tracer.start_span("solwyn.run")

        def fail():
            raise ValueError("PRIVATE")

        monkeypatch.setattr(span, "is_recording", fail)
        tracer = SimpleNamespace(start_span=lambda *args, **kwargs: span)
        monkeypatch.setattr(spans.api, "get_tracer", lambda *args, **kwargs: tracer)
    with (
        pytest.raises(ValueError) if outcome == "error" else nullcontext(),
        solwyn.run("PRIVATE"),
    ):
        if outcome == "error":
            raise ValueError("PRIVATE")
    run = _runs(spans)[0]
    assert run.status.status_code == (
        trace.StatusCode.OK if outcome == "success" else trace.StatusCode.ERROR
    )
    assert run.status.description is None
    assert not run.events
    assert spans.ends[run.context.span_id] == 1
