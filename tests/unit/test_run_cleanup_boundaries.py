"""Cleanup must stay safe after handled exceptions, slow exporters, and fork."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextvars import copy_context

import pytest
from conftest import VALID_API_KEY

import solwyn
from solwyn._lifecycle import _drain_run_cleanups
from solwyn.budget import AsyncBudgetEnforcer
from solwyn.testing import FakeControlPlane

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("started", [False, True])
def test_same_context_gc_after_handled_generator_exit_never_enters_enforcer(started):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc, sys
import solwyn
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane
gc.disable()
plane = FakeControlPlane()
enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
class Stream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        if sys.argv[1] == 'True':
            handle = solwyn.start_run('stream')
            try:
                yield handle.run_id
            except GeneratorExit:
                pass
            handle.finish()
        else:
            with solwyn.run('stream') as rid:
                try:
                    yield rid
                except GeneratorExit:
                    pass
stream = Stream()
rid = next(stream.chunks)
result = enforcer.check_budget(estimated_input_tokens=1, model='gpt-5.5',
                             provider='openai', agent_run_id=rid)
del stream
with enforcer._state_lock:
    gc.collect()
enforcer.close()
assert len(plane.lease_surrenders) == 1
assert solwyn.current_run()[0] is None
""",
            str(started),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.asyncio
async def test_async_holder_surrenders_an_idle_stream_from_the_cleanup_worker():
    plane = FakeControlPlane()
    enforcer = AsyncBudgetEnforcer(plane.api_url, VALID_API_KEY, transport=plane.transport)

    def chunks():
        with solwyn.run("stream") as rid:
            yield rid

    stream = chunks()
    rid = copy_context().run(next, stream)
    result = await enforcer.check_budget(
        estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id=rid
    )
    try:
        copy_context().run(stream.close)
        async with asyncio.timeout(2):
            while not plane.lease_surrenders:
                await asyncio.sleep(0.01)
        assert [s.lease_id for s in plane.lease_surrenders] == [result.lease_id]
    finally:
        await enforcer.close()


def test_same_context_finish_while_handling_generator_exit_is_valid():
    handle = solwyn.start_run("ordinary")
    try:
        try:
            raise GeneratorExit
        finally:
            handle.finish()
    except GeneratorExit:
        pass
    _drain_run_cleanups()
    assert handle._finished
    assert solwyn.current_run()[0] is None


def test_finish_with_a_live_activation_still_rejects_during_generator_exit():
    handle = solwyn.create_run("detached")
    with handle.activate():
        try:
            try:
                raise GeneratorExit
            finally:
                with pytest.raises(RuntimeError, match="activations are still active"):
                    handle.finish()
        except GeneratorExit:
            pass
    handle.finish()


def test_slow_span_export_cannot_hold_a_reporter_close_past_its_deadline():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc, os, threading, time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
import solwyn
from solwyn.reporter import MetadataReporter
from solwyn.testing import FakeControlPlane
entered, release = threading.Event(), threading.Event()
class Exporter(SpanExporter):
    def export(self, spans):
        entered.set()
        release.wait(3)
        return SpanExportResult.SUCCESS
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(Exporter()))
trace.set_tracer_provider(provider)
os.environ['SOLWYN_OTEL_SPANS'] = 'true'
plane = FakeControlPlane()
reporter = MetadataReporter(plane.api_url, plane.api_key, transport=plane.transport)
handle = solwyn.create_run('abandoned')
del handle
gc.collect()
assert entered.wait(2)
timer = threading.Timer(.5, release.set)
timer.start()
started = time.monotonic()
reporter.close(timeout=.01)
elapsed = time.monotonic() - started
release.set()
timer.join()
assert elapsed < .2, elapsed
provider.shutdown()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=7,
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_retains_cleanup_already_started_by_a_worker():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, signal, time
from contextvars import copy_context
import solwyn
from solwyn import _run
from solwyn._lifecycle import _drain_run_cleanups
handle = solwyn.create_run('detached')
def chunks():
    with handle.activate():
        yield
stream = chunks()
entered = copy_context()
entered.run(next, stream)
record = entered.run(_run._run_frames.get)[-1].record
with record.lock:
    copy_context().run(stream.close)
    time.sleep(.25)
    pid = os.fork()
    if pid == 0:
        signal.alarm(2)
        _drain_run_cleanups()
        handle.finish()
        assert not handle._active_activations
        os._exit(0)
_, status = os.waitpid(pid, 0)
_drain_run_cleanups()
handle.finish()
assert os.waitstatus_to_exitcode(status) == 0, status
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=7,
    )


def test_concurrent_holder_registration_cannot_interrupt_generator_cleanup():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, threading
from contextvars import copy_context
import solwyn
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane
plane = FakeControlPlane()
holders = [BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
           for _ in range(100)]
stopped = threading.Event()
thread_errors = []
threading.excepthook = lambda args: thread_errors.append(args.exc_value)
def construct():
    while not stopped.is_set() and len(holders) < 1000:
        holders.append(BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport))
def chunks():
    with solwyn.run('stream'):
        yield
sys.setswitchinterval(.000001)
thread = threading.Thread(target=construct)
thread.start()
try:
    for _ in range(300):
        stream = chunks()
        copy_context().run(next, stream)
        copy_context().run(stream.close)
finally:
    stopped.set()
    thread.join(3)
for holder in holders:
    holder.close()
assert not thread_errors, thread_errors
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
