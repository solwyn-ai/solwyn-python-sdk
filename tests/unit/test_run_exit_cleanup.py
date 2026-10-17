"""Normal exits finish immediately; finalizers hand off without changing end times."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("processor", ["simple", "batch"])
@pytest.mark.parametrize("kind", ["scope", "started", "detached"])
@pytest.mark.parametrize("terminal", ["exhausted", "exception"])
def test_generator_exit_exports_run_span_before_return(processor, kind, terminal):
    subprocess.run(
        [sys.executable, "-c", _NORMAL_EXIT, processor, kind, terminal],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


_TRACING = """
import gc, os, sys, time
from contextvars import copy_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import solwyn
from solwyn import _lifecycle
gc.disable()
os.environ['SOLWYN_OTEL_SPANS'] = 'true'
provider = TracerProvider()
exporter = InMemorySpanExporter()
processor = (BatchSpanProcessor(exporter, schedule_delay_millis=60000)
             if sys.argv[1] == 'batch' else SimpleSpanProcessor(exporter))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
"""

_NORMAL_EXIT = (
    _TRACING
    + """
kind, terminal = sys.argv[2:4]
exit_started = None
handle = solwyn.create_run('callback') if kind == 'detached' else None
def pipeline():
    global exit_started
    if kind == 'scope':
        with solwyn.run('pipeline'):
            yield 1
            exit_started = time.time_ns()
            if terminal == 'exception':
                raise ValueError()
    else:
        run = handle or solwyn.start_run('callback')
        try:
            yield 1
            if terminal == 'exception':
                raise ValueError()
        finally:
            exit_started = time.time_ns()
            run.finish()
            assert run._finished
try:
    for _ in pipeline():
        pass
except ValueError:
    assert terminal == 'exception'
returned = time.time_ns()
assert provider.force_flush()
spans = exporter.get_finished_spans()
assert len(spans) == 1, spans
assert exit_started <= spans[0].end_time <= returned, (exit_started, spans[0].end_time, returned)
provider.shutdown()
"""
)


@pytest.mark.parametrize("kind", ["scope", "started", "detached", "abandoned"])
@pytest.mark.parametrize("terminal", ["close", "gc"])
def test_deferred_span_uses_exit_timestamp_after_delayed_drain(kind, terminal):
    subprocess.run(
        [sys.executable, "-c", _DEFERRED_EXIT, "simple", kind, terminal],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


_DEFERRED_EXIT = (
    _TRACING
    + """
kind, terminal = sys.argv[2:4]
handle = solwyn.create_run('detached') if kind in ('detached', 'abandoned') else None
class Stream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        if kind == 'started':
            run = solwyn.start_run('started')
            try:
                yield
            finally:
                run.finish()
        elif handle:
            try:
                with handle.activate():
                    yield
            finally:
                if kind != 'abandoned':
                    handle.finish()
        else:
            with solwyn.run('stream'):
                yield
stream = Stream()
copy_context().run(next, stream.chunks)
with _lifecycle._run_cleanup_lock, _lifecycle._run_span_cleanup_lock:
    before = time.time_ns()
    if kind == 'abandoned':
        # The generator's activation shares the handle; close it before
        # dropping the final reference to the detached handle itself.
        copy_context().run(stream.chunks.close)
        _lifecycle._drain_run_cleanups()
        del stream, handle
        gc.collect()
    elif terminal == 'close':
        copy_context().run(stream.chunks.close)
    else:
        del stream
        gc.collect()
    returned = time.time_ns()
    assert not exporter.get_finished_spans()
    time.sleep(.05)
_lifecycle._drain_run_cleanups()
_lifecycle._drain_run_span_cleanups()
spans = exporter.get_finished_spans()
assert len(spans) == 1, spans
assert before <= spans[0].end_time <= returned, (before, spans[0].end_time, returned)
provider.shutdown()
"""
)


@pytest.mark.parametrize("boundary", ["exit", "close"])
def test_shutdown_drains_pending_run_spans_and_real_leases(boundary):
    subprocess.run(
        [sys.executable, "-c", _SHUTDOWN, "simple", boundary],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


_SHUTDOWN = (
    _TRACING
    + """
from unittest.mock import MagicMock
from solwyn.testing import FakeControlPlane
plane = FakeControlPlane()
raw = MagicMock(spec=['close'])
raw.__class__.__module__ = 'openai'
raw.__class__.__name__ = 'OpenAI'
client = plane.wrap(raw)
def pipeline():
    with solwyn.run('stream') as rid:
        yield rid
stream = pipeline()
rid = copy_context().run(next, stream)
result = client._solwyn_budget.check_budget(estimated_input_tokens=1, model='gpt-5.5',
                                          provider='openai', agent_run_id=rid)
with _lifecycle._run_cleanup_lock:
    copy_context().run(stream.close)
    if sys.argv[2] == 'exit':
        _lifecycle._exit_flush_all()
    else:
        client.close()
    assert len(exporter.get_finished_spans()) == 1
    assert [s.lease_id for s in plane.lease_surrenders] == [result.lease_id]
provider.shutdown()
"""
)


def test_ordinary_handles_and_streams_start_no_cleanup_workers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc, os, threading
from unittest.mock import MagicMock
import solwyn
from solwyn import _run_control
from solwyn.providers._accumulator import StreamUsageAccumulator
from solwyn.stream import SyncStreamWrapper
gc.disable()
os.environ['SOLWYN_OTEL_SPANS'] = 'false'
handle = solwyn.create_run('ordinary')
with handle.activate():
    pass
handle.finish()
def pipeline():
    with solwyn.run('ordinary'):
        yield
for _ in pipeline():
    pass
termination = _run_control._acquire_termination_handle('stream')
stream = SyncStreamWrapper(iter(()), MagicMock(spec=StreamUsageAccumulator),
                           lambda *args: None, lambda *args: None,
                           abort_release=termination.release)
list(stream)
assert not [t.name for t in threading.enumerate() if t.name.startswith('solwyn-run-')]
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_activation_in_a_gc_finally_never_joins_a_blocked_cleanup():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc, threading, time
from contextvars import copy_context
import solwyn
from solwyn import _lifecycle, _run
gc.disable()
first, second = solwyn.create_run('first'), solwyn.create_run('second')
entered = threading.Event()
def blocked():
    entered.set()
    with _run._generator_runs_lock:
        pass
class Stream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        try:
            yield
        finally:
            with second.activate():
                pass
stream = Stream()
next(stream.chunks)
del stream
with _run._generator_runs_lock:
    _lifecycle._defer_run_cleanup(blocked)
    worker = threading.Thread(target=_lifecycle._drain_run_cleanups)
    worker.start()
    assert entered.wait(2)
    gc.collect()
worker.join(2)
assert not worker.is_alive()
_lifecycle._drain_run_cleanups()
first.finish()
second.finish()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_gc_of_a_coroutine_with_a_contextmanager_run_never_enters_enforcer():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import contextlib, gc, types
import solwyn
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane
gc.disable()
plane = FakeControlPlane()
enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
@contextlib.contextmanager
def traced():
    with solwyn.run('coroutine') as rid:
        yield rid
@types.coroutine
def suspend():
    yield
async def work(box):
    with traced() as rid:
        enforcer.check_budget(estimated_input_tokens=1, model='gpt-5.5',
                              provider='openai', agent_run_id=rid)
        await suspend()
box = []
coroutine = work(box)
box.append(coroutine)
coroutine.send(None)
del coroutine, box
with enforcer._state_lock:
    gc.collect()
enforcer.close()
assert len(plane.lease_surrenders) == 1
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_first_finalizer_handoff_never_waits_for_threading_locks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc, threading, time
from contextvars import copy_context
import solwyn
from solwyn import _lifecycle
gc.disable()
def pipeline():
    with solwyn.run('stream'):
        yield
stream = pipeline()
copy_context().run(next, stream)
# Thread.start() needs this lock. The first handoff must return even if
# another thread holds it and cannot release it until finalization returns.
held, release = threading.Event(), threading.Event()
def hold():
    with threading._active_limbo_lock:
        held.set()
        release.wait(6)
holder = threading.Thread(target=hold)
holder.start()
assert held.wait(1)
started = time.monotonic()
copy_context().run(stream.close)
assert time.monotonic() - started < 1
assert not release.is_set()
release.set()
holder.join(1)
_lifecycle._drain_run_cleanups()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_collection_on_another_thread_keeps_normal_run_exit_synchronous():
    subprocess.run(
        [sys.executable, "-c", _CONCURRENT_GC, "simple"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


_CONCURRENT_GC = (
    _TRACING
    + """
import threading
entered, release = threading.Event(), threading.Event()
observed = []
class Cycle:
    def __del__(self):
        observed.append(_lifecycle._run_gc_active())
        entered.set()
        release.wait(3)
cycle = Cycle()
cycle.self = cycle
del cycle
collector = threading.Thread(target=gc.collect)
collector.start()
try:
    assert entered.wait(1)
    with solwyn.run('ordinary'):
        pass
    assert len(exporter.get_finished_spans()) == 1
    assert not _lifecycle._run_gc_active()
finally:
    release.set()
    collector.join(1)
assert observed == [True]
provider.shutdown()
"""
)


def test_interpreter_exit_drains_spans_without_a_client_or_another_sdk_entry():
    subprocess.run(
        [sys.executable, "-c", _INTERPRETER_EXIT, "simple"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


_INTERPRETER_EXIT = (
    _TRACING
    + """
import atexit
def verify():
    spans = exporter.get_finished_spans()
    if len(spans) != 1 or not before <= spans[0].end_time <= returned:
        os._exit(10)
# Register before the first run, so verification follows Solwyn's exit hook
# and precedes the application's tracer shutdown.
atexit.register(verify)
def pipeline():
    with solwyn.run('stream'):
        yield
stream = pipeline()
copy_context().run(next, stream)
# Keep both workers out until the actual atexit hook drains on this thread.
_lifecycle._run_cleanup_lock.acquire()
_lifecycle._run_span_cleanup_lock.acquire()
before = time.time_ns()
copy_context().run(stream.close)
returned = time.time_ns()
# Let the export worker proceed at shutdown, before Solwyn's exit hook waits
# on its barrier. Holding a private exporter lock across close is unsupported.
atexit.register(_lifecycle._run_span_cleanup_lock.release)
"""
)


@pytest.mark.parametrize("boundary", ["exit", "close"])
@pytest.mark.parametrize("exporting", [False, True])
def test_stuck_span_export_cannot_block_shutdown_of_spend_and_leases(boundary, exporting):
    subprocess.run(
        [sys.executable, "-c", _STUCK_EXPORTER, boundary, str(exporting)],
        check=True,
        capture_output=True,
        text=True,
        timeout=4,
    )


_STUCK_EXPORTER = """
import gc, os, sys, threading, time
from unittest.mock import MagicMock
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
import solwyn
from solwyn import _lifecycle
from solwyn.testing import FakeControlPlane
gc.disable()
os.environ['SOLWYN_OTEL_SPANS'] = 'true'
entered, release = threading.Event(), threading.Event()
class Exporter(SpanExporter):
    def export(self, spans):
        entered.set()
        release.wait()
        return SpanExportResult.SUCCESS
provider = TracerProvider(shutdown_on_exit=False)
provider.add_span_processor(SimpleSpanProcessor(Exporter()))
trace.set_tracer_provider(provider)
plane = FakeControlPlane()
raw = MagicMock(spec=['close'])
raw.__class__.__module__ = 'openai'
raw.__class__.__name__ = 'OpenAI'
client = plane.wrap(raw)
handle = solwyn.create_run('abandoned')
result = client._solwyn_budget.check_budget(estimated_input_tokens=1, model='gpt-5.5',
                                          provider='openai', agent_run_id=handle.run_id)
exporting = sys.argv[2] == 'True'
if not exporting:
    _lifecycle._run_span_cleanup_lock.acquire()
del handle
gc.collect()
if exporting:
    assert entered.wait(1)
try:
    started = time.monotonic()
    if sys.argv[1] == 'exit':
        _lifecycle._exit_flush_all()
    else:
        client.close()
    assert time.monotonic() - started < 2
    assert client._solwyn_reporter._close_is_completed()
    assert [s.lease_id for s in plane.lease_surrenders] == [result.lease_id]
finally:
    if not exporting:
        _lifecycle._run_span_cleanup_lock.release()
    release.set()
provider.shutdown()
"""
