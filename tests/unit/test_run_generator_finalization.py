"""Finalizing a disconnected stream must never enter budget or tracing locks."""

from __future__ import annotations

import subprocess
import sys
import threading
from contextvars import copy_context

import pytest
from conftest import VALID_API_KEY

import solwyn
from solwyn import _lifecycle
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["scope", "started", "detached"])
@pytest.mark.parametrize(
    "held_lock", ["enforcer", "lifecycle", "span", "record", "scope", "handle"]
)
def test_gc_cleanup_with_real_enforcer_never_waits_for_a_held_lock(kind, held_lock):
    subprocess.run(
        [sys.executable, "-c", _FINALIZATION_PROBE, kind, held_lock],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("kind", ["scope", "started", "detached"])
def test_gc_on_the_thread_holding_the_real_enforcer_lock_never_deadlocks(kind):
    subprocess.run(
        [sys.executable, "-c", _FINALIZATION_PROBE, kind, "enforcer", "same-thread"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


_FINALIZATION_PROBE = """
import gc
import os
import sys
import threading
import time
from contextvars import copy_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import solwyn
from solwyn import _lifecycle, _run
from solwyn._lifecycle import _drain_run_cleanups, _drain_run_span_cleanups
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane

gc.disable()
os.environ['SOLWYN_OTEL_SPANS'] = 'true'
provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
plane = FakeControlPlane()
enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
kind, lock_kind = sys.argv[1:3]
scope = solwyn.run('stream')
handles = []

class ChatStream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        handle = solwyn.create_run('stream') if kind == 'detached' else None
        if kind == 'started':
            handle = solwyn.start_run('stream')
            handles.append(handle)
            try:
                yield handle.run_id
            finally:
                handle.finish()
        else:
            if handle:
                handles.append(handle)
            with handle.activate() if handle else scope as rid:
                yield rid
            if handle:
                handle.finish()

stream = ChatStream()
entered = copy_context()
rid = entered.run(next, stream.chunks)
record = entered.run(_run._run_frames.get)[-1].record
result = enforcer.check_budget(estimated_input_tokens=10, model='gpt-5.5',
                             provider='openai', agent_run_id=rid)
assert result.lease_id is not None
handle = handles[0] if handles else None
lock = {
    'enforcer': enforcer._state_lock,
    'lifecycle': _lifecycle._registry_lock,
    'span': record.span._span._lock,
    'record': record.lock,
    'scope': scope._records_lock,
    'handle': handle._state_lock if handle else scope._records_lock,
}[lock_kind]

# Hold even reentrant locks on ANOTHER thread: the GC thread must not wait.
locked, release = threading.Event(), threading.Event()
def hold():
    with lock:
        locked.set()
        release.wait(6)
del entered, record, stream
if len(sys.argv) > 3:
    with enforcer._state_lock:
        gc.collect()
else:
    thread = threading.Thread(target=hold)
    thread.start()
    assert locked.wait(2)
    started = time.monotonic()
    gc.collect()
    elapsed = time.monotonic() - started
    assert elapsed < 1, elapsed
    release.set()
    thread.join(2)
if kind == 'detached':
    handle.finish()
enforcer.close()
_drain_run_cleanups()
_drain_run_span_cleanups()
assert [s.lease_id for s in plane.lease_surrenders] == [result.lease_id]
assert len(exporter.get_finished_spans()) == 1
assert solwyn.current_run() == (None, None)
provider.shutdown()
"""


def test_pending_surrender_finishes_before_a_later_grant_for_the_same_run():
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(plane.api_url, VALID_API_KEY, transport=plane.transport)

    def chunks():
        with solwyn.run("stream") as rid:
            yield rid

    stream = chunks()
    rid = copy_context().run(next, stream)
    first = enforcer.check_budget(
        estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id=rid
    )
    try:
        with _lifecycle._run_cleanup_lock:
            copy_context().run(stream.close)
            second = enforcer.check_budget(
                estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id=rid
            )
        assert second.lease_id != first.lease_id
        assert [s.lease_id for s in plane.lease_surrenders] == [first.lease_id]
        assert len(plane.lease_grants) == 2
    finally:
        enforcer.close()
    assert [s.lease_id for s in plane.lease_surrenders] == [first.lease_id, second.lease_id]


def test_idle_stream_is_surrendered_without_another_sdk_entry():
    import httpx

    plane = FakeControlPlane()
    surrendered = threading.Event()

    def dispatch(request):
        response = plane.transport.handle_request(request)
        if request.url.path.endswith("/surrender"):
            surrendered.set()
        return response

    enforcer = BudgetEnforcer(plane.api_url, VALID_API_KEY, transport=httpx.MockTransport(dispatch))

    def chunks():
        with solwyn.run("stream") as rid:
            yield rid

    stream = chunks()
    rid = copy_context().run(next, stream)
    result = enforcer.check_budget(
        estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id=rid
    )
    try:
        copy_context().run(stream.close)
        assert surrendered.wait(2)
        assert [s.lease_id for s in plane.lease_surrenders] == [result.lease_id]
    finally:
        enforcer.close()


def test_stale_context_cleanup_cannot_surrender_a_successor_lease():
    plane = FakeControlPlane()
    enforcer = BudgetEnforcer(plane.api_url, VALID_API_KEY, transport=plane.transport)
    scope = solwyn.run("stream")

    def chunks():
        with scope as rid:
            yield rid

    stream = chunks()
    entered = copy_context()
    rid = entered.run(next, stream)

    def check():
        return enforcer.check_budget(
            estimated_input_tokens=1, model="gpt-5.5", provider="openai", agent_run_id=rid
        )

    try:
        first = check()
        copy_context().run(stream.close)
        successor = check()
        entered.run(scope.__exit__, None, None, None)
        assert check().lease_id == successor.lease_id
        assert len(plane.lease_grants) == 2
        assert [s.lease_id for s in plane.lease_surrenders] == [first.lease_id]
    finally:
        enforcer.close()


def test_natural_gc_during_real_budget_checks_completes():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
from contextvars import copy_context
import solwyn
from solwyn.budget import BudgetEnforcer
from solwyn.testing import FakeControlPlane

plane = FakeControlPlane()
enforcer = BudgetEnforcer(plane.api_url, plane.api_key, transport=plane.transport)
class ChatStream:
    def __init__(self):
        self.chunks = self.generate()
    def generate(self):
        with solwyn.run('abandoned'):
            yield 1

with solwyn.run('request') as rid:
    for _ in range(5000):
        stream = ChatStream()
        copy_context().run(next, stream.chunks)
        del stream
        result = enforcer.check_budget(estimated_input_tokens=1, model='gpt-5.5',
                                     provider='openai', agent_run_id=rid)
        assert result.allowed
gc.collect()
enforcer.close()
assert len(plane.lease_grants) >= 1
assert len(plane.lease_surrenders) >= 1
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
