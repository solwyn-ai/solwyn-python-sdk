"""Fork safety (R4): a forked child repairs its reporter/enforcer/breaker state.

Scenario-3 pin: a child forked from a process with a live reporter inherits a
DEAD flush thread. Without ``os.register_at_fork`` relaunching it, the child's
queued settlement is never delivered (os._exit bypasses the exit hook too). Plus
in-process ``_reset_after_fork_in_child`` unit tests for each participating class.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
from conftest import VALID_API_KEY, make_mock_client

from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.client import Solwyn
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

_URL = "https://api.test.solwyn.ai"


def _quiet(**kwargs) -> MetadataReporter:
    """A sync reporter with its background flush thread stopped (_shutdown SET)."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    return reporter


def _unstarted(**kwargs) -> MetadataReporter:
    """A sync reporter whose thread exited but whose _shutdown stays UNSET."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        reporter = MetadataReporter(_URL, VALID_API_KEY, **kwargs)
    reporter._thread.join(timeout=2.0)
    return reporter


# ---------------------------------------------------------------------------
# Scenario-3 pin: subprocess that forks
# ---------------------------------------------------------------------------


class _RecordingServer:
    """Records POST (path, call_id) tuples; 200s every request."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []
        recorder = self.seen

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                call_id = None
                try:
                    body = json.loads(raw)
                    if isinstance(body, dict):
                        call_id = body.get("call_id")
                except Exception:
                    pass
                recorder.append((self.path, call_id))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ingested":1,"rejected":[]}')

            def log_message(self, *_args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# call_id is pinned to the canonical UUID text form on the confirm wire, so the
# child's two ids are fixed UUIDs. They are literals inside the child source
# (it runs in a fresh interpreter with no test helpers), and the parent asserts
# on the same constants.
CHILD_CALL_ID = "11111111-1111-4111-8111-111111111111"
PARENT_CALL_ID = "22222222-2222-4222-8222-222222222222"

_CHILD = """
import os, sys, time
from datetime import UTC, datetime

import solwyn.reporter as r
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent, ProviderName


def confirm(cid):
    return BudgetConfirmRequest(
        reservation_id="res_" + cid, model="gpt-5.5", provider=ProviderName.OPENAI,
        call_id=cid, token_details=TokenDetails(input_tokens=10, output_tokens=5),
    )


def event(cid):
    return MetadataEvent(
        model="gpt-5.5", provider=ProviderName.OPENAI, input_tokens=100, output_tokens=50,
        latency_ms=1.0, status="success", is_model_fallback=False, sdk_instance_id="i",
        timestamp=datetime.now(UTC), call_id=cid,
    )


rep = r.MetadataReporter("http://127.0.0.1:{port}", "sk_test_key", flush_interval=0.2)
pid = os.fork()
if pid == 0:
    # Forked child: only a RELAUNCHED flush thread can deliver this — os._exit
    # forbids the atexit rescue, and the inherited thread is dead.
    rep.report_settlement(confirm("{child_id}"), event("{child_id}"))
    time.sleep(1.5)  # >> several flush intervals
    os._exit(0)
else:
    _, status = os.waitpid(pid, 0)
    # Report whether the child died on a signal: forking a multi-threaded
    # process is unreliable on some platforms (notably macOS, where the child
    # can SIGSEGV before running any Python). That is a platform artifact, not
    # a delivery failure, so the test retries such attempts.
    sys.stderr.write("CHILD_SIGNALED=%s\\n" % os.WIFSIGNALED(status))
    sys.stderr.flush()
    rep.report_settlement(confirm("{parent_id}"), event("{parent_id}"))
    rep.close()
"""

# Forking a multi-threaded process can crash the child on macOS; retry those
# attempts (they fail fast) so the pin measures OUR relaunch logic, not the
# platform's fork reliability. On Linux — the platform CI actually gates on —
# fork-in-threaded-process is the supported norm (it is multiprocessing's
# default start method), so a signaled child there is OUR at-fork bug and must
# FAIL, never retry into a skip: ten crashes turning the gate into a skip
# would let an at-fork regression leave CI green (#6 review pin).
_FORK_ATTEMPTS = 10
_FORK_CRASH_IS_PLATFORM_ARTIFACT = sys.platform != "linux"


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() unavailable on this platform")
def test_forked_child_flush_thread_delivers() -> None:
    server = _RecordingServer()
    crashed_attempts = 0
    try:
        for _ in range(_FORK_ATTEMPTS):
            server.seen.clear()
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _CHILD.format(
                        port=server.port, child_id=CHILD_CALL_ID, parent_id=PARENT_CALL_ID
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"child failed: {result.stderr[-800:]}"

            deadline = time.monotonic() + 5.0
            confirm_ids: set[str | None] = set()
            while time.monotonic() < deadline:
                confirm_ids = {cid for path, cid in server.seen if "budgets/confirm" in path}
                if {CHILD_CALL_ID, PARENT_CALL_ID} <= confirm_ids:
                    break
                time.sleep(0.05)

            if CHILD_CALL_ID in confirm_ids:
                break
            if "CHILD_SIGNALED=True" in result.stderr:
                if not _FORK_CRASH_IS_PLATFORM_ARTIFACT:
                    pytest.fail(
                        "the forked child died on a signal on Linux — the at-fork "
                        f"repair hooks must not crash the child: {result.stderr[-800:]}"
                    )
                # The forked child segfaulted before running any Python.
                crashed_attempts += 1
                continue
            # A healthy child that did not deliver is a real regression.
            pytest.fail(
                "the forked child ran but its settlement was never delivered — a "
                "relaunched flush thread is required; confirms seen: "
                f"{sorted(map(str, confirm_ids))}"
            )
        else:
            # Only reachable off-Linux: every attempt crashed pre-exec.
            pytest.skip(
                f"all {crashed_attempts}/{_FORK_ATTEMPTS} fork attempts crashed the "
                f"child ({sys.platform} fork-in-threaded-process instability; "
                "on Linux this is a hard failure)"
            )

        assert CHILD_CALL_ID in confirm_ids, (
            "the forked child's settlement was never delivered — a relaunched "
            f"flush thread is required; confirms seen: {sorted(map(str, confirm_ids))}"
        )
        assert PARENT_CALL_ID in confirm_ids, (
            f"the parent's settlement was not delivered; confirms seen: "
            f"{sorted(map(str, confirm_ids))}"
        )
    finally:
        server.close()


# ---------------------------------------------------------------------------
# In-process _reset_after_fork_in_child unit tests (no fork)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_circuit_breaker_reset_replaces_lock_keeps_health() -> None:
    breaker = CircuitBreaker(failure_threshold=2, name="control-plane")
    breaker.record_failure()
    old_lock = breaker._lock
    failures_before = breaker.failure_count
    state_before = breaker.state

    breaker._reset_after_fork_in_child()

    assert breaker._lock is not old_lock  # lock replaced
    # Health carries over — the child inherits the provider-health view.
    assert breaker.failure_count == failures_before
    assert breaker.state == state_before


@pytest.mark.unit
def test_sync_reporter_reset_relaunches_thread_and_swaps_client() -> None:
    reporter = _unstarted(flush_interval=3600.0)  # _shutdown UNSET, old thread exited
    old_http = reporter._http
    old_lock = reporter._in_flight_lock
    reporter._in_flight = 5

    reporter._reset_after_fork_in_child()

    assert reporter._http is not old_http
    assert reporter._in_flight_lock is not old_lock
    assert reporter._in_flight == 0
    assert reporter._breaker_worker is None
    # The reset defers the relaunch (starting a thread in the fork handler is
    # fragile); the next enqueue relaunches a fresh live flush thread.
    assert not reporter._thread.is_alive()
    assert reporter._needs_thread_restart is True
    reporter._ensure_thread()
    assert reporter._thread.is_alive()

    reporter._shutdown.set()
    reporter._thread.join(timeout=2.0)
    reporter._http.close()
    old_http.close()


@pytest.mark.unit
def test_sync_reporter_reset_closed_stays_closed() -> None:
    reporter = _quiet(flush_interval=3600.0)  # _shutdown SET
    old_http = reporter._http

    reporter._reset_after_fork_in_child()

    assert reporter._shutdown.is_set()
    assert not reporter._thread.is_alive()  # no relaunch for a closed reporter
    reporter._http.close()
    old_http.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_reporter_reset_clears_loop_state_and_swaps_client() -> None:
    reporter = AsyncMetadataReporter(_URL, VALID_API_KEY)
    old_http = reporter._http
    old_lock = reporter._breaker_project_lock
    reporter._in_flight = 3
    # Simulate a started reporter (real tasks/events belong to the parent loop).
    reporter._flush_task = object()  # type: ignore[assignment]
    reporter._breaker_task = object()  # type: ignore[assignment]
    reporter._shutdown_event = object()  # type: ignore[assignment]

    reporter._reset_after_fork_in_child()

    assert reporter._http is not old_http
    assert reporter._breaker_project_lock is not old_lock
    assert reporter._in_flight == 0
    assert reporter._flush_task is None
    assert reporter._breaker_task is None
    assert reporter._shutdown_event is None

    await reporter._http.aclose()
    await old_http.aclose()


@pytest.mark.unit
def test_solwyn_base_reset_replaces_client_locks() -> None:
    # _SolwynBase's two locks guard pure in-process state, but a user thread can
    # hold _solwyn_breaker_lock on the lazy provider-breaker creation path; inheriting it
    # held would deadlock the child's next breaker lookup.
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(make_mock_client(), api_key=VALID_API_KEY, model="gpt-5.5")
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)

    old_signal_lock = solwyn._solwyn_signal_lock
    old_breaker_lock = solwyn._solwyn_breaker_lock
    breakers_before = dict(solwyn._solwyn_circuit_breakers)

    solwyn._reset_after_fork_in_child()

    assert solwyn._solwyn_signal_lock is not old_signal_lock
    assert solwyn._solwyn_breaker_lock is not old_breaker_lock
    # Breaker objects (and their health) are legitimately inherited.
    assert solwyn._solwyn_circuit_breakers == breakers_before

    solwyn.close()


@pytest.mark.unit
def test_sync_budget_enforcer_reset_swaps_lock_and_client() -> None:
    enforcer = BudgetEnforcer(_URL, VALID_API_KEY)
    old_lock = enforcer._state_lock
    old_http = enforcer._http

    enforcer._reset_after_fork_in_child()

    assert enforcer._state_lock is not old_lock
    assert enforcer._http is not old_http
    enforcer.close()
    old_http.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_budget_enforcer_reset_swaps_lock_and_client() -> None:
    enforcer = AsyncBudgetEnforcer(_URL, VALID_API_KEY)
    old_lock = enforcer._state_lock
    old_http = enforcer._http

    enforcer._reset_after_fork_in_child()

    assert enforcer._state_lock is not old_lock
    assert enforcer._http is not old_http
    await enforcer.close()
    await old_http.aclose()


@pytest.mark.unit
def test_circuit_breaker_reset_frees_orphaned_half_open_probe() -> None:
    """P1 review pin: a HALF_OPEN probe in flight at fork time belongs to a
    PARENT thread; inherited set, admit() refuses forever in the child (state
    only advances via an admitted call) — permanently wedging the control-plane
    breaker: fail-open budget checks and HELD confirms with no recovery."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, success_threshold=1)
    breaker.record_failure()  # -> OPEN, immediately recovery-eligible
    admission = breaker.admit()  # -> HALF_OPEN, consumes the single probe slot
    assert admission.allowed and admission.owns_probe

    breaker._reset_after_fork_in_child()

    follow_up = breaker.admit()
    assert follow_up.allowed, (
        "child breaker must not stay wedged behind the parent's orphaned probe"
    )


_REGISTRY_LOCK_CHILD = """
import os
import sys
import threading
import time

import solwyn._lifecycle as lifecycle
from solwyn.circuit_breaker import CircuitBreaker

CircuitBreaker(name="parent")  # installs the at_fork hooks

held = threading.Event()

def _hold_registry_lock() -> None:
    with lifecycle._registry_lock:
        held.set()
        time.sleep(0.4)

threading.Thread(target=_hold_registry_lock, daemon=True).start()
held.wait(timeout=5.0)

pid = os.fork()
if pid == 0:
    # Pre-fix: the child inherits _registry_lock held by a thread that does
    # not exist here, and this construction deadlocks forever.
    done = threading.Event()

    def _construct() -> None:
        CircuitBreaker(name="child")
        done.set()

    threading.Thread(target=_construct, daemon=True).start()
    os._exit(0 if done.wait(timeout=3.0) else 7)
else:
    _, status = os.waitpid(pid, 0)
    sys.exit(os.waitstatus_to_exitcode(status))
"""


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_survives_held_registry_lock() -> None:
    """P1 review pin: _registry_lock is the one lock fork repair cannot repair
    (it guards the repair registry itself). The at_fork acquire/release triple
    holds it across the fork; pre-fix, a child forked while another thread
    registers an SDK object deadlocks on its own next construction."""
    for _ in range(_FORK_ATTEMPTS):
        result = subprocess.run(
            [sys.executable, "-c", _REGISTRY_LOCK_CHILD],
            timeout=30,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        if result.returncode == 7:
            pytest.fail(
                "forked child deadlocked constructing an SDK object "
                "while _registry_lock was inherited held"
            )
        if not _FORK_CRASH_IS_PLATFORM_ARTIFACT:
            pytest.fail(
                f"forked child crashed on Linux (returncode {result.returncode}) — "
                f"the at-fork hooks must not crash the child: {result.stderr[-800:]}"
            )
        # Off-Linux: the macOS multi-threaded-fork SIGSEGV artifact —
        # retry, mirroring test_forked_child_flush_thread_delivers.
    pytest.skip(f"all fork attempts crashed pre-exec ({sys.platform} platform artifact)")
