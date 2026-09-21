"""Bounded courtesy release work, with real admission and transport boundaries."""

from __future__ import annotations

import asyncio
import json
import ssl
import threading
import time
import uuid
from unittest.mock import patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import LeaseSurrenderRequest
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer

pytestmark = pytest.mark.unit


def grant(run_id: str) -> dict[str, object]:
    return {
        "eligible": True,
        "allowed": True,
        "lease_id": "lease_" + run_id,
        "generation": 1,
        "granted_tokens": 100_000,
        "refresh_interval_s": 300,
        "lease_length_s": 600,
        "headroom_share_tokens": 50_000,
        "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
        "final_grant": False,
        "project_id": VALID_PROJECT_ID,
        "mode": "alert_only",
        "budget_limit": 100,
        "current_usage": 0,
        "remaining_budget": 100,
    }


def arguments(run_id: str) -> dict[str, object]:
    return dict(
        agent_run_id=run_id,
        call_id=str(uuid.uuid4()),
        model="test-model",
        provider="openai",
        estimated_input_tokens=1,
        estimated_output_bound=1,
    )


def response(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("check"):
        return httpx.Response(200, json=ALLOW_BUDGET_RESPONSE)
    return httpx.Response(200, json=grant(json.loads(request.content)["agent_run_id"]))


def test_sync_burst_has_fixed_workers_and_bounded_pending() -> None:
    gate = threading.Event()
    four_started = threading.Event()
    lock = threading.Lock()
    active = peak = sent = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, sent
        if not request.url.path.endswith("surrender"):
            return response(request)
        with lock:
            active += 1
            peak = max(peak, active)
            sent += 1
            if active == 4:
                four_started.set()
        try:
            assert gate.wait(5)
        finally:
            with lock:
                active -= 1
        return httpx.Response(200, json={})

    enforcer = BudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle)
    )
    try:
        for i in range(128):
            assert enforcer.check_budget(**arguments(str(i))).lease_id
        for i in range(4):
            enforcer.surrender_run(str(i))
        assert four_started.wait(3)
        for i in range(4, 128):
            enforcer.surrender_run(str(i))
        assert peak == 4
        assert len(enforcer._release_threads) == 4
        assert len(enforcer._releases_owed) == 64
        assert enforcer.release_counts["queue_full"] == 60
    finally:
        gate.set()
        enforcer.close()
    assert sent + enforcer.release_counts.get("queue_full", 0) == 128


async def test_async_burst_has_fixed_workers_and_bounded_pending() -> None:
    gate = asyncio.Event()
    four_started = asyncio.Event()
    active = peak = sent = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, sent
        if not request.url.path.endswith("surrender"):
            return response(request)
        active += 1
        peak = max(peak, active)
        sent += 1
        if active == 4:
            four_started.set()
        try:
            await gate.wait()
        finally:
            active -= 1
        return httpx.Response(200, json={})

    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle)
    )
    try:
        for i in range(128):
            assert (await enforcer.check_budget(**arguments(str(i)))).lease_id
        for i in range(4):
            enforcer.surrender_run(str(i))
        await asyncio.wait_for(four_started.wait(), 3)
        for i in range(4, 128):
            enforcer.surrender_run(str(i))
        await asyncio.sleep(0)
        assert peak == 4
        assert len(enforcer._release_tasks) == 4
        assert len(enforcer._releases_owed) == 64
        assert enforcer.release_counts["queue_full"] == 60
    finally:
        gate.set()
        await enforcer.close()
    assert sent + enforcer.release_counts.get("queue_full", 0) == 128


async def test_expired_native_releases_construct_no_clients_or_ssl_contexts() -> None:
    enforcer = AsyncBudgetEnforcer("https://offline.invalid", VALID_API_KEY)
    request = LeaseSurrenderRequest(
        lease_id="expired", holder_id="test", generation=1, spent_tokens=0
    )
    try:
        with patch.object(ssl, "create_default_context", wraps=ssl.create_default_context) as tls:
            for _ in range(128):
                await enforcer._surrender_payloads([request], time.monotonic() - 1)
            assert tls.call_count == 0
    finally:
        await enforcer.close()


async def drain(enforcer: AsyncBudgetEnforcer) -> None:
    while enforcer._release_tasks:
        await asyncio.wait_for(asyncio.gather(*enforcer._release_tasks), 3)
        await asyncio.sleep(0)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_open_breaker_avoids_native_release_setup(async_mode: bool) -> None:
    from solwyn.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(name="control-plane", failure_threshold=1)
    cls = AsyncBudgetEnforcer if async_mode else BudgetEnforcer
    enforcer = cls("https://offline.invalid", VALID_API_KEY, control_plane_breaker=breaker)
    breaker.record_failure()
    request = LeaseSurrenderRequest(lease_id="held", holder_id="test", generation=1, spent_tokens=0)
    try:
        with patch.object(ssl, "create_default_context", wraps=ssl.create_default_context) as tls:
            for _ in range(128):
                if async_mode:
                    await enforcer._surrender_payloads([request], time.monotonic() + 1)
                else:
                    enforcer._surrender_payloads([request], time.monotonic() + 1)
            assert tls.call_count == 0
    finally:
        if async_mode:
            await enforcer.close()
        else:
            enforcer.close()


async def test_native_tls_initialization_is_single_off_loop_owner_and_pool_reused() -> None:
    gate = threading.Event()
    entered = threading.Event()
    tls_threads: list[int] = []
    client_count = 0
    sent = 0
    original_tls = ssl.create_default_context
    original_client = httpx.AsyncClient.__init__
    enforcer = AsyncBudgetEnforcer("https://offline.invalid", VALID_API_KEY)

    def tls(*args: object, **kwargs: object) -> ssl.SSLContext:
        tls_threads.append(threading.get_ident())
        entered.set()
        assert gate.wait(3)
        return original_tls(*args, **kwargs)

    def client_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        nonlocal client_count
        client_count += 1
        original_client(self, *args, **kwargs)

    async def handle(_self: object, request: httpx.Request) -> httpx.Response:
        nonlocal sent
        if request.url.path.endswith("surrender"):
            sent += 1
            return httpx.Response(200, json={})
        return response(request)

    try:
        with (
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", handle),
            patch.object(ssl, "create_default_context", tls),
            patch.object(httpx.AsyncClient, "__init__", client_init),
        ):
            for i in range(32):
                await enforcer.check_budget(**arguments(str(i)))
            enforcer.surrender_run("0")
            assert await asyncio.to_thread(entered.wait, 2)
            # A ready callback must run while native trust-root loading is HELD.
            ready = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(ready.set_result, True)
            assert await ready
            for i in range(1, 32):
                enforcer.surrender_run(str(i))
            await asyncio.sleep(0)
            assert len(enforcer._release_tasks) == 4
            assert client_count == 0
            assert tls_threads == [tls_threads[0]]
            assert tls_threads[0] != threading.get_ident()
            gate.set()
            await drain(enforcer)
            # Separate bursts reuse the same native client and trust context.
            for i in range(32, 64):
                await enforcer.check_budget(**arguments(str(i)))
                enforcer.surrender_run(str(i))
                await drain(enforcer)
            assert sent == 64
            assert client_count == 1
            assert len(tls_threads) == 1
            client = enforcer._release_http_task.result()
            assert not client.is_closed
            await enforcer.close()
            assert client.is_closed
    finally:
        gate.set()
        await enforcer.close()


def test_sync_close_does_not_wait_for_held_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    import solwyn.budget as budget

    monkeypatch.setattr(budget, "_SURRENDER_TIMEOUT_S", 0.01)
    entered, gate, closed = threading.Event(), threading.Event(), threading.Event()
    enforcer = BudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(response)
    )
    original = httpx.Client.__init__

    def construct(self: httpx.Client, *args: object, **kwargs: object) -> None:
        entered.set()
        assert gate.wait(3)
        original(self, *args, **kwargs)

    try:
        enforcer.check_budget(**arguments("constructor"))
        with patch.object(httpx.Client, "__init__", construct):
            enforcer.surrender_run("constructor")
            assert entered.wait(2)
            worker = next(iter(enforcer._release_threads))
            closer = threading.Thread(target=lambda: (enforcer.close(), closed.set()), daemon=True)
            closer.start()
            assert closed.wait(1), "close must not acquire the live constructor's lock"
            assert worker in enforcer._release_threads
            gate.set()
            worker.join(2)
            closer.join(2)
            assert not worker.is_alive()
            assert enforcer._release_http is None
    finally:
        gate.set()
        enforcer.close()


@pytest.mark.parametrize("async_mode", [False, True])
async def test_active_release_remains_fenced_after_wait_timeout(async_mode: bool) -> None:
    entered, gate = threading.Event(), threading.Event()
    async_gate, async_entered = asyncio.Event(), asyncio.Event()
    grants = checks = 0

    def handle_sync(request: httpx.Request) -> httpx.Response:
        nonlocal grants, checks
        if request.url.path.endswith("surrender"):
            entered.set()
            assert gate.wait(4)
            return httpx.Response(200, json={})
        if request.url.path.endswith("check"):
            checks += 1
        else:
            grants += 1
        return response(request)

    async def handle_async(request: httpx.Request) -> httpx.Response:
        nonlocal grants, checks
        if request.url.path.endswith("surrender"):
            async_entered.set()
            await async_gate.wait()
            return httpx.Response(200, json={})
        if request.url.path.endswith("check"):
            checks += 1
        else:
            grants += 1
        return response(request)

    cls = AsyncBudgetEnforcer if async_mode else BudgetEnforcer
    enforcer = cls(
        "https://offline.invalid",
        VALID_API_KEY,
        transport=httpx.MockTransport(handle_async if async_mode else handle_sync),
    )
    try:
        if async_mode:
            await enforcer.check_budget(**arguments("fenced"))
        else:
            enforcer.check_budget(**arguments("fenced"))
        enforcer.surrender_run("fenced")
        if async_mode:
            await asyncio.wait_for(async_entered.wait(), 2)
        else:
            assert entered.wait(2)
        for _ in range(2):
            if async_mode:
                result = await enforcer.check_budget(**arguments("fenced"), timeout=0.01)
            else:
                result = enforcer.check_budget(**arguments("fenced"), timeout=0.01)
            assert result.allowed and result.lease_id is None
            assert "fenced" in enforcer._run_releases
            assert not enforcer._lease_grants_in_flight
        assert (grants, checks) == (1, 2)
        if async_mode:
            async_gate.set()
            await drain(enforcer)
            result = await enforcer.check_budget(**arguments("fenced"))
        else:
            gate.set()
            for worker in list(enforcer._release_threads):
                worker.join(2)
            result = enforcer.check_budget(**arguments("fenced"))
        assert result.lease_id is not None
        assert grants == 2
    finally:
        gate.set()
        async_gate.set()
        if async_mode:
            await enforcer.close()
        else:
            enforcer.close()


async def test_failed_initializer_can_recover_without_losing_later_releases() -> None:
    attempts = sent = 0
    original = httpx.AsyncClient.__init__

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        if request.url.path.endswith("surrender"):
            sent += 1
            return httpx.Response(200, json={})
        return response(request)

    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle)
    )

    def construct(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("synthetic setup failure")
        original(self, *args, **kwargs)

    try:
        with patch.object(httpx.AsyncClient, "__init__", construct):
            for i in range(2):
                await enforcer.check_budget(**arguments(str(i)))
                enforcer.surrender_run(str(i))
                await drain(enforcer)
        assert attempts == 2
        assert sent == 1
        assert not enforcer._release_jobs
    finally:
        await enforcer.close()


def test_loop_shutdown_does_not_resurrect_release_tasks_or_lose_queued_work() -> None:
    class HeldTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.sent = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("surrender"):
                self.sent += 1
                await asyncio.Event().wait()
            return response(request)

    transport = HeldTransport()
    enforcer = AsyncBudgetEnforcer("https://offline.invalid", VALID_API_KEY, transport=transport)

    async def produce() -> None:
        for i in range(8):
            await enforcer.check_budget(**arguments(str(i)))
            enforcer.surrender_run(str(i))
        while transport.sent < 4:
            await asyncio.sleep(0)

    asyncio.run(produce())
    assert not enforcer._release_tasks
    assert transport.sent == 4
    assert len(enforcer._releases_owed) == 4
    assert len(enforcer.lease_surrender_payloads()) == 4
    assert not enforcer._release_jobs
    asyncio.run(enforcer.close())


async def test_cancelled_constructor_awaiter_does_not_duplicate_native_setup() -> None:
    entered, gate = threading.Event(), threading.Event()
    calls = 0
    original = ssl.create_default_context
    enforcer = AsyncBudgetEnforcer("https://offline.invalid", VALID_API_KEY)

    def tls(*args: object, **kwargs: object) -> ssl.SSLContext:
        nonlocal calls
        calls += 1
        entered.set()
        assert gate.wait(3)
        return original(*args, **kwargs)

    async def handle(_self: object, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("surrender"):
            return httpx.Response(200, json={})
        return response(request)

    try:
        with (
            patch.object(ssl, "create_default_context", tls),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", handle),
        ):
            await enforcer.check_budget(**arguments("cancel-setup"))
            enforcer.surrender_run("cancel-setup")
            assert await asyncio.to_thread(entered.wait, 2)
            worker = next(iter(enforcer._release_tasks))
            initialization = enforcer._release_http_task
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
            await asyncio.sleep(0)
            await enforcer.check_budget(**arguments("next-setup"))
            enforcer.surrender_run("next-setup")
            await asyncio.sleep(0)
            assert enforcer._release_http_task is initialization
            assert not initialization.done()
            assert calls == 1
            gate.set()
            await drain(enforcer)
            assert calls == 1
    finally:
        gate.set()
        await enforcer.close()


async def test_cancellation_resistant_request_keeps_its_capacity_until_actual_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import solwyn.budget as budget

    monkeypatch.setattr(budget, "_SURRENDER_TIMEOUT_S", 0.01)
    active = peak = 0
    entered = asyncio.Event()
    finish = asyncio.Event()
    cancelled = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        if not request.url.path.endswith("surrender"):
            return response(request)
        active += 1
        peak = max(peak, active)
        if active == 4:
            entered.set()
        try:
            while not finish.is_set():
                try:
                    await finish.wait()
                except asyncio.CancelledError:
                    cancelled.set()
        finally:
            active -= 1
        return httpx.Response(200, json={})

    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle)
    )
    try:
        for i in range(8):
            await enforcer.check_budget(**arguments(str(i)))
            enforcer.surrender_run(str(i))
        await asyncio.wait_for(entered.wait(), 2)
        pool = enforcer._release_http_task.result()
        await enforcer.close()
        assert cancelled.is_set()
        assert active == 4
        assert len(enforcer._release_tasks) == 4
        assert len(enforcer._release_jobs) == 4
        assert not pool.is_closed
        assert enforcer.release_counts["expired"] == 4
        finish.set()
        await drain(enforcer)
        await asyncio.wait_for(enforcer._release_cleanup_task, 2)
        assert active == 0 and peak == 4
        assert pool.is_closed
        assert not enforcer._release_jobs
    finally:
        finish.set()
        await enforcer.close()


def test_off_loop_queue_saturation_is_bounded_and_expired_work_cannot_release_successors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import solwyn.budget as budget

    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(response)
    )
    with enforcer._state_lock:
        for i in range(128):
            enforcer._owe_release_locked(
                str(i),
                LeaseSurrenderRequest(
                    lease_id=str(i), holder_id="test", generation=1, spent_tokens=0
                ),
            )
    enforcer._dispatch_owed_releases()  # No running loop: no new tasks or initializer.
    assert not enforcer._release_tasks
    assert len(enforcer._releases_owed) == 64
    assert len(enforcer._release_jobs) == 64
    assert enforcer.release_counts["queue_full"] == 64
    now = time.monotonic()
    with monkeypatch.context() as m:
        m.setattr(budget.time, "monotonic", lambda: now + 10)
        assert enforcer.lease_surrender_payloads() == []
    assert enforcer.release_counts["expired"] == 64
    assert not enforcer._run_releases
    asyncio.run(enforcer.close())


@pytest.mark.parametrize("async_mode", [False, True])
async def test_exact_duplicates_coalesce_without_losing_successor_fences(async_mode: bool) -> None:
    cls = AsyncBudgetEnforcer if async_mode else BudgetEnforcer
    enforcer = cls(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(response)
    )
    first = LeaseSurrenderRequest(
        lease_id="same-row", holder_id="test", generation=1, spent_tokens=2
    )
    second = first.model_copy(update={"generation": 2, "spent_tokens": 3})
    try:
        with enforcer._state_lock:
            enforcer._owe_release_locked("run", first)
            enforcer._owe_release_locked("run", first)
            enforcer._owe_release_locked("run", second)
            assert len(enforcer._releases_owed) == 2
            old = enforcer._next_release_locked()
            successor = enforcer._next_release_locked()
            enforcer._finish_release_locked(old)
            assert list(enforcer._run_releases["run"].values()) == [successor]
            enforcer._finish_release_locked(old)  # Late duplicate completion is a no-op.
            assert list(enforcer._run_releases["run"].values()) == [successor]
            enforcer._finish_release_locked(successor)
            assert not enforcer._run_releases
    finally:
        if async_mode:
            await enforcer.close()
        else:
            enforcer.close()


@pytest.mark.parametrize("swallow_cancel", [False, True])
def test_shutdown_cancellation_never_claims_next_queued_request(swallow_cancel: bool) -> None:
    requests: list[str] = []

    class Transport(httpx.BaseTransport, httpx.AsyncBaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("surrender"):
                requests.append(json.loads(request.content)["lease_id"])
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    if not swallow_cancel:
                        raise
                return httpx.Response(200, json={})
            return response(request)

    enforcer = AsyncBudgetEnforcer("https://offline.invalid", VALID_API_KEY, transport=Transport())

    async def produce() -> None:
        for i in range(8):
            await enforcer.check_budget(**arguments(str(i)))
            enforcer.surrender_run(str(i))
        while len(requests) < 4:
            await asyncio.sleep(0)

    asyncio.run(produce())
    assert len(requests) == 4
    assert not enforcer._release_tasks
    assert len(enforcer.lease_surrender_payloads()) == 4
    asyncio.run(enforcer.close())


@pytest.mark.parametrize("async_mode", [False, True])
async def test_worker_start_failure_parks_bounded_fenced_work_for_retry(async_mode: bool) -> None:
    sent = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        if request.url.path.endswith("surrender"):
            sent += 1
            return httpx.Response(200, json={})
        return response(request)

    cls = AsyncBudgetEnforcer if async_mode else BudgetEnforcer
    enforcer = cls("https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle))
    try:
        if async_mode:
            await enforcer.check_budget(**arguments("start-failure"))
            target, method = asyncio.get_running_loop(), "create_task"
        else:
            enforcer.check_budget(**arguments("start-failure"))
            target, method = threading.Thread, "start"
        with patch.object(target, method, side_effect=RuntimeError("synthetic start failure")):
            enforcer.surrender_run("start-failure")
            assert len(enforcer._releases_owed) == 1
            assert len(enforcer._run_releases["start-failure"]) == 1
            assert not next(iter(enforcer._run_releases["start-failure"].values())).done.is_set()
            assert sent == 0
        enforcer._dispatch_owed_releases()
        if async_mode:
            await drain(enforcer)
        else:
            for worker in list(enforcer._release_threads):
                worker.join(2)
        assert sent == 1
        assert not enforcer._run_releases
        assert enforcer.release_counts["dispatch_failed"] == 1
    finally:
        if async_mode:
            await enforcer.close()
        else:
            enforcer.close()


async def test_late_grant_during_pool_teardown_transfers_cleanup_to_new_pool() -> None:
    release_count = 0
    late_entered, late_finish = asyncio.Event(), asyncio.Event()
    closing_entered, closing_finish = asyncio.Event(), asyncio.Event()
    pools: list[httpx.AsyncClient] = []
    original_constructor = httpx.AsyncClient.__init__

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal release_count
        if request.url.path.endswith("surrender"):
            release_count += 1
            return httpx.Response(200, json={})
        if (
            request.url.path.endswith("lease")
            and json.loads(request.content)["agent_run_id"] == "late"
        ):
            late_entered.set()
            await late_finish.wait()
        return response(request)

    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=httpx.MockTransport(handle)
    )

    def construct(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        original_constructor(self, *args, **kwargs)
        pools.append(self)

    try:
        with patch.object(httpx.AsyncClient, "__init__", construct):
            await enforcer.check_budget(**arguments("first"))
            enforcer.surrender_run("first")
            await drain(enforcer)
            old_pool = pools[0]
            original_close = old_pool.aclose

            async def hold_close() -> None:
                closing_entered.set()
                await closing_finish.wait()
                await original_close()

            late = asyncio.create_task(enforcer.check_budget(**arguments("late")))
            await asyncio.wait_for(late_entered.wait(), 2)
            with patch.object(old_pool, "aclose", hold_close):
                closing = asyncio.create_task(enforcer.close())
                await asyncio.wait_for(closing_entered.wait(), 2)
                late_finish.set()
                await asyncio.wait_for(late, 2)
                closing_finish.set()
                await asyncio.wait_for(closing, 2)
            await asyncio.wait_for(enforcer._release_cleanup_task, 2)
            assert release_count == 2
            assert len(pools) == 2
            assert all(pool.is_closed for pool in pools)
            assert not enforcer._release_tasks
            assert enforcer._release_http_task is None
    finally:
        closing_finish.set()
        late_finish.set()
        await enforcer.close()


@pytest.mark.parametrize("async_mode", [False, True])
async def test_real_slow_response_body_cannot_extend_close_or_free_live_sync_slots(
    async_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import solwyn.budget as budget

    monkeypatch.setattr(budget, "_SURRENDER_TIMEOUT_S", 0.05)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    entered, finish = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = peak = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal active, peak
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("surrender"):
                body = json.dumps(grant(payload["agent_run_id"])).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    entered.set()
            try:
                # Continual body progress is below the per-read timeout; only
                # the SDK's outer close deadline can bound this wait.
                self.send_response(200)
                self.send_header("Content-Length", "1002")
                self.end_headers()
                written = 0
                while not finish.is_set() and written < 1000:
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    written += 1
                    finish.wait(0.003)
                self.wfile.write(b" " * (1000 - written) + b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    active -= 1

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cls = AsyncBudgetEnforcer if async_mode else BudgetEnforcer
    enforcer = cls(f"http://127.0.0.1:{server.server_port}", VALID_API_KEY)
    try:
        for i in range(4):
            if async_mode:
                await enforcer.check_budget(**arguments(str(i)))
            else:
                enforcer.check_budget(**arguments(str(i)))
            enforcer.surrender_run(str(i))
        assert await asyncio.to_thread(entered.wait, 3)
        assert peak == 4
        if async_mode:
            await asyncio.wait_for(enforcer.close(), 1)
        else:
            closed = threading.Event()
            closer = threading.Thread(target=lambda: (enforcer.close(), closed.set()), daemon=True)
            closer.start()
            assert await asyncio.to_thread(closed.wait, 1)
            assert len(enforcer._release_threads) == 4
            assert len(enforcer._release_jobs) == 4
            assert not enforcer._release_http.is_closed
            closer.join(1)
    finally:
        finish.set()
        if async_mode:
            await enforcer.close()
            if enforcer._release_cleanup_task is not None:
                await asyncio.wait_for(enforcer._release_cleanup_task, 3)
        else:
            for worker in list(enforcer._release_threads):
                worker.join(3)
            enforcer.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(2)
    assert not enforcer._release_jobs


def test_sync_pool_teardown_retains_worker_slot_until_late_grants_can_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import solwyn.budget as budget

    monkeypatch.setattr(budget, "_SURRENDER_TIMEOUT_S", 0.05)
    release_entered, release_finish = threading.Event(), threading.Event()
    late_entered, late_finish = threading.Event(), threading.Event()
    close_entered, close_finish = threading.Event(), threading.Event()
    lock = threading.Lock()
    grants = 0
    sent: list[str] = []
    old_pool: httpx.Client | None = None
    original_close = httpx.Client.close

    def handle(_transport: object, request: httpx.Request) -> httpx.Response:
        nonlocal grants
        payload = json.loads(request.content)
        if request.url.path.endswith("surrender"):
            with lock:
                sent.append(payload["lease_id"])
            if payload["lease_id"] == "lease_initial":
                release_entered.set()
                assert release_finish.wait(5)
            return httpx.Response(200, json={})
        if request.url.path.endswith("lease") and payload["agent_run_id"].startswith("late"):
            with lock:
                grants += 1
                if grants == 4:
                    late_entered.set()
            assert late_finish.wait(5)
        return response(request)

    def close(client: httpx.Client) -> None:
        if client is old_pool:
            close_entered.set()
            assert close_finish.wait(5)
        original_close(client)

    with (
        patch.object(httpx.HTTPTransport, "handle_request", handle),
        patch.object(httpx.Client, "close", close),
    ):
        enforcer = BudgetEnforcer("https://offline.invalid", VALID_API_KEY)
        enforcer.check_budget(**arguments("initial"))
        enforcer.surrender_run("initial")
        assert release_entered.wait(2)
        old_pool = enforcer._release_http
        old_worker = next(iter(enforcer._release_threads))
        callers = [
            threading.Thread(
                target=lambda i=i: enforcer.check_budget(**arguments(f"late{i}")), daemon=True
            )
            for i in range(4)
        ]
        try:
            for caller in callers:
                caller.start()
            assert late_entered.wait(2)
            enforcer.close()
            release_finish.set()
            assert close_entered.wait(2)
            late_finish.set()
            for caller in callers:
                caller.join(2)
                assert not caller.is_alive()
            # Teardown is still underlying worker work. Its slot cannot be
            # lent to a fifth worker waiting for this same pool lock.
            assert old_worker.is_alive()
            assert old_worker in enforcer._release_threads
            assert len(enforcer._release_threads | {old_worker}) == 4
            assert len(enforcer._releases_owed) == 1
        finally:
            release_finish.set()
            late_finish.set()
            close_finish.set()
            for worker in [old_worker, *callers, *list(enforcer._release_threads)]:
                worker.join(3)
            enforcer.close()
        assert sorted(sent) == [
            "lease_initial",
            "lease_late0",
            "lease_late1",
            "lease_late2",
            "lease_late3",
        ]
        assert not enforcer._release_threads
        assert enforcer._release_http is None
