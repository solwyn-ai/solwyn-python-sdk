"""Slow run surrender must not consume settlement delivery's execution capacity."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn import run
from solwyn._token_details import TokenDetails
from solwyn._types import MetadataEvent
from solwyn.budget import AsyncBudgetEnforcer, BudgetCheckResult, BudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

pytestmark = pytest.mark.unit
_COUNT = 128


class _HeldReleasePlane(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """One shared in-process plane, with release responses separately held."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.release_gate = threading.Event()
        self.release_gate_async = asyncio.Event()
        self.four_releases = threading.Event()
        self.four_releases_async = asyncio.Event()
        self.all_ingested = threading.Event()
        self.all_ingested_async = asyncio.Event()
        self.active_releases = 0
        self.peak_releases = 0
        self.releases: list[str] = []
        self.confirms: list[str] = []
        self.ingested: list[str] = []
        self.ingest_requests = 0
        self.grants = 0

    def _begin_release(self, request: httpx.Request) -> None:
        with self.lock:
            self.releases.append(json.loads(request.content)["lease_id"])
            self.active_releases += 1
            self.peak_releases = max(self.peak_releases, self.active_releases)
            if self.active_releases == 4:
                self.four_releases.set()
                self.four_releases_async.set()

    def _respond(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        with self.lock:
            if request.url.path.endswith("/lease"):
                self.grants += 1
                return httpx.Response(
                    200,
                    json={
                        "eligible": True,
                        "allowed": True,
                        "lease_id": "lease_" + payload["agent_run_id"],
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
                    },
                )
            if request.url.path.endswith("/confirm"):
                self.confirms.append(payload["call_id"])
                return httpx.Response(200, json={})
            assert request.url.path.endswith("/ingest")
            identities = [event["call_id"] for event in payload]
            assert set(identities) <= set(self.confirms)
            self.ingested.extend(identities)
            self.ingest_requests += 1
            if len(self.ingested) == _COUNT:
                self.all_ingested.set()
                self.all_ingested_async.set()
            return httpx.Response(202, json={"rejected": []})

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/surrender"):
            return self._respond(request)
        self._begin_release(request)
        try:
            assert self.release_gate.wait(10)
        finally:
            with self.lock:
                self.active_releases -= 1
        return httpx.Response(200, json={})

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/surrender"):
            return self._respond(request)
        self._begin_release(request)
        try:
            await self.release_gate_async.wait()
        finally:
            with self.lock:
                self.active_releases -= 1
        return httpx.Response(200, json={})


def _arguments(run_id: str, index: int) -> dict:
    return dict(
        agent_run_id=run_id,
        call_id=str(UUID(int=index + 1)),
        model="synthetic",
        provider="openai",
        estimated_input_tokens=1,
        estimated_output_bound=1,
    )


def _settle(enforcer, reporter, admitted: BudgetCheckResult, run_id: str, index: int) -> None:
    assert admitted.lease_id is not None
    call_id = str(UUID(int=index + 1))
    confirm = enforcer.build_confirm_request(
        model="synthetic",
        token_details=TokenDetails(input_tokens=1, output_tokens=1),
        provider="openai",
        call_id=call_id,
        lease_id=admitted.lease_id,
        lease_claim_token=admitted.lease_claim_token,
    )
    event = MetadataEvent(
        model="synthetic",
        provider="openai",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        status="success",
        is_model_fallback=False,
        sdk_instance_id="combined-delivery-test",
        timestamp=datetime.now(UTC),
        call_id=call_id,
        agent_run_id=run_id,
    )
    reporter.report_settlement(confirm, event)


def _assert_progress(plane, reporter, enforcer) -> None:
    assert plane.grants == _COUNT
    assert plane.confirms == plane.ingested == [str(UUID(int=i + 1)) for i in range(_COUNT)]
    assert _COUNT // 4 <= plane.ingest_requests <= _COUNT
    assert reporter.dropped_counts == {}
    assert not reporter._queue and not reporter._settlement_queue
    assert plane.peak_releases == plane.active_releases == len(plane.releases) == 4
    assert len(enforcer._releases_owed) == 64
    assert enforcer.release_counts["queue_full"] == 60


def test_sync_run_exit_saturation_keeps_reporter_ingesting() -> None:
    plane = _HeldReleasePlane()
    breaker = CircuitBreaker(name="control-plane")
    enforcer = BudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=plane, control_plane_breaker=breaker
    )
    reporter = MetadataReporter(
        "https://offline.invalid",
        VALID_API_KEY,
        transport=plane,
        control_plane_breaker=breaker,
        batch_size=4,
        max_queue_size=8,
        flush_interval=0.001,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    try:
        for index in range(_COUNT):
            with run("synthetic-combined-delivery") as run_id:
                admitted = enforcer.check_budget(**_arguments(run_id, index))
                _settle(enforcer, reporter, admitted, run_id, index)
            # Ordinary run scope exit dispatches surrender via the lifecycle
            # registry. Hold every release worker before saturating the queue.
            if index == 3:
                assert plane.four_releases.wait(5)
        assert plane.all_ingested.wait(5)
        _assert_progress(plane, reporter, enforcer)
        assert len(enforcer._release_threads) == 4
    finally:
        plane.release_gate.set()
        reporter.close()
        enforcer.close()
    assert len(plane.releases) + enforcer.release_counts["queue_full"] == _COUNT
    assert len(plane.releases) == len(set(plane.releases))


async def test_async_run_exit_saturation_keeps_reporter_and_ready_loop_progressing() -> None:
    plane = _HeldReleasePlane()
    breaker = CircuitBreaker(name="control-plane")
    enforcer = AsyncBudgetEnforcer(
        "https://offline.invalid", VALID_API_KEY, transport=plane, control_plane_breaker=breaker
    )
    reporter = AsyncMetadataReporter(
        "https://offline.invalid",
        VALID_API_KEY,
        transport=plane,
        control_plane_breaker=breaker,
        batch_size=4,
        max_queue_size=8,
        flush_interval=0.001,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
    )
    try:
        for index in range(_COUNT):
            with run("synthetic-combined-delivery") as run_id:
                admitted = await enforcer.check_budget(**_arguments(run_id, index))
                _settle(enforcer, reporter, admitted, run_id, index)
            if index == 3:
                await asyncio.wait_for(plane.four_releases_async.wait(), 5)
        # The ready callback and all durable metadata complete while every
        # release request remains held; neither relies on a timing threshold.
        ready = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(ready.set_result, None)
        await ready
        await asyncio.wait_for(plane.all_ingested_async.wait(), 5)
        _assert_progress(plane, reporter, enforcer)
        assert len(enforcer._release_tasks) == 4
    finally:
        plane.release_gate_async.set()
        await reporter.close()
        await enforcer.close()
    assert len(plane.releases) + enforcer.release_counts["queue_full"] == _COUNT
    assert len(plane.releases) == len(set(plane.releases))
