"""Bounded serial delivery turns, with arrivals synchronized to successful sends."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest

from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, MetadataEvent
from solwyn.reporter import AsyncMetadataReporter, MetadataReporter

pytestmark = pytest.mark.unit


def _pair(index: int) -> tuple[BudgetConfirmRequest, MetadataEvent]:
    call_id = str(UUID(int=index + 1))
    return (
        BudgetConfirmRequest(
            reservation_id="synthetic",
            model="synthetic",
            provider="openai",
            call_id=call_id,
            token_details=TokenDetails(input_tokens=1, output_tokens=1),
        ),
        MetadataEvent(
            model="synthetic",
            provider="openai",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            status="success",
            is_model_fallback=False,
            sdk_instance_id="fairness-test",
            timestamp=datetime.now(UTC),
            call_id=call_id,
        ),
    )


def _reporter(mode: str, handler, **kwargs):
    options = dict(
        api_url="https://offline.invalid",
        api_key="synthetic-unused",
        transport=httpx.MockTransport(handler),
        flush_interval=3600,
        breaker_reporting_enabled=False,
        report_untracked_surfaces=False,
        **kwargs,
    )
    if mode == "sync":
        with patch.object(MetadataReporter, "_flush_loop"):
            reporter = MetadataReporter(**options)
        reporter._thread.join(timeout=2)
        return reporter
    reporter = AsyncMetadataReporter(**options)
    # Drive rounds explicitly, without a second owner or wall-clock scheduler.
    reporter._ensure_started = lambda: None
    return reporter


async def _flush(reporter, **kwargs):
    if isinstance(reporter, AsyncMetadataReporter):
        return await reporter._flush_remaining(**kwargs)
    return reporter._flush_remaining(**kwargs)


async def _close(reporter):
    if isinstance(reporter, AsyncMetadataReporter):
        await reporter.close()
    else:
        reporter.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("capacity", [4, 10_000])
@pytest.mark.parametrize("arrivals_per_send", [0, 1, 2])
@pytest.mark.parametrize("kind", ["settlement", "confirm"])
async def test_ready_metadata_gets_a_turn_while_confirm_producer_is_active(
    mode, capacity, arrivals_per_send, kind, monkeypatch
):
    # A small control bound also exercises counted overload independently of
    # metadata capacity. The production default is exercised below.
    monkeypatch.setattr("solwyn.reporter._MAX_PENDING_CONTROL", 8)
    generated: list[str] = []
    confirmed: list[str] = []
    ingested: list[str] = []
    first_ingest_at = None
    first_generated = None
    peak_event_queue = 0
    peak_control_queue = 0
    limit = 80 if arrivals_per_send else 4

    def produce():
        nonlocal peak_event_queue, peak_control_queue
        confirm, event = _pair(len(generated))
        generated.append(event.call_id)
        if kind == "settlement":
            reporter.report_settlement(confirm, event)
        else:
            reporter.report_confirm(confirm)
        peak_event_queue = max(peak_event_queue, len(reporter._queue))
        peak_control_queue = max(
            peak_control_queue, len(reporter._settlement_queue), len(reporter._confirm_queue)
        )

    def handle(request):
        nonlocal first_ingest_at, first_generated
        if request.url.path.endswith("/confirm"):
            confirmed.append(json.loads(request.content)["call_id"])
            for _ in range(arrivals_per_send):
                if len(generated) < limit:
                    produce()
        else:
            batch = json.loads(request.content)
            if first_ingest_at is None:
                first_ingest_at = len(confirmed)
                first_generated = len(generated)
            ingested.extend(event["call_id"] for event in batch)
            if kind == "settlement" and arrivals_per_send < 2:
                assert set(ingested) <= set(confirmed)
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter(mode, handle, batch_size=4, max_queue_size=capacity)
    if kind == "confirm":
        reporter.report(_pair(999)[1])
    for _ in range(4):
        produce()
    try:
        await _flush(reporter)
        assert first_ingest_at == 4
        if arrivals_per_send:
            assert first_generated < limit, "ingest waited for the producer to stop"
        for _ in range(limit):
            if not (reporter._queue or reporter._settlement_queue or reporter._confirm_queue):
                break
            await _flush(reporter)
        assert (
            not reporter._queue and not reporter._settlement_queue and not reporter._confirm_queue
        )
        assert confirmed == sorted(confirmed)
        assert len(confirmed) == len(set(confirmed))
        assert len(ingested) == len(set(ingested))
        control_kind = "settlement_confirm" if kind == "settlement" else "confirm"
        assert len(confirmed) + reporter.dropped_counts.get(f"{control_kind}.overflow", 0) == limit
        if kind == "settlement":
            assert len(ingested) + reporter.dropped_counts.get("event.overflow", 0) == limit
            if arrivals_per_send < 2:
                assert ingested == generated
                assert reporter.dropped_counts == {}
        else:
            assert ingested == [_pair(999)[1].call_id]
        assert peak_event_queue <= capacity
        assert peak_control_queue <= 8
    finally:
        await _close(reporter)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_event_arrivals_cannot_extend_a_cycle_forever(mode):
    ingested = []
    generated = 0

    def produce():
        nonlocal generated
        reporter.report(_pair(generated)[1])
        generated += 1

    def handle(request):
        ingested.extend(json.loads(request.content))
        for _ in range(4):
            if generated < 40:
                produce()
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter(mode, handle, batch_size=4)
    for _ in range(4):
        produce()
    try:
        await _flush(reporter)
        assert len(ingested) == 4
        assert generated == 8
        assert len(reporter._queue) == 4
    finally:
        # Stop the producer before exercising the final finite drain.
        generated = 40
        await _close(reporter)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_default_capacity_finite_backlog_finishes_in_serial_bounded_rounds(mode):
    confirmed = []
    ingested = []
    first_ingest_at = None

    def handle(request):
        nonlocal first_ingest_at
        if request.url.path.endswith("/confirm"):
            confirmed.append(json.loads(request.content)["call_id"])
        else:
            if first_ingest_at is None:
                first_ingest_at = len(confirmed)
            ingested.extend(event["call_id"] for event in json.loads(request.content))
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter(mode, handle)
    for index in range(200):
        reporter.report_settlement(*_pair(index))
    try:
        for _ in range(4):
            await _flush(reporter)
        assert first_ingest_at == 50
        assert confirmed == ingested == [_pair(index)[1].call_id for index in range(200)]
        assert reporter.dropped_counts == {}
    finally:
        await _close(reporter)


async def test_sync_close_during_successful_confirm_preserves_ready_events():
    entered = threading.Event()
    resume = threading.Event()
    ingested = []
    confirms = []

    def handle(request):
        if request.url.path.endswith("/confirm"):
            if not confirms:
                entered.set()
                assert resume.wait(5)
            confirms.append(json.loads(request.content)["call_id"])
        else:
            ingested.extend(event["call_id"] for event in json.loads(request.content))
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter("sync", handle, batch_size=50, max_queue_size=2)
    reporter.report(_pair(100)[1])
    reporter.report(_pair(101)[1])
    for index in range(8):
        reporter.report_settlement(*_pair(index))
    reporter.flush_interval = 0.001
    reporter._thread = reporter._launch_thread()
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(asyncio.to_thread(reporter.close, 5))
        assert await asyncio.to_thread(reporter._shutdown.wait, 5)
        # close's ordinary worker join must already protect ready metadata.
        # Acquiring the ownership gate serializes with close's transition.
        while not reporter._final_delivery_started:
            await asyncio.sleep(0)
        resume.set()
        await closing
        assert set(ingested) == {_pair(index)[1].call_id for index in [*range(8), 100, 101]}
        assert len(ingested) == 10
        assert len(confirms) == 8
        assert reporter.dropped_counts == {}
    finally:
        resume.set()
        await asyncio.to_thread(reporter.close)


async def test_async_close_during_successful_confirm_preserves_ready_events():
    entered = asyncio.Event()
    resume = asyncio.Event()
    ingested = []
    confirms = []

    async def handle(request):
        if request.url.path.endswith("/confirm"):
            if not confirms:
                entered.set()
                await resume.wait()
            confirms.append(json.loads(request.content)["call_id"])
        else:
            ingested.extend(event["call_id"] for event in json.loads(request.content))
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter("async", handle, batch_size=50, max_queue_size=2)
    reporter.report(_pair(100)[1])
    reporter.report(_pair(101)[1])
    for index in range(8):
        reporter.report_settlement(*_pair(index))
    reporter.flush_interval = 0.001
    reporter.start()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(reporter.close(5))
        await asyncio.sleep(0)
        assert reporter._closed and reporter._final_delivery_started
        resume.set()
        await closing
        assert set(ingested) == {_pair(index)[1].call_id for index in [*range(8), 100, 101]}
        assert len(ingested) == 10
        assert len(confirms) == 8
        assert reporter.dropped_counts == {}
    finally:
        resume.set()
        await reporter.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("disposition", ["idle", "retry", "held", "event_retry"])
async def test_idle_and_parked_heads_do_not_request_immediate_polling(mode, disposition):
    from solwyn.circuit_breaker import CircuitBreaker

    requests = []

    def handle(request):
        requests.append(request.url.path)
        return httpx.Response(503, json={})

    breaker = CircuitBreaker(name="control-plane", failure_threshold=1, recovery_timeout=3600)
    reporter = _reporter(mode, handle, batch_size=1, control_plane_breaker=breaker)
    if disposition == "held":
        breaker.record_failure()
    if disposition in {"held", "retry"}:
        reporter.report_settlement(*_pair(0))
        reporter.report_settlement(*_pair(1))
    elif disposition == "event_retry":
        reporter.report(_pair(0)[1])
    try:
        assert await _flush(reporter) is False
        attempted = len(requests)
        assert await _flush(reporter) is False
        assert len(requests) == attempted
        if disposition != "idle":
            assert reporter._queue or reporter._settlement_queue
    finally:
        await _close(reporter)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_standalone_and_paired_confirm_stages_each_get_bounded_turns(mode):
    confirmations = []
    ingests = []
    first_ingest_at = None

    def handle(request):
        nonlocal first_ingest_at
        if request.url.path.endswith("/confirm"):
            confirmations.append(json.loads(request.content)["call_id"])
        else:
            if first_ingest_at is None:
                first_ingest_at = len(confirmations)
            batch = [event["call_id"] for event in json.loads(request.content)]
            assert set(batch) <= set(confirmations)
            ingests.extend(batch)
        return httpx.Response(202, json={"rejected": []})

    reporter = _reporter(mode, handle, batch_size=4)
    for index in range(16):
        reporter.report_confirm(_pair(index + 100)[0])
        reporter.report_settlement(*_pair(index))
    try:
        assert await _flush(reporter) is True
        assert first_ingest_at == 8
        assert confirmations == [
            *[_pair(index + 100)[0].call_id for index in range(4)],
            *[_pair(index)[0].call_id for index in range(4)],
        ]
        assert len(ingests) == 4
        for _ in range(3):
            await _flush(reporter)
        assert len(confirmations) == 32
        assert ingests == [_pair(index)[1].call_id for index in range(16)]
        assert reporter.dropped_counts == {}
    finally:
        await _close(reporter)
