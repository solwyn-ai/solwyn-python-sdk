"""Control-plane background I/O must not retain the application's OTel context."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import UTC, datetime

import httpx
import pytest
from conftest import call_uuid

from solwyn._otel import capture_trace
from solwyn._surfaces import SurfaceContext
from solwyn._types import MetadataEvent, ProviderName
from solwyn.budget import AsyncBudgetEnforcer
from solwyn.circuit_breaker import CircuitBreaker
from solwyn.reporter import AsyncMetadataReporter
from solwyn.testing import FakeControlPlane

trace = pytest.importorskip("opentelemetry.trace")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_http_has_empty_context_while_preflight_keeps_caller_context() -> None:
    sentinel = ContextVar("application-private-context", default=None)
    seen = []
    plane = FakeControlPlane(granted_tokens=20)

    class RecordingTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
        def handle_request(self, request):
            return plane.transport.handle_request(request)

        async def handle_async_request(self, request):
            seen.append((request.url.path, sentinel.get(), capture_trace()))
            return await plane.transport.handle_async_request(request)

    transport = RecordingTransport()
    enforcer = AsyncBudgetEnforcer(plane.api_url, plane.api_key, transport=transport)
    breaker = CircuitBreaker(name="openai")
    reporter = AsyncMetadataReporter(
        plane.api_url,
        plane.api_key,
        transport=transport,
        flush_interval=0.01,
        sdk_instance_id="context-isolation",
        breaker_snapshots=lambda: [(ProviderName.OPENAI, breaker.get_state())],
    )
    reporter.observe_project_id(plane.project_id)
    caller = trace.SpanContext(trace_id=0xABC, span_id=0xDE, is_remote=False)
    token = sentinel.set("caller")
    try:
        with trace.use_span(trace.NonRecordingSpan(caller)):
            budget = await enforcer.check_budget(
                agent_run_id="run-context-test",
                call_id=call_uuid("context-admit"),
                model="gpt-5.5",
                provider="openai",
                estimated_input_tokens=1,
                estimated_output_bound=16,
            )
            assert budget.lease_id is not None
            reporter.report(
                MetadataEvent(
                    model="gpt-5.5",
                    provider="openai",
                    input_tokens=1,
                    output_tokens=0,
                    latency_ms=1,
                    status="success",
                    is_model_fallback=False,
                    sdk_instance_id="context-isolation",
                    timestamp=datetime.now(UTC),
                    call_id=call_uuid("background-context"),
                )
            )
            reporter.observe_untracked_surface(
                context=SurfaceContext(
                    provider="openai", dialect="openai", client_shape="openai_sdk", mode="async"
                ),
                surface="responses.create",
                rule_kind="unmetered_spend",
                capability_scope="operation",
                posture="warn",
                seen_at=datetime.now(UTC),
            )
            # Renewal starts at 75% depletion; surrender exercises release workers,
            # initialization of their pool, cleanup and pool closure as well.
            if enforcer._renewal_tasks:
                await asyncio.gather(*enforcer._renewal_tasks)
            enforcer.surrender_run("run-context-test")
            await enforcer.close()
            await reporter.close()
    finally:
        sentinel.reset(token)
        await enforcer.close()
        await reporter.close()
    grant = [entry for entry in seen if entry[0].endswith("/lease")]
    background = [entry for entry in seen if not entry[0].endswith("/lease")]
    assert len(grant) == 1
    assert grant[0][1] == "caller" and grant[0][2] is not None
    assert {path.rsplit("/", 1)[-1] for path, _, _ in background} >= {
        "renew",
        "surrender",
        "ingest",
        "breaker-reports",
        "untracked-surfaces",
    }
    assert all(value is None and snapshot is None for _, value, snapshot in background)
