"""Decisions metering against genuine OpenAI objects and offline HTTP transports."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from solwyn import run
from solwyn._types import CallStatus
from solwyn.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    UntrackedSpendSurfaceError,
)
from solwyn.providers.openai import OpenAIAdapter
from solwyn.testing import FakeControlPlane

openai = pytest.importorskip("openai", minversion="3.26.0")
decision_types = pytest.importorskip("openai.types.decision")
pytestmark = pytest.mark.unit

MODEL = "gpt-6-luna"
REQUEST = {
    "model": MODEL,
    "input": "a" * 40,
    "questions": [{"type": "predicate", "instructions": "b" * 20}],
}


def _decision():
    return {
        "model": MODEL,
        "answers": [{"type": "predicate", "name": None, "probability": 0.9}],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 20},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 160,
        },
    }


@asynccontextmanager
async def _wrapped(
    is_async, *, response=None, status=200, plane=None, transport_error=None, **options
):
    seen = []

    def handle(request):
        seen.append(request)
        if transport_error is not None:
            raise transport_error
        payload = response() if callable(response) else response
        return httpx.Response(status, json=_decision() if payload is None else payload)

    plane = plane or FakeControlPlane()
    if is_async:
        raw = openai.AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        wrapper = plane.wrap_async(raw, on_unmetered="raise", **options)
    else:
        raw = openai.OpenAI(
            api_key="offline", http_client=httpx.Client(transport=httpx.MockTransport(handle))
        )
        wrapper = plane.wrap(raw, on_unmetered="raise", **options)
    try:
        yield wrapper, plane, seen
    finally:
        if is_async:
            await wrapper.close()
        else:
            wrapper.close()


async def _create(wrapper, is_async, **kwargs):
    result = wrapper.decisions.create(**kwargs)
    return await result if is_async else result


def _require_admin_usage_decisions():
    resources = pytest.importorskip("openai.resources.admin.organization.usage")
    if not hasattr(resources.Usage, "decisions"):
        pytest.skip("This OpenAI version has no historical Decisions usage report")


def test_decision_usage_reuses_responses_extraction():
    response = decision_types.Decision.model_validate(_decision())

    usage = OpenAIAdapter().extract_usage(response)

    assert usage.input_tokens == 120
    assert usage.output_tokens == 40
    assert usage.total_tokens == 160
    assert usage.cached_input_tokens == 30
    assert usage.cache_creation_5m_tokens == 20
    assert usage.reasoning_tokens == 10
    assert not usage.is_estimated


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_allow_checks_and_settles_text_with_run_attribution(is_async):
    payload = _decision()
    payload["model"] = "gpt-6-luna-resolved"
    async with _wrapped(is_async, response=payload) as (wrapper, plane, seen):
        with run("parent") as parent_id, run("decisions", tags={"job": "test"}) as run_id:
            response = await _create(wrapper, is_async, **REQUEST, solwyn_tags={"step": "classify"})

    assert isinstance(response, decision_types.Decision)
    assert len(seen) == 1
    assert seen[0].url.path == "/v1/decisions"
    assert seen[0].headers["authorization"] == "Bearer offline"
    assert json.loads(seen[0].content) == REQUEST
    assert len(plane.checks) == len(plane.confirms) == len(plane.ingested) == 1
    check, confirm, event = plane.checks[0], plane.confirms[0], plane.ingested[0]
    assert check.model == MODEL
    assert check.estimated_input_tokens == 15  # input + question, 60 chars / 4
    assert check.modality == confirm.modality == event.modality == "text"
    assert check.provider == confirm.provider == event.provider == "openai"
    assert check.agent_run_id == event.agent_run_id == run_id
    assert event.parent_agent_run_id == parent_id
    assert event.tags == {"job": "test", "step": "classify"}
    assert confirm.call_id == event.call_id
    assert confirm.model == event.model == response.model
    assert confirm.token_details == event.token_details
    assert event.input_tokens == 120 and event.output_tokens == 40
    assert event.token_details.cached_input_tokens == 30
    assert event.token_details.cache_creation_5m_tokens == 20
    assert event.token_details.reasoning_tokens == 10
    assert not event.token_details.is_estimated
    assert event.status == CallStatus.SUCCESS
    assert not event.is_model_fallback
    assert plane.untracked_reports == []
    wire = json.dumps([item.model_dump(mode="json") for item in (check, confirm, event)])
    assert REQUEST["input"] not in wire
    assert REQUEST["questions"][0]["instructions"] not in wire
    assert "probability" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_budget_deny_never_dispatches_or_settles(is_async):
    async with _wrapped(is_async) as (wrapper, plane, seen):
        plane.deny_next()
        with pytest.raises(BudgetExceededError):
            await _create(wrapper, is_async, **REQUEST)

    assert seen == []
    assert len(plane.checks) == 1
    assert plane.checks[0].estimated_input_tokens == 15
    assert plane.confirms == []
    assert len(plane.ingested) == 1
    assert plane.ingested[0].status == CallStatus.BUDGET_DENIED
    assert plane.ingested[0].modality == "text"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("usage", [None, {}], ids=["missing", "zeroed"])
async def test_decisions_unusable_usage_settles_marked_request_estimate(is_async, usage):
    payload = _decision()
    payload["usage"] = usage
    async with _wrapped(is_async, response=payload) as (wrapper, plane, _):
        await _create(wrapper, is_async, **REQUEST)

    assert len(plane.confirms) == 1
    details = plane.confirms[0].token_details
    assert details.input_tokens == 15
    assert details.output_tokens == 0
    assert details.is_estimated
    assert plane.ingested[0].token_details == details


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("iterator_questions", [False, True], ids=["no-text", "iterator"])
async def test_decisions_missing_usage_with_no_readable_text_still_settles(
    is_async, iterator_questions
):
    payload = _decision()
    payload["usage"] = None
    questions = [{"type": "predicate", "instructions": ""}]
    if iterator_questions:
        questions = iter(REQUEST["questions"])
    plane = FakeControlPlane(granted_tokens=60, final_grant=True)
    async with _wrapped(is_async, response=payload, plane=plane, lease_output_bound_default=20) as (
        wrapper,
        _,
        seen,
    ):
        with run("decisions-image-only") as run_id:
            await _create(
                wrapper,
                is_async,
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "https://example.test/image.png"}
                        ],
                    }
                ],
                questions=questions,
            )
            state = wrapper._solwyn_budget._lease.state_for(run_id)
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == 20

    assert len(plane.confirms) == len(plane.ingested) == len(seen) == 1
    details = plane.confirms[0].token_details
    assert details.is_estimated
    assert details.input_tokens == 1  # Existing heuristic minimum for unmeasurable text.
    assert details.output_tokens == 0
    assert plane.ingested[0].token_details == details
    assert json.loads(seen[0].content)["questions"] == (
        REQUEST["questions"] if iterator_questions else [{"type": "predicate", "instructions": ""}]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("missing_usage", [False, True], ids=["measured", "estimated"])
async def test_decisions_lease_keeps_output_bound_when_usage_is_unmeasured(is_async, missing_usage):
    payload = _decision()
    payload["usage"] = (
        None
        if missing_usage
        else {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        }
    )
    plane = FakeControlPlane(granted_tokens=60, final_grant=True)
    async with _wrapped(is_async, response=payload, plane=plane, lease_output_bound_default=20) as (
        wrapper,
        _,
        seen,
    ):
        with run("decisions-lease") as run_id:
            await _create(wrapper, is_async, **REQUEST)
            state = wrapper._solwyn_budget._lease.state_for(run_id)
            assert state.spent_tokens_since_report == (35 if missing_usage else 3)
            await _create(wrapper, is_async, **REQUEST)

    assert len(seen) == len(plane.confirms) == 2
    assert plane.confirms[0].lease_id is not None
    # An exhausted final lease returns to an authoritative per-call check.
    assert len(plane.checks) == (1 if missing_usage else 0)
    assert (plane.confirms[1].lease_id is None) == missing_usage


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("provider", ["azure_openai", "groq", "openai_compatible"])
async def test_other_providers_retain_acknowledged_unmetered_decisions(is_async, provider):
    async with _wrapped(
        is_async, provider=provider, acknowledge_untracked=["decisions.create"]
    ) as (wrapper, plane, seen):
        await _create(wrapper, is_async, **REQUEST)

    assert len(seen) == 1
    assert plane.checks == plane.confirms == plane.ingested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_acknowledged_raw_decisions_mirror_does_not_settle(is_async):
    async with _wrapped(is_async, acknowledge_untracked=["decisions.with_raw_response.create"]) as (
        wrapper,
        plane,
        seen,
    ):
        response = wrapper.decisions.with_raw_response.create(**REQUEST)
        if is_async:
            response = await response
        parsed = response.parse()
        assert isinstance(parsed, decision_types.Decision)

    assert len(seen) == 1
    assert plane.checks == plane.confirms == plane.ingested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_acknowledged_historical_decisions_usage_is_never_settled(is_async):
    _require_admin_usage_decisions()
    payload = {"object": "page", "data": [], "has_more": False, "next_page": None}
    async with _wrapped(
        is_async,
        response=payload,
        acknowledge_untracked=["admin.organization.usage.decisions"],
    ) as (wrapper, plane, seen):
        wrapper.admin_api_key = "offline-admin"
        response = wrapper.admin.organization.usage.decisions(start_time=0)
        if is_async:
            await response

    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/v1/organization/usage/decisions"
    assert seen[0].headers["authorization"] == "Bearer offline-admin"
    assert plane.checks == plane.confirms == plane.ingested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_provider_error_does_not_dispatch_a_fallback(is_async):
    fallback_calls = []

    def handle(request):
        fallback_calls.append(request)
        return httpx.Response(200, json=_decision())

    if is_async:
        raw_fallback = openai.AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
    else:
        raw_fallback = openai.OpenAI(
            api_key="offline", http_client=httpx.Client(transport=httpx.MockTransport(handle))
        )
    try:
        async with _wrapped(
            is_async,
            status=500,
            response={"error": {"message": "offline error", "type": "server_error"}},
            fallback=[(raw_fallback, MODEL)],
        ) as (wrapper, plane, seen):
            with pytest.raises(openai.InternalServerError):
                await _create(wrapper, is_async, **REQUEST)

        assert len(seen) == 1
        assert fallback_calls == plane.confirms == []
        assert len(plane.ingested) == 1
        assert plane.ingested[0].status == CallStatus.ERROR
    finally:
        if is_async:
            await raw_fallback.close()
        else:
            raw_fallback.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("fault", ["connect", "read", "interrupt", "preparation-interrupt"])
async def test_decisions_dispatch_errors_retire_lease_claims_without_refunding_unknown_spend(
    is_async, fault, monkeypatch
):
    interrupt = (
        asyncio.CancelledError("private interruption")
        if is_async
        else KeyboardInterrupt("private interruption")
    )
    if fault == "connect":
        transport_error = httpx.ConnectTimeout("private transport error")
    elif fault == "read":
        transport_error = httpx.ReadTimeout("private transport error")
    else:
        transport_error = interrupt
    expected_error = type(interrupt) if "interrupt" in fault else openai.APITimeoutError
    unknown_spend = fault in {"read", "interrupt"}
    plane = FakeControlPlane(granted_tokens=35, final_grant=True)
    async with _wrapped(
        is_async, plane=plane, transport_error=transport_error, lease_output_bound_default=20
    ) as (wrapper, _, seen):
        if fault == "preparation-interrupt":

            def stop_before_dispatch(**_kwargs):
                raise interrupt

            monkeypatch.setattr(wrapper._solwyn_client, "with_options", stop_before_dispatch)
        with run("decisions-dispatch-error") as run_id:
            with pytest.raises(expected_error):
                await _create(wrapper, is_async, **REQUEST)
            state = wrapper._solwyn_budget._lease.state_for(run_id)
            assert state.reserved_tokens == 0
            assert state.spent_tokens_since_report == (35 if unknown_spend else 0)
            if unknown_spend:
                plane.deny_next()
                with pytest.raises(BudgetExceededError):
                    await _create(wrapper, is_async, **REQUEST)

    assert len(seen) == (0 if fault == "preparation-interrupt" else 1)
    assert plane.confirms == []
    errors = [event for event in plane.ingested if event.status == CallStatus.ERROR]
    assert len(errors) == 1
    assert errors[0].lease_id is not None
    assert errors[0].possibly_succeeded is (True if unknown_spend else None)
    assert "private" not in errors[0].model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_emit_content_free_call_spans(is_async, monkeypatch):
    from solwyn import _otel

    trace = pytest.importorskip("opentelemetry.trace")
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    provider = sdk.TracerProvider()
    exporter = memory.InMemorySpanExporter()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_otel, "_trace_api", trace)
    monkeypatch.setattr(_otel, "_trace_api_resolved", True)
    monkeypatch.setattr(_otel, "_call_tracer", provider.get_tracer("decisions-test"))
    try:
        async with _wrapped(is_async, otel_spans=True) as (wrapper, plane, _):
            await _create(wrapper, is_async, **REQUEST)

        calls = [span for span in exporter.get_finished_spans() if span.name == "solwyn.call"]
        assert len(calls) == 1
        assert calls[0].status.status_code == trace.StatusCode.OK
        assert calls[0].attributes["solwyn.call_id"] == plane.ingested[0].call_id
        assert REQUEST["input"] not in str(dict(calls[0].attributes))
        assert REQUEST["questions"][0]["instructions"] not in str(dict(calls[0].attributes))
        assert not calls[0].events
    finally:
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "path",
    [
        "decisions.with_raw_response.create",
        "decisions.with_streaming_response.create",
        "with_raw_response.decisions.create",
        "with_streaming_response.decisions.create",
        "admin.organization.usage.decisions",
    ],
)
async def test_decisions_raw_mirrors_and_admin_report_keep_exact_untracked_posture(is_async, path):
    if path == "admin.organization.usage.decisions":
        _require_admin_usage_decisions()
    async with _wrapped(is_async) as (wrapper, plane, seen):
        with pytest.raises(UntrackedSpendSurfaceError):
            value = wrapper
            for name in path.split("."):
                value = getattr(value, name)

    assert seen == plane.checks == plane.confirms == plane.ingested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_stale_acknowledgment_is_rejected_as_metered(is_async):
    raw = (openai.AsyncOpenAI if is_async else openai.OpenAI)(api_key="offline")
    plane = FakeControlPlane()
    try:
        with pytest.raises(ConfigurationError, match="classified as metered"):
            (plane.wrap_async if is_async else plane.wrap)(
                raw, acknowledge_untracked=["decisions.create"]
            )
    finally:
        if is_async:
            await raw.close()
        else:
            raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("field", ["model", "input", "questions", "stream"])
async def test_decisions_extra_body_cannot_override_metering_fields(is_async, field):
    async with _wrapped(is_async) as (wrapper, plane, seen):
        with pytest.raises(ConfigurationError, match="extra_body"):
            await _create(wrapper, is_async, **REQUEST, extra_body={field: "override"})

    assert seen == plane.checks == plane.confirms == plane.ingested == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_tool_runner_meters_each_decision_with_the_wrapped_client(is_async):
    runners = pytest.importorskip("openai.lib.decisions_tool_runner")
    actions = []

    def response():
        payload = _decision()
        choice = "done" if actions else "act"
        payload["answers"] = [
            {
                "name": "next_action",
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": [
                    {"value": "act", "probability": float(choice == "act")},
                    {"value": "done", "probability": float(choice == "done")},
                ],
            }
        ]
        return payload

    def execute():
        actions.append("executed")

    async def execute_async():
        execute()

    def candidates(_observation):
        return [
            {
                "id": "act",
                "description": "Take an action",
                "execute": execute_async if is_async else execute,
            },
            {"id": "done", "description": "Finish", "done": True},
        ]

    async def observe_async():
        return "observation"

    async def candidates_async(observation):
        return candidates(observation)

    async with _wrapped(is_async, response=response) as (wrapper, plane, seen):
        if is_async:
            runner = runners.AsyncDecisionsToolRunner(
                observe=observe_async, candidates=candidates_async
            )
            result = await runner.run(wrapper, {"model": MODEL, "input": "task"})
        else:
            runner = runners.DecisionsToolRunner(
                observe=lambda: "observation", candidates=candidates
            )
            result = runner.run(wrapper, {"model": MODEL, "input": "task"})

    assert result["status"] == "unverified"
    assert actions == ["executed"]
    assert len(result["decisions"]) == 2
    assert len(seen) == len(plane.checks) == len(plane.confirms) == len(plane.ingested) == 2
    assert len({event.call_id for event in plane.ingested}) == 2
