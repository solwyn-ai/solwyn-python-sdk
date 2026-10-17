"""Native Anthropic spans correlate runs without adding provider-wire fields."""

from __future__ import annotations

import importlib
import inspect
import json
import os
from types import ModuleType
from typing import Any

import pytest

import solwyn
from solwyn.testing import FakeControlPlane

anthropic = pytest.importorskip("anthropic", minversion="1.14.0")


def _test_dependency(name: str) -> ModuleType:
    # Once Anthropic supports telemetry, missing dev dependencies must fail CI.
    if os.environ.get("CI"):
        return importlib.import_module(name)
    return pytest.importorskip(name)


httpx2 = _test_dependency("httpx2")
otel_api = _test_dependency("opentelemetry.trace")
otel_trace = _test_dependency("opentelemetry.sdk.trace")
otel_export = _test_dependency("opentelemetry.sdk.trace.export")
otel_memory = _test_dependency("opentelemetry.sdk.trace.export.in_memory_span_exporter")


def _message() -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _response(request: Any, streaming: bool) -> Any:
    if not streaming:
        return httpx2.Response(
            200, request=request, json=_message(), headers={"request-id": "req_test"}
        )
    message = {**_message(), "content": [], "stop_reason": None}
    events = [
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(
        200,
        request=request,
        text=body,
        headers={"content-type": "text/event-stream", "request-id": "req_test"},
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["message", "stream"])
@pytest.mark.parametrize("trace_correlation", [False, True], ids=["trace-off", "trace-on"])
@pytest.mark.parametrize(
    ("configuration", "customer_conversation"),
    [
        ("mapping", None),
        ("customer-id", "customer-conversation"),
        ("customer-empty", ""),
        ("customer-none", None),
        ("default", None),
        ("none", None),
        ("disabled", None),
    ],
    ids=["run-id", "customer-id", "customer-empty", "customer-none", "default", "none", "disabled"],
)
async def test_native_span_conversation_id_never_reaches_provider_request(
    is_async: bool,
    streaming: bool,
    trace_correlation: bool,
    configuration: str,
    customer_conversation: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = otel_memory.InMemorySpanExporter()
    tracer_provider = otel_trace.TracerProvider()
    tracer_provider.add_span_processor(otel_export.SimpleSpanProcessor(exporter))
    # Default/None clients use the application's global provider. Patch its
    # public lookup so each case uses a real provider without global registration.
    monkeypatch.setattr(otel_api, "get_tracer_provider", lambda: tracer_provider)
    for name in (
        "ANTHROPIC_OPEN_TELEMETRY",
        "ANTHROPIC_OPEN_TELEMETRY_PROPAGATION",
        "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE",
        "ANTHROPIC_OPEN_TELEMETRY_TRACES_MAX_CONTENT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)
    seen_requests: list[Any] = []
    seen_options: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen_requests.append(request)
        return _response(request, streaming)

    # Anthropic 1.14+ uses httpx2; its MockTransport is the provider's own
    # in-memory HTTP seam, with no socket or global httpx aliasing.
    http_client_type = httpx2.AsyncClient if is_async else httpx2.Client
    provider_type = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
    options: dict[str, Any] = {
        "tracer_provider": tracer_provider,
        "propagation": False,
        "traces": {"content_mode": "metadata_only", "max_content_bytes": 128},
    }
    if configuration in {"customer-id", "customer-empty", "customer-none"}:
        options["conversation_id"] = customer_conversation
    constructor_options: dict[str, Any] = {}
    if configuration == "none":
        constructor_options["open_telemetry"] = None
    elif configuration == "disabled":
        constructor_options["open_telemetry"] = False
    elif configuration != "default":
        constructor_options["open_telemetry"] = options
    provider = provider_type(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=http_client_type(transport=httpx2.MockTransport(handler)),
        **constructor_options,
    )
    # Pin the public property path against a genuine SDK client, rather than
    # shadowing its descriptor with a synthetic instance attribute.
    assert type(inspect.getattr_static(provider, "open_telemetry")) is property
    original = provider.open_telemetry
    with_options = provider.with_options

    def capture_options(**kwargs: Any) -> Any:
        seen_options.append(kwargs)
        return with_options(**kwargs)

    monkeypatch.setattr(provider, "with_options", capture_options)
    plane = FakeControlPlane()
    wrap = plane.wrap_async if is_async else plane.wrap
    wrapper = wrap(
        provider,
        anthropic_conversation_id=True,
        trace_correlation=trace_correlation,
        lease_enabled=False,
    )
    try:
        with (
            tracer_provider.get_tracer(__name__).start_as_current_span("caller") as caller_span,
            solwyn.run("private run name", tags={"private": "private tag"}) as run_id,
        ):
            caller_context = caller_span.get_span_context()
            response = wrapper.messages.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                stream=streaming,
            )
            if is_async:
                response = await response
            if streaming:
                if is_async:
                    assert [event.type async for event in response][-1] == "message_stop"
                else:
                    assert [event.type for event in response][-1] == "message_stop"
            else:
                assert response.content[0].text == "ok"
            assert otel_api.get_current_span() is caller_span

        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "anthropic.messages.create"
        ]
        expected_id = run_id if customer_conversation is None else customer_conversation
        if configuration == "disabled":
            assert spans == []
        else:
            assert len(spans) == 1
            assert spans[0].attributes["gen_ai.conversation.id"] == expected_id
            assert spans[0].parent == caller_context
        assert len(seen_options) == 1
        if customer_conversation is not None:
            assert "open_telemetry" not in seen_options[0]
        else:
            assert seen_options[0]["open_telemetry"] == {**original, "conversation_id": run_id}
            if "tracer_provider" in original:
                assert seen_options[0]["open_telemetry"]["tracer_provider"] is tracer_provider
        assert provider.open_telemetry == original
        assert len(seen_requests) == 1
        request = seen_requests[0]
        wire = (
            json.dumps(dict(request.headers)).encode() + request.content + str(request.url).encode()
        )
        assert b"conversation_id" not in wire
        assert b"open_telemetry" not in wire
        if expected_id:
            assert expected_id.encode() not in wire
        assert run_id.encode() not in wire
    finally:
        if is_async:
            await wrapper.close()
        else:
            wrapper.close()
        tracer_provider.shutdown()

    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert event.agent_run_id == run_id
    if trace_correlation:
        assert event.trace_id == otel_api.format_trace_id(caller_context.trace_id)
        assert event.span_id == otel_api.format_span_id(caller_context.span_id)
        assert event.trace_sampled == caller_context.trace_flags.sampled
    else:
        assert (event.trace_id, event.span_id, event.trace_sampled) == (None, None, None)
