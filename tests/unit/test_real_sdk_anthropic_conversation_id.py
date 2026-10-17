"""Native Anthropic spans correlate runs without adding provider-wire fields."""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

import solwyn
from solwyn.testing import FakeControlPlane

anthropic = pytest.importorskip("anthropic", minversion="1.14.0")
httpx2 = pytest.importorskip("httpx2")
otel_trace = pytest.importorskip("opentelemetry.sdk.trace")
otel_export = pytest.importorskip("opentelemetry.sdk.trace.export")
otel_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")


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
@pytest.mark.parametrize("customer_conversation", [False, True], ids=["run-id", "customer-id"])
async def test_native_span_conversation_id_never_reaches_provider_request(
    is_async: bool, streaming: bool, customer_conversation: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = otel_memory.InMemorySpanExporter()
    tracer_provider = otel_trace.TracerProvider()
    tracer_provider.add_span_processor(otel_export.SimpleSpanProcessor(exporter))
    seen_requests: list[Any] = []
    seen_options: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen_requests.append(request)
        return _response(request, streaming)

    # Anthropic 1.15 uses httpx2; its MockTransport is the provider's own
    # in-memory HTTP seam, with no socket or global httpx aliasing.
    http_client_type = httpx2.AsyncClient if is_async else httpx2.Client
    provider_type = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
    options: dict[str, Any] = {
        "tracer_provider": tracer_provider,
        "propagation": False,
        "traces": {"content_mode": "metadata_only", "max_content_bytes": 128},
    }
    if customer_conversation:
        options["conversation_id"] = "customer-conversation"
    provider = provider_type(
        api_key="sk-ant-test",
        max_retries=0,
        open_telemetry=options,
        http_client=http_client_type(transport=httpx2.MockTransport(handler)),
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
    wrapper = wrap(provider, anthropic_conversation_id=True, lease_enabled=False)
    try:
        with solwyn.run("private run name", tags={"private": "private tag"}) as run_id:
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

        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "anthropic.messages.create"
        ]
        assert len(spans) == 1
        expected_id = "customer-conversation" if customer_conversation else run_id
        assert spans[0].attributes["gen_ai.conversation.id"] == expected_id
        assert len(seen_options) == 1
        if customer_conversation:
            assert "open_telemetry" not in seen_options[0]
        else:
            assert seen_options[0]["open_telemetry"] == {**original, "conversation_id": run_id}
            assert seen_options[0]["open_telemetry"]["tracer_provider"] is tracer_provider
        assert provider.open_telemetry == original
        assert len(seen_requests) == 1
        request = seen_requests[0]
        wire = (
            json.dumps(dict(request.headers)).encode() + request.content + str(request.url).encode()
        )
        assert b"conversation_id" not in wire
        assert expected_id.encode() not in wire
        assert run_id.encode() not in wire
    finally:
        if is_async:
            await wrapper.close()
        else:
            wrapper.close()
        tracer_provider.shutdown()
