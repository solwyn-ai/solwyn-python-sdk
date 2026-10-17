"""Hermetic tests of correlation against the optional OpenTelemetry API/SDK."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

trace = pytest.importorskip("opentelemetry.trace")
sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

from solwyn._otel import capture_trace  # noqa: E402


@pytest.mark.unit
def test_no_provider_or_active_span() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from solwyn._otel import capture_trace; assert capture_trace() is None",
        ],
        check=True,
    )


@pytest.mark.unit
def test_invalid_span_returns_none() -> None:
    with trace.use_span(trace.INVALID_SPAN):
        assert capture_trace() is None


@pytest.mark.unit
@pytest.mark.parametrize("sampled", [False, True])
def test_fixed_width_lowercase_hex_and_sampled(sampled: bool) -> None:
    context = trace.SpanContext(
        trace_id=0xABC,
        span_id=0xDE,
        is_remote=False,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED if sampled else 0),
    )
    # A strict mock permits exactly the one span operation Solwyn may perform.
    span = Mock(spec_set=trace.Span)
    span.get_span_context.return_value = context
    with trace.use_span(span):
        snapshot = capture_trace()
    assert snapshot == ("00000000000000000000000000000abc", "00000000000000de", sampled)
    assert span.mock_calls == [("get_span_context", (), {})]


@pytest.mark.unit
def test_application_tracer_provider_without_global_registration() -> None:
    provider = sdk_trace.TracerProvider()
    try:
        with provider.get_tracer(__name__).start_as_current_span("application") as span:
            expected = span.get_span_context()
            snapshot = capture_trace()
        assert snapshot is not None
        assert snapshot.trace_id == trace.format_trace_id(expected.trace_id)
        assert snapshot.span_id == trace.format_span_id(expected.span_id)
    finally:
        provider.shutdown()


@pytest.mark.unit
def test_broken_custom_span_cannot_break_provider_call() -> None:
    span = Mock(spec_set=trace.Span)
    span.get_span_context.side_effect = RuntimeError("unavailable")
    with trace.use_span(span):
        assert capture_trace() is None


@pytest.mark.unit
def test_httpx2_alias_before_solwyn_import_keeps_transport_classification_and_delivery() -> None:
    pytest.importorskip("httpx2")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
from types import SimpleNamespace
import httpx2
httpx2.alias_httpx()
import httpx
import solwyn
from opentelemetry import trace
from solwyn.providers._errors import classify_exception, Disposition
from solwyn.testing import FakeControlPlane

assert httpx is httpx2
assert classify_exception(httpx.ConnectTimeout('connect')) is Disposition.FAILOVER
assert classify_exception(httpx.ReadTimeout('read')) is Disposition.POST_SEND_AMBIGUOUS
assert classify_exception(httpx2.ConnectError('connect')) is Disposition.FAILOVER

class OpenAI:
    __module__ = 'openai._client'
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def with_options(self, **kwargs):
        return self
    def create(self, **kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1))
    def close(self):
        pass
class AsyncOpenAI(OpenAI):
    __module__ = 'openai._client'
    async def create(self, **kwargs):
        return super().create(**kwargs)
    async def close(self):
        pass

context = trace.SpanContext(trace_id=0xABC, span_id=0xDE, is_remote=False)
plane = FakeControlPlane()
with trace.use_span(trace.NonRecordingSpan(context)):
    with plane.wrap(OpenAI()) as wrapped:
        wrapped.chat.completions.create(model='gpt-5.5', messages=[])
    async def run():
        async with plane.wrap_async(AsyncOpenAI()) as wrapped:
            await wrapped.chat.completions.create(model='gpt-5.5', messages=[])
    asyncio.run(run())
assert len(plane.ingested) == len(plane.confirms) == 2
assert all(event.trace_id == '00000000000000000000000000000abc' for event in plane.ingested)
""",
        ],
        check=True,
    )


@pytest.mark.unit
def test_httpx2_alias_after_solwyn_import_reports_required_ordering() -> None:
    pytest.importorskip("httpx2")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import solwyn
import httpx2
try:
    httpx2.alias_httpx()
except RuntimeError as exc:
    assert 'before any' in str(exc)
else:
    raise AssertionError('late alias must reject already imported httpx classes')
""",
        ],
        check=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize("abandon", [False, True])
def test_real_stream_keeps_callers_span_when_consumed_on_another_thread(abandon: bool) -> None:
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    from solwyn.testing import FakeControlPlane

    data = {
        "id": "chat-trace",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.5",
        "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    body = f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n".encode()
    raw = openai.OpenAI(
        api_key="offline",
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(
                lambda request: httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                )
            ),
        ),
    )
    plane = FakeControlPlane()
    wrapper = plane.wrap(raw)
    provider = sdk_trace.TracerProvider()
    try:
        with provider.get_tracer(__name__).start_as_current_span("caller") as span:
            context = span.get_span_context()
            stream = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)

        def consume():
            other = trace.SpanContext(trace_id=1, span_id=2, is_remote=False)
            with trace.use_span(trace.NonRecordingSpan(other)):
                if abandon:
                    stream.close()
                else:
                    list(stream)

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(consume).result()
    finally:
        wrapper.close()
        provider.shutdown()
    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert event.trace_id == trace.format_trace_id(context.trace_id)
    assert event.span_id == trace.format_span_id(context.span_id)
    assert event.trace_sampled is context.trace_flags.sampled
