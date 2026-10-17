"""Hermetic span lifecycle tests with the optional OTel SDK/exporter."""

from __future__ import annotations

import asyncio
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY
from test_otel_population import _budget, _raw_client, _response

import solwyn
from solwyn import _otel
from solwyn._base import MediaSurfaceSpec
from solwyn._types import ProviderName
from solwyn.config import ProviderEntry, SolwynConfig
from solwyn.testing import FakeControlPlane

trace = pytest.importorskip("opentelemetry.trace")
sdk = pytest.importorskip("opentelemetry.sdk.trace")
export = pytest.importorskip("opentelemetry.sdk.trace.export")
memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

_IDENTITY = {
    "solwyn.call_id",
    "solwyn.provider.requested",
    "solwyn.model.requested",
    "solwyn.budget.mode",
}
_BUDGET = {"solwyn.budget.allowed", "solwyn.lease_funded"}
_ATTEMPT = {"solwyn.provider.served", "solwyn.model.served", "solwyn.attempt_index"}
_DENIAL = {"solwyn.deny_source", "solwyn.denied_by_period"}


@pytest.fixture
def spans(monkeypatch):
    provider = sdk.TracerProvider()
    exporter = memory.InMemorySpanExporter()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    started_calls = []

    class StartRecorder(sdk.SpanProcessor):
        def on_start(self, span, parent_context=None):
            if span.name == "solwyn.call":
                started_calls.append(span)

    provider.add_span_processor(StartRecorder())
    tracer = provider.get_tracer("span-tests")
    api = ModuleType("test_trace_api")
    api.__dict__.update(vars(trace))
    api.get_tracer = lambda *args, **kwargs: tracer
    monkeypatch.setattr(_otel, "_trace_api", api)
    monkeypatch.setattr(_otel, "_trace_api_resolved", True)
    monkeypatch.setattr(_otel, "_call_tracer", None, raising=False)
    ends = Counter()
    original_end = sdk.Span.end

    def end(span, *args, **kwargs):
        ends[span.get_span_context().span_id] += 1
        return original_end(span, *args, **kwargs)

    monkeypatch.setattr(sdk.Span, "end", end)
    yield SimpleNamespace(
        tracer=tracer,
        api=api,
        exporter=exporter,
        ends=ends,
        started_calls=started_calls,
        provider=provider,
        calls=lambda: [s for s in exporter.get_finished_spans() if s.name == "solwyn.call"],
    )
    provider.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("yes", True)])
def test_span_config_environment_and_override(monkeypatch, value, expected):
    monkeypatch.setenv("SOLWYN_OTEL_SPANS", value)
    assert (
        SolwynConfig(
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
            api_key=VALID_API_KEY,
        ).otel_spans
        is expected
    )
    assert SolwynConfig(
        providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
        api_key=VALID_API_KEY,
        otel_spans=not expected,
    ).otel_spans is (not expected)


@pytest.mark.unit
def test_spans_default_off(monkeypatch):
    monkeypatch.delenv("SOLWYN_OTEL_SPANS", raising=False)
    assert (
        SolwynConfig(
            providers=[ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5")],
            api_key=VALID_API_KEY,
        ).otel_spans
        is False
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("media", [False, True], ids=["chat", "media"])
@pytest.mark.parametrize(
    "outcome", ["success", "error", "denial", "run_stop", "post_stop", "interrupt"]
)
async def test_call_span_lifecycle(spans, async_mode, media, outcome):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    open_at_preflight = []

    def dispatch(*args, **kwargs):
        # This simulates independent provider instrumentation at its public seam.
        with spans.tracer.start_as_current_span("provider.native"):
            if outcome == "error":
                raise ValueError("PRIVATE provider error")
            if outcome == "interrupt":
                raise asyncio.CancelledError() if async_mode else KeyboardInterrupt()
        return _response()

    raw.chat.completions.create.side_effect = dispatch
    raw.embeddings.create.side_effect = dispatch

    def with_options(**kwargs):
        with spans.tracer.start_as_current_span("provider.with_options"):
            pass
        return raw

    raw.with_options.side_effect = with_options

    def check(*args, **kwargs):
        open_at_preflight.append(
            len(spans.started_calls) == 1 and spans.started_calls[0].is_recording()
        )
        with spans.tracer.start_as_current_span("budget.httpx"):
            pass
        if outcome == "post_stop":
            from solwyn._run_control import mark_terminated

            mark_terminated(kwargs["agent_run_id"], reason="manual_kill", source="server")
        return _budget(denied=outcome == "denial")

    expected_error = outcome != "success"
    try:
        with (
            patch.object(
                wrapper._solwyn_budget,
                "check_budget",
                new=(
                    AsyncMock(side_effect=check)
                    if async_mode
                    else MagicMock(spec=[], side_effect=check)
                ),
            ),
            spans.tracer.start_as_current_span("caller") as caller,
            solwyn.run("PRIVATE run name", tags={"private": "tag"}) as run_id,
        ):
            caller_id = caller.get_span_context().span_id
            run_span = trace.get_current_span()
            if outcome == "run_stop":
                from solwyn._run_control import mark_terminated

                mark_terminated(run_id, reason="velocity:repeat_size", source="local_velocity")
            try:
                if media:
                    result = wrapper._media_call(
                        MediaSurfaceSpec(
                            surface="embeddings",
                            modality="embedding",
                            extract_usage=lambda response: None,
                            measure_request=lambda kwargs: None,
                        ),
                        model="gpt-5.5",
                        input="PRIVATE prompt",
                    )
                else:
                    result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
                if async_mode:
                    await result
                assert not expected_error
            except (Exception, KeyboardInterrupt, asyncio.CancelledError):
                assert expected_error
            assert trace.get_current_span() is run_span
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()

    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    assert call.kind == trace.SpanKind.INTERNAL
    assert call.parent.span_id == run_span.get_span_context().span_id
    run = next(s for s in spans.exporter.get_finished_spans() if s.name == "solwyn.run")
    assert run.parent.span_id == caller_id
    assert spans.ends[call.context.span_id] == 1
    assert call.status.status_code == (
        trace.StatusCode.ERROR if expected_error else trace.StatusCode.OK
    )
    assert call.status.description is None
    assert not call.events
    for child in spans.exporter.get_finished_spans():
        if child.name in {"provider.native", "provider.with_options", "budget.httpx"}:
            assert child.parent.span_id == call.context.span_id
    assert open_at_preflight == ([] if outcome == "run_stop" else [True])
    assert call.attributes["solwyn.agent_run_id"] == run_id
    assert call.attributes["solwyn.provider.requested"] == "openai"
    assert call.attributes["solwyn.model.requested"] == "gpt-5.5"
    assert all(key.startswith("solwyn.") for key in call.attributes)
    assert "PRIVATE" not in str(dict(call.attributes))
    expected_keys = _IDENTITY | {"solwyn.agent_run_id"}
    if outcome != "run_stop":
        expected_keys |= _BUDGET
    else:
        expected_keys |= {"solwyn.budget.allowed"}
    if outcome in {"run_stop", "post_stop", "denial"}:
        expected_keys |= _DENIAL
        assert call.attributes["solwyn.budget.allowed"] is False
    else:
        expected_keys |= _ATTEMPT
    assert set(call.attributes) == expected_keys
    assert bool(plane.ingested) is not (media and outcome == "interrupt")
    assert all(
        event.span_id == trace.format_span_id(call.context.span_id) for event in plane.ingested
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("terminal", ["normal", "close", "gc", "error", "interrupt"])
async def test_stream_span_is_not_current_and_ends_once(spans, async_mode, terminal):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)

    def chunks():
        assert trace.get_current_span() is consumer
        yield _response()
        if terminal == "error":
            raise ValueError("PRIVATE stream error")
        if terminal == "interrupt":
            raise asyncio.CancelledError() if async_mode else KeyboardInterrupt()

    async def async_chunks():
        for chunk in chunks():
            yield chunk

    raw.chat.completions.create.return_value = async_chunks() if async_mode else chunks()
    try:
        with spans.tracer.start_as_current_span("caller") as caller:
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            if async_mode:
                result = await result
            assert trace.get_current_span() is caller
            assert not spans.calls()
        with spans.tracer.start_as_current_span("consumer") as consumer:
            if terminal == "gc":
                del result
                gc.collect()
            elif terminal == "close":
                if async_mode:
                    await result.close()
                    await result.close()
                else:
                    result.close()
                    result.close()
            else:
                with (
                    pytest.raises((ValueError, KeyboardInterrupt, asyncio.CancelledError))
                    if terminal in {"error", "interrupt"}
                    else nullcontext()
                ):
                    if async_mode:
                        async for _ in result:
                            assert trace.get_current_span() is consumer
                    else:
                        for _ in result:
                            assert trace.get_current_span() is consumer
                if async_mode:
                    await result.close()
                else:
                    result.close()
            assert trace.get_current_span() is consumer
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    assert spans.ends[call.context.span_id] == 1
    assert not call.events
    assert call.status.description is None
    assert call.status.status_code == (
        trace.StatusCode.ERROR if terminal in {"error", "interrupt", "gc"} else trace.StatusCode.OK
    )
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["message", "stream"])
@pytest.mark.parametrize("correlation", [False, True], ids=["trace-off", "trace-on"])
async def test_native_anthropic_conversation_span_is_a_child_of_call(
    spans, async_mode, streaming, correlation
):
    from test_real_sdk_anthropic_conversation_id import (
        _response as anthropic_response,
    )
    from test_real_sdk_anthropic_conversation_id import anthropic, httpx2

    requests = []

    def handler(request):
        requests.append(request)
        return anthropic_response(request, streaming)

    transport = httpx2.MockTransport(handler)
    http = (httpx2.AsyncClient if async_mode else httpx2.Client)(transport=transport)
    raw = (anthropic.AsyncAnthropic if async_mode else anthropic.Anthropic)(
        api_key="offline",
        http_client=http,
        open_telemetry={"tracer_provider": spans.provider, "propagation": False},
    )
    original_options = dict(raw.open_telemetry)
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw,
        otel_spans=True,
        anthropic_conversation_id=True,
        trace_correlation=correlation,
        lease_enabled=False,
    )
    try:
        with (
            spans.tracer.start_as_current_span("caller") as caller,
            solwyn.run("PRIVATE run") as run,
        ):
            run_span = trace.get_current_span()
            result = wrapper.messages.create(
                model="claude-sonnet-5",
                max_tokens=16,
                messages=[{"role": "user", "content": "offline"}],
                stream=streaming,
            )
            if async_mode:
                result = await result
            if streaming:
                if async_mode:
                    assert [event.type async for event in result][-1] == "message_stop"
                else:
                    assert [event.type for event in result][-1] == "message_stop"
            else:
                assert result.usage.input_tokens == 10
            assert trace.get_current_span() is run_span
    finally:
        await _close(wrapper, async_mode)

    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    native = [
        span
        for span in spans.exporter.get_finished_spans()
        if span.name == "anthropic.messages.create"
    ]
    assert len(native) == 1
    assert native[0].parent == call.context
    assert native[0].attributes["gen_ai.conversation.id"] == run
    assert call.parent == run_span.get_span_context()
    assert run_span.parent == caller.get_span_context()
    assert call.attributes["solwyn.agent_run_id"] == run
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT | {"solwyn.agent_run_id"}
    assert not any(key.startswith("gen_ai.") for key in call.attributes)
    assert spans.ends[call.context.span_id] == 1
    assert not call.events
    assert raw.open_telemetry == original_options
    assert len(plane.ingested) == 1
    assert plane.ingested[0].span_id == (
        trace.format_span_id(call.context.span_id) if correlation else None
    )
    assert len(requests) == 1
    request = requests[0]
    assert run not in str(request.headers) + str(request.url)
    assert run.encode() not in request.content


async def _close(wrapper, async_mode):
    if async_mode:
        await wrapper.close()
    else:
        wrapper.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("mode", ["off", "noop", "drop", "record_only", "absent"])
@pytest.mark.parametrize("surface", ["chat", "stream", "media"])
async def test_inactive_spans_preserve_caller_context_and_correlation(
    spans, monkeypatch, async_mode, mode, surface
):
    sampling = pytest.importorskip("opentelemetry.sdk.trace.sampling")
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=mode != "off")
    caller_context = None

    def dispatch(*args, **kwargs):
        assert trace.get_current_span().get_span_context() == caller_context
        return _response()

    raw.chat.completions.create.side_effect = dispatch
    raw.embeddings.create.side_effect = dispatch
    if surface == "stream":

        def chunks():
            yield _response()

        async def async_chunks():
            yield _response()

        raw.chat.completions.create.side_effect = lambda **kwargs: (
            async_chunks() if async_mode else chunks()
        )

    class RecordOnly(sampling.Sampler):
        def should_sample(self, *args, **kwargs):
            return sampling.SamplingResult(sampling.Decision.RECORD_ONLY)

        def get_description(self):
            return "record-only"

    noop = trace.NoOpTracerProvider()
    drop = sdk.TracerProvider(sampler=sampling.ALWAYS_OFF)
    record_only = sdk.TracerProvider(sampler=RecordOnly())
    if mode in {"noop", "drop", "record_only"}:
        tracer = {"noop": noop, "drop": drop, "record_only": record_only}[mode].get_tracer("idle")
        monkeypatch.setattr(spans.api, "get_tracer", lambda *args, **kwargs: tracer)
    elif mode == "off":
        monkeypatch.setattr(
            spans.api, "get_tracer", MagicMock(spec=[], side_effect=AssertionError("disabled"))
        )
    else:
        monkeypatch.setattr(_otel, "_trace_api", None)
        monkeypatch.setattr(_otel, "_trace_api_resolved", True)

    def unexpected_context(*args, **kwargs):
        raise AssertionError("inactive spans must use the shared empty context")

    monkeypatch.setattr(_otel._CallSpan, "current", unexpected_context)
    monkeypatch.setattr(_otel._CallSpan, "finishing", unexpected_context)
    from solwyn import client

    reasons = MagicMock(spec=[], side_effect=client._success_failover_reason)
    monkeypatch.setattr(client, "_success_failover_reason", reasons)

    try:
        with spans.tracer.start_as_current_span("caller") as caller, solwyn.run("PRIVATE run"):
            caller_context = caller.get_span_context()
            if surface == "media":
                result = wrapper.embeddings.create(model="text-embedding-3-small", input="x")
            else:
                result = wrapper.chat.completions.create(
                    model="gpt-5.5", messages=[], stream=surface == "stream"
                )
            if async_mode:
                result = await result
            if surface == "stream":
                if async_mode:
                    assert len([c async for c in result]) == 1
                else:
                    assert len(list(result)) == 1
                    result.close()
                if async_mode:
                    await result.close()
            assert trace.get_current_span() is caller
    finally:
        await _close(wrapper, async_mode)
        drop.shutdown()
        record_only.shutdown()
    assert not spans.calls()
    assert all(s.name != "solwyn.run" for s in spans.exporter.get_finished_spans())
    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert event.span_id == (
        None if mode == "absent" else trace.format_span_id(caller_context.span_id)
    )
    assert event.trace_id == (
        None if mode == "absent" else trace.format_trace_id(caller_context.trace_id)
    )
    # The receipt still computes its reason once; no span-only computation.
    assert reasons.call_count == (0 if surface == "media" else 1)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cross_provider", [False, True])
@pytest.mark.parametrize("outcome", ["success", "all_failed", "ambiguous"])
async def test_failover_uses_one_span_for_all_attempts(
    spans, async_mode, streaming, cross_provider, outcome
):
    plane = FakeControlPlane()
    raw, fallback = _raw_client(async_mode), _raw_client(async_mode)
    fallback_provider = "groq" if cross_provider else "openai"
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw,
        otel_spans=True,
        provider="openai",
        fallback=[(fallback, "fallback-model", {}, fallback_provider)],
        failover_idempotency="always" if outcome == "ambiguous" else "safe",
    )
    parents = []

    def primary_dispatch(**kwargs):
        parents.append(trace.get_current_span().get_span_context())
        raise (
            httpx.ReadTimeout("PRIVATE timeout")
            if outcome == "ambiguous"
            else httpx.ConnectError("PRIVATE connection")
        )

    def fallback_dispatch(**kwargs):
        parents.append(trace.get_current_span().get_span_context())
        if outcome == "all_failed":
            raise httpx.ConnectError("PRIVATE connection")

        def chunks():
            yield _response()

        async def async_chunks():
            yield _response()

        return (async_chunks() if async_mode else chunks()) if streaming else _response()

    raw.chat.completions.create.side_effect = primary_dispatch
    fallback.chat.completions.create.side_effect = fallback_dispatch
    try:
        with pytest.raises(httpx.ConnectError) if outcome == "all_failed" else nullcontext():
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=streaming)
            if async_mode:
                result = await result
            if streaming:
                assert not spans.calls()
                if async_mode:
                    assert len([c async for c in result]) == 1
                else:
                    assert len(list(result)) == 1
    finally:
        await _close(wrapper, async_mode)
    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    assert spans.ends[call.context.span_id] == 1
    assert all(parent.span_id == call.context.span_id for parent in parents)
    assert len(parents) == 2
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT | {"solwyn.failover.reason"}
    assert call.attributes["solwyn.provider.served"] == fallback_provider
    assert call.attributes["solwyn.model.served"] == "fallback-model"
    assert call.attributes["solwyn.attempt_index"] == 1
    assert call.attributes["solwyn.failover.reason"] == (
        "primary_error" if cross_provider else "model_fallback"
    )
    assert not call.events
    assert call.status.status_code == (
        trace.StatusCode.ERROR if outcome == "all_failed" else trace.StatusCode.OK
    )
    assert call.status.description is None
    assert len(plane.ingested) == 2
    assert all(e.span_id == trace.format_span_id(call.context.span_id) for e in plane.ingested)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "terminal",
    [
        "normal",
        "close",
        "gc",
        "entered_gc",
        "entry_error",
        "entry_interrupt",
        "wrap_error",
        "stream_error",
    ],
)
async def test_responses_manager_span_lifetime(spans, async_mode, terminal):
    raw = _raw_client(async_mode)

    def chunks():
        yield _response()
        if terminal == "stream_error":
            raise ValueError("PRIVATE stream error")

    async def async_chunks():
        for chunk in chunks():
            yield chunk

    def enter():
        with spans.tracer.start_as_current_span("native.responses"):
            if terminal == "entry_error":
                raise httpx.ReadTimeout("PRIVATE entry error")
            if terminal == "entry_interrupt":
                raise asyncio.CancelledError() if async_mode else GeneratorExit()
        return async_chunks() if async_mode else chunks()

    if async_mode:
        manager = MagicMock(spec=["__aenter__", "__aexit__"])
        manager.__aenter__ = AsyncMock(side_effect=enter)
        manager.__aexit__ = AsyncMock(return_value=False)
    else:
        manager = MagicMock(spec=["__enter__", "__exit__"])
        manager.__enter__.side_effect = enter
        manager.__exit__.return_value = False
    raw.responses = SimpleNamespace(stream=MagicMock(spec=[], return_value=manager))
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw, provider="openai", otel_spans=True
    )
    try:
        with spans.tracer.start_as_current_span("caller"):
            result = wrapper._intercepted_call(
                _surface="responses",
                _responses_leaf="stream",
                _force_stream=True,
                model="gpt-5.5",
                input=[],
            )
            if async_mode:
                result = await result
            assert not spans.calls()
        with spans.tracer.start_as_current_span("consumer") as consumer:
            if terminal == "gc":
                del result
                gc.collect()
            elif terminal == "close":
                if async_mode:
                    await result.close()
                    await result.close()
                else:
                    result.close()
                    result.close()
            elif terminal == "entered_gc":
                stream = await result.__aenter__() if async_mode else result.__enter__()
                del result
                gc.collect()
                assert not spans.calls()  # Inner event stream now owns the span.
                del stream
                gc.collect()
            else:
                errors = (
                    httpx.ReadTimeout,
                    ValueError,
                    RuntimeError,
                    GeneratorExit,
                    asyncio.CancelledError,
                )
                with (
                    patch.object(
                        wrapper._solwyn_runtimes[0].adapter,
                        "create_stream_accumulator",
                        side_effect=RuntimeError("PRIVATE wrapper"),
                    )
                    if terminal == "wrap_error"
                    else nullcontext(),
                    pytest.raises(errors) if terminal != "normal" else nullcontext(),
                ):
                    if async_mode:
                        async with result as stream:
                            assert trace.get_current_span() is consumer
                            async for _ in stream:
                                pass
                    else:
                        with result as stream:
                            assert trace.get_current_span() is consumer
                            list(stream)
            assert trace.get_current_span() is consumer
    finally:
        await _close(wrapper, async_mode)
    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    assert spans.ends[call.context.span_id] == 1
    assert call.status.status_code == (
        trace.StatusCode.OK if terminal == "normal" else trace.StatusCode.ERROR
    )
    assert not call.events
    assert call.status.description is None
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT
    for child in spans.exporter.get_finished_spans():
        if child.name == "native.responses":
            assert child.parent.span_id == call.context.span_id


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("surface", ["chat", "media", "responses"])
@pytest.mark.parametrize("spans_enabled", [False, True], ids=["spans-off", "spans-on"])
@pytest.mark.parametrize("correlation", [False, True], ids=["trace-off", "trace-on"])
async def test_real_httpx_preflight_span_is_a_child(
    spans, async_mode, surface, spans_enabled, correlation
):
    instrumentation = pytest.importorskip("opentelemetry.instrumentation.httpx")
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = _response()
    raw.embeddings.create.return_value = _response()
    raw.responses = SimpleNamespace(create=raw.chat.completions.create)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw, otel_spans=spans_enabled, trace_correlation=correlation
    )
    for http_client in (wrapper._solwyn_budget._http, wrapper._solwyn_reporter._http):
        instrumentation.HTTPXClientInstrumentor.instrument_client(
            http_client, tracer_provider=spans.provider
        )
    with spans.tracer.start_as_current_span("caller") as caller:
        try:
            if surface == "media":
                result = wrapper.embeddings.create(model="text-embedding-3-small", input="x")
            elif surface == "responses":
                result = wrapper._intercepted_call(_surface="responses", model="gpt-5.5", input=[])
            else:
                result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            if async_mode:
                await result
            assert trace.get_current_span() is caller
        finally:
            # Closing under the caller also exercises the reporter's isolated
            # final flush, rather than masking context leaks by exiting first.
            await _close(wrapper, async_mode)
        assert trace.get_current_span() is caller
    assert len(spans.calls()) == int(spans_enabled)
    call_parent = spans.calls()[0].context if spans_enabled else caller.get_span_context()
    http_spans = [s for s in spans.exporter.get_finished_spans() if s.kind == trace.SpanKind.CLIENT]
    preflight = [
        s
        for s in http_spans
        if s.attributes.get("url.full", s.attributes.get("http.url", "")).endswith("/budgets/check")
    ]
    background = [s for s in http_spans if s not in preflight]
    assert len(preflight) == 1
    assert preflight[0].parent == call_parent
    assert preflight[0].context.trace_id == call_parent.trace_id
    assert len(plane.confirms) == len(plane.ingested) == 1
    assert len(background) >= 2  # Confirmation and metadata ingestion.
    assert all(s.parent is None for s in background)
    event = plane.ingested[0]
    assert event.span_id == (trace.format_span_id(call_parent.span_id) if correlation else None)
    assert event.trace_id == (trace.format_trace_id(call_parent.trace_id) if correlation else None)


@pytest.mark.unit
def test_call_trace_snapshot_does_not_attach_span(spans, monkeypatch):
    attachments = []
    use_span = spans.api.use_span

    def attach(span, *args, **kwargs):
        attachments.append(span.get_span_context())
        return use_span(span, *args, **kwargs)

    monkeypatch.setattr(spans.api, "use_span", attach)
    with (
        spans.tracer.start_as_current_span("caller") as caller,
        _otel._start_call_span() as owner,
    ):
        snapshot = owner.capture_trace()
        assert snapshot.trace_id == trace.format_trace_id(caller.get_span_context().trace_id)
        assert snapshot.span_id == trace.format_span_id(spans.started_calls[0].context.span_id)
        assert trace.get_current_span() is caller
    assert attachments == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, asyncio.CancelledError, GeneratorExit])
async def test_preflight_baseexception_ends_span(spans, async_mode, media, exception):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        with (
            patch.object(
                wrapper._solwyn_budget,
                "check_budget",
                new=(
                    AsyncMock(side_effect=exception)
                    if async_mode
                    else MagicMock(spec=[], side_effect=exception)
                ),
            ),
            pytest.raises(exception),
        ):
            result = (
                wrapper.embeddings.create(model="text-embedding-3-small", input="x")
                if media
                else wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            )
            if async_mode:
                await result
    finally:
        await _close(wrapper, async_mode)
    assert len(spans.calls()) == 1
    call = spans.calls()[0]
    assert spans.ends[call.context.span_id] == 1
    assert call.status.status_code == trace.StatusCode.ERROR
    assert not call.events
    assert call.status.description is None
    assert set(call.attributes) == _IDENTITY


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["wrap", "finalize", "callback"])
async def test_stream_bookkeeping_failure_ends_span(spans, async_mode, failure):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)

    def chunks():
        yield _response()

    async def async_chunks():
        yield _response()

    raw.chat.completions.create.return_value = async_chunks() if async_mode else chunks()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        with (
            patch.object(
                wrapper._solwyn_runtimes[0].adapter,
                "create_stream_accumulator",
                side_effect=ValueError("PRIVATE wrap"),
            )
            if failure == "wrap"
            else nullcontext(),
            pytest.raises(ValueError) if failure == "wrap" else nullcontext(),
        ):
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            if async_mode:
                result = await result
        if failure != "wrap":
            if failure == "callback":
                result._on_complete = (
                    AsyncMock(side_effect=GeneratorExit)
                    if async_mode
                    else MagicMock(spec=[], side_effect=GeneratorExit)
                )
                target, name, exception = result, "_on_complete", GeneratorExit
            else:
                target, name, exception = result._accumulator, "finalize", ValueError
            with patch.object(target, name, side_effect=exception), pytest.raises(exception):
                if async_mode:
                    await result.close()
                else:
                    result.close()
            if async_mode:
                await result.close()
            else:
                result.close()
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert len(spans.calls()) == 1
    assert spans.ends[call.context.span_id] == 1
    assert call.status.status_code == trace.StatusCode.ERROR
    assert not call.events


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("source", ["server", "local_velocity"])
async def test_run_stop_during_stream_ends_with_error(spans, async_mode, source):
    from solwyn._run_control import mark_terminated
    from solwyn.exceptions import RunStoppedError

    plane = FakeControlPlane()
    raw = _raw_client(async_mode)

    def chunks():
        yield _response()

    async def async_chunks():
        yield _response()

    raw.chat.completions.create.return_value = async_chunks() if async_mode else chunks()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        with solwyn.run("PRIVATE run") as run_id:
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            if async_mode:
                result = await result
            mark_terminated(run_id, reason="manual_kill", source=source)
            with pytest.raises(RunStoppedError):
                if async_mode:
                    await result.__anext__()
                else:
                    next(result)
            if async_mode:
                await result.close()
            else:
                result.close()
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert len(spans.calls()) == 1
    assert spans.ends[call.context.span_id] == 1
    assert call.status.status_code == trace.StatusCode.ERROR
    assert call.attributes["solwyn.budget.allowed"] is True
    assert "solwyn.deny_source" not in call.attributes
    assert "solwyn.denied_by_period" not in call.attributes
    assert not call.events


@pytest.mark.unit
def test_snapshot_is_captured_once_before_cross_thread_stream_consumption(spans, monkeypatch):
    import threading

    plane = FakeControlPlane()
    raw = _raw_client(False)
    raw.chat.completions.create.return_value = iter([_response()])
    wrapper = plane.wrap(raw, otel_spans=True)
    captures = []
    capture_trace = _otel.capture_trace

    def capture(span=None):
        captures.append(threading.get_ident())
        return capture_trace(span)

    monkeypatch.setattr(_otel, "capture_trace", capture)
    try:
        with spans.tracer.start_as_current_span("caller") as caller:
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert len(executor.submit(list, result).result(timeout=5)) == 1
    finally:
        wrapper.close()
    call = spans.calls()[0]
    assert captures == [threading.get_ident()]
    event = plane.ingested[0]
    assert event.trace_id == trace.format_trace_id(caller.get_span_context().trace_id)
    assert event.span_id == trace.format_span_id(call.context.span_id)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("correlation", [False, True])
async def test_nested_run_identity_and_independent_correlation_flag(spans, async_mode, correlation):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = _response()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw, otel_spans=True, trace_correlation=correlation
    )
    try:
        with solwyn.run("PRIVATE parent") as parent, solwyn.run("PRIVATE child") as child:
            with patch("solwyn._otel.capture_trace", wraps=_otel.capture_trace) as capture:
                result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
                if async_mode:
                    await result
            assert capture.call_count == (1 if correlation else 0)
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert call.attributes["solwyn.agent_run_id"] == child
    assert call.attributes["solwyn.parent_agent_run_id"] == parent
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT | {
        "solwyn.agent_run_id",
        "solwyn.parent_agent_run_id",
    }
    assert (plane.ingested[0].span_id is not None) is correlation


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("terminal", ["no_candidates", "deadline"])
async def test_admitted_call_without_a_dispatch_ends_once(spans, async_mode, terminal):
    from solwyn.client import Deadline
    from solwyn.exceptions import ProviderUnavailableError

    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw,
        otel_spans=True,
        circuit_breaker_failure_threshold=1,
    )
    if terminal == "no_candidates":
        wrapper._get_circuit_breaker("openai").record_failure()
    try:
        with (
            patch.object(Deadline, "expired", return_value=True)
            if terminal == "deadline"
            else nullcontext(),
            pytest.raises(ProviderUnavailableError),
        ):
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            if async_mode:
                await result
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert len(spans.calls()) == 1
    assert spans.ends[call.context.span_id] == 1
    assert set(call.attributes) == _IDENTITY | _BUDGET
    assert call.status.status_code == trace.StatusCode.ERROR
    assert not call.events
    assert not plane.ingested
    raw.chat.completions.create.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_cost_routed_failure_followed_by_primary_has_no_stale_reason(spans, async_mode):
    plane = FakeControlPlane(price_hints={"openai": 5.0, "groq": 1.0})
    raw, fallback = _raw_client(async_mode), _raw_client(async_mode)
    raw.chat.completions.create.return_value = _response()
    fallback.chat.completions.create.side_effect = httpx.ConnectError("PRIVATE failure")
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw,
        provider="openai",
        otel_spans=True,
        selection_policy=solwyn.CostPolicy(),
        fallback=[(fallback, "fallback-model", {}, "groq")],
    )
    try:
        result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
        if async_mode:
            await result
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert len(spans.calls()) == 1
    assert len(plane.ingested) == 2
    assert call.attributes["solwyn.provider.served"] == "openai"
    assert call.attributes["solwyn.attempt_index"] == 0
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT
    assert call.status.status_code == trace.StatusCode.OK


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_lease_funded_attribute_is_a_boolean_only(spans, async_mode):
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = _response()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        with solwyn.run("PRIVATE lease"):
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            if async_mode:
                await result
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    assert call.attributes["solwyn.lease_funded"] is True
    assert plane.ingested[0].lease_id
    assert set(call.attributes) == _IDENTITY | _BUDGET | _ATTEMPT | {"solwyn.agent_run_id"}


@pytest.mark.unit
def test_attribute_contract_is_closed():
    assert (
        _IDENTITY
        | _BUDGET
        | _ATTEMPT
        | _DENIAL
        | {
            "solwyn.agent_run_id",
            "solwyn.parent_agent_run_id",
            "solwyn.failover.reason",
        }
    ) == _otel.CALL_SPAN_ATTRIBUTES


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_provider_stream_context_entry_failure_ends_span(spans, async_mode):
    class BadEntry:
        def __enter__(self):
            raise ValueError("PRIVATE context entry")

        async def __aenter__(self):
            raise asyncio.CancelledError()

    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = BadEntry()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
        if async_mode:
            result = await result
        with pytest.raises(asyncio.CancelledError if async_mode else ValueError):
            if async_mode:
                await result.__aenter__()
            else:
                result.__enter__()
        assert len(spans.calls()) == 1
        assert spans.calls()[0].status.status_code == trace.StatusCode.ERROR
        assert spans.ends[spans.calls()[0].context.span_id] == 1
    finally:
        await _close(wrapper, async_mode)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("terminal", ["close", "exit", "provider_exit"])
@pytest.mark.parametrize(
    "exception", [ValueError, KeyboardInterrupt, asyncio.CancelledError, GeneratorExit]
)
async def test_provider_cleanup_failure_ends_span_with_error(
    spans, async_mode, responses, terminal, exception
):
    class Stream:
        def __iter__(self):
            return iter([])

        async def __aiter__(self):
            for _ in []:
                yield _response()

        def close(self):
            if terminal != "provider_exit":
                raise exception()

        async def aclose(self):
            if terminal != "provider_exit":
                raise exception()

        def __exit__(self, *args):
            if terminal == "provider_exit":
                raise exception()
            return False

        async def __aexit__(self, *args):
            if terminal == "provider_exit":
                raise exception()
            return False

    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = Stream()
    if async_mode:
        manager = MagicMock(spec=["__aenter__", "__aexit__"])
        manager.__aenter__ = AsyncMock(return_value=Stream())
        manager.__aexit__ = AsyncMock(side_effect=exception)
    else:
        manager = MagicMock(spec=["__enter__", "__exit__"])
        manager.__enter__.return_value = Stream()
        manager.__exit__.side_effect = exception
    raw.responses = SimpleNamespace(stream=MagicMock(spec=[], return_value=manager))
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(
        raw, otel_spans=True, provider="openai"
    )
    try:
        if responses:
            result = wrapper._intercepted_call(
                _surface="responses",
                _responses_leaf="stream",
                _force_stream=True,
                model="gpt-5.5",
                input=[],
            )
        else:
            result = wrapper.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
        if async_mode:
            result = await result
        if responses:
            if async_mode:
                await result.__aenter__()
            else:
                result.__enter__()
        # Direct __exit__ suppresses ordinary close errors by established policy;
        # even those suppressed errors must still mark the span ERROR.
        suppresses = terminal == "exit" and not responses and exception is ValueError
        with nullcontext() if suppresses else pytest.raises(exception):
            if terminal == "close":
                if async_mode:
                    await result.close()
                else:
                    result.close()
            elif async_mode:
                await result.__aexit__(None, None, None)
            else:
                result.__exit__(None, None, None)
        assert len(spans.calls()) == 1
        call = spans.calls()[0]
        assert call.status.status_code == trace.StatusCode.ERROR
        assert spans.ends[call.context.span_id] == 1
        assert not call.events
        assert call.status.description is None
    finally:
        await _close(wrapper, async_mode)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["get_span_context", "is_recording"])
def test_interruption_while_adopting_span_still_ends_it(spans, monkeypatch, method):
    span = spans.tracer.start_span("solwyn.call")
    context = span.get_span_context()
    original = getattr(span, method)

    def interrupt():
        # Restore before end/export so the observer can count the real end.
        monkeypatch.setattr(span, method, original)
        raise KeyboardInterrupt()

    monkeypatch.setattr(span, method, interrupt)
    tracer = SimpleNamespace(start_span=MagicMock(spec=[], return_value=span))
    monkeypatch.setattr(spans.api, "get_tracer", lambda *args, **kwargs: tracer)
    with pytest.raises(KeyboardInterrupt):
        _otel._start_call_span()
    assert spans.ends[context.span_id] == 1
    assert len(spans.calls()) == 1
    assert spans.calls()[0].status.status_code == trace.StatusCode.ERROR
    assert spans.calls()[0].status.description is None
    assert not spans.calls()[0].events


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["success", "error", "adoption_error"])
def test_status_objects_are_portable_to_older_otel_sdks(spans, monkeypatch, outcome):
    original_set_status = sdk.Span.set_status

    def old_set_status(span, status):
        # SDK <1.12 requires Status and cannot safely export a bare StatusCode.
        if not isinstance(status, trace.Status):
            raise TypeError("Status required")
        return original_set_status(span, status)

    monkeypatch.setattr(sdk.Span, "set_status", old_set_status)
    if outcome == "adoption_error":
        span = spans.tracer.start_span("solwyn.call")

        def fail():
            raise ValueError()

        monkeypatch.setattr(span, "is_recording", fail)
        tracer = SimpleNamespace(start_span=MagicMock(spec=[], return_value=span))
        monkeypatch.setattr(spans.api, "get_tracer", lambda *args, **kwargs: tracer)
    owner = _otel._start_call_span()
    owner.end(error=outcome == "error")
    call = spans.calls()[0]
    assert call.status.status_code == (
        trace.StatusCode.OK if outcome == "success" else trace.StatusCode.ERROR
    )
    assert call.status.description is None
    assert not call.events


@pytest.mark.unit
@pytest.mark.parametrize("first", ["call", "run"])
def test_call_and_run_spans_share_versioned_tracer_and_reset_after_fork(spans, monkeypatch, first):
    monkeypatch.setenv("SOLWYN_OTEL_SPANS", "true")
    lookups = []

    def get_tracer(name, instrumenting_library_version=None):
        lookups.append((name, instrumenting_library_version))
        return spans.provider.get_tracer(name, instrumenting_library_version)

    monkeypatch.setattr(spans.api, "get_tracer", get_tracer)

    def start(kind):
        if kind == "run":
            _otel._start_run_span("run-tracer-cache", None).end()
        else:
            _otel._start_call_span().end()

    start(first)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(start, ["call", "run"] * 6))
    assert lookups == [("solwyn", solwyn.__version__)]
    finished = spans.exporter.get_finished_spans()
    assert len(finished) == 13
    assert {s.name for s in finished} == {"solwyn.call", "solwyn.run"}
    assert all(s.instrumentation_scope.name == "solwyn" for s in finished)
    assert all(s.instrumentation_scope.version == solwyn.__version__ for s in finished)
    _otel._reset_after_fork()
    start("run")
    start("call")
    assert lookups == [("solwyn", solwyn.__version__), ("solwyn", solwyn.__version__)]


@pytest.mark.unit
def test_call_tracer_resolution_can_reenter_without_recursion(spans, monkeypatch):
    nested = []

    def get_tracer(*args, **kwargs):
        nested.append(_otel._start_call_span())
        nested.append(_otel._start_run_span("run-tracer-reentry", None))
        return spans.tracer

    monkeypatch.setattr(spans.api, "get_tracer", get_tracer)
    monkeypatch.setenv("SOLWYN_OTEL_SPANS", "true")
    _otel._start_call_span().end()
    assert nested == [_otel._INACTIVE_CALL_SPAN, _otel._INACTIVE_RUN_SPAN]
    assert len(spans.calls()) == 1
    assert all(s.name != "solwyn.run" for s in spans.exporter.get_finished_spans())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("run_scope", [False, True])
async def test_native_anthropic_span_is_a_child(spans, async_mode, run_scope):
    anthropic = pytest.importorskip("anthropic", minversion="1.14.0")
    httpx2 = pytest.importorskip("httpx2")
    response_body = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "offline"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=response_body))
    http = (httpx2.AsyncClient if async_mode else httpx2.Client)(transport=transport)
    raw = (anthropic.AsyncAnthropic if async_mode else anthropic.Anthropic)(
        api_key="offline",
        http_client=http,
        open_telemetry={"tracer_provider": spans.provider},
    )
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, otel_spans=True)
    try:
        with (
            spans.tracer.start_as_current_span("caller") as caller,
            solwyn.run("PRIVATE run") if run_scope else nullcontext() as run_id,
        ):
            active_span = trace.get_current_span()
            result = wrapper.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4,
                messages=[{"role": "user", "content": "offline"}],
            )
            if async_mode:
                result = await result
            assert result.usage.input_tokens == 3
            assert trace.get_current_span() is active_span
    finally:
        await _close(wrapper, async_mode)
    call = spans.calls()[0]
    native = [
        s for s in spans.exporter.get_finished_spans() if s.name == "anthropic.messages.create"
    ]
    assert len(native) == 1
    assert native[0].parent.span_id == call.context.span_id
    assert call.parent.span_id == active_span.get_span_context().span_id
    if run_scope:
        run = next(s for s in spans.exporter.get_finished_spans() if s.name == "solwyn.run")
        assert run.parent.span_id == caller.get_span_context().span_id
        assert dict(run.attributes) == {"solwyn.agent_run_id": run_id}
        assert spans.ends[run.context.span_id] == 1
    assert plane.ingested[0].span_id == trace.format_span_id(call.context.span_id)
    assert not call.events
    assert set(call.attributes) == (
        _IDENTITY | _BUDGET | _ATTEMPT | ({"solwyn.agent_run_id"} if run_scope else set())
    )
