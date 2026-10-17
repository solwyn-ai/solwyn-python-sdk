"""Call-start trace identity survives dispatch, failover and deferred settlement."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_PROJECT_ID

import solwyn
from solwyn._base import MediaSurfaceSpec
from solwyn._types import BudgetMode, CallStatus
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.testing import FakeControlPlane

_CALL_TRACE = ("0" * 29 + "abc", "0" * 14 + "de", True)
_OTHER_TRACE = ("f" * 32, "f" * 16, False)
_current_trace: ContextVar[tuple[str, str, bool] | None] = ContextVar("test-trace", default=None)


def _raw_client(async_mode: bool) -> MagicMock:
    cls = type(
        "AsyncOpenAI" if async_mode else "OpenAI",
        (),
        {
            "__module__": "openai._client",
            "chat": None,
            "embeddings": None,
            "with_options": None,
            "close": None,
        },
    )
    raw = cls()
    raw.with_options = MagicMock(spec=[], return_value=raw)
    raw.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=AsyncMock() if async_mode else MagicMock(spec=[]),
        )
    )
    raw.embeddings = SimpleNamespace(create=AsyncMock() if async_mode else MagicMock(spec=[]))
    raw.close = AsyncMock() if async_mode else MagicMock(spec=[])
    return raw


def _budget(*, denied: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        allowed=not denied,
        reservation_id="res_trace" if not denied else None,
        project_id=VALID_PROJECT_ID,
        mode=BudgetMode.HARD_DENY,
        budget_limit=10.0,
        current_usage=10.0 if denied else 0.0,
        remaining_budget=0.0 if denied else 10.0,
        price_hints=None,
        failover_tuning_allowed=None,
        deny_source="server" if denied else None,
        deny_reason="budget_exceeded" if denied else None,
        denied_by_period="monthly" if denied else None,
    )


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=4,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
    )


async def _chunks():
    yield _response()


def _broken_chunks():
    yield _response()
    raise ValueError("stream failure")


async def _async_broken_chunks():
    for chunk in _broken_chunks():
        yield chunk


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "error",
        "ambiguous",
        "denial",
        "run_stop",
        "failover",
        "stream",
        "abandon",
        "media",
        "media_error",
        "media_denial",
        "media_run_stop",
        "media_post_stop",
        "stream_error",
        "stream_wrap_error",
        "interrupt",
        "failover_ambiguous",
        "post_stop",
    ],
)
async def test_every_outcome_uses_one_call_start_snapshot(async_mode: bool, outcome: str) -> None:
    from solwyn._otel import _TraceSnapshot

    raw = _raw_client(async_mode)
    fallback = _raw_client(async_mode)
    wrapper_type = AsyncSolwyn if async_mode else Solwyn
    plane = FakeControlPlane(project_id=VALID_PROJECT_ID)
    events = []
    with patch("solwyn.reporter.MetadataReporter._flush_loop", return_value=None):
        wrapper = wrapper_type(
            raw,
            provider="openai",
            api_key=plane.api_key,
            api_url=plane.api_url,
            control_plane_transport=plane.transport,
            fallback=[(fallback, "gpt-5.5")] if outcome.startswith("failover") else [],
            failover_idempotency="always" if outcome == "failover_ambiguous" else "safe",
        )
    if not async_mode:
        wrapper._solwyn_reporter._shutdown.set()
        wrapper._solwyn_reporter._thread.join(timeout=2)

    def capture():
        value = _current_trace.get()
        return _TraceSnapshot(*value) if value is not None else None

    def dispatch(*args, **kwargs):
        _current_trace.set(_OTHER_TRACE)
        if outcome in {"error", "media_error"}:
            raise ValueError("provider failure")
        if outcome in {"ambiguous", "failover_ambiguous"}:
            raise httpx.ReadTimeout("provider timeout")
        if outcome == "interrupt":
            raise asyncio.CancelledError() if async_mode else KeyboardInterrupt()
        if outcome == "failover":
            raise httpx.ConnectError("provider unavailable")
        if outcome == "stream_error":
            return _async_broken_chunks() if async_mode else _broken_chunks()
        if outcome in {"stream", "abandon", "stream_wrap_error"}:
            return _chunks() if async_mode else iter([_response()])
        return _response()

    raw.chat.completions.create.side_effect = dispatch
    raw.embeddings.create.side_effect = dispatch
    fallback.chat.completions.create.return_value = _response()
    token = _current_trace.set(_CALL_TRACE)

    def check(*args, **kwargs):
        if outcome in {"post_stop", "media_post_stop"}:
            from solwyn._run_control import mark_terminated

            mark_terminated(kwargs["agent_run_id"], reason="manual_kill", source="server")
        return _budget(denied=outcome in {"denial", "media_denial"})

    try:
        with (
            patch("solwyn._otel.capture_trace", side_effect=capture) as captured,
            patch.object(
                wrapper._solwyn_budget,
                "check_budget",
                new=AsyncMock(side_effect=check)
                if async_mode
                else MagicMock(spec=[], side_effect=check),
            ),
            patch.object(wrapper._solwyn_reporter, "report", side_effect=events.append),
            patch.object(
                wrapper._solwyn_reporter,
                "report_settlement",
                side_effect=lambda confirm, event: events.append(event),
            ),
            solwyn.run("trace-population") as run_id,
            patch.object(
                wrapper._solwyn_runtimes[0].adapter,
                "create_stream_accumulator",
                side_effect=RuntimeError("wrapper failure"),
            )
            if outcome == "stream_wrap_error"
            else nullcontext(),
        ):
            if outcome in {"run_stop", "media_run_stop"}:
                from solwyn._run_control import mark_terminated

                mark_terminated(run_id, reason="velocity:repeat_size", source="local_velocity")
            if outcome.startswith("media"):
                spec = MediaSurfaceSpec(
                    surface="embeddings",
                    modality="embedding",
                    extract_usage=lambda response: None,
                    measure_request=lambda kwargs: None,
                )
                result = wrapper._media_call(spec, model="text-embedding-3-small", input="x")
            else:
                result = wrapper.chat.completions.create(
                    model="gpt-5.5",
                    messages=[],
                    stream=outcome.startswith("stream") or outcome == "abandon",
                )
            if async_mode:
                result = await result
            if outcome.startswith("stream") or outcome == "abandon":
                if async_mode:
                    if outcome == "abandon":
                        await result.aclose()
                    else:
                        async for _ in result:
                            pass
                else:
                    # The consumer has no copy of the caller's contextvars.
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        executor.submit(
                            result.close if outcome == "abandon" else list,
                            *(() if outcome == "abandon" else (result,)),
                        ).result()
    except (Exception, KeyboardInterrupt, asyncio.CancelledError):
        if outcome not in {
            "error",
            "media_error",
            "ambiguous",
            "denial",
            "run_stop",
            "stream_error",
            "stream_wrap_error",
            "interrupt",
            "post_stop",
            "media_denial",
            "media_run_stop",
            "media_post_stop",
        }:
            raise
    finally:
        _current_trace.reset(token)
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    assert captured.call_count == 1
    assert len(events) == (2 if outcome.startswith("failover") else 1)
    for event in events:
        assert (event.trace_id, event.span_id, event.trace_sampled) == _CALL_TRACE
    if outcome in {"ambiguous", "stream_error", "stream_wrap_error", "interrupt"}:
        assert events[0].possibly_succeeded is True
    if outcome in {
        "denial",
        "run_stop",
        "post_stop",
        "media_denial",
        "media_run_stop",
        "media_post_stop",
    }:
        assert events[0].status == CallStatus.BUDGET_DENIED


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("surface", ["chat", "stream", "media"])
@pytest.mark.parametrize(
    "snapshot",
    [
        ("1" * 33, "2" * 16, True),
        ("1" * 32, "0" * 16, True),
        ("1" * 32, "2" * 16, "true"),
    ],
    ids=["malformed-trace", "malformed-span", "malformed-sampled"],
)
async def test_malformed_snapshot_preserves_response_and_settlement(
    async_mode: bool, surface: str, snapshot: tuple[object, object, object]
) -> None:
    from solwyn._otel import _TraceSnapshot

    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    response = _response()
    raw.chat.completions.create.return_value = (
        (_chunks() if async_mode else iter([response])) if surface == "stream" else response
    )
    raw.embeddings.create.return_value = response
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw)
    try:
        with patch("solwyn._otel.capture_trace", return_value=_TraceSnapshot(*snapshot)):
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
                    assert len([chunk async for chunk in result]) == 1
                else:
                    assert list(result) == [response]
            else:
                assert result is response
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    assert len(plane.ingested) == len(plane.confirms) == 1
    assert {"trace_id", "span_id", "trace_sampled"}.isdisjoint(plane.ingested[0].model_dump())


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("media", [False, True])
async def test_disabled_correlation_never_captures(async_mode: bool, media: bool) -> None:
    plane = FakeControlPlane()
    raw = _raw_client(async_mode)
    raw.chat.completions.create.return_value = _response()
    raw.embeddings.create.return_value = _response()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, trace_correlation=False)
    try:
        with patch(
            "solwyn._otel.capture_trace", side_effect=AssertionError("capture disabled")
        ) as capture:
            if media:
                result = wrapper._media_call(
                    MediaSurfaceSpec(
                        surface="embeddings",
                        modality="embedding",
                        extract_usage=lambda response: None,
                        measure_request=lambda kwargs: None,
                    ),
                    model="text-embedding-3-small",
                    input="x",
                )
            else:
                result = wrapper.chat.completions.create(model="gpt-5.5", messages=[])
            if async_mode:
                await result
            capture.assert_not_called()
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    assert len(plane.ingested) == 1
    assert plane.ingested[0].trace_id is plane.ingested[0].span_id is None
    assert plane.ingested[0].trace_sampled is None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["entry", "wrap", "consume"])
async def test_deferred_responses_stream_failures_keep_call_trace(
    async_mode: bool, failure: str
) -> None:
    from solwyn._otel import _TraceSnapshot

    raw = _raw_client(async_mode)
    if async_mode:
        manager = MagicMock(spec=["__aenter__", "__aexit__"])
        manager.__aenter__ = AsyncMock(return_value=_async_broken_chunks())
        manager.__aexit__ = AsyncMock(return_value=False)
        enter = manager.__aenter__
    else:
        manager = MagicMock(spec=["__enter__", "__exit__"])
        manager.__enter__.return_value = _broken_chunks()
        manager.__exit__.return_value = False
        enter = manager.__enter__
    if failure == "entry":
        enter.side_effect = httpx.ReadTimeout("stream entry")
    raw.responses = SimpleNamespace(stream=MagicMock(spec=[], return_value=manager))
    plane = FakeControlPlane()
    wrapper = (plane.wrap_async if async_mode else plane.wrap)(raw, provider="openai")
    try:
        with patch(
            "solwyn._otel.capture_trace", return_value=_TraceSnapshot(*_CALL_TRACE)
        ) as capture:
            result = wrapper._intercepted_call(
                _surface="responses",
                _responses_leaf="stream",
                _force_stream=True,
                model="gpt-5.5",
                input=[],
            )
            if async_mode:
                result = await result
        capture.assert_called_once()
        # Manager entry and reconciliation happen after call capture.
        with (
            patch("solwyn._otel.capture_trace", side_effect=AssertionError("late capture")),
            patch.object(
                wrapper._solwyn_runtimes[0].adapter,
                "create_stream_accumulator",
                side_effect=RuntimeError("wrap failure"),
            )
            if failure == "wrap"
            else nullcontext(),
            pytest.raises((ValueError, RuntimeError, httpx.ReadTimeout)),
        ):
            if async_mode:
                async with result as stream:
                    async for _ in stream:
                        pass
            else:
                with result as stream:
                    list(stream)
    finally:
        if async_mode:
            await wrapper.close()
        else:
            wrapper.close()
    assert len(plane.ingested) == 1
    event = plane.ingested[0]
    assert (event.trace_id, event.span_id, event.trace_sampled) == _CALL_TRACE
    assert event.status is CallStatus.ERROR
    assert event.possibly_succeeded is True
