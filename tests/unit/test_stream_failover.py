"""Streaming failover + idempotency hardening.

Three mechanisms under test:

  1. First-chunk MATERIALIZATION. ``_materialize_stream`` /
     ``_materialize_stream_async`` pull the first chunk eagerly so a Google lazy
     generator's establishment error surfaces INSIDE the candidate-walk except
     (failover-eligible) exactly like OpenAI/Anthropic's eager raise_for_status.
     No double-emit: the materialized first chunk is chained back so the wrapper
     observes + yields it exactly once.

  2. Cross-provider TEXT stream normalization. The fail-loud
     is lifted for plain-text cross-provider streaming; it stays only for the
     TOOL case (``cross_provider_tool_stream``). The wrapper accumulates the RAW
     served chunk but yields caller-dialect chunks via a chunk_translator.

  3. First-byte rule is structural: once the wrapper is returned, a mid-stream
     error goes to on_error (breaker health failure, NO failover) and the
     ORIGINAL exception surfaces.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._types import CallStatus, CircuitState
from solwyn.client import (
    AsyncSolwyn,
    Solwyn,
    _materialize_stream,
    _materialize_stream_async,
)
from solwyn.exceptions import UntranslatableRequestError
from solwyn.providers import _translation

# ── establishment-error helpers ──────────────────────────────────────────


class _Status(Exception):
    """Duck-typed transport error carrying an HTTP ``status_code`` (429 = FAILOVER)."""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def _google_lazy_stream_raising(exc: Exception):
    """A Google-style lazy generator: raises on FIRST next() (establishment)."""

    def _gen():
        raise exc
        yield  # pragma: no cover - makes this a generator

    return _gen()


def _google_lazy_stream(chunks: list[object]):
    """A Google-style lazy generator yielding ``chunks`` (no eager network)."""

    def _gen():
        yield from chunks

    return _gen()


# ── _materialize_stream (sync) ───────────────────────────────────────────


@pytest.mark.unit
class TestMaterializeStreamSync:
    def test_propagates_establishment_error_on_first_pull(self) -> None:
        # A lazy generator that raises on first next() must PROPAGATE that error
        # out of _materialize_stream (so the candidate walk can fail over).
        stream = _google_lazy_stream_raising(_Status(429))
        with pytest.raises(_Status):
            _materialize_stream(stream)

    def test_no_double_emit_first_chunk_yielded_once(self) -> None:
        chunks = [SimpleNamespace(i=0), SimpleNamespace(i=1), SimpleNamespace(i=2)]
        materialized = _materialize_stream(_google_lazy_stream(chunks))
        assert list(materialized) == chunks

    def test_empty_stream_returns_empty_iterator(self) -> None:
        materialized = _materialize_stream(_google_lazy_stream([]))
        assert list(materialized) == []


@pytest.mark.unit
class TestMaterializeStreamAsync:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_propagates_establishment_error_on_first_pull(self) -> None:
        async def _raising():
            raise _Status(429)
            yield  # pragma: no cover

        # The eager anext runs INSIDE the await, so the establishment error
        # surfaces at the await site (the dispatch try).
        with pytest.raises(_Status):
            await _materialize_stream_async(_raising())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_double_emit_first_chunk_yielded_once(self) -> None:
        chunks = [SimpleNamespace(i=0), SimpleNamespace(i=1)]

        async def _gen():
            for c in chunks:
                yield c

        materialized = await _materialize_stream_async(_gen())
        collected = [c async for c in materialized]
        assert collected == chunks

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        async def _gen():
            return
            yield  # pragma: no cover

        materialized = await _materialize_stream_async(_gen())
        collected = [c async for c in materialized]
        assert collected == []


# ── client fakes ─────────────────────────────────────────────────────────


def _openai_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _anthropic_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client
    return client


def _google_client() -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "google.genai.client"
    client.__class__.__name__ = "Client"
    client.with_options.return_value = client
    return client


def _allow_budget(reservation_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    solwyn._solwyn_reporter.report = MagicMock()
    return solwyn


def _make_async_solwyn(client: object, **overrides: object) -> AsyncSolwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter.report = MagicMock()
    return solwyn


def _close(solwyn: Solwyn) -> None:
    solwyn._solwyn_reporter._http.close()
    solwyn._solwyn_budget._http.close()


async def _aclose(solwyn: AsyncSolwyn) -> None:
    await solwyn._solwyn_reporter._http.aclose()
    await solwyn._solwyn_budget._http.aclose()


def _async_iter(chunks: list[Any]) -> AsyncIterator[Any]:
    """Wrap a list of chunks as a NON-lazy async iterator (already established)."""

    async def _gen() -> AsyncIterator[Any]:
        for c in chunks:
            yield c

    return _gen()


def _google_lazy_async_stream_raising(exc: Exception) -> AsyncIterator[Any]:
    """A Google-style async lazy generator: raises on FIRST anext (establishment)."""

    async def _gen() -> AsyncIterator[Any]:
        raise exc
        yield  # pragma: no cover - makes this an async generator

    return _gen()


# ── OpenAI stream chunk builders (served dialect) ────────────────────────


def _openai_text_chunk(text: str | None, finish: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(role=None, content=text, tool_calls=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], model="gpt-5.5", usage=None)


def _anthropic_text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _anthropic_message_start(input_tokens: int) -> SimpleNamespace:
    # The accumulator reads input usage from message_start.message.usage; the
    # translator maps this structural event to ZERO caller chunks.
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens, cache_read_input_tokens=0)
        ),
    )


def _anthropic_message_delta(output_tokens: int) -> SimpleNamespace:
    # Output usage + the terminal stop_reason. Translates to ONE caller finish
    # chunk; the accumulator reads output_tokens off it.
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="end_turn"),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


def _google_text_chunk(text: str | None, finish: str | None = None) -> SimpleNamespace:
    # A Google streaming chunk: candidates[0].content.parts[*].text plus an
    # optional finish_reason. usage_metadata is absent on text chunks here.
    part = SimpleNamespace(text=text, function_call=None, inline_data=None, file_data=None)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part] if text is not None else []),
        finish_reason=finish,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


# ── Google establishment-error -> failover (materialization) ─────────────


@pytest.mark.unit
class TestGoogleStreamingEstablishmentFailover:
    def test_google_lazy_establishment_error_fails_over(self) -> None:
        # Google PRIMARY: generate_content_stream returns a lazy generator that
        # raises on first next(). Materialization surfaces it in the candidate
        # walk -> failover to OpenAI fallback (same dialect? no — cross-provider,
        # but text, so it must NOT fail loud). Use OpenAI primary -> Google
        # fallback to keep the served dialect for the wrapper simple instead.
        google = _google_client()
        google.models.generate_content_stream.return_value = _google_lazy_stream_raising(
            _Status(429)
        )
        openai = _openai_client()
        openai.chat.completions.create.return_value = iter(
            [_openai_text_chunk("hello"), _openai_text_chunk(None, finish="stop")]
        )

        solwyn = _make_solwyn(
            google,
            model="gemini-3.5-flash",
            fallback=[(openai, "gpt-5.5")],
        )

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.models.generate_content_stream(**request)
            chunks = list(stream)

        # Google failed to establish; OpenAI fallback served. Chunks are in the
        # caller (Google) dialect after cross-provider translation.
        assert len(chunks) >= 1
        google.models.generate_content_stream.assert_called_once()
        openai.chat.completions.create.assert_called_once()
        _close(solwyn)


# ── cross-provider streaming INTO / OUT OF Google uses the right method ───
# Fix [A]: the served Google hop's stream-method choice must be driven by the
# dispatch-level is_streaming boolean, NOT by the original _force_stream flag /
# canonical.stream. When the caller streamed via OpenAI/Anthropic stream=True and
# fails over to Google, _force_stream is False but is_streaming is True — Google
# must still STREAM (generate_content_stream), never generate_content.


@pytest.mark.unit
class TestCrossProviderStreamingIntoGoogle:
    def test_openai_429_failover_into_google_actually_streams(self) -> None:
        # OpenAI PRIMARY streamed via stream=True (so _force_stream is False);
        # 429 pre-send -> failover to Google. The served Google hop must call
        # generate_content_stream (is_streaming True), NOT generate_content.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        google = _google_client()
        google.models.generate_content_stream.return_value = iter(
            [_google_text_chunk("hel"), _google_text_chunk("lo", finish="STOP")]
        )

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(google, "gemini-3.5-flash")],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        # Google STREAMED (not generate_content); non-streaming must never be called.
        google.models.generate_content_stream.assert_called_once()
        google.models.generate_content.assert_not_called()
        # No leaked `stream` kwarg reached Google's streaming method.
        assert "stream" not in google.models.generate_content_stream.call_args.kwargs
        # Caller wrote OpenAI dialect -> wrapper yields OpenAI-shaped chunks.
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hello"
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_openai_429_failover_into_google_actually_streams(self) -> None:
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        google = _google_client()
        google.models.generate_content_stream = AsyncMock(
            return_value=_async_iter(
                [_google_text_chunk("hel"), _google_text_chunk("lo", finish="STOP")]
            )
        )

        solwyn = _make_async_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(google, "gemini-3.5-flash")],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            chunks = [c async for c in stream]

        google.models.generate_content_stream.assert_awaited_once()
        google.models.generate_content.assert_not_called()
        assert "stream" not in google.models.generate_content_stream.call_args.kwargs
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hello"
        await _aclose(solwyn)


@pytest.mark.unit
class TestGooglePrimaryStreamingFailoverIntoOpenAI:
    def test_google_primary_stream_429_failover_into_openai_streams(self) -> None:
        # Google PRIMARY streamed via generate_content_stream (_force_stream True);
        # eager 429 -> failover to OpenAI. OpenAI must stream (stream=True in
        # call_kwargs) and the caller (Google dialect) sees Google-shaped chunks.
        google = _google_client()
        google.models.generate_content_stream.side_effect = _Status(429)
        openai = _openai_client()
        openai.chat.completions.create.return_value = iter(
            [_openai_text_chunk("hel"), _openai_text_chunk("lo", finish="stop")]
        )

        solwyn = _make_solwyn(
            google,
            model="gemini-3.5-flash",
            fallback=[(openai, "gpt-5.5")],
        )

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.models.generate_content_stream(**request)
            chunks = list(stream)

        google.models.generate_content_stream.assert_called_once()
        openai.chat.completions.create.assert_called_once()
        # OpenAI served as a STREAM.
        assert openai.chat.completions.create.call_args.kwargs.get("stream") is True
        # Caller wrote Google dialect -> wrapper yields Google-shaped chunks.
        texts = [
            part.text
            for c in chunks
            for cand in (c.candidates or [])
            for part in (cand.content.parts or [])
            if getattr(part, "text", None)
        ]
        assert "".join(texts) == "hello"
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_google_primary_stream_429_failover_into_openai_streams(self) -> None:
        google = _google_client()
        google.models.generate_content_stream = AsyncMock(side_effect=_Status(429))
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(
            return_value=_async_iter(
                [_openai_text_chunk("hel"), _openai_text_chunk("lo", finish="stop")]
            )
        )

        solwyn = _make_async_solwyn(
            google,
            model="gemini-3.5-flash",
            fallback=[(openai, "gpt-5.5")],
        )

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.models.generate_content_stream(**request)
            chunks = [c async for c in stream]

        google.models.generate_content_stream.assert_awaited_once()
        openai.chat.completions.create.assert_awaited_once()
        assert openai.chat.completions.create.call_args.kwargs.get("stream") is True
        texts = [
            part.text
            for c in chunks
            for cand in (c.candidates or [])
            for part in (cand.content.parts or [])
            if getattr(part, "text", None)
        ]
        assert "".join(texts) == "hello"
        await _aclose(solwyn)


# ── cross-provider TEXT stream: fail-loud LIFTED ─────────────────────────


@pytest.mark.unit
class TestCrossProviderTextStreamingNormalizes:
    def test_openai_to_anthropic_text_stream_yields_openai_dialect(self) -> None:
        # OpenAI primary 429s; Anthropic fallback serves a text stream. The
        # caller wrote OpenAI dialect, so wrapper-yielded chunks must be OpenAI
        # chunks (choices[0].delta.content), translated from Anthropic chunks.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter(
            [_anthropic_text_chunk("hel"), _anthropic_text_chunk("lo")]
        )

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        anthropic.messages.create.assert_called_once()
        # Caller-dialect (OpenAI) chunks: each exposes choices[0].delta.content.
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hello"
        _close(solwyn)

    def test_cross_provider_tool_stream_still_fails_loud(self) -> None:
        # A cross-provider streaming hop carrying TOOLS cannot be normalized:
        # must fail loud upfront with cross_provider_tool_stream.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("x")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError) as exc_info,
        ):
            solwyn.chat.completions.create(**request)

        assert exc_info.value.feature == "cross_provider_tool_stream"
        # The foreign-dialect stream was never served.
        anthropic.messages.create.assert_not_called()
        _close(solwyn)


# ── same-dialect streaming pays ZERO translation ─────────────────────────


@pytest.mark.unit
class TestSameDialectStreamingNoTranslation:
    def test_same_provider_model_swap_stream_is_passthrough(self) -> None:
        client = _openai_client()
        client.chat.completions.create.side_effect = [
            _Status(429),
            iter([_openai_text_chunk("a"), _openai_text_chunk(None, finish="stop")]),
        ]

        solwyn = _make_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            patch.object(
                _translation,
                "translate_stream_chunk",
                wraps=_translation.translate_stream_chunk,
            ) as translate,
        ):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        # Same-dialect: the chunk translator is NEVER invoked (zero overhead),
        # and raw chunks pass straight through.
        translate.assert_not_called()
        assert chunks[0].choices[0].delta.content == "a"
        _close(solwyn)


# ── first-byte rule: mid-stream error never fails over ───────────────────


@pytest.mark.unit
class TestMidStreamErrorNeverFailsOver:
    def test_mid_stream_error_surfaces_original_no_failover(self) -> None:
        openai = _openai_client()

        def _exploding_stream():
            yield _openai_text_chunk("ok")
            raise ConnectionError("mid-stream reset")

        openai.chat.completions.create.return_value = _exploding_stream()
        anthropic = _anthropic_client()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            with pytest.raises(ConnectionError, match="mid-stream reset"):
                list(stream)

        # The fallback was NEVER attempted — mid-stream errors don't fail over.
        anthropic.messages.create.assert_not_called()
        errors = [
            c.args[0]
            for c in solwyn._solwyn_reporter.report.call_args_list
            if c.args[0].status is CallStatus.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        _close(solwyn)

    def test_openai_first_chunk_error_after_establishment_no_failover(self) -> None:
        # Fix [B]: OpenAI .create(stream=True) establishes eagerly, so a read
        # error on the FIRST chunk happens AFTER a successful connect. We must NOT
        # pre-pull (materialize) OpenAI streams — doing so would misclassify this
        # post-connect first-chunk error as pre-send and fail over (double-spend
        # risk). Instead the error surfaces via the wrapper's on_error when the
        # caller iterates; the ORIGINAL exception reaches the caller, no failover.
        openai = _openai_client()

        def _first_chunk_raises():
            raise ConnectionError("first-chunk read reset")
            yield  # pragma: no cover - makes this a generator

        openai.chat.completions.create.return_value = _first_chunk_raises()
        anthropic = _anthropic_client()

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            # The wrapper IS returned (establishment succeeded); the error only
            # surfaces on iteration — proving OpenAI was NOT materialized.
            stream = solwyn.chat.completions.create(**request)
            with pytest.raises(ConnectionError, match="first-chunk read reset"):
                list(stream)

        # No failover; the served-provider breaker recorded a health failure via
        # on_error (NOT a pre-send candidate-walk failure -> no double dispatch).
        anthropic.messages.create.assert_not_called()
        openai.chat.completions.create.assert_called_once()
        assert openai_cb.failure_count == 1
        errors = [
            c.args[0]
            for c in solwyn._solwyn_reporter.report.call_args_list
            if c.args[0].status is CallStatus.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_mid_stream_error_surfaces_original_no_failover(self) -> None:
        # Async mirror: chunk 1 yields, then the served stream raises mid-stream.
        # The wrapper fires on_error (breaker health failure) and re-raises the
        # ORIGINAL exception — NO failover (the first byte already left).
        openai = _openai_client()

        async def _exploding_stream() -> AsyncIterator[Any]:
            yield _openai_text_chunk("ok")
            raise ConnectionError("mid-stream reset")

        openai.chat.completions.create = AsyncMock(return_value=_exploding_stream())
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock()

        solwyn = _make_async_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            with pytest.raises(ConnectionError, match="mid-stream reset"):
                _ = [c async for c in stream]

        # The fallback was NEVER attempted; the served-provider breaker recorded a
        # health failure via on_error.
        anthropic.messages.create.assert_not_awaited()
        assert openai_cb.failure_count == 1
        errors = [
            c.args[0]
            for c in solwyn._solwyn_reporter.report.call_args_list
            if c.args[0].status is CallStatus.ERROR
        ]
        assert len(errors) == 1
        assert errors[0].possibly_succeeded is True
        await _aclose(solwyn)


# ── eager (OpenAI/Anthropic) establishment-error failover ────────────────


@pytest.mark.unit
class TestEagerEstablishmentFailover:
    def test_openai_primary_eager_establishment_error_fails_over(self) -> None:
        # OpenAI dispatch raises eagerly at establishment (429) BEFORE any chunk.
        # The candidate walk fails over to a same-provider gpt-5.4-mini streaming
        # swap; the served stream is the fallback's (no chunk left the primary).
        client = _openai_client()
        client.chat.completions.create.side_effect = [
            _Status(429),
            iter([_openai_text_chunk("served"), _openai_text_chunk(None, finish="stop")]),
        ]

        solwyn = _make_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        # The fallback model served; the stream is its chunks.
        assert client.chat.completions.create.call_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-5.4-mini"
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "served"
        _close(solwyn)

    def test_anthropic_primary_eager_establishment_error_fails_over(self) -> None:
        # Anthropic primary 429s eagerly; same-provider claude swap serves the
        # stream. No chunk left the primary before failover.
        client = _anthropic_client()
        client.messages.create.side_effect = [
            _Status(429),
            iter([_anthropic_text_chunk("ok"), _anthropic_message_delta(5)]),
        ]

        solwyn = _make_solwyn(
            client,
            model="claude-sonnet-5",
            fallback=[(client, "claude-haiku-4-5", {"max_tokens": 256})],
            default_params={"max_tokens": 256},
        )

        request = {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        assert client.messages.create.call_count == 2
        assert client.messages.create.call_args_list[1].kwargs["model"] == "claude-haiku-4-5"
        # Same-dialect: raw Anthropic chunks pass straight through.
        assert chunks[0].delta.text == "ok"
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_openai_eager_establishment_error_fails_over(self) -> None:
        # Async mirror of the eager-establishment failover.
        client = _openai_client()
        served_stream = _async_iter(
            [_openai_text_chunk("served"), _openai_text_chunk(None, finish="stop")]
        )
        client.chat.completions.create = AsyncMock(side_effect=[_Status(429), served_stream])

        solwyn = _make_async_solwyn(client, model="gpt-5.5", fallback=[(client, "gpt-5.4-mini")])

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            chunks = [c async for c in stream]

        assert client.chat.completions.create.await_count == 2
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-5.4-mini"
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "served"
        await _aclose(solwyn)


# ── Google async lazy establishment-error failover (materialization) ──────


@pytest.mark.unit
class TestAsyncGoogleStreamingEstablishmentFailover:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_google_lazy_establishment_error_fails_over(self) -> None:
        # Async mirror of the sync Google case: the primary lazy async generator
        # raises on first anext (establishment). Materialization (await anext
        # inside dispatch) surfaces it in the candidate walk -> failover to the
        # OpenAI fallback, whose stream is served in the caller's Google dialect.
        google = _google_client()
        google.models.generate_content_stream = AsyncMock(
            return_value=_google_lazy_async_stream_raising(_Status(429))
        )
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(
            return_value=_async_iter(
                [_openai_text_chunk("hello"), _openai_text_chunk(None, finish="stop")]
            )
        )

        solwyn = _make_async_solwyn(
            google,
            model="gemini-3.5-flash",
            fallback=[(openai, "gpt-5.5")],
        )

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.models.generate_content_stream(**request)
            chunks = [c async for c in stream]

        # Google failed to establish; OpenAI fallback served (cross-provider,
        # text -> normalized to the caller's Google dialect).
        assert len(chunks) >= 1
        google.models.generate_content_stream.assert_awaited_once()
        openai.chat.completions.create.assert_awaited_once()
        await _aclose(solwyn)


# ── materialized Google stream forwards close() to the ORIGINAL generator ─
# Fix [C]: the materialized wrapper's close()/aclose() must reach the REAL
# provider stream so abandoning a stream releases the connection (no leak).


class _ClosableGoogleStream:
    """A Google-style lazy generator with an observable close()."""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self.close_calls = 0

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.close_calls += 1


class _AclosableGoogleStream:
    """A Google-style async lazy generator with an observable aclose()."""

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks
        self.aclose_calls = 0

    async def __aiter__(self):
        for c in self._chunks:
            yield c

    async def aclose(self) -> None:
        self.aclose_calls += 1


@pytest.mark.unit
class TestMaterializedGoogleStreamForwardsClose:
    def test_materialize_close_forwards_to_original_generator(self) -> None:
        # _materialize_stream must return an object whose close() forwards to the
        # ORIGINAL provider stream's close() (not an itertools.chain that drops it).
        original = _ClosableGoogleStream([SimpleNamespace(i=0), SimpleNamespace(i=1)])
        materialized = _materialize_stream(original)
        # First chunk is buffered + replayed once, in order.
        first = next(iter(materialized))
        assert first.i == 0
        materialized.close()
        assert original.close_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_materialize_async_aclose_forwards_to_original_generator(self) -> None:
        original = _AclosableGoogleStream([SimpleNamespace(i=0), SimpleNamespace(i=1)])
        materialized = await _materialize_stream_async(original)
        ait = materialized.__aiter__()
        first = await ait.__anext__()
        assert first.i == 0
        await materialized.aclose()
        assert original.aclose_calls == 1

    def test_abandoning_materialized_google_stream_closes_original_once(self) -> None:
        # End-to-end: a Google-served stream the caller breaks out of and closes
        # must (a) call the ORIGINAL generator's close() exactly once and (b)
        # settle on_complete exactly once (no double settle).
        google = _google_client()
        original = _ClosableGoogleStream(
            [_google_text_chunk("a"), _google_text_chunk("b"), _google_text_chunk("c")]
        )
        google.models.generate_content_stream.return_value = original

        solwyn = _make_solwyn(google, model="gemini-3.5-flash")
        confirms: list[Any] = []
        solwyn._solwyn_reporter.report_settlement = lambda req, event: confirms.append(req)

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                return_value=_allow_budget(reservation_id="resv_g"),
            ),
            solwyn.models.generate_content_stream(**request) as stream,
        ):
            for _chunk in stream:
                break  # abandon after the first chunk

        # The ORIGINAL provider generator was closed exactly once (connection
        # released) and settlement fired exactly once (single confirm).
        assert original.close_calls == 1
        assert len(confirms) == 1
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_abandoning_materialized_google_stream_aclose_once(self) -> None:
        google = _google_client()
        original = _AclosableGoogleStream(
            [_google_text_chunk("a"), _google_text_chunk("b"), _google_text_chunk("c")]
        )
        google.models.generate_content_stream = AsyncMock(return_value=original)

        solwyn = _make_async_solwyn(google, model="gemini-3.5-flash")
        confirms: list[Any] = []
        solwyn._solwyn_reporter.report_settlement = lambda req, event: confirms.append(req)

        request = {
            "model": "gemini-3.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": {"max_output_tokens": 256},
        }

        with patch.object(
            solwyn._solwyn_budget,
            "check_budget",
            new=AsyncMock(return_value=_allow_budget("resv_g")),
        ):
            stream = await solwyn.models.generate_content_stream(**request)
            async with stream:
                async for _chunk in stream:
                    break  # abandon after the first chunk

        assert original.aclose_calls == 1
        # Settlement rides report_settlement exactly once (the blocking
        # confirm_cost path no longer exists).
        assert len(confirms) == 1
        await _aclose(solwyn)


# ── no double-emit / no splice: 3 chunks observed once, in order ─────────


class _CountingAccumulator:
    """Wraps a real accumulator, counting observe() calls per identity."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.observed: list[Any] = []

    def observe(self, chunk: Any) -> None:
        self.observed.append(chunk)
        self._inner.observe(chunk)

    def finalize(self) -> Any:
        return self._inner.finalize()

    def get_service_tier(self) -> str | None:
        return self._inner.get_service_tier()


@pytest.mark.unit
class TestNoDoubleEmitThreeChunks:
    def test_three_chunk_stream_yields_each_once_in_order(self) -> None:
        # A 3-chunk stream (the materialized first chunk + two more) must be
        # yielded to the user exactly 3 times, in order, with the accumulator
        # observing each chunk exactly once (no splice, no re-observe).
        c0 = _openai_text_chunk("a")
        c1 = _openai_text_chunk("b")
        c2 = SimpleNamespace(
            choices=[],
            model="gpt-5.5",
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )
        client = _openai_client()
        client.chat.completions.create.return_value = iter([c0, c1, c2])

        solwyn = _make_solwyn(client, model="gpt-5.5")

        # Wrap the served runtime's accumulator so we can count observe() calls.
        rt = solwyn._solwyn_runtimes[0]
        real_factory = rt.adapter.create_stream_accumulator
        counting: list[_CountingAccumulator] = []

        def _factory(*, estimated_input_tokens: int = 0) -> _CountingAccumulator:
            acc = _CountingAccumulator(real_factory(estimated_input_tokens=estimated_input_tokens))
            counting.append(acc)
            return acc

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            patch.object(rt.adapter, "create_stream_accumulator", _factory),
        ):
            stream = solwyn.chat.completions.create(**request)
            yielded = list(stream)

        # Exactly the 3 served chunks, in order, each yielded once.
        assert yielded == [c0, c1, c2]
        # The accumulator observed each chunk exactly once (incl. the
        # materialized first chunk — no double-observe).
        assert counting[0].observed == [c0, c1, c2]
        _close(solwyn)


# ── cross-provider TEXT stream: usage settles served provider + metadata ─


@pytest.mark.unit
class TestCrossProviderStreamSettlement:
    def test_served_anthropic_usage_settles_and_metadata_attributed(self) -> None:
        # OpenAI primary 429s; Anthropic fallback streams text. The user iterates
        # OPENAI-dialect chunks, but the accumulator settles the SERVED (Anthropic)
        # usage and the success MetadataEvent is attributed to Anthropic with
        # is_provider_fallback True.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter(
            [
                _anthropic_message_start(input_tokens=31),
                _anthropic_text_chunk("hel"),
                _anthropic_text_chunk("lo"),
                _anthropic_message_delta(output_tokens=9),
            ]
        )

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list[Any] = []
        solwyn._solwyn_reporter.report = lambda e: events.append(e)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        # Caller sees OpenAI-dialect text deltas carrying the Anthropic text.
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hello"

        # The success MetadataEvent (fired on settle) is attributed to the SERVED
        # Anthropic provider with the served usage and is_provider_fallback True.
        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        ev = success[0]
        assert ev.provider.value == "anthropic"
        assert ev.is_provider_fallback is True
        assert ev.requested_provider.value == "openai"
        assert ev.input_tokens == 31
        assert ev.output_tokens == 9
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_served_anthropic_usage_settles_and_metadata_attributed(self) -> None:
        # Async mirror of the cross-provider settlement.
        openai = _openai_client()
        openai.chat.completions.create = AsyncMock(side_effect=_Status(429))
        anthropic = _anthropic_client()
        anthropic.messages.create = AsyncMock(
            return_value=_async_iter(
                [
                    _anthropic_message_start(input_tokens=31),
                    _anthropic_text_chunk("hel"),
                    _anthropic_text_chunk("lo"),
                    _anthropic_message_delta(output_tokens=9),
                ]
            )
        )

        solwyn = _make_async_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )
        events: list[Any] = []
        solwyn._solwyn_reporter.report = lambda e: events.append(e)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            chunks = [c async for c in stream]

        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "hello"

        success = [e for e in events if e.status.value == "success"]
        assert len(success) == 1
        ev = success[0]
        assert ev.provider.value == "anthropic"
        assert ev.is_provider_fallback is True
        assert ev.input_tokens == 31
        assert ev.output_tokens == 9
        await _aclose(solwyn)


# ── streaming idempotency matrix ─────────────────────────────────────────


@pytest.mark.unit
class TestStreamingIdempotencyMatrix:
    def test_safe_default_post_send_ambiguous_streaming_does_not_failover(self) -> None:
        # A post-send-ambiguous establishment failure (read timeout) on a
        # streaming hop must NOT cross providers under the default "safe" policy —
        # the ORIGINAL exception re-raises and the fallback is never served.
        class APITimeoutError(Exception):
            """Local stand-in classified POST_SEND_AMBIGUOUS by MRO name."""

        openai = _openai_client()
        original = APITimeoutError("read timed out")
        openai.chat.completions.create.side_effect = original
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("x")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(APITimeoutError) as exc_info,
        ):
            solwyn.chat.completions.create(**request)

        assert exc_info.value is original
        anthropic.messages.create.assert_not_called()
        _close(solwyn)

    def test_429_pre_send_streaming_does_failover(self) -> None:
        # A 429 is a provable pre-send rejection -> the streaming hop crosses to
        # Anthropic and serves a normalized text stream.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429)
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("served")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        anthropic.messages.create.assert_called_once()
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "served"
        _close(solwyn)

    def test_always_allows_ambiguous_streaming_failover(self) -> None:
        # failover_idempotency="always" lets a post-send-ambiguous streaming
        # failure cross providers.
        class APITimeoutError(Exception):
            """Local stand-in classified POST_SEND_AMBIGUOUS by MRO name."""

        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("served")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            failover_idempotency="always",
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            chunks = list(stream)

        anthropic.messages.create.assert_called_once()
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "served"
        _close(solwyn)

    def test_never_disables_cross_provider_streaming_failover(self) -> None:
        # failover_idempotency="never" forbids ANY cross-provider hop, even a 429.
        # The original error re-raises and the fallback is never served. BUT the
        # primary breaker STILL tracks health ("never = breaker still tracks
        # health, no cross-provider failover") — so it records a failure.
        openai = _openai_client()
        openai.chat.completions.create.side_effect = _Status(429, "rate limited")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("x")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
            failover_idempotency="never",
        )
        openai_cb = solwyn._get_circuit_breaker("openai")

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(_Status, match="rate limited"),
        ):
            solwyn.chat.completions.create(**request)

        anthropic.messages.create.assert_not_called()
        # The primary breaker tracked health: exactly one recorded failure.
        assert openai_cb.failure_count == 1
        _close(solwyn)

    def test_per_call_solwyn_idempotent_override_streaming_and_stripped(self) -> None:
        # Per-call solwyn_idempotent=True escalates "safe" to ambiguous-OK for
        # this streaming call, and the flag is stripped before dispatch on BOTH
        # hops (never forwarded to a provider SDK).
        class APITimeoutError(Exception):
            """Local stand-in classified POST_SEND_AMBIGUOUS by MRO name."""

        openai = _openai_client()
        openai.chat.completions.create.side_effect = APITimeoutError("read timed out")
        anthropic = _anthropic_client()
        anthropic.messages.create.return_value = iter([_anthropic_text_chunk("served")])

        solwyn = _make_solwyn(
            openai,
            model="gpt-5.5",
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 256})],
        )

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(solwyn_idempotent=True, **request)
            chunks = list(stream)

        anthropic.messages.create.assert_called_once()
        texts = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert "".join(texts) == "served"
        # solwyn_idempotent stripped before dispatch on BOTH hops.
        assert "solwyn_idempotent" not in openai.chat.completions.create.call_args.kwargs
        assert "solwyn_idempotent" not in anthropic.messages.create.call_args.kwargs
        _close(solwyn)


# ── abandoned-stream settlement ──────────────────────────────────────────


@pytest.mark.unit
class TestAbandonedStreamSettlement:
    def test_early_break_then_close_settles_once_via_report_settlement(self) -> None:
        # A caller that breaks early and closes the stream still settles via
        # close()/__exit__: on_complete fires ONCE with whatever usage was
        # observed, and settlement is fire-and-forget via reporter.report_settlement.
        client = _openai_client()
        # A long stream the caller abandons after the first chunk.
        client.chat.completions.create.return_value = iter(
            [_openai_text_chunk("a"), _openai_text_chunk("b"), _openai_text_chunk("c")]
        )

        solwyn = _make_solwyn(client, model="gpt-5.5")
        # A reservation makes on_complete build + fire-and-forget a confirm.
        confirms: list[Any] = []
        solwyn._solwyn_reporter.report_settlement = lambda req, event: confirms.append(req)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                return_value=_allow_budget(reservation_id="resv_1"),
            ),
            solwyn.chat.completions.create(**request) as stream,
        ):
            for _chunk in stream:
                break  # abandon after the first chunk

        # Settled exactly once: a single fire-and-forget confirm was enqueued for
        # the reservation (no duplicate confirm / double-spend).
        assert len(confirms) == 1
        assert confirms[0].reservation_id == "resv_1"
        _close(solwyn)

    def test_explicit_close_after_break_does_not_double_settle(self) -> None:
        # Breaking inside a `with` block, THEN calling close() again, must still
        # only settle once (close() is idempotent; no second confirm).
        client = _openai_client()
        client.chat.completions.create.return_value = iter(
            [_openai_text_chunk("a"), _openai_text_chunk("b")]
        )

        solwyn = _make_solwyn(client, model="gpt-5.5")
        confirms: list[Any] = []
        solwyn._solwyn_reporter.report_settlement = lambda req, event: confirms.append(req)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget,
            "check_budget",
            return_value=_allow_budget(reservation_id="resv_2"),
        ):
            stream = solwyn.chat.completions.create(**request)
            for _chunk in stream:
                break
            stream.close()
            stream.close()  # extra close() — must NOT settle again

        assert len(confirms) == 1
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_early_break_then_close_settles_once(self) -> None:
        # Async mirror: break early, then `async with`/close() settles once.
        client = _openai_client()
        client.chat.completions.create = AsyncMock(
            return_value=_async_iter(
                [_openai_text_chunk("a"), _openai_text_chunk("b"), _openai_text_chunk("c")]
            )
        )

        solwyn = _make_async_solwyn(client, model="gpt-5.5")
        confirms: list[Any] = []
        solwyn._solwyn_reporter.report_settlement = lambda req, event: confirms.append(req)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget,
            "check_budget",
            new=AsyncMock(return_value=_allow_budget("resv_3")),
        ):
            stream = await solwyn.chat.completions.create(**request)
            async with stream:
                async for _chunk in stream:
                    break  # abandon after the first chunk

        # Settled exactly once via report_settlement (the blocking confirm_cost
        # path no longer exists).
        assert len(confirms) == 1
        await _aclose(solwyn)


# ── fix [A]: a streaming success credits the served breaker EXACTLY ONCE ──
#
# On a streaming success the served breaker's record_success() must run ONLY
# in the stream wrapper's on_complete (when the stream settles), NOT also at
# the dispatch site. Otherwise a single streaming probe double-credits a
# HALF_OPEN breaker and closes it after one drain (success_threshold defaults
# to 2), defeating anti-flap recovery.


def _force_half_open(cb: Any) -> None:
    """Drive a breaker into HALF_OPEN with a clean probe slate (success_count==0)."""
    cb.state = CircuitState.HALF_OPEN
    cb.success_count = 0
    cb.failure_count = 0


@pytest.mark.unit
class TestStreamingSuccessSingleBreakerCredit:
    def test_halfopen_streaming_probe_credits_once_not_closed(self) -> None:
        # A served breaker in HALF_OPEN with success_threshold=2: ONE streaming
        # call drained to completion is a SINGLE successful probe. It must leave
        # the breaker HALF_OPEN with success_count==1 — NOT CLOSED. A second
        # successful probe is required to close it.
        client = _openai_client()
        client.chat.completions.create.return_value = iter(
            [_openai_text_chunk("a"), _openai_text_chunk(None, finish="stop")]
        )

        solwyn = _make_solwyn(client, model="gpt-5.5", circuit_breaker_success_threshold=2)
        cb = solwyn._get_circuit_breaker("openai")
        _force_half_open(cb)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            list(stream)  # drain the stream to completion -> on_complete settles

        # ONE streaming probe -> ONE success credit. Still HALF_OPEN, not CLOSED.
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1
        _close(solwyn)

    def test_streaming_establish_then_midstream_error_is_single_failure(self) -> None:
        # A stream that ESTABLISHES then errors mid-flight must record EXACTLY
        # one FAILURE (on_error) and ZERO successes on the served breaker — no
        # spurious dispatch-site success preceding it.
        client = _openai_client()

        def _exploding_stream():
            yield _openai_text_chunk("ok")
            raise ConnectionError("mid-stream reset")

        client.chat.completions.create.return_value = _exploding_stream()

        solwyn = _make_solwyn(client, model="gpt-5.5")
        cb = solwyn._get_circuit_breaker("openai")
        record_success = MagicMock(wraps=cb.record_success)
        record_failure = MagicMock(wraps=cb.record_failure)
        cb.record_success = record_success  # type: ignore[method-assign]
        cb.record_failure = record_failure  # type: ignore[method-assign]

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**request)
            with pytest.raises(ConnectionError, match="mid-stream reset"):
                list(stream)

        # Exactly one failure (on_error), zero successes (no dispatch-site credit).
        record_success.assert_not_called()
        record_failure.assert_called_once()
        _close(solwyn)

    def test_nonstreaming_success_still_credits_once(self) -> None:
        # Regression guard: the NON-streaming single-success path is unchanged —
        # one non-streaming success still credits the breaker exactly once at the
        # dispatch site (there is no on_complete on the non-streaming path).
        client = _openai_client()
        client.chat.completions.create.return_value = SimpleNamespace(
            model="gpt-5.5",
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

        solwyn = _make_solwyn(client, model="gpt-5.5", circuit_breaker_success_threshold=2)
        cb = solwyn._get_circuit_breaker("openai")
        _force_half_open(cb)

        request = {"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]}
        with patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**request)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1
        _close(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_halfopen_streaming_probe_credits_once_not_closed(self) -> None:
        # Async mirror of the HALF_OPEN single-probe credit.
        client = _openai_client()
        client.chat.completions.create = AsyncMock(
            return_value=_async_iter(
                [_openai_text_chunk("a"), _openai_text_chunk(None, finish="stop")]
            )
        )

        solwyn = _make_async_solwyn(client, model="gpt-5.5", circuit_breaker_success_threshold=2)
        cb = solwyn._get_circuit_breaker("openai")
        _force_half_open(cb)

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            _ = [c async for c in stream]

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1
        await _aclose(solwyn)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_streaming_midstream_error_is_single_failure(self) -> None:
        # Async mirror: establish then error mid-flight -> one failure, zero successes.
        client = _openai_client()

        async def _exploding_stream() -> AsyncIterator[Any]:
            yield _openai_text_chunk("ok")
            raise ConnectionError("mid-stream reset")

        client.chat.completions.create = AsyncMock(return_value=_exploding_stream())

        solwyn = _make_async_solwyn(client, model="gpt-5.5")
        cb = solwyn._get_circuit_breaker("openai")
        record_success = MagicMock(wraps=cb.record_success)
        record_failure = MagicMock(wraps=cb.record_failure)
        cb.record_success = record_success  # type: ignore[method-assign]
        cb.record_failure = record_failure  # type: ignore[method-assign]

        request = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with patch.object(
            solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow_budget())
        ):
            stream = await solwyn.chat.completions.create(**request)
            with pytest.raises(ConnectionError, match="mid-stream reset"):
                _ = [c async for c in stream]

        record_success.assert_not_called()
        record_failure.assert_called_once()
        await _aclose(solwyn)
