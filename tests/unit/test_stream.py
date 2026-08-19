"""Tests for SyncStreamWrapper and AsyncStreamWrapper."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeAccumulator:
    """Test accumulator that returns preset TokenDetails."""

    def __init__(self, result: TokenDetails | None = None) -> None:
        self._result = result or TokenDetails(input_tokens=100, output_tokens=50)
        self.observed: list = []

    def observe(self, chunk: object) -> None:
        self.observed.append(chunk)

    def finalize(self) -> TokenDetails:
        return self._result

    def get_service_tier(self) -> str | None:
        return None


class _CloseableSyncStream:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self.close_calls = 0

    def __iter__(self):
        yield from self._items

    def close(self) -> None:
        self.close_calls += 1


class _CloseableAsyncStream:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self.aclose_calls = 0

    async def __aiter__(self):
        for item in self._items:
            yield item

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _TrackedSyncStream:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self.pulled: list[object] = []
        self.close_calls = 0

    def __iter__(self):
        for item in self._items:
            self.pulled.append(item)
            yield item

    def close(self) -> None:
        self.close_calls += 1


class _TrackedAsyncStream:
    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)
        self.pulled: list[object] = []
        self.aclose_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = next(self._items, None)
        if item is None:
            raise StopAsyncIteration
        self.pulled.append(item)
        return item

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _CloseFailingSyncStream(_TrackedSyncStream):
    def __init__(self, items: list[object], close_exc: BaseException) -> None:
        super().__init__(items)
        self._close_exc = close_exc

    def close(self) -> None:
        super().close()
        raise self._close_exc


class _AcloseFailingAsyncStream(_TrackedAsyncStream):
    def __init__(self, items: list[object], close_exc: BaseException) -> None:
        super().__init__(items)
        self._close_exc = close_exc

    async def aclose(self) -> None:
        await super().aclose()
        raise self._close_exc


class _BlockingAsyncCleanup:
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self.allow = asyncio.Event()
        self.attempts = 0
        self.completed = 0
        self.active = 0
        self.max_active = 0

    async def run(self) -> None:
        self.attempts += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.attempts == 1:
            self.first_started.set()
        else:
            self.second_started.set()
        try:
            await self.allow.wait()
            self.completed += 1
        finally:
            self.active -= 1


class _BlockingAcloseStream:
    def __init__(self, cleanup: _BlockingAsyncCleanup) -> None:
        self._cleanup = cleanup

    async def __aiter__(self):
        return
        yield

    async def aclose(self) -> None:
        await self._cleanup.run()


class _BlockingAsyncCloseStream:
    def __init__(self, cleanup: _BlockingAsyncCleanup) -> None:
        self._cleanup = cleanup

    async def __aiter__(self):
        return
        yield

    async def close(self) -> None:
        await self._cleanup.run()


def test_fake_accumulator_satisfies_stream_usage_accumulator_protocol() -> None:
    assert isinstance(FakeAccumulator(), StreamUsageAccumulator)


# ---------------------------------------------------------------------------
# SyncStreamWrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncStreamWrapperHappyPath:
    """Wrapper yields all chunks and calls on_complete after exhaustion."""

    def test_yields_all_chunks(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [
            SimpleNamespace(content="a"),
            SimpleNamespace(content="b"),
            SimpleNamespace(content="c"),
        ]
        acc = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=chunks,
            accumulator=acc,
            on_complete=on_complete,
            on_error=on_error,
        )

        collected = list(wrapper)

        assert collected == chunks
        assert acc.observed == chunks

    def test_calls_on_complete_with_token_details(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        expected_td = TokenDetails(input_tokens=150, output_tokens=83)
        acc = FakeAccumulator(result=expected_td)
        on_complete = MagicMock()
        on_error = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=[SimpleNamespace()],
            accumulator=acc,
            on_complete=on_complete,
            on_error=on_error,
        )
        list(wrapper)  # exhaust

        on_complete.assert_called_once()
        call_args = on_complete.call_args
        assert call_args[0][0] is expected_td  # first positional arg
        assert isinstance(call_args[0][1], float)  # elapsed_ms
        assert call_args[0][1] >= 0
        on_error.assert_not_called()

    def test_empty_stream_still_calls_on_complete(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        acc = FakeAccumulator()
        on_complete = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=[],
            accumulator=acc,
            on_complete=on_complete,
            on_error=MagicMock(),
        )
        list(wrapper)

        on_complete.assert_called_once()

    def test_on_complete_exception_suppressed_on_direct_iteration(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        def exploding_on_complete(token_details, elapsed_ms):
            raise RuntimeError("callback boom")

        wrapper = SyncStreamWrapper(
            stream=[SimpleNamespace()],
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=MagicMock(),
        )

        assert list(wrapper) == [SimpleNamespace()]


@pytest.mark.unit
class TestSyncStreamWrapperErrorPath:
    """On error, wrapper calls on_error and re-raises."""

    def test_error_during_iteration(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        def exploding_stream():
            yield SimpleNamespace(content="ok")
            raise ConnectionError("provider down")

        acc = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=exploding_stream(),
            accumulator=acc,
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ConnectionError, match="provider down"):
            list(wrapper)

        on_error.assert_called_once()
        assert isinstance(on_error.call_args[0][0], ConnectionError)
        on_complete.assert_not_called()

    def test_on_error_exception_suppressed_and_stream_error_preserved(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        def exploding_stream():
            yield SimpleNamespace(content="ok")
            raise ConnectionError("provider down")

        def exploding_on_error(exc):
            raise RuntimeError("callback boom")

        wrapper = SyncStreamWrapper(
            stream=exploding_stream(),
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=exploding_on_error,
        )

        with pytest.raises(ConnectionError, match="provider down"):
            list(wrapper)

    @pytest.mark.parametrize("failure_stage", ["iterator", "observer", "translator"])
    def test_chunk_processing_error_settles_as_error(self, failure_stage: str) -> None:
        from solwyn.stream import SyncStreamWrapper

        failure = ValueError(f"{failure_stage} failed")

        class BrokenIterator:
            def __iter__(self):
                raise failure

        accumulator = FakeAccumulator()
        if failure_stage == "observer":
            accumulator.observe = MagicMock(side_effect=failure)

        def translate(chunk: object) -> list[object]:
            if failure_stage == "translator":
                raise failure
            return [chunk]

        on_complete = MagicMock()
        on_error = MagicMock()
        stream = BrokenIterator() if failure_stage == "iterator" else [object()]
        wrapper = SyncStreamWrapper(
            stream=stream,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            chunk_translator=translate,
        )

        with pytest.raises(ValueError, match=rf"{failure_stage} failed") as exc_info:
            next(wrapper)

        on_error.assert_called_once_with(exc_info.value)
        on_complete.assert_not_called()


@pytest.mark.unit
class TestSyncStreamWrapperContextManager:
    """Wrapper delegates __enter__/__exit__ to the underlying stream."""

    def test_delegates_context_manager(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        inner = MagicMock()
        inner.__iter__ = MagicMock(return_value=iter([]))
        inner.__enter__ = MagicMock(return_value=inner)
        inner.__exit__ = MagicMock(return_value=False)

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )

        with wrapper as w:
            assert w is wrapper

        inner.__enter__.assert_called_once()
        inner.__exit__.assert_called_once()

    def test_works_without_context_manager_on_inner(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        wrapper = SyncStreamWrapper(
            stream=[SimpleNamespace()],
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )

        # Should not raise even though list has no __enter__
        with wrapper:
            list(wrapper)


@pytest.mark.unit
class TestSyncStreamWrapperPassthrough:
    """Wrapper delegates unknown attributes to the underlying stream."""

    def test_getattr_passthrough(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        inner = MagicMock()
        inner.response = SimpleNamespace(status_code=200)

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )

        assert wrapper.response.status_code == 200


@pytest.mark.unit
class TestSyncStreamWrapperAbort:
    """Early abort settles reservation with partial data via close()."""

    def test_close_fires_on_complete_with_partial_data(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [
            SimpleNamespace(content="a"),
            SimpleNamespace(content="b"),
            SimpleNamespace(content="c"),
        ]
        acc = FakeAccumulator()
        on_complete = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=iter(chunks),
            accumulator=acc,
            on_complete=on_complete,
            on_error=MagicMock(),
        )

        # Consume only first chunk, then close
        it = iter(wrapper)
        next(it)
        wrapper.close()

        on_complete.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        on_complete = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=[SimpleNamespace()],
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=MagicMock(),
        )
        list(wrapper)  # exhaust — triggers on_complete once

        wrapper.close()  # second close — should NOT trigger on_complete again
        assert on_complete.call_count == 1

    def test_context_manager_exit_calls_close(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        on_complete = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=iter([SimpleNamespace(), SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=MagicMock(),
        )

        with wrapper:
            it = iter(wrapper)
            next(it)  # consume only one chunk
            # exit context without exhausting

        # close() should have been called by __exit__
        on_complete.assert_called_once()

    def test_break_in_for_loop_with_context_manager(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        on_complete = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=iter([SimpleNamespace(i=0), SimpleNamespace(i=1), SimpleNamespace(i=2)]),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=MagicMock(),
        )

        with wrapper as stream:
            for chunk in stream:
                if chunk.i == 1:
                    break

        on_complete.assert_called_once()

    def test_abort_check_closes_and_settles_only_previously_observed_chunks(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [object() for _ in range(5)]
        inner = _TrackedSyncStream(chunks)
        accumulator = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()
        stopped = RuntimeError("run stopped")

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            abort_check=lambda: stopped if len(accumulator.observed) == 2 else None,
        )

        assert next(wrapper) is chunks[0]
        assert next(wrapper) is chunks[1]
        with pytest.raises(RuntimeError) as exc_info:
            next(wrapper)

        assert exc_info.value is stopped
        assert inner.pulled == chunks[:3]
        assert accumulator.observed == chunks[:2]
        assert inner.close_calls == 1
        on_complete.assert_called_once()
        on_error.assert_not_called()

    def test_first_abort_is_terminal_after_the_seam_clears(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [object() for _ in range(5)]
        inner = _TrackedSyncStream(chunks)
        accumulator = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()
        translator = MagicMock(side_effect=lambda chunk: [chunk])
        stopped = RuntimeError("run stopped")
        abort_enabled = True

        def abort_check() -> Exception | None:
            if abort_enabled and len(accumulator.observed) == 2:
                return stopped
            return None

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            chunk_translator=translator,
            abort_check=abort_check,
        )

        assert next(wrapper) is chunks[0]
        assert next(wrapper) is chunks[1]
        with pytest.raises(RuntimeError) as first_abort:
            next(wrapper)
        assert first_abort.value is stopped

        abort_enabled = False
        with pytest.raises(RuntimeError) as reentry:
            next(wrapper)
        assert reentry.value is stopped

        pending = object()
        wrapper._pending_chunks.append(pending)
        with pytest.raises(RuntimeError) as pending_reentry:
            next(wrapper)
        assert pending_reentry.value is stopped
        assert list(wrapper._pending_chunks) == [pending]
        assert inner.pulled == chunks[:3]
        assert accumulator.observed == chunks[:2]
        assert [call.args[0] for call in translator.call_args_list] == chunks[:2]
        assert inner.close_calls == 1
        on_complete.assert_called_once()
        on_error.assert_not_called()

    def test_abort_propagates_base_exception_raised_by_provider_close(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [object() for _ in range(4)]
        interrupt = KeyboardInterrupt()
        inner = _CloseFailingSyncStream(chunks, interrupt)
        accumulator = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()
        stopped = RuntimeError("run stopped")

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            abort_check=lambda: stopped if len(accumulator.observed) == 1 else None,
        )

        assert next(wrapper) is chunks[0]
        with pytest.raises(KeyboardInterrupt) as interrupted:
            next(wrapper)

        assert interrupted.value is interrupt
        assert inner.close_calls == 1
        on_complete.assert_called_once()
        on_error.assert_not_called()

        with pytest.raises(RuntimeError) as reentry:
            next(wrapper)
        assert reentry.value is stopped
        assert inner.close_calls == 1
        on_complete.assert_called_once()

    def test_abort_suppresses_ordinary_provider_close_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [SimpleNamespace(content="chunk-body-never-logged") for _ in range(3)]
        inner = _CloseFailingSyncStream(chunks, ConnectionResetError("provider socket gone"))
        accumulator = FakeAccumulator()
        on_complete = MagicMock()
        on_error = MagicMock()
        stopped = RuntimeError("run stopped")

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            abort_check=lambda: stopped if len(accumulator.observed) == 1 else None,
        )

        assert next(wrapper) is chunks[0]
        with (
            caplog.at_level(logging.WARNING, logger="solwyn.stream"),
            pytest.raises(RuntimeError) as exc_info,
        ):
            next(wrapper)

        assert exc_info.value is stopped
        assert inner.close_calls == 1
        on_complete.assert_called_once()
        on_error.assert_not_called()

        records = [record for record in caplog.records if record.name == "solwyn.stream"]
        assert len(records) == 1
        assert records[0].args == ("ConnectionResetError",)
        message = records[0].getMessage()
        assert "ConnectionResetError" in message
        assert "chunk-body-never-logged" not in message
        assert "provider socket gone" not in message

    @pytest.mark.parametrize("abort_mode", ["default", "explicit_none", "always_none"])
    def test_no_abort_check_preserves_legacy_iteration(self, abort_mode: str) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [object(), object()]
        kwargs: dict[str, object] = {}
        if abort_mode == "explicit_none":
            kwargs["abort_check"] = None
        elif abort_mode == "always_none":
            kwargs["abort_check"] = lambda: None
        accumulator = FakeAccumulator()
        on_complete = MagicMock()
        wrapper = SyncStreamWrapper(
            stream=chunks,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=MagicMock(),
            **kwargs,
        )

        assert list(wrapper) == chunks
        assert accumulator.observed == chunks
        on_complete.assert_called_once()

    def test_abort_on_first_boundary_settles_zero_observed_usage(self) -> None:
        from solwyn.stream import SyncStreamWrapper

        chunks = [object(), object()]
        inner = _TrackedSyncStream(chunks)
        accumulator = FakeAccumulator(result=TokenDetails())
        on_complete = MagicMock()
        stopped = RuntimeError("stop immediately")
        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=MagicMock(),
            abort_check=lambda: stopped,
        )

        with pytest.raises(RuntimeError) as exc_info:
            next(wrapper)

        assert exc_info.value is stopped
        assert inner.pulled == chunks[:1]
        assert accumulator.observed == []
        assert inner.close_calls == 1
        on_complete.assert_called_once()
        settled = on_complete.call_args.args[0]
        assert settled.input_tokens == 0
        assert settled.output_tokens == 0


# ---------------------------------------------------------------------------
# AsyncStreamWrapper
# ---------------------------------------------------------------------------


async def _aiter(items):
    """Helper: async generator from a list."""
    for item in items:
        yield item


@pytest.mark.unit
class TestAsyncStreamWrapper:
    """Async version yields chunks and calls async on_complete."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_yields_all_chunks(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [SimpleNamespace(content="a"), SimpleNamespace(content="b")]
        acc = FakeAccumulator()
        on_complete = AsyncMock()
        on_error = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=_aiter(chunks),
            accumulator=acc,
            on_complete=on_complete,
            on_error=on_error,
        )

        collected = [c async for c in wrapper]

        assert collected == chunks
        assert acc.observed == chunks

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calls_on_complete(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        expected_td = TokenDetails(input_tokens=200, output_tokens=100)
        acc = FakeAccumulator(result=expected_td)
        on_complete = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace()]),
            accumulator=acc,
            on_complete=on_complete,
            on_error=AsyncMock(),
        )
        _ = [c async for c in wrapper]

        on_complete.assert_called_once()
        assert on_complete.call_args[0][0] is expected_td

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_on_complete_exception_suppressed_on_direct_iteration(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        async def exploding_on_complete(token_details, elapsed_ms):
            raise RuntimeError("async callback boom")

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=AsyncMock(),
        )

        assert [c async for c in wrapper] == [SimpleNamespace()]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_during_iteration(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        async def exploding():
            yield SimpleNamespace()
            raise ConnectionError("boom")

        on_complete = AsyncMock()
        on_error = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=exploding(),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ConnectionError, match="boom"):
            _ = [c async for c in wrapper]

        on_error.assert_called_once()
        on_complete.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_on_error_exception_suppressed_and_stream_error_preserved(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        async def exploding():
            yield SimpleNamespace()
            raise ConnectionError("boom")

        async def exploding_on_error(exc):
            raise RuntimeError("async callback boom")

        wrapper = AsyncStreamWrapper(
            stream=exploding(),
            accumulator=FakeAccumulator(),
            on_complete=AsyncMock(),
            on_error=exploding_on_error,
        )

        with pytest.raises(ConnectionError, match="boom"):
            _ = [c async for c in wrapper]

    @pytest.mark.parametrize("failure_stage", ["iterator", "observer", "translator"])
    @pytest.mark.asyncio
    async def test_chunk_processing_error_settles_as_error(self, failure_stage: str) -> None:
        from solwyn.stream import AsyncStreamWrapper

        failure = ValueError(f"{failure_stage} failed")

        class BrokenAsyncIterator:
            def __aiter__(self):
                raise failure

        accumulator = FakeAccumulator()
        if failure_stage == "observer":
            accumulator.observe = MagicMock(side_effect=failure)

        def translate(chunk: object) -> list[object]:
            if failure_stage == "translator":
                raise failure
            return [chunk]

        on_complete = AsyncMock()
        on_error = AsyncMock()
        stream = BrokenAsyncIterator() if failure_stage == "iterator" else _aiter([object()])
        wrapper = AsyncStreamWrapper(
            stream=stream,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            chunk_translator=translate,
        )

        with pytest.raises(ValueError, match=rf"{failure_stage} failed") as exc_info:
            await anext(wrapper)

        on_error.assert_awaited_once_with(exc_info.value)
        on_complete.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        inner = AsyncMock()
        inner.__aiter__ = MagicMock(return_value=_aiter([]))
        inner.__aenter__ = AsyncMock(return_value=inner)
        inner.__aexit__ = AsyncMock(return_value=False)

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=AsyncMock(),
            on_error=AsyncMock(),
        )

        async with wrapper:
            pass

        inner.__aenter__.assert_called_once()
        inner.__aexit__.assert_called_once()


@pytest.mark.unit
class TestAsyncStreamWrapperAbort:
    """Early abort settles reservation via close()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_fires_on_complete(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        on_complete = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=AsyncMock(),
        )

        # Consume one chunk manually via __aiter__
        ait = wrapper.__aiter__()
        await ait.__anext__()
        await wrapper.close()

        on_complete.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        on_complete = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=AsyncMock(),
        )
        _ = [c async for c in wrapper]  # exhaust

        await wrapper.close()  # second call
        assert on_complete.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_exit_calls_close(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        on_complete = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=AsyncMock(),
        )

        async with wrapper:
            ait = wrapper.__aiter__()
            await ait.__anext__()  # consume one, then exit context

        on_complete.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_abort_check_closes_and_settles_only_previously_observed_chunks(
        self,
    ) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [object() for _ in range(5)]
        inner = _TrackedAsyncStream(chunks)
        accumulator = FakeAccumulator()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        stopped = RuntimeError("run stopped")
        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            abort_check=lambda: stopped if len(accumulator.observed) == 2 else None,
        )

        assert await anext(wrapper) is chunks[0]
        assert await anext(wrapper) is chunks[1]
        with pytest.raises(RuntimeError) as exc_info:
            await anext(wrapper)

        assert exc_info.value is stopped
        assert inner.pulled == chunks[:3]
        assert accumulator.observed == chunks[:2]
        assert inner.aclose_calls == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_first_abort_is_terminal_after_the_seam_clears(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [object() for _ in range(5)]
        inner = _TrackedAsyncStream(chunks)
        accumulator = FakeAccumulator()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        translator = MagicMock(side_effect=lambda chunk: [chunk])
        stopped = RuntimeError("run stopped")
        abort_enabled = True

        def abort_check() -> Exception | None:
            if abort_enabled and len(accumulator.observed) == 2:
                return stopped
            return None

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            chunk_translator=translator,
            abort_check=abort_check,
        )

        assert await anext(wrapper) is chunks[0]
        assert await anext(wrapper) is chunks[1]
        with pytest.raises(RuntimeError) as first_abort:
            await anext(wrapper)
        assert first_abort.value is stopped

        abort_enabled = False
        with pytest.raises(RuntimeError) as reentry:
            await anext(wrapper)
        assert reentry.value is stopped

        pending = object()
        wrapper._pending_chunks.append(pending)
        with pytest.raises(RuntimeError) as pending_reentry:
            await anext(wrapper)
        assert pending_reentry.value is stopped
        assert list(wrapper._pending_chunks) == [pending]
        assert inner.pulled == chunks[:3]
        assert accumulator.observed == chunks[:2]
        assert [call.args[0] for call in translator.call_args_list] == chunks[:2]
        assert inner.aclose_calls == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_abort_propagates_cancellation_raised_by_provider_aclose(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [object() for _ in range(4)]
        cancelled = asyncio.CancelledError()
        inner = _AcloseFailingAsyncStream(chunks, cancelled)
        accumulator = FakeAccumulator()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        stopped = RuntimeError("run stopped")

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=on_error,
            abort_check=lambda: stopped if len(accumulator.observed) == 1 else None,
        )

        assert await anext(wrapper) is chunks[0]
        with pytest.raises(asyncio.CancelledError) as interrupted:
            await anext(wrapper)

        assert interrupted.value is cancelled
        assert inner.aclose_calls == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

        with pytest.raises(RuntimeError) as reentry:
            await anext(wrapper)
        assert reentry.value is stopped
        on_complete.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("abort_mode", ["default", "explicit_none", "always_none"])
    async def test_no_abort_check_preserves_legacy_iteration(self, abort_mode: str) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [object(), object()]
        kwargs: dict[str, object] = {}
        if abort_mode == "explicit_none":
            kwargs["abort_check"] = None
        elif abort_mode == "always_none":
            kwargs["abort_check"] = lambda: None
        accumulator = FakeAccumulator()
        on_complete = AsyncMock()
        wrapper = AsyncStreamWrapper(
            stream=_aiter(chunks),
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=AsyncMock(),
            **kwargs,
        )

        assert [chunk async for chunk in wrapper] == chunks
        assert accumulator.observed == chunks
        on_complete.assert_awaited_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_abort_on_first_boundary_settles_zero_observed_usage(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        chunks = [object(), object()]
        inner = _TrackedAsyncStream(chunks)
        accumulator = FakeAccumulator(result=TokenDetails())
        on_complete = AsyncMock()
        stopped = RuntimeError("stop immediately")
        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=accumulator,
            on_complete=on_complete,
            on_error=AsyncMock(),
            abort_check=lambda: stopped,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await anext(wrapper)

        assert exc_info.value is stopped
        assert inner.pulled == chunks[:1]
        assert accumulator.observed == []
        assert inner.aclose_calls == 1
        on_complete.assert_awaited_once()
        settled = on_complete.await_args.args[0]
        assert settled.input_tokens == 0
        assert settled.output_tokens == 0


# ---------------------------------------------------------------------------
# Finding 002: __exit__ / __aexit__ swallow on_complete exceptions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExitSwallowsCallbackExceptions:
    """__exit__ and __aexit__ must not propagate on_complete exceptions."""

    def test_sync_exit_suppresses_callback_on_partial_consume(self) -> None:
        """__exit__ swallows on_complete exception when stream not fully consumed."""
        from solwyn.stream import SyncStreamWrapper

        def exploding_on_complete(token_details, elapsed_ms):
            raise RuntimeError("callback boom")

        # Exit before exhausting — __exit__ calls close() -> _settle() -> on_complete raises
        wrapper = SyncStreamWrapper(
            stream=iter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=MagicMock(),
        )
        with wrapper:
            it = iter(wrapper)
            next(it)  # consume one chunk, leave one unconsumed

        # If we get here without RuntimeError, __exit__ correctly swallowed it

    def test_sync_exit_suppresses_callback_exception_with_no_body_exception(self) -> None:
        """Explicit: __exit__ swallows RuntimeError from on_complete."""
        from solwyn.stream import SyncStreamWrapper

        callback_calls: list[str] = []

        def exploding_on_complete(token_details, elapsed_ms):
            callback_calls.append("called")
            raise RuntimeError("callback failure")

        wrapper = SyncStreamWrapper(
            stream=iter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=MagicMock(),
        )

        # No exception from body; __exit__ calls close() which calls on_complete (raises)
        with wrapper:
            it = iter(wrapper)
            next(it)  # partial consume — on_complete not yet fired

        # RuntimeError from callback should be swallowed, not propagated
        assert callback_calls == ["called"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_aexit_suppresses_callback_exception(self) -> None:
        """__aexit__ swallows exceptions from async on_complete."""
        from solwyn.stream import AsyncStreamWrapper

        callback_calls: list[str] = []

        async def exploding_on_complete(token_details, elapsed_ms):
            callback_calls.append("called")
            raise RuntimeError("async callback failure")

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=AsyncMock(),
        )

        async with wrapper:
            ait = wrapper.__aiter__()
            await ait.__anext__()  # partial consume — on_complete not yet fired

        # RuntimeError from async callback should be swallowed
        assert callback_calls == ["called"]

    def test_sync_exit_body_exception_takes_priority_over_callback_exception(self) -> None:
        """When body raises AND on_complete raises, body exception propagates.

        Both __exit__ exceptions (from body) and on_complete exceptions (from callback)
        occur during the same __exit__ call. Python's exception handling prioritizes
        the original exception from the with-body, suppressing the callback exception.
        """
        from solwyn.stream import SyncStreamWrapper

        callback_calls: list[str] = []

        def exploding_on_complete(token_details, elapsed_ms):
            callback_calls.append("on_complete_called")
            raise RuntimeError("callback error")

        wrapper = SyncStreamWrapper(
            stream=iter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=MagicMock(),
        )

        # Both body and callback raise — body exception must propagate
        with pytest.raises(ValueError, match="body error"):  # noqa: SIM117
            with wrapper:
                it = iter(wrapper)
                next(it)  # partial consume — on_complete fires in __exit__
                raise ValueError("body error")

        # Verify callback was called (and raised, but was swallowed)
        assert callback_calls == ["on_complete_called"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_aexit_body_exception_takes_priority_over_callback_exception(
        self,
    ) -> None:
        """When async body raises AND async on_complete raises, body exception propagates.

        Async variant: both exceptions occur during the same __aexit__ call. The original
        exception from the async with-body takes priority over callback exceptions.
        """
        from solwyn.stream import AsyncStreamWrapper

        callback_calls: list[str] = []

        async def exploding_on_complete(token_details, elapsed_ms):
            callback_calls.append("on_complete_called")
            raise RuntimeError("async callback error")

        wrapper = AsyncStreamWrapper(
            stream=_aiter([SimpleNamespace(), SimpleNamespace()]),
            accumulator=FakeAccumulator(),
            on_complete=exploding_on_complete,
            on_error=AsyncMock(),
        )

        # Both body and callback raise — body exception must propagate
        with pytest.raises(ValueError, match="async body error"):
            async with wrapper:
                ait = wrapper.__aiter__()
                await ait.__anext__()  # partial consume — on_complete fires in __aexit__
                raise ValueError("async body error")

        # Verify callback was called (and raised, but was swallowed)
        assert callback_calls == ["on_complete_called"]


# ---------------------------------------------------------------------------
# Finding 004: _settle_error fires on_error exactly once (symmetric with _settle)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSettleErrorSymmetry:
    """_settle_error fires on_error exactly once, even on concurrent calls."""

    def test_on_error_called_exactly_once_on_stream_error(self) -> None:
        """Error path calls on_error once and not on_complete."""
        from solwyn.stream import SyncStreamWrapper

        def exploding_stream():
            yield SimpleNamespace()
            raise ValueError("stream failed")

        on_complete = MagicMock()
        on_error = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=exploding_stream(),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ValueError, match="stream failed"):
            list(wrapper)

        on_error.assert_called_once()
        on_complete.assert_not_called()

    def test_close_after_error_does_not_call_on_complete(self) -> None:
        """After error settlement, calling close() must not fire on_complete."""
        from solwyn.stream import SyncStreamWrapper

        def exploding_stream():
            yield SimpleNamespace()
            raise ValueError("stream failed")

        on_complete = MagicMock()
        on_error = MagicMock()

        wrapper = SyncStreamWrapper(
            stream=exploding_stream(),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ValueError):
            list(wrapper)

        wrapper.close()  # _settled is True — must be a no-op

        on_error.assert_called_once()
        on_complete.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_on_error_called_exactly_once(self) -> None:
        """Async error path calls on_error once and not on_complete."""
        from solwyn.stream import AsyncStreamWrapper

        async def exploding():
            yield SimpleNamespace()
            raise ValueError("async stream failed")

        on_complete = AsyncMock()
        on_error = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=exploding(),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ValueError, match="async stream failed"):
            _ = [c async for c in wrapper]

        on_error.assert_called_once()
        on_complete.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_close_after_error_does_not_call_on_complete(self) -> None:
        """After async error settlement, close() must not fire on_complete."""
        from solwyn.stream import AsyncStreamWrapper

        async def exploding():
            yield SimpleNamespace()
            raise ValueError("async stream failed")

        on_complete = AsyncMock()
        on_error = AsyncMock()

        wrapper = AsyncStreamWrapper(
            stream=exploding(),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(ValueError):
            _ = [c async for c in wrapper]

        await wrapper.close()  # must be a no-op

        on_error.assert_called_once()
        on_complete.assert_not_called()


# ---------------------------------------------------------------------------
# close()/aclose() forward cleanup to inner stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncStreamWrapperInnerClose:
    """close() forwards cleanup to the inner stream's close() method."""

    def test_close_calls_inner_close(self) -> None:
        """wrapper.close() must invoke inner.close() exactly once."""
        from solwyn.stream import SyncStreamWrapper

        inner = MagicMock()
        inner.__iter__ = MagicMock(return_value=iter([]))

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )
        wrapper.close()

        inner.close.assert_called_once()

    def test_context_manager_closes_inner_without_exit(self) -> None:
        """with wrapper: closes an inner object that has close() but no __exit__."""
        from solwyn.stream import SyncStreamWrapper

        class CloseOnlyStream:
            """Exposes close() but no __exit__."""

            def __init__(self, items):
                self._iter = iter(items)
                self.close_called = 0

            def __iter__(self):
                return self._iter

            def close(self):
                self.close_called += 1

        inner = CloseOnlyStream([SimpleNamespace(), SimpleNamespace()])

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )

        with wrapper:
            it = iter(wrapper)
            next(it)  # consume one chunk, then exit early

        assert inner.close_called >= 1

    def test_close_idempotent_inner_called_multiple_times_is_ok(self) -> None:
        """Repeated wrapper.close() calls don't raise even if inner.close() is called each time."""
        from solwyn.stream import SyncStreamWrapper

        inner = MagicMock()
        inner.__iter__ = MagicMock(return_value=iter([]))

        wrapper = SyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )
        wrapper.close()
        wrapper.close()  # must not raise
        wrapper.close()

        # Inner close may be called multiple times — that's acceptable
        assert inner.close.call_count >= 1


@pytest.mark.unit
class TestAsyncStreamWrapperInnerClose:
    """close() forwards cleanup to the inner stream's aclose() / close() method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_calls_inner_aclose(self) -> None:
        """await wrapper.close() must invoke inner.aclose() when present."""
        from solwyn.stream import AsyncStreamWrapper

        inner = AsyncMock()
        inner.__aiter__ = MagicMock(return_value=_aiter([]))

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=AsyncMock(),
            on_error=AsyncMock(),
        )
        await wrapper.close()

        inner.aclose.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager_closes_inner_with_aclose_but_no_aexit(self) -> None:
        """async with wrapper: closes inner via aclose() when __aexit__ is absent."""
        from solwyn.stream import AsyncStreamWrapper

        class AcloseOnlyStream:
            """Exposes aclose() but no __aexit__."""

            def __init__(self, items):
                self._items = items
                self.aclose_called = 0

            async def __aiter__(self):
                for item in self._items:
                    yield item

            async def aclose(self):
                self.aclose_called += 1

        inner = AcloseOnlyStream([SimpleNamespace(), SimpleNamespace()])

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=AsyncMock(),
            on_error=AsyncMock(),
        )

        async with wrapper:
            ait = wrapper.__aiter__()
            await ait.__anext__()  # partial consume then exit early

        assert inner.aclose_called >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_falls_back_to_sync_close_when_no_aclose(self) -> None:
        """await wrapper.close() falls back to inner.close() when aclose() is absent."""
        from solwyn.stream import AsyncStreamWrapper

        class SyncCloseOnlyStream:
            """Has close() but no aclose()."""

            def __init__(self):
                self.close_called = 0

            async def __aiter__(self):
                return
                yield  # make it an async generator

            def close(self):
                self.close_called += 1

        inner = SyncCloseOnlyStream()

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=AsyncMock(),
            on_error=AsyncMock(),
        )
        await wrapper.close()

        assert inner.close_called == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_awaits_async_close_when_no_aclose(self) -> None:
        """await wrapper.close() awaits an awaitable returned by inner.close()."""
        from solwyn.stream import AsyncStreamWrapper

        class AsyncCloseOnlyStream:
            """Matches Together AsyncStream: async close() but no aclose()."""

            def __init__(self) -> None:
                self.close_called = 0

            async def __aiter__(self):
                yield SimpleNamespace()
                yield SimpleNamespace()

            async def close(self) -> None:
                self.close_called += 1

        inner = AsyncCloseOnlyStream()
        completion_count = 0

        async def on_complete(_token_details: TokenDetails, _elapsed_ms: float) -> None:
            nonlocal completion_count
            completion_count += 1

        async def on_error(_exc: Exception) -> None:
            raise AssertionError("early close must not settle as an error")

        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )
        iterator = wrapper.__aiter__()
        await iterator.__anext__()

        await wrapper.close()
        await iterator.aclose()

        assert inner.close_called == 1
        assert completion_count == 1

    @pytest.mark.parametrize(
        "stream_type",
        [_BlockingAcloseStream, _BlockingAsyncCloseStream],
        ids=["aclose", "async-close"],
    )
    @pytest.mark.asyncio
    async def test_cancelled_provider_cleanup_is_retryable_and_serialized(
        self,
        stream_type: type[_BlockingAcloseStream] | type[_BlockingAsyncCloseStream],
    ) -> None:
        from solwyn.stream import AsyncStreamWrapper

        cleanup = _BlockingAsyncCleanup()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        wrapper = AsyncStreamWrapper(
            stream=stream_type(cleanup),
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        first_close = asyncio.create_task(wrapper.close())
        await cleanup.first_started.wait()
        second_close = asyncio.create_task(wrapper.close())
        await asyncio.sleep(0)
        assert cleanup.attempts == 1

        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close
        try:
            await asyncio.wait_for(cleanup.second_started.wait(), timeout=1)
        finally:
            cleanup.allow.set()
        await second_close
        await wrapper.close()

        assert cleanup.attempts == 2
        assert cleanup.completed == 1
        assert cleanup.max_active == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_settlement_cancellation_leaves_provider_cleanup_retryable(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        inner = _CloseableAsyncStream([])
        settlement_started = asyncio.Event()
        never_finish_settlement = asyncio.Event()
        completion_calls = 0

        async def on_complete(_details: TokenDetails, _elapsed_ms: float) -> None:
            nonlocal completion_calls
            completion_calls += 1
            settlement_started.set()
            await never_finish_settlement.wait()

        on_error = AsyncMock()
        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        first_close = asyncio.create_task(wrapper.close())
        await settlement_started.wait()
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close
        await wrapper.close()
        await wrapper.close()

        assert completion_calls == 1
        assert inner.aclose_calls == 1
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_cleanup_error_is_retryable_without_double_settlement(self) -> None:
        from solwyn.stream import AsyncStreamWrapper

        class FailsOnceStream:
            def __init__(self) -> None:
                self.aclose_calls = 0

            async def __aiter__(self):
                return
                yield

            async def aclose(self) -> None:
                self.aclose_calls += 1
                if self.aclose_calls == 1:
                    raise RuntimeError("cleanup failed")

        inner = FailsOnceStream()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        wrapper = AsyncStreamWrapper(
            stream=inner,
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

        with pytest.raises(RuntimeError, match="cleanup failed"):
            await wrapper.close()
        await wrapper.close()
        await wrapper.close()

        assert inner.aclose_calls == 2
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()


@pytest.mark.unit
class TestSyncResponsesStreamManagerLifecycle:
    @staticmethod
    def _wrap(source: object):
        from solwyn.stream import SyncStreamWrapper

        return SyncStreamWrapper(
            stream=source,
            accumulator=FakeAccumulator(),
            on_complete=MagicMock(),
            on_error=MagicMock(),
        )

    def test_enter_and_close_serialize_without_losing_the_opened_stream(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        enter_started = threading.Event()
        allow_enter = threading.Event()
        close_done = threading.Event()
        event = object()

        class BlockingManager:
            def __init__(self) -> None:
                self.stream = _CloseableSyncStream([event])
                self.exit_calls: list[tuple[object, ...]] = []

            def __enter__(self):
                enter_started.set()
                if not allow_enter.wait(timeout=2):
                    raise TimeoutError("test did not release manager entry")
                return self.stream

            def __exit__(self, *args: object) -> None:
                self.exit_calls.append(args)
                self.stream.close()

        manager = BlockingManager()
        wrapper = _SyncResponsesStreamManagerWrapper(
            manager,
            self._wrap,
            on_error=MagicMock(),
            on_entry_error=MagicMock(),
            on_abandoned_before_entry=MagicMock(),
        )
        entered: list[object] = []
        errors: list[BaseException] = []

        def enter() -> None:
            try:
                entered.append(wrapper.__enter__())
            except BaseException as exc:
                errors.append(exc)

        def close() -> None:
            try:
                wrapper.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                close_done.set()

        enter_thread = threading.Thread(target=enter)
        close_thread = threading.Thread(target=close)
        enter_thread.start()
        assert enter_started.wait(timeout=1)
        close_thread.start()
        try:
            assert not close_done.wait(timeout=0.1)
        finally:
            allow_enter.set()
            enter_thread.join(timeout=2)
            close_thread.join(timeout=2)

        assert not errors
        assert len(entered) == 1
        assert list(entered[0]) == [event]
        assert len(manager.exit_calls) == 1
        assert manager.stream.close_calls == 1

    def test_reentry_is_rejected_and_close_is_idempotent(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        inner = _CloseableSyncStream([])
        manager = MagicMock()
        manager.__enter__.return_value = inner
        manager.__exit__.return_value = None
        wrapper = _SyncResponsesStreamManagerWrapper(
            manager,
            self._wrap,
            on_error=MagicMock(),
            on_entry_error=MagicMock(),
            on_abandoned_before_entry=MagicMock(),
        )

        wrapper.__enter__()
        with pytest.raises(RuntimeError, match="already entered"):
            wrapper.__enter__()
        wrapper.close()
        wrapper.close()

        manager.__enter__.assert_called_once_with()
        manager.__exit__.assert_called_once_with(None, None, None)

    def test_enter_after_close_is_rejected(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        manager = MagicMock()
        manager.__exit__.return_value = None
        wrap = MagicMock()
        abandoned = MagicMock()
        wrapper = _SyncResponsesStreamManagerWrapper(
            manager,
            wrap,
            on_error=MagicMock(),
            on_entry_error=MagicMock(),
            on_abandoned_before_entry=abandoned,
        )

        wrapper.close()
        wrapper.close()
        with pytest.raises(RuntimeError, match="closed"):
            wrapper.__enter__()

        manager.__enter__.assert_not_called()
        manager.__exit__.assert_called_once_with(None, None, None)
        # Nothing was dispatched: the reservation goes back and no stream is
        # ever wrapped or settled.
        abandoned.assert_called_once_with()
        wrap.assert_not_called()

    def test_gc_before_entry_releases_pre_dispatch_abort_handle(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        abort_release = MagicMock()
        wrapper = _SyncResponsesStreamManagerWrapper(
            MagicMock(),
            MagicMock(),
            on_error=MagicMock(),
            on_entry_error=MagicMock(),
            on_abandoned_before_entry=MagicMock(),
            abort_release=abort_release,
        )

        del wrapper
        gc.collect()

        abort_release.assert_called_once_with()

    def test_wrap_failure_closes_opened_manager_without_masking_original(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        original = ValueError("wrapping failed")
        inner = _CloseableSyncStream([])
        manager = MagicMock()
        manager.__enter__.return_value = inner
        manager.__exit__.side_effect = RuntimeError("cleanup failed")
        on_error = MagicMock()
        on_entry_error = MagicMock()
        wrapper = _SyncResponsesStreamManagerWrapper(
            manager,
            MagicMock(side_effect=original),
            on_error=on_error,
            on_entry_error=on_entry_error,
            on_abandoned_before_entry=MagicMock(),
        )

        with pytest.raises(ValueError, match="wrapping failed") as exc_info:
            wrapper.__enter__()
        wrapper.close()

        assert exc_info.value is original
        # The provider stream was already open: this is an established-stream
        # failure, not an entry failure.
        on_error.assert_called_once_with(original)
        on_entry_error.assert_not_called()
        manager.__exit__.assert_called_once()
        exit_args = manager.__exit__.call_args.args
        assert exit_args[0] is ValueError
        assert exit_args[1] is original

    def test_provider_entry_failure_uses_the_entry_error_path(self) -> None:
        from solwyn.stream import _SyncResponsesStreamManagerWrapper

        original = RuntimeError("entry failed")
        manager = MagicMock()
        manager.__enter__.side_effect = original
        manager.__exit__.return_value = None
        wrap = MagicMock()
        on_error = MagicMock()
        on_entry_error = MagicMock()
        wrapper = _SyncResponsesStreamManagerWrapper(
            manager,
            wrap,
            on_error=on_error,
            on_entry_error=on_entry_error,
            on_abandoned_before_entry=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="entry failed"):
            wrapper.__enter__()
        wrapper.close()

        on_entry_error.assert_called_once_with(original)
        on_error.assert_not_called()
        wrap.assert_not_called()
        manager.__exit__.assert_called_once()


@pytest.mark.unit
class TestAsyncResponsesStreamManagerLifecycle:
    @staticmethod
    def _wrap(source: object, on_complete: AsyncMock, on_error: AsyncMock):
        from solwyn.stream import AsyncStreamWrapper

        return AsyncStreamWrapper(
            stream=source,
            accumulator=FakeAccumulator(),
            on_complete=on_complete,
            on_error=on_error,
        )

    @pytest.mark.asyncio
    async def test_nested_reentry_is_rejected_without_poisoning_first_stream(self) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        event = object()
        inner = _CloseableAsyncStream([event])
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=inner)
        manager.__aexit__ = AsyncMock(return_value=None)
        on_complete = AsyncMock()
        on_error = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            lambda source: self._wrap(source, on_complete, on_error),
            on_error=on_error,
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=AsyncMock(),
        )

        first = await wrapper.__aenter__()
        with pytest.raises(RuntimeError, match="already entered"):
            await wrapper.__aenter__()
        assert [item async for item in first] == [event]
        await wrapper.__aexit__(None, None, None)

        manager.__aenter__.assert_awaited_once_with()
        manager.__aexit__.assert_awaited_once_with(None, None, None)
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_reentry_waits_then_rejects_without_provider_reuse(self) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        event = object()
        inner = _CloseableAsyncStream([event])
        enter_started = asyncio.Event()
        allow_enter = asyncio.Event()

        class OneShotManager:
            def __init__(self) -> None:
                self.enter_calls = 0
                self.exit_calls: list[tuple[object, ...]] = []

            async def __aenter__(self):
                self.enter_calls += 1
                if self.enter_calls > 1:
                    raise RuntimeError("one-shot manager reused")
                enter_started.set()
                await allow_enter.wait()
                return inner

            async def __aexit__(self, *args: object) -> None:
                self.exit_calls.append(args)
                await inner.aclose()

        manager = OneShotManager()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            lambda source: self._wrap(source, on_complete, on_error),
            on_error=on_error,
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=AsyncMock(),
        )

        first_task = asyncio.create_task(wrapper.__aenter__())
        await enter_started.wait()
        second_task = asyncio.create_task(wrapper.__aenter__())
        await asyncio.sleep(0)
        assert manager.enter_calls == 1
        allow_enter.set()

        first = await first_task
        with pytest.raises(RuntimeError, match="already entered"):
            await second_task
        assert [item async for item in first] == [event]
        await wrapper.close()

        assert manager.enter_calls == 1
        assert len(manager.exit_calls) == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enter_after_close_is_rejected_and_close_is_idempotent(self) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        manager = MagicMock()
        manager.__aenter__ = AsyncMock()
        manager.__aexit__ = AsyncMock(return_value=None)
        on_complete = AsyncMock()
        on_error = AsyncMock()
        abandoned = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            lambda source: self._wrap(source, on_complete, on_error),
            on_error=on_error,
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=abandoned,
        )

        await wrapper.close()
        await wrapper.close()
        with pytest.raises(RuntimeError, match="closed"):
            await wrapper.__aenter__()

        manager.__aenter__.assert_not_awaited()
        manager.__aexit__.assert_awaited_once_with(None, None, None)
        # Nothing was dispatched: the reservation goes back instead of settling.
        abandoned.assert_awaited_once_with()
        on_complete.assert_not_awaited()
        on_error.assert_not_awaited()

    def test_gc_before_entry_releases_pre_dispatch_abort_handle(self) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        abort_release = MagicMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            MagicMock(),
            MagicMock(),
            on_error=AsyncMock(),
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=AsyncMock(),
            abort_release=abort_release,
        )

        del wrapper
        gc.collect()

        abort_release.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cancelled_close_is_retryable_and_serializes_concurrent_closers(
        self,
    ) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        inner = _CloseableAsyncStream([])
        first_exit_started = asyncio.Event()
        second_exit_started = asyncio.Event()
        allow_exit = asyncio.Event()

        class RetryableExitManager:
            def __init__(self) -> None:
                self.exit_attempts = 0
                self.completed_exits = 0
                self.active_exits = 0
                self.max_active_exits = 0

            async def __aenter__(self):
                return inner

            async def __aexit__(self, *_args: object) -> None:
                self.exit_attempts += 1
                self.active_exits += 1
                self.max_active_exits = max(self.max_active_exits, self.active_exits)
                if self.exit_attempts == 1:
                    first_exit_started.set()
                else:
                    second_exit_started.set()
                try:
                    await allow_exit.wait()
                    await inner.aclose()
                    self.completed_exits += 1
                finally:
                    self.active_exits -= 1

        manager = RetryableExitManager()
        on_complete = AsyncMock()
        on_error = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            lambda source: self._wrap(source, on_complete, on_error),
            on_error=on_error,
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=AsyncMock(),
        )
        await wrapper.__aenter__()

        first_close = asyncio.create_task(wrapper.__aexit__(None, None, None))
        await first_exit_started.wait()
        second_close = asyncio.create_task(wrapper.close())
        await asyncio.sleep(0)
        assert manager.exit_attempts == 1

        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close
        try:
            await asyncio.wait_for(second_exit_started.wait(), timeout=1)
        finally:
            allow_exit.set()
        await second_close
        await wrapper.close()

        assert manager.exit_attempts == 2
        assert manager.completed_exits == 1
        assert manager.max_active_exits == 1
        assert inner.aclose_calls == 1
        on_complete.assert_awaited_once()
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_settlement_cancellation_still_closes_provider_without_double_accounting(
        self,
    ) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        inner = _CloseableAsyncStream([])
        settlement_started = asyncio.Event()
        never_finish_settlement = asyncio.Event()
        completion_calls = 0

        async def on_complete(_details: TokenDetails, _elapsed_ms: float) -> None:
            nonlocal completion_calls
            completion_calls += 1
            settlement_started.set()
            await never_finish_settlement.wait()

        on_error = AsyncMock()
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=inner)
        manager.__aexit__ = AsyncMock(return_value=None)
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            lambda source: self._wrap(source, on_complete, on_error),
            on_error=on_error,
            on_entry_error=AsyncMock(),
            on_abandoned_before_entry=AsyncMock(),
        )
        await wrapper.__aenter__()

        close_task = asyncio.create_task(wrapper.close())
        await settlement_started.wait()
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await wrapper.close()

        assert completion_calls == 1
        manager.__aexit__.assert_awaited_once_with(None, None, None)
        on_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wrap_failure_settles_error_and_closes_without_masking_original(
        self,
    ) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        original = ValueError("wrapping failed")
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=_CloseableAsyncStream([]))
        manager.__aexit__ = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        on_error = AsyncMock()
        on_entry_error = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            MagicMock(side_effect=original),
            on_error=on_error,
            on_entry_error=on_entry_error,
            on_abandoned_before_entry=AsyncMock(),
        )

        with pytest.raises(ValueError, match="wrapping failed") as exc_info:
            await wrapper.__aenter__()
        await wrapper.close()

        assert exc_info.value is original
        # The provider stream was already open: this is an established-stream
        # failure, not an entry failure.
        on_error.assert_awaited_once_with(original)
        on_entry_error.assert_not_awaited()
        manager.__aexit__.assert_awaited_once()
        exit_args = manager.__aexit__.await_args.args
        assert exit_args[0] is ValueError
        assert exit_args[1] is original

    @pytest.mark.asyncio
    async def test_provider_entry_failure_uses_the_entry_error_path(self) -> None:
        from solwyn.stream import _AsyncResponsesStreamManagerWrapper

        original = RuntimeError("entry failed")
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(side_effect=original)
        manager.__aexit__ = AsyncMock(return_value=None)
        wrap = MagicMock()
        on_error = AsyncMock()
        on_entry_error = AsyncMock()
        wrapper = _AsyncResponsesStreamManagerWrapper(
            manager,
            wrap,
            on_error=on_error,
            on_entry_error=on_entry_error,
            on_abandoned_before_entry=AsyncMock(),
        )

        with pytest.raises(RuntimeError, match="entry failed"):
            await wrapper.__aenter__()
        await wrapper.close()

        on_entry_error.assert_awaited_once_with(original)
        on_error.assert_not_awaited()
        wrap.assert_not_called()
        manager.__aexit__.assert_awaited_once()


@pytest.mark.unit
class TestStreamingSettlementAttribution:
    def test_run_tags_and_agent_run_survive_stream_settlement(self) -> None:
        from conftest import VALID_API_KEY

        from solwyn import run
        from solwyn.client import Solwyn

        provider = MagicMock()
        provider.__class__.__module__ = "openai._client"
        provider.__class__.__name__ = "OpenAI"
        provider.with_options.return_value = provider
        provider.chat.completions.create.return_value = iter(
            [
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                    service_tier=None,
                )
            ]
        )
        with patch("solwyn.reporter.MetadataReporter._flush_loop"):
            solwyn = Solwyn(provider, api_key=VALID_API_KEY, model="gpt-5.5")
        solwyn._solwyn_reporter._shutdown.set()
        solwyn._solwyn_reporter._thread.join(timeout=2.0)
        settlements: list[tuple[object, object]] = []
        solwyn._solwyn_reporter.report_settlement = lambda confirm, event: settlements.append(
            (confirm, event)
        )
        budget = SimpleNamespace(
            allowed=True,
            reservation_id="res_stream_attribution",
            project_id=None,
            budget_limit=100.0,
            current_usage=0.0,
            mode=SimpleNamespace(value="alert_only"),
            price_hints=None,
        )

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=budget),
            run("stream-agent", tags={"team": "research"}) as run_id,
        ):
            stream = solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(stream)

        assert len(settlements) == 1
        _confirm, event = settlements[0]
        assert event.tags == {"team": "research"}
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "stream-agent"
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()
