"""Tests for SyncStreamWrapper and AsyncStreamWrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
