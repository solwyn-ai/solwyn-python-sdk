"""Stream wrappers for intercepting streaming LLM responses.

These wrappers yield every chunk unchanged to the user while a
StreamUsageAccumulator silently observes usage data. After the stream
completes (or errors), a callback fires to settle the budget reservation
and report metadata.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Self, cast

from solwyn._token_details import TokenDetails
from solwyn.providers._accumulator import StreamUsageAccumulator

logger = logging.getLogger(__name__)


class SyncStreamWrapper:
    """Wraps a synchronous streaming response for token accumulation.

    Yields every chunk unchanged. After the iterator is exhausted,
    calls on_complete(token_details, elapsed_ms). If iteration raises,
    calls on_error(exception) and re-raises.

    If the caller breaks early or abandons the iterator, call close()
    explicitly or use the context manager (``with stream:``) to settle
    the budget reservation with whatever usage data was observed.

    Thread safety: _settle() and _settle_error() are protected by a
    threading.Lock so that concurrent calls (e.g. close() racing with
    iterator exhaustion) fire on_complete/on_error exactly once.

    Cross-provider stream normalization: an optional ``chunk_translator``
    maps ONE raw served chunk into a LIST of caller-dialect chunks. The
    accumulator ALWAYS observes the RAW served chunk (served-provider usage); the
    wrapper yields the translated chunks. ``None`` (default) is passthrough — the
    raw chunk is yielded unchanged (same-dialect / primary / same-provider swap).
    This wrapper never logs/stores/stringifies chunk content; it only ROUTES the
    raw chunk to ``chunk_translator`` (a content-privileged ``_translation`` seam).

    A keyword-only ``abort_check`` may cooperatively stop a run at a raw provider
    chunk boundary. The next chunk is pulled first, then the check runs before
    that chunk is observed or yielded. A returned exception closes and partially
    settles the stream before that exact exception is raised.
    """

    def __init__(
        self,
        stream: Any,
        accumulator: StreamUsageAccumulator,
        on_complete: Callable[[TokenDetails, float], None],
        on_error: Callable[[BaseException], None],
        chunk_translator: Callable[[Any], list[Any]] | None = None,
        *,
        abort_check: Callable[[], Exception | None] | None = None,
        abort_release: Callable[[], None] | None = None,
    ) -> None:
        self._stream = stream
        self._iterator: Iterator[Any] | None = None
        self._pending_chunks: deque[Any] = deque()
        self._accumulator = accumulator
        self._on_complete = on_complete
        self._on_error = on_error
        self._chunk_translator = chunk_translator
        self._abort_check = abort_check
        self._abort_finalizer = (
            weakref.finalize(self, abort_release) if abort_release is not None else None
        )
        self._abort_exc: Exception | None = None
        self._start_time = time.monotonic()
        self._settled = False
        self._lock = threading.Lock()

    def _latched_abort(self) -> Exception | None:
        with self._lock:
            return self._abort_exc

    def _latch_abort(self, exc: Exception) -> Exception:
        with self._lock:
            if self._abort_exc is None:
                self._abort_exc = exc
            return self._abort_exc

    def _settle(self) -> None:
        """Fire on_complete exactly once with accumulated data."""
        with self._lock:
            if self._settled:
                return
            self._settled = True
        self._release_abort_handle()
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        token_details = self._accumulator.finalize()
        try:
            self._on_complete(token_details, elapsed_ms)
        except Exception as cb_exc:
            # Log only the CALLBACK failure's class name — never a traceback
            # (which would capture the live exception context, possibly a provider
            # mid-stream exception whose str() embeds streamed content; fix [D]).
            logger.warning(
                "on_complete raised during stream settlement; suppressing (%s)",
                type(cb_exc).__name__,
            )

    def _settle_error(self, exc: BaseException) -> None:
        """Fire on_error exactly once. Mirrors _settle() for the error path."""
        with self._lock:
            if self._settled:
                return
            self._settled = True
        self._release_abort_handle()
        try:
            self._on_error(exc)
        except Exception as cb_exc:
            # Structural-only log of the CALLBACK failure; never a traceback (which,
            # during error settlement, holds the provider exception ``exc``; fix [D]).
            logger.warning(
                "on_error raised during stream settlement; suppressing (%s)",
                type(cb_exc).__name__,
            )

    def _release_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()

    def __next__(self) -> Any:
        """Return one observed caller-dialect chunk."""
        latched_abort = self._latched_abort()
        if latched_abort is not None:
            raise latched_abort
        while not self._pending_chunks:
            try:
                iterator = self._iterator
                if iterator is None:
                    iterator = iter(self._stream)
                    self._iterator = iterator
                chunk = next(iterator)
                if self._abort_check is not None:
                    abort_exc = self._abort_check()
                    if abort_exc is not None:
                        abort_exc = self._latch_abort(abort_exc)
                        try:
                            self.close()
                        finally:
                            raise abort_exc
                self._accumulator.observe(chunk)
                if self._chunk_translator is None:
                    return chunk
                self._pending_chunks.extend(self._chunk_translator(chunk))
            except StopIteration:
                self._settle()
                raise
            except Exception as exc:
                if self._latched_abort() is exc:
                    raise
                self._settle_error(exc)
                raise
        return self._pending_chunks.popleft()

    def __iter__(self) -> Iterator[Any]:
        return self

    def close(self) -> None:
        """Settle the budget reservation with whatever data we have, then
        forward cleanup to the inner stream's close() method.

        Safe to call multiple times — only the first call fires on_complete.
        Forwarding close() to the inner stream is also safe if called multiple
        times; well-behaved stream implementations are idempotent.
        Called automatically by __exit__ if the stream wasn't fully consumed.
        """
        self._settle()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __enter__(self) -> Self:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> bool | None:
        try:
            # self.close() now settles AND forwards to inner close().
            # Broad except is intentional: callback/provider-close failures
            # must not mask the application exception propagating through
            # __exit__ (if any).
            self.close()
        except Exception as cb_exc:
            # Structural-only: log the suppressed callback class, never a traceback
            # (fix [D]).
            logger.warning(
                "on_complete raised during __exit__; suppressing (%s)", type(cb_exc).__name__
            )
        if hasattr(self._stream, "__exit__"):
            return cast("bool | None", self._stream.__exit__(*args))
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncStreamWrapper:
    """Wraps an asynchronous streaming response for token accumulation.

    Async version of SyncStreamWrapper. on_complete and on_error are
    coroutines (awaited after stream completes or errors).

    Call close() or use ``async with stream:`` to settle on early abort.

    Settlement marks itself complete before awaiting its callback. Provider
    cleanup is serialized separately so cancellation leaves it retryable.

    Cross-provider stream normalization: see ``SyncStreamWrapper`` — the
    optional ``chunk_translator`` reshapes raw served chunks into caller-dialect
    chunks while the accumulator still observes the RAW served chunk.
    ``abort_check`` has the same post-pull, pre-observe boundary as the sync
    wrapper and is evaluated synchronously (registry reads perform no I/O).
    """

    def __init__(
        self,
        stream: Any,
        accumulator: StreamUsageAccumulator,
        on_complete: Callable[[TokenDetails, float], Awaitable[None]],
        on_error: Callable[[BaseException], Awaitable[None]],
        chunk_translator: Callable[[Any], list[Any]] | None = None,
        *,
        abort_check: Callable[[], Exception | None] | None = None,
        abort_release: Callable[[], None] | None = None,
    ) -> None:
        self._stream = stream
        self._iterator: AsyncIterator[Any] | None = None
        self._pending_chunks: deque[Any] = deque()
        self._accumulator = accumulator
        self._on_complete = on_complete
        self._on_error = on_error
        self._chunk_translator = chunk_translator
        self._abort_check = abort_check
        self._abort_finalizer = (
            weakref.finalize(self, abort_release) if abort_release is not None else None
        )
        self._abort_exc: Exception | None = None
        self._start_time = time.monotonic()
        self._settled = False
        self._cleanup_complete = False
        self._close_lock = asyncio.Lock()

    async def _settle(self) -> None:
        """Fire on_complete exactly once with accumulated data."""
        if self._settled:
            return
        self._settled = True
        self._release_abort_handle()
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        token_details = self._accumulator.finalize()
        try:
            await self._on_complete(token_details, elapsed_ms)
        except Exception as cb_exc:
            # Structural-only log of the CALLBACK failure; never a traceback (fix [D]).
            logger.warning(
                "on_complete raised during async stream settlement; suppressing (%s)",
                type(cb_exc).__name__,
            )

    async def _settle_error(self, exc: BaseException) -> None:
        """Fire on_error exactly once. Mirrors _settle() for the error path."""
        if self._settled:
            return
        self._settled = True
        self._release_abort_handle()
        try:
            await self._on_error(exc)
        except Exception as cb_exc:
            # Structural-only log; never a traceback (which holds provider ``exc``
            # during error settlement; fix [D]).
            logger.warning(
                "on_error raised during async stream settlement; suppressing (%s)",
                type(cb_exc).__name__,
            )

    def _release_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()

    async def __anext__(self) -> Any:
        """Return one observed caller-dialect chunk."""
        if self._abort_exc is not None:
            raise self._abort_exc
        while not self._pending_chunks:
            try:
                iterator = self._iterator
                if iterator is None:
                    iterator = self._stream.__aiter__()
                    self._iterator = iterator
                chunk = await iterator.__anext__()
                if self._abort_check is not None:
                    abort_exc = self._abort_check()
                    if abort_exc is not None:
                        if self._abort_exc is None:
                            self._abort_exc = abort_exc
                        abort_exc = self._abort_exc
                        try:
                            await self.close()
                        finally:
                            raise abort_exc
                self._accumulator.observe(chunk)
                if self._chunk_translator is None:
                    return chunk
                self._pending_chunks.extend(self._chunk_translator(chunk))
            except StopAsyncIteration:
                await self._settle()
                raise
            except Exception as exc:
                if self._abort_exc is exc:
                    raise
                await self._settle_error(exc)
                raise
        return self._pending_chunks.popleft()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def close(self) -> None:
        """Settle the budget reservation with whatever data we have, then
        forward cleanup to the inner stream's aclose() or close() method.

        Safe to call multiple times — only the first call fires on_complete.
        Forwarding to the inner stream is also safe if called multiple times;
        well-behaved stream implementations are idempotent.
        """
        async with self._close_lock:
            if self._cleanup_complete:
                return
            await self._settle()
            if hasattr(self._stream, "aclose"):
                await self._stream.aclose()
            elif hasattr(self._stream, "close"):
                close_result = self._stream.close()
                if inspect.isawaitable(close_result):
                    await close_result
            self._cleanup_complete = True

    async def aclose(self) -> None:
        """Support the async-iterator cleanup protocol."""
        await self.close()

    async def __aenter__(self) -> Self:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        try:
            # self.close() now settles AND forwards to inner aclose()/close().
            # Broad except is intentional: callback/provider-close failures
            # must not mask the application exception propagating through
            # __aexit__ (if any).
            await self.close()
        except Exception as cb_exc:
            # Structural-only: log the suppressed callback class, never a traceback
            # (fix [D]).
            logger.warning(
                "on_complete raised during __aexit__; suppressing (%s)", type(cb_exc).__name__
            )
        if hasattr(self._stream, "__aexit__"):
            return cast("bool | None", await self._stream.__aexit__(*args))
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _SyncResponsesEnteredStreamWrapper:
    """Add SDK Responses helpers without widening every stream's surface."""

    def __init__(self, stream: SyncStreamWrapper) -> None:
        self._stream = stream

    def __next__(self) -> Any:
        return next(self._stream)

    def __iter__(self) -> Iterator[Any]:
        return self

    def until_done(self) -> Self:
        """Consume through Solwyn's observer and preserve the SDK helper shape."""
        for _event in self:
            pass
        return self

    def get_final_response(self) -> Any:
        """Return the SDK result after every event has passed through the observer."""
        self.until_done()
        return self._stream.get_final_response()

    def __enter__(self) -> Self:
        self._stream.__enter__()
        return self

    def __exit__(self, *args: object) -> bool | None:
        return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncResponsesEnteredStreamWrapper:
    """Async Responses-only helper surface around the generic observer."""

    def __init__(self, stream: AsyncStreamWrapper) -> None:
        self._stream = stream

    async def __anext__(self) -> Any:
        return await anext(self._stream)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def until_done(self) -> Self:
        """Consume through Solwyn's observer and preserve the SDK helper shape."""
        async for _event in self:
            pass
        return self

    async def get_final_response(self) -> Any:
        """Return the SDK result after every event has passed through the observer."""
        await self.until_done()
        return await self._stream.get_final_response()

    async def __aenter__(self) -> Self:
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        return await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _SyncResponsesStreamManagerWrapper:
    """Wrap an SDK Responses manager while preserving its context lifecycle.

    The SDK manager opens the provider stream only in ``__enter__``. Solwyn
    therefore wraps the returned inner stream at that point, allowing the
    ordinary ``SyncStreamWrapper`` accumulator and settlement callbacks to
    observe terminal Responses events. Closing before entry dispatched no
    provider request at all, so it releases the reservation without settling
    or reporting; a provider entry failure takes the classified entry-error
    path, and any failure after the provider stream opens takes the
    established stream-error path.
    """

    def __init__(
        self,
        manager: Any,
        wrap_stream: Callable[[Any], SyncStreamWrapper],
        *,
        on_error: Callable[[BaseException], None],
        on_entry_error: Callable[[BaseException], None],
        on_abandoned_before_entry: Callable[[], None],
        abort_release: Callable[[], None] | None = None,
    ) -> None:
        self._manager = manager
        self._wrap_stream = wrap_stream
        self._on_error = on_error
        self._on_entry_error = on_entry_error
        self._on_abandoned_before_entry = on_abandoned_before_entry
        self._abort_finalizer = (
            weakref.finalize(self, abort_release) if abort_release is not None else None
        )
        self._stream: SyncStreamWrapper | None = None
        self._pre_entry_settled = False
        self._dispatched = False
        self._state = "new"
        self._lock = threading.RLock()

    def _wrapped_stream(self, source: Any = ()) -> SyncStreamWrapper:
        if self._stream is None:
            self._stream = self._wrap_stream(source)
            self._detach_abort_handle()
        return self._stream

    def _release_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()

    def _detach_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()

    def _settle_before_stream(
        self,
        exc: BaseException,
        handler: Callable[[BaseException], None],
    ) -> None:
        """Run at most ONE pre-entry reconciliation for this manager."""
        if self._stream is not None:
            self._stream._settle_error(exc)
            return
        if self._pre_entry_settled:
            return
        self._pre_entry_settled = True
        try:
            handler(exc)
        except BaseException as settlement_exc:
            logger.warning(
                "Responses entry settlement raised; suppressing (%s)",
                type(settlement_exc).__name__,
            )
        finally:
            self._release_abort_handle()

    def _settle_entry_error(self, exc: BaseException) -> None:
        """The provider request failed in ``__enter__``: classify it."""
        self._settle_before_stream(exc, self._on_entry_error)

    def _settle_wrap_error(self, exc: BaseException) -> None:
        """The provider stream opened; wrapping failed after establishment."""
        self._settle_before_stream(exc, self._on_error)

    def _release_before_entry(self) -> None:
        """Give the reservation back for a manager that never dispatched."""
        if self._pre_entry_settled:
            return
        self._pre_entry_settled = True
        try:
            self._on_abandoned_before_entry()
        finally:
            self._release_abort_handle()

    def _close_failed_entry(self, exc: BaseException) -> None:
        try:
            self._manager.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException as close_exc:
            logger.warning(
                "Responses manager close raised after entry failed; suppressing (%s)",
                type(close_exc).__name__,
            )

    def __enter__(self) -> _SyncResponsesEnteredStreamWrapper:
        # The lock spans provider entry and wrapping. A concurrent close must
        # never cache an empty abandonment wrapper while the real stream opens.
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("Responses stream manager is closed")
            if self._state != "new":
                raise RuntimeError("Responses stream manager is already entered")
            self._state = "entering"
            try:
                source = self._manager.__enter__()
            except BaseException as exc:
                self._state = "closed"
                self._settle_entry_error(exc)
                self._close_failed_entry(exc)
                raise
            self._dispatched = True

            try:
                stream = self._wrapped_stream(source)
                entered_stream = _SyncResponsesEnteredStreamWrapper(stream)
            except BaseException as exc:
                self._state = "closed"
                self._settle_wrap_error(exc)
                self._close_failed_entry(exc)
                raise

            self._state = "entered"
            return entered_stream

    def _finish(self, args: tuple[object, ...]) -> bool | None:
        with self._lock:
            if self._state == "closed":
                return False
            self._state = "closed"
            original_exception = bool(args and args[0] is not None)

            settlement_error: BaseException | None = None
            try:
                if self._dispatched:
                    self._wrapped_stream()._settle()
                else:
                    self._release_before_entry()
            except BaseException as exc:
                settlement_error = exc

            try:
                result = cast("bool | None", self._manager.__exit__(*args))
            except BaseException as exc:
                if original_exception:
                    logger.warning(
                        "Responses manager close raised during __exit__; suppressing (%s)",
                        type(exc).__name__,
                    )
                    result = False
                else:
                    raise

            if settlement_error is not None:
                if original_exception:
                    logger.warning(
                        "Responses stream settlement raised during __exit__; suppressing (%s)",
                        type(settlement_error).__name__,
                    )
                else:
                    raise settlement_error
            return result

    def close(self) -> None:
        """Reconcile once (settle, or release if never entered) and close."""
        self._finish((None, None, None))

    def __exit__(self, *args: object) -> bool | None:
        return self._finish(args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _AsyncResponsesStreamManagerWrapper:
    """Async counterpart of ``_SyncResponsesStreamManagerWrapper``."""

    def __init__(
        self,
        manager: Any,
        wrap_stream: Callable[[Any], AsyncStreamWrapper],
        *,
        on_error: Callable[[BaseException], Awaitable[None]],
        on_entry_error: Callable[[BaseException], Awaitable[None]],
        on_abandoned_before_entry: Callable[[], Awaitable[None]],
        abort_release: Callable[[], None] | None = None,
    ) -> None:
        self._manager = manager
        self._wrap_stream = wrap_stream
        self._on_error = on_error
        self._on_entry_error = on_entry_error
        self._on_abandoned_before_entry = on_abandoned_before_entry
        self._abort_finalizer = (
            weakref.finalize(self, abort_release) if abort_release is not None else None
        )
        self._stream: AsyncStreamWrapper | None = None
        self._pre_entry_settled = False
        self._dispatched = False
        self._state = "new"
        self._lock = asyncio.Lock()

    def _wrapped_stream(self, source: Any = ()) -> AsyncStreamWrapper:
        if self._stream is None:
            self._stream = self._wrap_stream(source)
            self._detach_abort_handle()
        return self._stream

    def _release_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()

    def _detach_abort_handle(self) -> None:
        finalizer = self._abort_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()

    @property
    def _cleanup_complete(self) -> bool:
        return self._state == "closed"

    async def _settle_before_stream(
        self,
        exc: BaseException,
        handler: Callable[[BaseException], Awaitable[None]],
    ) -> None:
        """Run at most ONE pre-entry reconciliation for this manager."""
        if self._stream is not None:
            await self._stream._settle_error(exc)
            return
        if self._pre_entry_settled:
            return
        self._pre_entry_settled = True
        try:
            await handler(exc)
        except BaseException as settlement_exc:
            logger.warning(
                "Responses entry settlement raised; suppressing (%s)",
                type(settlement_exc).__name__,
            )
        finally:
            self._release_abort_handle()

    async def _settle_entry_error(self, exc: BaseException) -> None:
        """The provider request failed in ``__aenter__``: classify it."""
        await self._settle_before_stream(exc, self._on_entry_error)

    async def _settle_wrap_error(self, exc: BaseException) -> None:
        """The provider stream opened; wrapping failed after establishment."""
        await self._settle_before_stream(exc, self._on_error)

    async def _release_before_entry(self) -> None:
        """Give the reservation back for a manager that never dispatched."""
        if self._pre_entry_settled:
            return
        self._pre_entry_settled = True
        try:
            await self._on_abandoned_before_entry()
        finally:
            self._release_abort_handle()

    async def _close_failed_entry(self, exc: BaseException) -> None:
        try:
            await self._manager.__aexit__(type(exc), exc, exc.__traceback__)
        except BaseException as close_exc:
            logger.warning(
                "Responses manager close raised after entry failed; suppressing (%s)",
                type(close_exc).__name__,
            )

    async def __aenter__(self) -> _AsyncResponsesEnteredStreamWrapper:
        async with self._lock:
            if self._state == "closed":
                raise RuntimeError("Responses stream manager is closed")
            if self._state != "new":
                raise RuntimeError("Responses stream manager is already entered")
            self._state = "entering"
            try:
                source = await self._manager.__aenter__()
            except BaseException as exc:
                self._state = "closed"
                await self._settle_entry_error(exc)
                await self._close_failed_entry(exc)
                raise
            self._dispatched = True

            try:
                stream = self._wrapped_stream(source)
                entered_stream = _AsyncResponsesEnteredStreamWrapper(stream)
            except BaseException as exc:
                self._state = "closed"
                await self._settle_wrap_error(exc)
                await self._close_failed_entry(exc)
                raise

            self._state = "entered"
            return entered_stream

    async def _finish(self, args: tuple[object, ...]) -> bool | None:
        async with self._lock:
            if self._state == "closed":
                return False
            # Cancellation releases the lock. Keep cleanup retryable until the
            # provider manager has actually finished closing its connection.
            self._state = "closing"
            original_exception = bool(args and args[0] is not None)

            settlement_error: BaseException | None = None
            try:
                if self._dispatched:
                    await self._wrapped_stream()._settle()
                else:
                    await self._release_before_entry()
            except BaseException as exc:
                settlement_error = exc

            try:
                result = cast("bool | None", await self._manager.__aexit__(*args))
            except BaseException as exc:
                if original_exception:
                    logger.warning(
                        "Responses manager close raised during __aexit__; suppressing (%s)",
                        type(exc).__name__,
                    )
                    result = False
                else:
                    raise
            else:
                self._state = "closed"

            if settlement_error is not None:
                if original_exception:
                    logger.warning(
                        "Responses stream settlement raised during __aexit__; suppressing (%s)",
                        type(settlement_error).__name__,
                    )
                else:
                    raise settlement_error
            return result

    async def close(self) -> None:
        """Reconcile once (settle, or release if never entered) and close."""
        await self._finish((None, None, None))

    async def __aexit__(self, *args: object) -> bool | None:
        return await self._finish(args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _DeferredAsyncResponsesStreamManagerWrapper:
    """Keep ``AsyncResponses.stream()`` synchronous while awaiting Solwyn I/O.

    OpenAI's async resource returns an async context manager directly; it is
    not itself an async method. Solwyn's budget check is asynchronous, so the
    public proxy returns this small bridge and resolves the pipeline-produced
    manager wrapper on first entry or explicit close.
    """

    def __init__(
        self,
        manager_factory: Callable[[], Awaitable[_AsyncResponsesStreamManagerWrapper]],
    ) -> None:
        self._manager_factory: (
            Callable[[], Awaitable[_AsyncResponsesStreamManagerWrapper]] | None
        ) = manager_factory
        self._manager: _AsyncResponsesStreamManagerWrapper | None = None
        self._lock = asyncio.Lock()
        self._state = "new"

    async def _resolve_locked(self) -> _AsyncResponsesStreamManagerWrapper:
        if self._manager is None:
            manager_factory = self._manager_factory
            if manager_factory is None:
                raise RuntimeError("Responses stream manager resolution lost its factory")
            self._manager = await manager_factory()
            self._manager_factory = None
        return self._manager

    async def __aenter__(self) -> _AsyncResponsesEnteredStreamWrapper:
        async with self._lock:
            if self._state == "closed":
                raise RuntimeError("Responses stream manager is closed")
            if self._state != "new":
                raise RuntimeError("Responses stream manager is already entered")
            manager = await self._resolve_locked()
            try:
                stream = await manager.__aenter__()
            except BaseException:
                self._state = "closed"
                raise
            self._state = "entered"
            return stream

    async def _finish_resolved(self, cleanup: Awaitable[Any]) -> Any:
        manager = self._manager
        if manager is None:
            raise RuntimeError("Responses stream manager resolution was lost")
        self._state = "closing"
        try:
            result = await cleanup
        except BaseException:
            if manager._cleanup_complete:
                self._state = "closed"
            raise
        self._state = "closed" if manager._cleanup_complete else "closing"
        return result

    async def __aexit__(self, *args: object) -> bool | None:
        async with self._lock:
            if self._state == "closed":
                return False
            if self._manager is None:
                self._state = "closed"
                self._manager_factory = None
                return False
            return cast(
                "bool | None",
                await self._finish_resolved(self._manager.__aexit__(*args)),
            )

    async def close(self) -> None:
        async with self._lock:
            if self._state == "closed":
                return
            if self._manager is None:
                self._state = "closed"
                self._manager_factory = None
                return
            await self._finish_resolved(self._manager.close())
