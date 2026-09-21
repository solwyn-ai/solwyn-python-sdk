"""Run-control ownership across actual client and stream-wrapper lifecycles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest

import solwyn
from solwyn import _run_control
from solwyn._run_control import clear_run_termination, mark_terminated, run_termination
from solwyn.exceptions import RunStoppedError
from solwyn.testing import FakeControlPlane


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()
    yield
    with _run_control._STATE.lock:
        _run_control._STATE._clear_for_test_locked()


class _ProviderStream:
    """Content-free provider stream with an observable cleanup boundary."""

    def __init__(self) -> None:
        self.error: BaseException | None = None
        self.close_calls = 0
        self.chunk = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1), choices=[]
        )

    def __iter__(self) -> _ProviderStream:
        return self

    def __next__(self) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return self.chunk

    def __aiter__(self) -> _ProviderStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        return next(self)

    def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.close()


class _Completions:
    def __init__(self, provider: OpenAI) -> None:
        self.provider = provider

    def create(self, **_kwargs: object) -> _ProviderStream:
        return self.provider.dispatch()


class _AsyncCompletions:
    def __init__(self, provider: OpenAI) -> None:
        self.provider = provider

    async def create(self, **_kwargs: object) -> _ProviderStream:
        return self.provider.dispatch()


class OpenAI:
    __module__ = "openai._client"

    def __init__(self) -> None:
        self.streams: list[_ProviderStream] = []
        self.dispatch: Callable[[], _ProviderStream] = self.new_stream
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.close_calls = 0

    def new_stream(self) -> _ProviderStream:
        stream = _ProviderStream()
        self.streams.append(stream)
        return stream

    def with_options(self, **_kwargs: object) -> OpenAI:
        return self

    def close(self) -> None:
        self.close_calls += 1


class AsyncOpenAI(OpenAI):
    __module__ = "openai._client"

    def __init__(self) -> None:
        super().__init__()
        self.chat = SimpleNamespace(completions=_AsyncCompletions(self))


def _stop_current_generation_and_evict(run_id: str) -> None:
    # A live server ALLOW must not hide stale current-generation ownership by
    # clearing a server stop before the next stream acquires its handle.
    mark_terminated(run_id, reason="current_stop", source="local_velocity")
    for index in range(_run_control._MAX_TERMINATED_RUNS + 1):
        mark_terminated(f"eviction-{index}", reason="unrelated_stop", source="server")
    assert run_termination(run_id) is None


@pytest.mark.unit
@pytest.mark.parametrize("disposition", ["close", "provider_error", "dispatch_error"])
def test_sync_last_current_owner_leaves_old_stream_stopped_through_shutdown(
    disposition: str,
) -> None:
    # Arrange an obsolete stopped stream that outlives the current generation.
    provider = OpenAI()
    plane = FakeControlPlane()
    with plane.wrap(provider) as client:
        with solwyn.run("sync-generation-lifetime") as run_id:
            old = client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            old_source = provider.streams[-1]
            mark_terminated(run_id, reason="original_stop", source="server")
            clear_run_termination(run_id)

            # Act: retire the last current owner after its stop leaves the LRU.
            if disposition == "dispatch_error":

                def failed_dispatch() -> _ProviderStream:
                    _stop_current_generation_and_evict(run_id)
                    raise RuntimeError("synthetic dispatch failure")

                provider.dispatch = failed_dispatch
                with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
                    client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
                provider.dispatch = provider.new_stream
            else:
                current = client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
                current_source = provider.streams[-1]
                _stop_current_generation_and_evict(run_id)
                if disposition == "provider_error":
                    current_source.error = RuntimeError("synthetic stream failure")
                    with pytest.raises(RuntimeError, match="synthetic stream failure"):
                        next(current)
                    assert len(_run_control._STATE.active_handles[run_id]) == 1
                current.close()

            # The obsolete owner cannot keep the current stop alive or lose its
            # own original stop when a new stream acquires the same run id.
            assert len(_run_control._STATE.active_handles[run_id]) == 1
            fresh = client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            assert next(fresh) is provider.streams[-1].chunk
            fresh.close()

        # Finishing a run and shutting down the client do not retire a stream
        # still owned by its caller.
        assert len(_run_control._STATE.active_handles[run_id]) == 1
    assert provider.close_calls == 1
    assert old_source.close_calls == 0
    assert len(_run_control._STATE.active_handles[run_id]) == 1

    with pytest.raises(RunStoppedError) as stopped:
        next(old)
    assert (stopped.value.reason, stopped.value.source) == ("original_stop", "server")
    assert old_source.close_calls == 1
    assert run_id not in _run_control._STATE.active_handles


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["close", "stream_cancel", "dispatch_cancel"])
async def test_async_last_current_owner_leaves_old_stream_stopped_through_shutdown(
    disposition: str,
) -> None:
    # Arrange the same lifetime overlap on the public async client path.
    provider = AsyncOpenAI()
    plane = FakeControlPlane()
    async with plane.wrap_async(provider) as client:
        async with solwyn.run("async-generation-lifetime") as run_id:
            old = await client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            old_source = provider.streams[-1]
            mark_terminated(run_id, reason="original_stop", source="server")
            clear_run_termination(run_id)

            # Act: cancellation must release only the current generation.
            if disposition == "dispatch_cancel":

                def cancelled_dispatch() -> _ProviderStream:
                    _stop_current_generation_and_evict(run_id)
                    raise asyncio.CancelledError

                provider.dispatch = cancelled_dispatch
                with pytest.raises(asyncio.CancelledError):
                    await client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
                provider.dispatch = provider.new_stream
            else:
                current = await client.chat.completions.create(
                    model="gpt-5.5", messages=[], stream=True
                )
                current_source = provider.streams[-1]
                _stop_current_generation_and_evict(run_id)
                if disposition == "stream_cancel":
                    current_source.error = asyncio.CancelledError()
                    with pytest.raises(asyncio.CancelledError):
                        await anext(current)
                    assert len(_run_control._STATE.active_handles[run_id]) == 1
                await current.aclose()

            # Assert a fresh stream starts clean while the old one stays live.
            assert len(_run_control._STATE.active_handles[run_id]) == 1
            fresh = await client.chat.completions.create(model="gpt-5.5", messages=[], stream=True)
            assert await anext(fresh) is provider.streams[-1].chunk
            await fresh.aclose()

        assert len(_run_control._STATE.active_handles[run_id]) == 1
    assert provider.close_calls == 1
    assert old_source.close_calls == 0
    assert len(_run_control._STATE.active_handles[run_id]) == 1

    with pytest.raises(RunStoppedError) as stopped:
        await anext(old)
    assert (stopped.value.reason, stopped.value.source) == ("original_stop", "server")
    assert old_source.close_calls == 1
    assert run_id not in _run_control._STATE.active_handles
