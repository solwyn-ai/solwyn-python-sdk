"""End-to-end mid-stream run aborts through the public client wrappers.

Covers the whole chain for a run terminated while its stream is being
consumed: the typed stop error, the provider stream's close, exactly one
partial settlement, and the released active-run watcher.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY

import solwyn as solwyn_pkg
from solwyn import _run_control
from solwyn._run import current_run
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import RunStoppedError


def _mock_openai_client() -> MagicMock:
    """A mock that adapter detection treats as ``openai.OpenAI()``."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    return client


def _stream_chunks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        )
        for text in ("one", "two", "three", "four")
    ]


class _FakeSyncProviderStream:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self.close_calls = 0

    def __iter__(self):
        yield from self._items

    def close(self) -> None:
        self.close_calls += 1


class _FakeAsyncProviderStream:
    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)
        self.aclose_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _allow_budget_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    # httpx.Response.json() is sync even on the async client.
    response.json = MagicMock(return_value=ALLOW_BUDGET_RESPONSE)
    response.raise_for_status = MagicMock()
    return response


def _make_solwyn(client: MagicMock) -> Solwyn:
    """A Solwyn wrapper whose settlement lands in the reporter's report()."""
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, api_key=VALID_API_KEY)

    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)

    def report_settlement(_request, event):
        solwyn._solwyn_reporter.report(event)

    solwyn._solwyn_reporter.report_settlement = report_settlement
    return solwyn


def _make_async_solwyn(client: MagicMock) -> AsyncSolwyn:
    solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)

    def report_settlement(_request, event):
        solwyn._solwyn_reporter.report(event)

    solwyn._solwyn_reporter.report_settlement = report_settlement
    return solwyn


@pytest.mark.unit
def test_sync_mid_stream_termination_stops_settles_and_releases_watcher() -> None:
    client = _mock_openai_client()
    chunks = _stream_chunks()
    provider_stream = _FakeSyncProviderStream(chunks)
    client.chat.completions.create.return_value = provider_stream

    solwyn = _make_solwyn(client)
    reported: list = []
    solwyn._solwyn_reporter.report = reported.append

    with (
        patch.object(solwyn._solwyn_budget._http, "post", return_value=_allow_budget_response()),
        solwyn_pkg.run("e2e-sync-mid-stream-abort"),
    ):
        run_id, _ = current_run()
        assert run_id is not None
        stream = solwyn.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        assert next(stream) is chunks[0]
        assert reported == []

        _run_control.mark_terminated(
            run_id,
            reason="velocity:test-e2e",
            source="local_velocity",
        )

        with pytest.raises(RunStoppedError) as exc_info:
            next(stream)

    assert exc_info.value.agent_run_id == run_id
    assert exc_info.value.reason == "velocity:test-e2e"
    assert exc_info.value.source == "local_velocity"
    assert provider_stream.close_calls == 1
    assert len(reported) == 1
    assert reported[0].agent_run_id == run_id
    assert run_id not in _run_control._STATE.active_handles

    solwyn.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_mid_stream_termination_stops_settles_and_releases_watcher() -> None:
    client = _mock_openai_client()
    chunks = _stream_chunks()
    provider_stream = _FakeAsyncProviderStream(chunks)
    client.chat.completions.create = AsyncMock(return_value=provider_stream)

    solwyn = _make_async_solwyn(client)
    reported: list = []
    solwyn._solwyn_reporter.report = reported.append

    with patch.object(solwyn._solwyn_budget._http, "post", return_value=_allow_budget_response()):
        async with solwyn_pkg.run("e2e-async-mid-stream-abort"):
            run_id, _ = current_run()
            assert run_id is not None
            stream = await solwyn.chat.completions.create(
                model="gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            assert await anext(stream) is chunks[0]
            assert reported == []

            _run_control.mark_terminated(
                run_id,
                reason="velocity:test-e2e",
                source="local_velocity",
            )

            with pytest.raises(RunStoppedError) as exc_info:
                await anext(stream)

    assert exc_info.value.agent_run_id == run_id
    assert exc_info.value.reason == "velocity:test-e2e"
    assert exc_info.value.source == "local_velocity"
    assert provider_stream.aclose_calls == 1
    assert len(reported) == 1
    assert reported[0].agent_run_id == run_id
    assert run_id not in _run_control._STATE.active_handles

    await solwyn._solwyn_budget._http.aclose()
    await solwyn._solwyn_reporter._http.aclose()
