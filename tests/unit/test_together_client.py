"""Client-level coverage for native Together sync and async clients.

The fakes in this module deliberately expose only the native Together chat
completions surface.  They exercise adapter detection, budget pre-flight,
dispatch, usage extraction, and success reporting without importing the
provider SDK.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn import _base
from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._types import CallStatus, ProviderName
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.providers.together import TogetherAdapter

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# Together's adapter-DECLARED untracked spend surfaces: provider-specific extras
# the central per-dialect map does not cover. audio/videos moved to the central
# openai map; embeddings/images (intercepted) and batches/fine_tuning are no
# longer warned.
DECLARED_UNMETERED_SURFACES = frozenset({"completions", "rerank", "code_interpreter", "evals"})
# Central openai-dialect surfaces every openai-dialect provider (Together
# included) warns for, sourced from _base._UNSHIPPED_SPEND_SURFACES["openai"].
# images are intercepted, so only audio/videos remain.
CENTRAL_OPENAI_SURFACES = frozenset({"audio", "videos"})
# Every surface that warns-once for a Together (openai-dialect) client.
WARNING_SURFACES = DECLARED_UNMETERED_SURFACES | CENTRAL_OPENAI_SURFACES
# Together-billed surfaces Solwyn passes through SILENTLY (truly-unrelated
# resources, not warned). embeddings and images are NOT here:
# they are intercepted, so they return the media proxy rather than passing through.
SILENT_SURFACES = frozenset({"batches", "fine_tuning"})


@pytest.fixture(autouse=True)
def _reset_spend_surface_latch() -> None:
    """Reset the per-process warn-once latch so warn tests stay order-independent."""
    _reset_unmetered_spend_warnings()


class _YieldingSet(set[str]):
    """Yield after an absent membership check to expose check/insert races."""

    def __contains__(self, item: object) -> bool:
        present = super().__contains__(item)
        if not present:
            time.sleep(0.05)
        return present


@pytest.mark.unit
def test_together_adapter_declares_unmetered_spend_surfaces() -> None:
    # Provider-specific extras only; audio/videos ride the central openai dialect
    # map, and embeddings/images (intercepted) + batches/fine_tuning are not warned.
    assert TogetherAdapter().unmetered_spend_surfaces == DECLARED_UNMETERED_SURFACES


def _completion_response() -> SimpleNamespace:
    """Return a Together v2 response with the flat cached-token shape."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=13,
            completion_tokens=5,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            cached_tokens=4,
        ),
    )


def _text_chunk(length: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="x" * length),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _usage_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=8,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            cached_tokens=6,
        ),
        service_tier="priority",
    )


class _TogetherCompletions:
    def __init__(self, response: object, order_log: list[str] | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self._order_log = order_log

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> object:
        if self._order_log is not None:
            self._order_log.append("dispatch")
        self.calls.append({"model": model, "messages_identity": id(messages), "stream": stream})
        return self.response


class Together:
    """Duck-typed native Together client; no provider dependency required."""

    __module__ = "together"

    def __init__(self, response: object, order_log: list[str] | None = None) -> None:
        self.completions = _TogetherCompletions(response, order_log)
        self.chat = SimpleNamespace(completions=self.completions)


FakeTogetherClient = Together


class _AsyncTogetherCompletions:
    def __init__(self, response: object, order_log: list[str] | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self._order_log = order_log

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> object:
        if self._order_log is not None:
            self._order_log.append("dispatch")
        self.calls.append({"model": model, "messages_identity": id(messages), "stream": stream})
        return self.response


class AsyncTogether:
    """Duck-typed native async Together client."""

    __module__ = "together"

    def __init__(self, response: object, order_log: list[str] | None = None) -> None:
        self.completions = _AsyncTogetherCompletions(response, order_log)
        self.chat = SimpleNamespace(completions=self.completions)


class _SyncResource:
    def create(self, **_kwargs: object) -> object:
        raise NotImplementedError


class _AsyncResource:
    async def create(self, **_kwargs: object) -> object:
        raise NotImplementedError


def _async_iter(chunks: list[object]) -> Any:
    async def _gen() -> Any:
        for chunk in chunks:
            yield chunk

    return _gen()


def _allow_budget(*, reservation_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(allowed=True, reservation_id=reservation_id, price_hints=None)


def _make_solwyn(client: object) -> Solwyn:
    with patch("solwyn.reporter.MetadataReporter._flush_loop", autospec=True):
        solwyn = Solwyn(client, api_key=VALID_API_KEY)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


def _make_async_solwyn(client: object) -> AsyncSolwyn:
    return AsyncSolwyn(client, api_key=VALID_API_KEY)


@pytest.mark.unit
def test_sync_unmetered_surface_warns_and_passes_through(caplog: pytest.LogCaptureFixture) -> None:
    # rerank is a Together-declared untracked surface: warn-once, pass through.
    client = FakeTogetherClient(_completion_response())
    resource = MagicMock(spec=_SyncResource)
    resource.create.return_value = object()
    client.rerank = resource
    solwyn = _make_solwyn(client)
    check_budget = MagicMock(spec=solwyn._budget.check_budget)
    report = MagicMock(spec=solwyn._reporter.report)
    report_settlement = MagicMock(spec=solwyn._reporter.report_settlement)

    with (
        caplog.at_level(logging.WARNING, logger="solwyn._base"),
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=report),
        patch.object(solwyn._reporter, "report_settlement", new=report_settlement),
    ):
        result = solwyn.rerank.create(model=MODEL, input="private-input-marker")

    assert result is resource.create.return_value
    assert resource.create.call_count == 1
    check_budget.assert_not_called()
    report.assert_not_called()
    report_settlement.assert_not_called()
    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert "together" in warning
    assert "rerank" in warning
    assert "no budget check" in warning
    assert "no cost event" in warning
    assert "tracking for this surface is coming" in warning.lower()
    assert "private-input-marker" not in warning
    solwyn.close()


@pytest.mark.unit
def test_sync_curated_silent_surfaces_pass_through_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # batches and fine_tuning (truly-unrelated resources) were dropped from
    # Together's warn set: they pass through SILENTLY. embeddings and
    # images are intercepted -> they return the media proxy, not the raw
    # attribute, and are silent too.
    client = FakeTogetherClient(_completion_response())
    for surface in SILENT_SURFACES:
        setattr(client, surface, object())
    client.embeddings = object()
    client.images = object()
    solwyn = _make_solwyn(client)

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        for surface in SILENT_SURFACES:
            assert getattr(solwyn, surface) is getattr(client, surface)
        # embeddings + images are intercepted, not passed through to the raw client.
        assert solwyn.embeddings is not client.embeddings
        assert solwyn.images is not client.images

    assert caplog.records == []
    solwyn.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_unmetered_surface_warns_and_passes_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # audio is a still-unshipped central openai surface: warn-once, pass
    # through. (images are intercepted and no longer warn.)
    client = AsyncTogether(_completion_response())
    resource = MagicMock(spec=_AsyncResource)
    resource.create.return_value = object()
    client.audio = resource
    solwyn = _make_async_solwyn(client)
    check_budget = AsyncMock(spec=solwyn._budget.check_budget)
    report = MagicMock(spec=solwyn._reporter.report)
    report_settlement = MagicMock(spec=solwyn._reporter.report_settlement)

    with (
        caplog.at_level(logging.WARNING, logger="solwyn._base"),
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=report),
        patch.object(solwyn._reporter, "report_settlement", new=report_settlement),
    ):
        result = await solwyn.audio.create(model=MODEL, prompt="private-input-marker")

    assert result is resource.create.return_value
    resource.create.assert_awaited_once()
    check_budget.assert_not_awaited()
    report.assert_not_called()
    report_settlement.assert_not_called()
    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert "together" in warning
    assert "audio" in warning
    assert "no budget check" in warning
    assert "no cost event" in warning
    assert "tracking for this surface is coming" in warning.lower()
    assert "private-input-marker" not in warning
    await solwyn.close()


@pytest.mark.unit
def test_sync_warns_once_for_each_warning_surface(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A Together (openai-dialect) client warns for the union of the central
    # openai map and its own declared surfaces; each warns exactly once per
    # process no matter how many times it is accessed.
    client = FakeTogetherClient(_completion_response())
    resources: dict[str, MagicMock] = {}
    for surface in WARNING_SURFACES:
        resource = MagicMock(spec=_SyncResource)
        resource.create.return_value = surface
        setattr(client, surface, resource)
        resources[surface] = resource
    solwyn = _make_solwyn(client)

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        for surface, resource in resources.items():
            first = getattr(solwyn, surface)
            second = getattr(solwyn, surface)
            assert first is resource
            assert second is resource
            assert first.create() == surface
            assert second.create() == surface

    assert len(caplog.records) == len(WARNING_SURFACES)
    warning_messages = [record.getMessage() for record in caplog.records]
    for surface in WARNING_SURFACES:
        assert sum(f"surface '{surface}'" in message for message in warning_messages) == 1
        assert resources[surface].create.call_count == 2
    solwyn.close()


@pytest.mark.unit
def test_sync_concurrent_unmetered_surface_access_warns_exactly_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_count = 8
    client = FakeTogetherClient(_completion_response())
    resource = object()
    client.audio = resource
    solwyn = _make_solwyn(client)
    # Swap the module-level latch for one that yields on an absent membership
    # check, exposing any check-then-insert race; the surrounding lock must still
    # serialize so exactly one thread warns. monkeypatch restores the original.
    monkeypatch.setattr(_base, "_warned_spend_surfaces", _YieldingSet())
    start = threading.Barrier(worker_count)

    def access_surface(_: int) -> object:
        start.wait(timeout=5)
        return solwyn.audio

    with (
        caplog.at_level(logging.WARNING, logger="solwyn._base"),
        ThreadPoolExecutor(max_workers=worker_count) as executor,
    ):
        results = list(executor.map(access_surface, range(worker_count)))

    assert all(result is resource for result in results)
    assert len(caplog.records) == 1
    assert "surface 'audio'" in caplog.records[0].getMessage()
    solwyn.close()


@pytest.mark.unit
def test_missing_unmetered_surface_does_not_warn_or_consume_latch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeTogetherClient(_completion_response())
    solwyn = _make_solwyn(client)

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        with pytest.raises(AttributeError):
            _ = solwyn.audio
        assert caplog.records == []

        resource = object()
        client.audio = resource
        assert solwyn.audio is resource

    assert len(caplog.records) == 1
    assert "surface 'audio'" in caplog.records[0].getMessage()
    solwyn.close()


@pytest.mark.unit
def test_unmetered_surface_warning_latch_is_per_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Per-PROCESS, not per-instance: a second client (even a fresh wrapper) does
    # NOT re-warn a surface already warned this process.
    client = FakeTogetherClient(_completion_response())
    resource = object()
    client.audio = resource
    first = _make_solwyn(client)
    second = _make_solwyn(client)

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert first.audio is resource
        assert first.audio is resource
        assert second.audio is resource

    assert len(caplog.records) == 1
    assert "surface 'audio'" in caplog.records[0].getMessage()
    first.close()
    second.close()


@pytest.mark.unit
def test_unrelated_surface_passes_through_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A surface that is neither in the central openai map nor any adapter's
    # declared set (moderations) passes through with no warning.
    OpenAI = type("OpenAI", (), {"__module__": "openai._client"})
    client = OpenAI()
    resource = object()
    client.moderations = resource
    solwyn = _make_solwyn(client)

    with caplog.at_level(logging.WARNING, logger="solwyn._base"):
        assert solwyn.moderations is resource

    assert caplog.records == []
    solwyn.close()


def _capture_settlements(
    target: list[tuple[Any, Any]],
) -> Callable[[Any, Any], None]:
    def capture(request: Any, event: Any) -> None:
        target.append((request, event))

    return capture


def _record_budget_check(
    order_log: list[str], response: SimpleNamespace
) -> Callable[..., SimpleNamespace]:
    def check(**_: object) -> SimpleNamespace:
        order_log.append("budget")
        return response

    return check


@pytest.mark.unit
def test_sync_together_runs_budget_dispatch_usage_and_success_pipeline() -> None:
    # Arrange
    order: list[str] = []
    response = _completion_response()
    client = FakeTogetherClient(response, order)
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_solwyn(client)
    reported: list[Any] = []
    check_budget = MagicMock(
        spec=solwyn._budget.check_budget,
        side_effect=_record_budget_check(order, _allow_budget()),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=reported.append),
    ):
        result = solwyn.chat.completions.create(model=MODEL, messages=messages)

    # Assert
    assert result is response
    assert order == ["budget", "dispatch"]
    assert check_budget.call_args.kwargs["provider"] == "together"
    assert check_budget.call_args.kwargs["model"] == MODEL
    assert check_budget.call_args.kwargs["estimated_input_tokens"] > 0
    assert client.completions.calls == [
        {"model": MODEL, "messages_identity": id(messages), "stream": False}
    ]
    assert len(reported) == 1
    event = reported[0]
    assert event.provider == ProviderName.TOGETHER
    assert event.status == CallStatus.SUCCESS
    assert event.input_tokens == 13
    assert event.output_tokens == 5
    assert event.token_details.cached_input_tokens == 4
    assert event.token_details.is_estimated is False
    solwyn.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_together_runs_budget_dispatch_usage_and_success_pipeline() -> None:
    # Arrange
    order: list[str] = []
    response = _completion_response()
    client = AsyncTogether(response, order)
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_async_solwyn(client)
    reported: list[Any] = []
    check_budget = AsyncMock(
        spec=solwyn._budget.check_budget,
        side_effect=_record_budget_check(order, _allow_budget()),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=reported.append),
    ):
        result = await solwyn.chat.completions.create(model=MODEL, messages=messages)

    # Assert
    assert result is response
    assert order == ["budget", "dispatch"]
    assert check_budget.call_args.kwargs["provider"] == "together"
    assert check_budget.call_args.kwargs["model"] == MODEL
    assert check_budget.call_args.kwargs["estimated_input_tokens"] > 0
    assert client.completions.calls == [
        {"model": MODEL, "messages_identity": id(messages), "stream": False}
    ]
    assert len(reported) == 1
    event = reported[0]
    assert event.provider == ProviderName.TOGETHER
    assert event.status == CallStatus.SUCCESS
    assert event.input_tokens == 13
    assert event.output_tokens == 5
    assert event.token_details.cached_input_tokens == 4
    assert event.token_details.is_estimated is False
    await solwyn.close()


@pytest.mark.unit
def test_sync_together_stream_settles_from_terminal_usage() -> None:
    # Arrange
    terminal_usage = _usage_chunk()
    client = FakeTogetherClient(iter([_text_chunk(8), terminal_usage]))
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_solwyn(client)
    reported: list[Any] = []
    settlements: list[tuple[Any, Any]] = []
    check_budget = MagicMock(
        spec=solwyn._budget.check_budget,
        return_value=_allow_budget(reservation_id="res_together_stream"),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=reported.append),
        patch.object(
            solwyn._reporter,
            "report_settlement",
            new=_capture_settlements(settlements),
        ),
    ):
        chunks = list(
            solwyn.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )
        )

    # Assert
    assert chunks[-1] is terminal_usage
    assert client.completions.calls == [
        {"model": MODEL, "messages_identity": id(messages), "stream": True}
    ]
    assert reported == []
    assert len(settlements) == 1
    confirm, event = settlements[0]
    assert confirm.provider == ProviderName.TOGETHER
    assert confirm.service_tier == "priority"
    assert confirm.token_details.input_tokens == 21
    assert confirm.token_details.output_tokens == 8
    assert confirm.token_details.cached_input_tokens == 6
    assert confirm.token_details.is_estimated is False
    assert event.provider == ProviderName.TOGETHER
    assert event.status == CallStatus.SUCCESS
    assert event.service_tier == "priority"
    assert event.token_details == confirm.token_details
    solwyn.close()


@pytest.mark.unit
def test_sync_together_stream_without_usage_settles_flagged_length_estimate() -> None:
    # Arrange
    client = FakeTogetherClient(iter([_text_chunk(24), _text_chunk(16)]))
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_solwyn(client)
    settlements: list[tuple[Any, Any]] = []
    check_budget = MagicMock(
        spec=solwyn._budget.check_budget,
        return_value=_allow_budget(reservation_id="res_together_estimated"),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(
            solwyn._reporter,
            "report_settlement",
            new=_capture_settlements(settlements),
        ),
    ):
        chunks = list(
            solwyn.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )
        )

    # Assert
    assert len(chunks) == 2
    assert len(settlements) == 1
    confirm, event = settlements[0]
    assert confirm.provider == ProviderName.TOGETHER
    assert confirm.token_details.input_tokens > 0
    assert confirm.token_details.output_tokens == 10
    assert confirm.token_details.is_estimated is True
    assert event.provider == ProviderName.TOGETHER
    assert event.status == CallStatus.SUCCESS
    assert event.token_details == confirm.token_details
    solwyn.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_together_stream_settles_from_terminal_usage() -> None:
    # Arrange
    terminal_usage = _usage_chunk()
    client = AsyncTogether(_async_iter([_text_chunk(8), terminal_usage]))
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_async_solwyn(client)
    settlements: list[tuple[Any, Any]] = []
    check_budget = AsyncMock(
        spec=solwyn._budget.check_budget,
        return_value=_allow_budget(reservation_id="res_together_async_stream"),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(
            solwyn._reporter,
            "report_settlement",
            new=_capture_settlements(settlements),
        ),
    ):
        stream = await solwyn.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    # Assert
    assert chunks[-1] is terminal_usage
    assert check_budget.call_args.kwargs["provider"] == "together"
    assert client.completions.calls == [
        {"model": MODEL, "messages_identity": id(messages), "stream": True}
    ]
    assert len(settlements) == 1
    confirm, event = settlements[0]
    assert confirm.provider == ProviderName.TOGETHER
    assert confirm.token_details.cached_input_tokens == 6
    assert confirm.token_details.is_estimated is False
    assert event.provider == ProviderName.TOGETHER
    assert event.status == CallStatus.SUCCESS
    assert event.token_details == confirm.token_details
    await solwyn.close()


@pytest.mark.unit
def test_sync_wrapper_with_async_together_returns_unawaited_coroutine() -> None:
    """Pin the current wrapper/client mispair contract without fixing it.

    Users must pair ``Solwyn`` with ``Together`` and ``AsyncSolwyn`` with
    ``AsyncTogether``.  The sync wrapper currently returns the async client's
    coroutine object; this characterization test closes it to avoid a runtime
    warning.
    """
    # Arrange
    client = AsyncTogether(_completion_response())
    solwyn = _make_solwyn(client)
    reported: list[Any] = []
    check_budget = MagicMock(
        spec=solwyn._budget.check_budget,
        return_value=_allow_budget(),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=reported.append),
    ):
        result = solwyn.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hello"}],
        )

    # Assert
    assert inspect.iscoroutine(result)
    assert client.completions.calls == []
    assert len(reported) == 1
    assert reported[0].provider == ProviderName.TOGETHER
    assert reported[0].status == CallStatus.SUCCESS
    assert reported[0].token_details.is_estimated is True
    result.close()
    solwyn.close()
