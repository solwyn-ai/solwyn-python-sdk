"""Client-level coverage for native Together sync and async clients.

The fakes in this module deliberately expose only the native Together chat
completions surface.  They exercise adapter detection, budget pre-flight,
dispatch, usage extraction, and success reporting without importing the
provider SDK.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._types import CallStatus, ProviderName
from solwyn.client import AsyncSolwyn, Solwyn

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


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
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> object:
        self.calls.append({"model": model, "messages_identity": id(messages), "stream": stream})
        return self.response


class Together:
    """Duck-typed native Together client; no provider dependency required."""

    __module__ = "together"

    def __init__(self, response: object) -> None:
        self.completions = _TogetherCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


FakeTogetherClient = Together


class _AsyncTogetherCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> object:
        self.calls.append({"model": model, "messages_identity": id(messages), "stream": stream})
        return self.response


class AsyncTogether:
    """Duck-typed native async Together client."""

    __module__ = "together"

    def __init__(self, response: object) -> None:
        self.completions = _AsyncTogetherCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


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


def _capture_settlements(
    target: list[tuple[Any, Any]],
) -> Callable[[Any, Any], None]:
    def capture(request: Any, event: Any) -> None:
        target.append((request, event))

    return capture


@pytest.mark.unit
def test_sync_together_runs_budget_dispatch_usage_and_success_pipeline() -> None:
    # Arrange
    response = _completion_response()
    client = FakeTogetherClient(response)
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
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
        result = solwyn.chat.completions.create(model=MODEL, messages=messages)

    # Assert
    assert result is response
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
    response = _completion_response()
    client = AsyncTogether(response)
    messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
    solwyn = _make_async_solwyn(client)
    reported: list[Any] = []
    check_budget = AsyncMock(
        spec=solwyn._budget.check_budget,
        return_value=_allow_budget(),
    )

    # Act
    with (
        patch.object(solwyn._budget, "check_budget", new=check_budget),
        patch.object(solwyn._reporter, "report", new=reported.append),
    ):
        result = await solwyn.chat.completions.create(model=MODEL, messages=messages)

    # Assert
    assert result is response
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
