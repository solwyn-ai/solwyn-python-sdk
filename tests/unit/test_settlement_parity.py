"""Settlement parity: every settlement path rides the reporter queue (PJ-1).

Non-streaming (sync + async), streaming, and the media lifecycle must ALL
settle a reservation the same way: build the confirm sans-I/O and enqueue it
with its metadata event as ONE ordered item via
``reporter.report_settlement(confirm, event)``. The caller's thread never
blocks on a Solwyn round-trip after the provider has answered, and the
blocking ``confirm_cost`` no longer exists on the enforcers — the reporter
queue is the ONLY settlement path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY

from solwyn._base import MediaSurfaceSpec
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetConfirmRequest, CallStatus, MetadataEvent
from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import AsyncSolwyn, Solwyn


def _openai_client(*, is_async: bool = False) -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI" if is_async else "OpenAI"
    client.with_options.return_value = client
    return client


def _openai_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _openai_stream_chunks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
            choices=[],
        ),
    ]


def _allow_budget(reservation_id: str | None = "res_123") -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=None,
        budget_limit=100.0,
        current_usage=0.0,
        mode=SimpleNamespace(value="alert_only"),
        price_hints=None,
    )


def _make_solwyn(client: object) -> Solwyn:
    # Patch the flush loop out (the thread exits immediately) WITHOUT setting
    # _shutdown — report_settlement drops items once shutdown is set.
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        return Solwyn(client, api_key=VALID_API_KEY, model="gpt-5.5")


def _make_async_solwyn(client: object) -> AsyncSolwyn:
    return AsyncSolwyn(client, api_key=VALID_API_KEY, model="gpt-5.5")


def _close(solwyn: Solwyn) -> None:
    solwyn._reporter._shutdown.set()
    solwyn._reporter._http.close()
    solwyn._budget._http.close()


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


def _assert_settled_exactly_once(
    settle: MagicMock, report: MagicMock, *, reservation_id: str = "res_123"
) -> MetadataEvent:
    """One report_settlement(confirm, event); no separate SUCCESS report()."""
    settle.assert_called_once()
    confirm, event = settle.call_args.args
    assert isinstance(confirm, BudgetConfirmRequest)
    assert isinstance(event, MetadataEvent)
    assert confirm.reservation_id == reservation_id
    # The confirm and its metadata event share the reconciliation join key.
    assert confirm.call_id == event.call_id
    assert event.status is CallStatus.SUCCESS
    # The SUCCESS event travels WITH the confirm as one ordered settlement;
    # report() must not double-report it.
    success_reports = [c for c in report.call_args_list if c.args[0].status is CallStatus.SUCCESS]
    assert success_reports == []
    return event


@pytest.mark.unit
class TestNoHotPathConfirm:
    """The blocking confirm method is gone — the queue is the only path."""

    def test_confirm_cost_removed_from_sync_enforcer(self) -> None:
        assert not hasattr(BudgetEnforcer, "confirm_cost"), (
            "BudgetEnforcer.confirm_cost is a blocking POST on the caller's "
            "thread — settlement must go through reporter.report_settlement()"
        )

    def test_confirm_cost_removed_from_async_enforcer(self) -> None:
        assert not hasattr(AsyncBudgetEnforcer, "confirm_cost"), (
            "AsyncBudgetEnforcer.confirm_cost gates the response on a Solwyn "
            "round-trip — settlement must go through reporter.report_settlement()"
        )


@pytest.mark.unit
class TestSyncNonStreamingSettlement:
    def test_settles_via_reporter_exactly_once(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
        ):
            response = solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert response.choices[0].message.content == "ok"
        event = _assert_settled_exactly_once(settle, report)
        assert event.input_tokens == 10
        assert event.output_tokens == 5
        _close(solwyn)

    def test_no_reservation_reports_without_settlement(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_response()
        solwyn = _make_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget(None)),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)

        settle.assert_not_called()
        success_reports = [
            c for c in report.call_args_list if c.args[0].status is CallStatus.SUCCESS
        ]
        assert len(success_reports) == 1
        _close(solwyn)


@pytest.mark.unit
class TestAsyncNonStreamingSettlement:
    @pytest.mark.asyncio
    async def test_settles_via_reporter_exactly_once(self) -> None:
        client = _openai_client(is_async=True)
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        solwyn = _make_async_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", AsyncMock(return_value=_allow_budget())),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
        ):
            response = await solwyn.chat.completions.create(**_PLAIN_REQUEST)

        assert response.choices[0].message.content == "ok"
        _assert_settled_exactly_once(settle, report)
        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


@pytest.mark.unit
class TestStreamingSettlement:
    """Streaming already settles via the queue — parity's fixed point."""

    def test_sync_stream_settles_via_reporter_exactly_once(self) -> None:
        client = _openai_client()
        client.chat.completions.create.return_value = _openai_stream_chunks()
        solwyn = _make_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
        ):
            stream = solwyn.chat.completions.create(**_PLAIN_REQUEST, stream=True)
            settle.assert_not_called()  # settles on completion, not establishment
            chunks = list(stream)

        assert len(chunks) == 2
        event = _assert_settled_exactly_once(settle, report)
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        _close(solwyn)


@pytest.mark.unit
class TestMediaSettlement:
    """The media lifecycle settles through the same queue as chat."""

    @staticmethod
    def _spec() -> MediaSurfaceSpec:
        def extract(response: object) -> TokenDetails | None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            return TokenDetails(input_tokens=usage.prompt_tokens)

        return MediaSurfaceSpec(
            surface="embeddings",
            modality="embedding",
            extract_usage=extract,
            measure_request=lambda _kwargs: None,
        )

    @staticmethod
    def _route_to_embeddings(surface, client, kwargs, *, timeout, max_retries):
        return client.embeddings.create, dict(kwargs)

    def test_sync_media_settles_via_reporter_exactly_once(self) -> None:
        client = _openai_client()
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7), data=[])
        client.embeddings.create.return_value = resp
        solwyn = _make_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
            patch.object(
                solwyn._runtimes[0].adapter, "prepare_media_call", self._route_to_embeddings
            ),
        ):
            result = solwyn._media_call(self._spec(), model="text-embedding-3-small", input="hello")

        assert result is resp
        confirm, event = settle.call_args.args
        assert confirm.modality == "embedding"
        _assert_settled_exactly_once(settle, report)
        _close(solwyn)

    @pytest.mark.asyncio
    async def test_async_media_settles_via_reporter_exactly_once(self) -> None:
        client = _openai_client(is_async=True)
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7), data=[])
        client.embeddings.create = AsyncMock(return_value=resp)
        solwyn = _make_async_solwyn(client)

        settle = MagicMock()
        report = MagicMock()
        with (
            patch.object(solwyn._budget, "check_budget", AsyncMock(return_value=_allow_budget())),
            patch.object(solwyn._reporter, "report_settlement", settle),
            patch.object(solwyn._reporter, "report", report),
            patch.object(
                solwyn._runtimes[0].adapter, "prepare_media_call", self._route_to_embeddings
            ),
        ):
            result = await solwyn._media_call(
                self._spec(), model="text-embedding-3-small", input="hello"
            )

        assert result is resp
        _assert_settled_exactly_once(settle, report)
        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()
