"""Settlement parity: every settlement path rides the reporter queue (PJ-1).

Non-streaming (sync + async), streaming, and the media lifecycle must ALL
settle a reservation the same way: build the confirm sans-I/O and enqueue it
with its metadata event as ONE ordered item via
``reporter.report_settlement(confirm, event)``. The caller's thread never
blocks on a Solwyn round-trip after the provider has answered, and the
blocking ``confirm_cost`` no longer exists on the enforcers — the reporter
queue is the ONLY settlement path.

Seams are service boundaries only: the provider SDK client is conftest's
``make_mock_client`` stand-in, and Solwyn's control-plane HTTP rides an
``httpx.MockTransport`` recorder — the real enforcer, reporter queue, and
adapters all run. Parity is asserted on the wire: exactly one
``/budgets/confirm`` per reservation, exactly one SUCCESS event in
``/metadata/ingest``, sharing the ``call_id`` join key, confirm first.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, make_mock_client

from solwyn.budget import AsyncBudgetEnforcer, BudgetEnforcer
from solwyn.client import AsyncSolwyn, Solwyn


class _ControlPlaneRecorder:
    """Record control-plane traffic at the HTTP transport boundary.

    Serves ``/budgets/check`` (allow, with this recorder's ``reservation_id``),
    ``/budgets/confirm``, ``/metadata/ingest``, and breaker reports; every
    request lands in ``requests`` as an ordered ``(path, body)`` pair.
    """

    def __init__(self, *, reservation_id: str | None = "res_123") -> None:
        self.reservation_id = reservation_id
        self.requests: list[tuple[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((path, body))
        if path.endswith("/budgets/check"):
            return httpx.Response(
                200, json={**ALLOW_BUDGET_RESPONSE, "reservation_id": self.reservation_id}
            )
        if path.endswith("/budgets/confirm"):
            return httpx.Response(200, json={"status": "confirmed"})
        if path.endswith("/metadata/ingest"):
            return httpx.Response(202, json={"rejected": []})
        if path.endswith("/breaker-reports"):
            return httpx.Response(202, json={})
        return httpx.Response(404, json={})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), timeout=5.0)

    def aclient(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), timeout=5.0)

    @property
    def confirms(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path.endswith("/budgets/confirm")]

    @property
    def events(self) -> list[dict[str, Any]]:
        return [
            event
            for path, body in self.requests
            if path.endswith("/metadata/ingest")
            for event in body
        ]

    def _first_index(self, suffix: str) -> int:
        return next(i for i, (path, _) in enumerate(self.requests) if path.endswith(suffix))


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


def _make_solwyn(client: object, recorder: _ControlPlaneRecorder) -> Solwyn:
    solwyn = Solwyn(client, api_key=VALID_API_KEY, model="gpt-5.5")
    solwyn._budget._http.close()
    solwyn._budget._http = recorder.client()
    solwyn._reporter._http.close()
    solwyn._reporter._http = recorder.client()
    return solwyn


async def _make_async_solwyn(client: object, recorder: _ControlPlaneRecorder) -> AsyncSolwyn:
    solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, model="gpt-5.5")
    await solwyn._budget._http.aclose()
    solwyn._budget._http = recorder.aclient()
    await solwyn._reporter._http.aclose()
    solwyn._reporter._http = recorder.aclient()
    return solwyn


_PLAIN_REQUEST = {
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


def _assert_settled_exactly_once(
    recorder: _ControlPlaneRecorder, *, reservation_id: str = "res_123"
) -> dict[str, Any]:
    """One wire confirm for the reservation; one SUCCESS event, confirm first."""
    assert len(recorder.confirms) == 1
    confirm = recorder.confirms[0]
    assert confirm["reservation_id"] == reservation_id
    # Exactly one SUCCESS event reached ingest — the event travels WITH the
    # confirm as one ordered settlement, never as a second report().
    success_events = [e for e in recorder.events if e["status"] == "success"]
    assert len(success_events) == 1
    event = success_events[0]
    # The confirm and its metadata event share the reconciliation join key.
    assert event["call_id"] == confirm["call_id"]
    # Confirm-before-metadata order on the wire.
    assert recorder._first_index("/budgets/confirm") < recorder._first_index("/metadata/ingest")
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
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)

        response = solwyn.chat.completions.create(**_PLAIN_REQUEST)
        # The provider has answered; nothing has touched the confirm wire yet.
        assert recorder.confirms == []
        solwyn.close()  # drains the settlement queue to the wire

        assert response.choices[0].message.content == "ok"
        event = _assert_settled_exactly_once(recorder)
        assert event["input_tokens"] == 10
        assert event["output_tokens"] == 5

    def test_no_reservation_reports_without_settlement(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder(reservation_id=None)
        solwyn = _make_solwyn(client, recorder)

        solwyn.chat.completions.create(**_PLAIN_REQUEST)
        solwyn.close()

        assert recorder.confirms == []
        success_events = [e for e in recorder.events if e["status"] == "success"]
        assert len(success_events) == 1


@pytest.mark.unit
class TestAsyncNonStreamingSettlement:
    @pytest.mark.asyncio
    async def test_settles_via_reporter_exactly_once(self) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        recorder = _ControlPlaneRecorder()
        solwyn = await _make_async_solwyn(client, recorder)

        response = await solwyn.chat.completions.create(**_PLAIN_REQUEST)
        assert recorder.confirms == []
        await solwyn.close()

        assert response.choices[0].message.content == "ok"
        event = _assert_settled_exactly_once(recorder)
        assert event["input_tokens"] == 10
        assert event["output_tokens"] == 5


@pytest.mark.unit
class TestStreamingSettlement:
    """Streaming already settles via the queue — parity's fixed point."""

    def test_sync_stream_settles_via_reporter_exactly_once(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_stream_chunks()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)

        stream = solwyn.chat.completions.create(**_PLAIN_REQUEST, stream=True)
        chunks = list(stream)
        solwyn.close()

        assert len(chunks) == 2
        # Token counts come from the stream's final usage chunk: settlement was
        # built at completion, not establishment.
        event = _assert_settled_exactly_once(recorder)
        assert event["input_tokens"] == 100
        assert event["output_tokens"] == 50


@pytest.mark.unit
class TestMediaSettlement:
    """The media lifecycle settles through the same queue as chat.

    Uses the REAL embeddings proxy surface (``solwyn.embeddings.create``), so
    the adapter's ``prepare_media_call`` routing runs unmocked.
    """

    def test_sync_media_settles_via_reporter_exactly_once(self) -> None:
        client = make_mock_client()
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7), data=[])
        client.embeddings.create.return_value = resp
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)

        result = solwyn.embeddings.create(model="text-embedding-3-small", input="hello")
        solwyn.close()

        assert result is resp
        event = _assert_settled_exactly_once(recorder)
        assert recorder.confirms[0]["modality"] == "embedding"
        assert event["input_tokens"] == 7

    @pytest.mark.asyncio
    async def test_async_media_settles_via_reporter_exactly_once(self) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7), data=[])
        client.embeddings.create = AsyncMock(return_value=resp)
        recorder = _ControlPlaneRecorder()
        solwyn = await _make_async_solwyn(client, recorder)

        result = await solwyn.embeddings.create(model="text-embedding-3-small", input="hello")
        await solwyn.close()

        assert result is resp
        event = _assert_settled_exactly_once(recorder)
        assert recorder.confirms[0]["modality"] == "embedding"
        assert event["input_tokens"] == 7
