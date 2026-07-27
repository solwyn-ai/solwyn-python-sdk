"""R5: post-success bookkeeping must never destroy a paid response.

Helper-level tests, plus end-to-end coverage that a paid, successful response
still reaches the caller (and still settles, with degraded usage) when the
sync or async non-streaming success block's adapter bookkeeping raises.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, make_mock_client

from solwyn._token_details import TokenDetails
from solwyn.client import (
    AsyncSolwyn,
    Solwyn,
    _extract_usage_fail_soft,
    _safe_extract_region,
    _safe_extract_service_tier,
)


def _runtime(adapter: MagicMock) -> MagicMock:
    rt = MagicMock()
    rt.adapter = adapter
    rt.sdk_client = MagicMock()
    return rt


@pytest.mark.unit
class TestExtractUsageFailSoft:
    def test_provider_reported_usage_passes_through(self) -> None:
        adapter = MagicMock()
        reported = TokenDetails(input_tokens=10, output_tokens=5)
        adapter.extract_usage.return_value = reported
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=99)

        assert result is reported
        assert result.is_estimated is False

    def test_adapter_estimate_preferred_when_present(self) -> None:
        # Mirrors the existing inline behavior: a non-None estimate REPLACES
        # the extracted details (compat provider omitted its usage block).
        adapter = MagicMock()
        adapter.extract_usage.return_value = TokenDetails()
        estimated = TokenDetails(input_tokens=99, output_tokens=0, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=99)

        assert result is estimated

    def test_extract_usage_raise_degrades_to_adapter_estimate(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("unexpected shape")
        estimated = TokenDetails(input_tokens=42, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=42)

        assert result is estimated

    def test_both_raises_degrade_to_synthetic_estimate(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.side_effect = ValueError("boom too")

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=123)

        assert result.input_tokens == 123
        assert result.output_tokens == 0
        assert result.is_estimated is True

    def test_raise_then_none_estimate_degrades_to_synthetic(self) -> None:
        adapter = MagicMock()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=7)

        assert result.input_tokens == 7
        assert result.is_estimated is True


@pytest.mark.unit
class TestSafeExtractRegionAndTier:
    def test_region_raise_degrades_to_none(self) -> None:
        adapter = MagicMock()
        adapter.extract_region.side_effect = AttributeError("no client attr")
        assert _safe_extract_region(_runtime(adapter)) is None

    def test_region_passthrough(self) -> None:
        adapter = MagicMock()
        adapter.extract_region.return_value = "us-east-1"
        assert _safe_extract_region(_runtime(adapter)) == "us-east-1"

    def test_tier_raise_degrades_to_none(self) -> None:
        adapter = MagicMock()
        adapter.extract_service_tier.side_effect = KeyError("service_tier")
        assert _safe_extract_service_tier(_runtime(adapter), object()) is None

    def test_tier_passthrough(self) -> None:
        adapter = MagicMock()
        adapter.extract_service_tier.return_value = "priority"
        assert _safe_extract_service_tier(_runtime(adapter), object()) == "priority"


# --- End-to-end harness, duplicated from tests/unit/test_settlement_parity.py ---
#
# Copied rather than imported: these are private test-module fixtures, not a
# shared library. Duplication here documents that this suite's assertions
# stand on their own without a cross-test-module dependency.


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


def _openai_response() -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-5.5",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


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


@pytest.mark.unit
class TestPaidResponseSurvivesBookkeepingFailure:
    def test_sync_extract_usage_raise_returns_response_and_settles_estimated(
        self,
    ) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with patch.object(type(adapter), "extract_usage", side_effect=RuntimeError("shape drift")):
            response = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            # The paid response reached the caller — R5's core assertion.
            assert response.choices[0].message.content == "ok"
        solwyn.close()  # drain the settlement queue to the wire

        # Settlement still happened, with degraded (estimated) usage.
        assert len(recorder.confirms) == 1
        success_events = [e for e in recorder.events if e["status"] == "success"]
        assert len(success_events) == 1
        assert success_events[0]["token_details"]["is_estimated"] is True

    def test_sync_tier_and_region_raise_settle_with_none(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with (
            patch.object(type(adapter), "extract_service_tier", side_effect=KeyError("tier")),
            patch.object(type(adapter), "extract_region", side_effect=AttributeError("region")),
        ):
            response = solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert response.choices[0].message.content == "ok"
        solwyn.close()

        assert len(recorder.confirms) == 1  # reservation NOT leaked

        # The 'with none' half the test name promises: MetadataEvent's
        # None-skipping serializer OMITS provider_region/service_tier from
        # the wire entirely when they're None, rather than emitting a
        # literal null — so absence, not a None value, is the contract.
        success_events = [e for e in recorder.events if e["status"] == "success"]
        assert len(success_events) == 1
        event = success_events[0]
        assert "provider_region" not in event
        assert "service_tier" not in event

    async def test_async_extract_usage_raise_returns_response_and_settles(
        self,
    ) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        recorder = _ControlPlaneRecorder()
        solwyn = await _make_async_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with patch.object(type(adapter), "extract_usage", side_effect=RuntimeError("shape drift")):
            response = await solwyn.chat.completions.create(**_PLAIN_REQUEST)
            assert response.choices[0].message.content == "ok"
        await solwyn.close()

        assert len(recorder.confirms) == 1
        success_events = [e for e in recorder.events if e["status"] == "success"]
        assert success_events[0]["token_details"]["is_estimated"] is True
