"""R5: post-success bookkeeping must never destroy a paid response.

Helper-level tests, plus end-to-end coverage that a paid, successful response
still reaches the caller (and still settles, with degraded usage) when the
sync or async non-streaming success block's adapter bookkeeping raises.

Degrading must not weaken enforcement either: a lease-funded call whose usage
was never measurable settles at its reserved bound rather than handing back
output authority the paid response already consumed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID, make_mock_client

import solwyn as solwyn_pkg
from solwyn._base import MediaSurfaceSpec
from solwyn._registry import ProviderRuntime
from solwyn._token_details import TokenDetails
from solwyn._types import ProviderEntry, ProviderName
from solwyn.client import (
    AsyncSolwyn,
    Solwyn,
    _extract_usage_fail_soft,
    _safe_extract_region,
    _safe_extract_service_tier,
)
from solwyn.providers._protocol import ProviderAdapter


def _adapter() -> MagicMock:
    """An adapter double constrained to the real ProviderAdapter interface."""
    return MagicMock(spec=ProviderAdapter)


def _runtime(adapter: MagicMock) -> ProviderRuntime:
    """A REAL ProviderRuntime — only the adapter and SDK client are doubles."""
    return ProviderRuntime(
        entry=ProviderEntry(provider=ProviderName.OPENAI, model="gpt-5.5"),
        sdk_client=make_mock_client(),
        adapter=adapter,
    )


@pytest.mark.unit
class TestExtractUsageFailSoft:
    def test_provider_reported_usage_passes_through(self) -> None:
        adapter = _adapter()
        reported = TokenDetails(input_tokens=10, output_tokens=5)
        adapter.extract_usage.return_value = reported
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=99)

        assert result.token_details is reported
        assert result.token_details.is_estimated is False
        assert result.unmeasured is False

    def test_adapter_estimate_preferred_when_present(self) -> None:
        # Mirrors the existing inline behavior: a non-None estimate REPLACES
        # the extracted details (compat provider omitted its usage block).
        adapter = _adapter()
        adapter.extract_usage.return_value = TokenDetails()
        estimated = TokenDetails(input_tokens=99, output_tokens=0, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=99)

        assert result.token_details is estimated
        # An ADAPTER estimate is a real measurement of the response — the lease
        # trues up against it normally.
        assert result.unmeasured is False

    def test_zero_usage_passes_through_without_empty_usage_policy(self) -> None:
        adapter = _adapter()
        empty = TokenDetails()
        adapter.extract_usage.return_value = empty
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=99)

        assert result.token_details is empty
        assert result.unmeasured is False

    def test_empty_usage_policy_degrades_zero_usage_to_synthetic_estimate(self) -> None:
        adapter = _adapter()
        adapter.extract_usage.return_value = TokenDetails()
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(
            _runtime(adapter),
            object(),
            estimated_input_tokens=99,
            estimate_empty_usage=True,
        )

        assert result.token_details == TokenDetails(
            input_tokens=99,
            output_tokens=0,
            is_estimated=True,
        )
        assert result.unmeasured is True

    def test_empty_usage_policy_keeps_adapter_estimate_unmeasured(self) -> None:
        adapter = _adapter()
        adapter.extract_usage.return_value = TokenDetails()
        estimated = TokenDetails(input_tokens=99, output_tokens=7, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(
            _runtime(adapter),
            object(),
            estimated_input_tokens=99,
            estimate_empty_usage=True,
        )

        assert result.token_details is estimated
        assert result.unmeasured is True

    def test_extract_usage_raise_degrades_to_adapter_estimate(self) -> None:
        adapter = _adapter()
        adapter.extract_usage.side_effect = RuntimeError("unexpected shape")
        estimated = TokenDetails(input_tokens=42, is_estimated=True)
        adapter.estimate_missing_usage.return_value = estimated

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=42)

        assert result.token_details is estimated
        assert result.unmeasured is False

    def test_both_raises_degrade_to_synthetic_estimate(self) -> None:
        adapter = _adapter()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.side_effect = ValueError("boom too")

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=123)

        assert result.token_details.input_tokens == 123
        assert result.token_details.output_tokens == 0
        assert result.token_details.is_estimated is True
        # Nothing was measured — the flag that keeps a lease from re-lending
        # the output allowance this response already consumed.
        assert result.unmeasured is True

    def test_raise_then_none_estimate_degrades_to_synthetic(self) -> None:
        adapter = _adapter()
        adapter.extract_usage.side_effect = RuntimeError("boom")
        adapter.estimate_missing_usage.return_value = None

        result = _extract_usage_fail_soft(_runtime(adapter), object(), estimated_input_tokens=7)

        assert result.token_details.input_tokens == 7
        assert result.token_details.is_estimated is True
        assert result.unmeasured is True


@pytest.mark.unit
class TestSafeExtractRegionAndTier:
    def test_region_raise_degrades_to_none(self) -> None:
        adapter = _adapter()
        adapter.extract_region.side_effect = AttributeError("no client attr")
        assert _safe_extract_region(_runtime(adapter)) is None

    def test_region_passthrough(self) -> None:
        adapter = _adapter()
        adapter.extract_region.return_value = "us-east-1"
        assert _safe_extract_region(_runtime(adapter)) == "us-east-1"

    def test_tier_raise_degrades_to_none(self) -> None:
        adapter = _adapter()
        adapter.extract_service_tier.side_effect = KeyError("service_tier")
        assert _safe_extract_service_tier(_runtime(adapter), object()) is None

    def test_tier_passthrough(self) -> None:
        adapter = _adapter()
        adapter.extract_service_tier.return_value = "priority"
        assert _safe_extract_service_tier(_runtime(adapter), object()) == "priority"


# --- End-to-end harness, duplicated from tests/unit/test_settlement_parity.py ---
#
# Copied rather than imported: these are private test-module fixtures, not a
# shared library. Duplication here documents that this suite's assertions
# stand on their own without a cross-test-module dependency.


_GRANTED_TOKENS = 100_000

# The declared per-call output cap: also the lease reserve's output half, so
# the lease-funded assertions below can name the exact bound at stake.
_OUTPUT_BOUND = 256


def _grant_payload() -> dict[str, Any]:
    """An eligible, allowed lease grant with room to spare (PJ-2 shape)."""
    return {
        "eligible": True,
        "allowed": True,
        "lease_id": "lease_1",
        "generation": 1,
        "granted_tokens": _GRANTED_TOKENS,
        "refresh_interval_s": 300.0,
        "lease_length_s": 600.0,
        "headroom_share_tokens": 50_000,
        "posture": {"mode": "alert_only", "on_unreachable": "fail_open"},
        "final_grant": False,
        "project_id": VALID_PROJECT_ID,
        "mode": "alert_only",
        "budget_limit": 100.0,
        "current_usage": 20.0,
        "remaining_budget": 80.0,
    }


class _ControlPlaneRecorder:
    """Record control-plane traffic at the HTTP transport boundary.

    Serves ``/budgets/check`` (allow, with this recorder's ``reservation_id``),
    the ``/budgets/lease`` grant + surrender pair, ``/budgets/confirm``,
    ``/metadata/ingest``, and breaker reports; every request lands in
    ``requests`` as an ordered ``(path, body)`` pair.
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
        if path.endswith("/budgets/lease"):
            return httpx.Response(200, json=_grant_payload())
        if path.endswith("/budgets/lease/surrender"):
            return httpx.Response(200, json={})
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


def _openai_stream_chunks() -> list[SimpleNamespace]:
    # Copied from tests/unit/test_settlement_parity.py:95-110.
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


@pytest.mark.unit
class TestLeaseFundedUnmeasurableCall:
    """A call whose usage was never measurable must not refund its allowance.

    The fail-soft synthetic tier reports the pre-flight INPUT estimate and no
    output at all. Trued up naively, that hands a lease back the entire output
    reservation a paid response already consumed, and later admissions re-lend
    already-spent authority past the run's hard token cap. The local
    reservation settles at its bound instead; the wire confirm keeps the honest
    ``is_estimated`` under-measure for the cloud to reconcile.
    """

    def test_sync_unmeasurable_usage_settles_the_lease_at_its_reserved_bound(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with (
            patch.object(type(adapter), "extract_usage", side_effect=RuntimeError("shape drift")),
            solwyn_pkg.run("fail-soft-lease"),
        ):
            run_id = solwyn_pkg.current_run()[0]
            response = solwyn.chat.completions.create(
                max_completion_tokens=_OUTPUT_BOUND, **_PLAIN_REQUEST
            )
            assert response.choices[0].message.content == "ok"
            # Snapshot BEFORE close(): the lease is surrendered on shutdown.
            state = solwyn._budget._lease.state_for(run_id)
            assert state is not None
            granted_remaining = state.granted_remaining_tokens
            spent = state.spent_tokens_since_report
            open_reservations = dict(state.reservations)
        solwyn.close()  # drain the settlement queue to the wire

        _assert_lease_held_at_bound(recorder, granted_remaining, spent, open_reservations)

    async def test_async_unmeasurable_usage_settles_the_lease_at_its_reserved_bound(
        self,
    ) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        client.chat.completions.create = AsyncMock(return_value=_openai_response())
        recorder = _ControlPlaneRecorder()
        solwyn = await _make_async_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with (
            patch.object(type(adapter), "extract_usage", side_effect=RuntimeError("shape drift")),
            solwyn_pkg.run("fail-soft-lease-async"),
        ):
            run_id = solwyn_pkg.current_run()[0]
            response = await solwyn.chat.completions.create(
                max_completion_tokens=_OUTPUT_BOUND, **_PLAIN_REQUEST
            )
            assert response.choices[0].message.content == "ok"
            state = solwyn._budget._lease.state_for(run_id)
            assert state is not None
            granted_remaining = state.granted_remaining_tokens
            spent = state.spent_tokens_since_report
            open_reservations = dict(state.reservations)
        await solwyn.close()

        _assert_lease_held_at_bound(recorder, granted_remaining, spent, open_reservations)

    def test_sync_measured_usage_still_trues_the_lease_down(self) -> None:
        # The twin that proves the floor is scoped to the UNMEASURABLE tier:
        # a provider-reported response still hands its unspent bound back.
        client = make_mock_client()
        client.chat.completions.create.return_value = _openai_response()
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)

        with solwyn_pkg.run("measured-lease"):
            run_id = solwyn_pkg.current_run()[0]
            solwyn.chat.completions.create(max_completion_tokens=_OUTPUT_BOUND, **_PLAIN_REQUEST)
            state = solwyn._budget._lease.state_for(run_id)
            assert state is not None
            granted_remaining = state.granted_remaining_tokens
            spent = state.spent_tokens_since_report
        solwyn.close()

        # _openai_response() reports 10 in / 5 out — the whole 256-token bound
        # minus the 5 actually produced goes back to the lease.
        assert spent == 15
        assert granted_remaining == _GRANTED_TOKENS - 15
        # is_estimated is omitted from the wire when False — measured usage.
        assert "is_estimated" not in recorder.confirms[0]["token_details"]


def _assert_lease_held_at_bound(
    recorder: _ControlPlaneRecorder,
    granted_remaining: int,
    spent: int,
    open_reservations: dict[str, Any],
) -> None:
    """Assert the wire confirm and the local counters for an unmeasurable call."""
    assert len(recorder.confirms) == 1
    confirm = recorder.confirms[0]
    # Wire shape is UNCHANGED: the lease settlement key plus the honest,
    # explicitly-estimated under-measure the cloud reconciles.
    assert confirm["lease_id"] == "lease_1"
    # The settlement key is exclusive, and an absent key is OMITTED (not null).
    assert "reservation_id" not in confirm
    details = confirm["token_details"]
    assert details["is_estimated"] is True
    assert details["output_tokens"] == 0
    reported = details["input_tokens"] + details["output_tokens"]

    # Local authority is charged the RESERVED bound, not the under-measure:
    # the output allowance the paid response consumed is never re-lent.
    reserved = reported + _OUTPUT_BOUND
    assert spent == reserved
    assert granted_remaining == _GRANTED_TOKENS - reserved
    assert reported < reserved  # the gap this test exists to keep unrefunded
    assert open_reservations == {}


def _route_media_to_embeddings(surface, client, kwargs, *, timeout, max_retries):
    """Stand-in ``prepare_media_call``: route the embeddings surface to the SDK."""
    return client.embeddings.create, dict(kwargs)


@pytest.mark.unit
class TestStreamingAndMediaBookkeepingFailSoft:
    def test_stream_region_raise_still_yields_stream_and_settles(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.return_value = iter(_openai_stream_chunks())
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with patch.object(type(adapter), "extract_region", side_effect=AttributeError("region")):
            stream = solwyn.chat.completions.create(stream=True, **_PLAIN_REQUEST)
            chunks = list(stream)  # consuming settles via on_complete
        solwyn.close()

        assert len(chunks) == 2
        assert len(recorder.confirms) == 1

    def test_error_event_region_raise_preserves_original_error(self) -> None:
        client = make_mock_client()
        client.chat.completions.create.side_effect = RuntimeError("provider down")
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)
        adapter = solwyn._runtimes[0].adapter

        with (
            patch.object(type(adapter), "extract_region", side_effect=AttributeError("region")),
            pytest.raises(RuntimeError, match="provider down"),
        ):
            solwyn.chat.completions.create(**_PLAIN_REQUEST)
        solwyn.close()

    def test_media_extract_usage_raise_still_returns_response_and_settles(self) -> None:
        client = make_mock_client()
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=42))
        client.embeddings.create.return_value = resp
        recorder = _ControlPlaneRecorder()
        solwyn = _make_solwyn(client, recorder)

        def _raising_extract_usage(_response: Any) -> TokenDetails | None:
            raise RuntimeError("usage extraction failed")

        spec = MediaSurfaceSpec(
            surface="embeddings",
            modality="embedding",
            extract_usage=_raising_extract_usage,
            measure_request=lambda _kwargs: TokenDetails(input_tokens=7),
        )

        with patch.object(
            solwyn._runtimes[0].adapter, "prepare_media_call", _route_media_to_embeddings
        ):
            result = solwyn._media_call(spec, model="text-embedding-3-small", input="hello world")
        solwyn.close()

        # The paid media response reached the caller despite the raise — R5's
        # core assertion, mirrored from the chat success-block tests above.
        assert result is resp
        assert len(recorder.confirms) == 1

    async def test_async_media_extract_usage_raise_still_returns_response_and_settles(
        self,
    ) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=42))
        client.embeddings.create = AsyncMock(return_value=resp)
        recorder = _ControlPlaneRecorder()
        solwyn = await _make_async_solwyn(client, recorder)

        def _raising_extract_usage(_response: Any) -> TokenDetails | None:
            raise RuntimeError("usage extraction failed")

        spec = MediaSurfaceSpec(
            surface="embeddings",
            modality="embedding",
            extract_usage=_raising_extract_usage,
            measure_request=lambda _kwargs: TokenDetails(input_tokens=7),
        )

        with patch.object(
            solwyn._runtimes[0].adapter, "prepare_media_call", _route_media_to_embeddings
        ):
            result = await solwyn._media_call(
                spec, model="text-embedding-3-small", input="hello world"
            )
        await solwyn.close()

        assert result is resp
        assert len(recorder.confirms) == 1
