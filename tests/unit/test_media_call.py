"""Tests for the ``_media_call`` lifecycle (non-chat media surfaces).

The lifecycle is FOUNDATION: no proxy or real surface exists yet, so these
tests drive ``_media_call`` directly with a fake ``MediaSurfaceSpec`` and a
patched ``prepare_media_call`` seam. They pin the lean shape the plan mandates:
estimate -> budget check -> primary-only call -> extract/measure -> confirm +
report, with ``is_model_fallback`` always False and no candidate walk.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._base import MediaSurfaceSpec
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, CallStatus
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import BudgetExceededError, UnsupportedSurfaceError


def _extract_prompt_tokens(response: object) -> TokenDetails | None:
    """A fake per-surface extractor: read prompt_tokens off the usage block."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenDetails(input_tokens=usage.prompt_tokens)


def _spec(*, extract=_extract_prompt_tokens, measure=lambda _kwargs: None) -> MediaSurfaceSpec:
    return MediaSurfaceSpec(
        surface="embeddings",
        modality="text-embedding",
        extract_usage=extract,
        measure_request=measure,
    )


def _route_to_embeddings(surface, client, kwargs, *, timeout, max_retries):
    """Stand-in ``prepare_media_call``: route the embeddings surface to the SDK."""
    return client.embeddings.create, dict(kwargs)


def _allow(reservation_id: str | None = "res_media") -> SimpleNamespace:
    return SimpleNamespace(allowed=True, reservation_id=reservation_id, price_hints=None)


def _deny() -> SimpleNamespace:
    return SimpleNamespace(
        allowed=False,
        reservation_id=None,
        price_hints=None,
        project_id=VALID_PROJECT_ID,
        budget_limit=10.0,
        current_usage=10.0,
        mode=BudgetMode.HARD_DENY,
    )


def _sync_client() -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=42))
    client.embeddings.create.return_value = resp
    return client, resp


def _build_sync(client: MagicMock, **overrides) -> Solwyn:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, api_key=VALID_API_KEY, **overrides)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


@pytest.mark.unit
class TestMediaCallSync:
    def test_success_confirms_and_reports_primary_only(self) -> None:
        client, resp = _sync_client()
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route_to_embeddings),
        ):
            result = solwyn._media_call(
                _spec(), model="text-embedding-3-small", input="hello world"
            )

        assert result is resp
        client.embeddings.create.assert_called_once()

        # Budget checked against the PRIMARY only — no failover chain hinted.
        assert check.call_args.kwargs["provider"] == "openai"
        assert "fallback_providers" not in check.call_args.kwargs
        assert "fallback_models" not in check.call_args.kwargs

        # Confirmed with the extractor's quantity, never a provider fallback.
        confirm.assert_called_once()
        assert confirm.call_args.args[2].input_tokens == 42
        assert confirm.call_args.kwargs["is_provider_fallback"] is False

        # Metadata: SUCCESS, foundation flags off.
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.is_model_fallback is False
        assert event.is_provider_fallback is False
        assert event.input_tokens == 42

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_request_derived_quantity_when_response_has_no_usage(self) -> None:
        client, _ = _sync_client()
        client.embeddings.create.return_value = SimpleNamespace(usage=None)
        solwyn = _build_sync(client)
        spec = _spec(measure=lambda _kwargs: TokenDetails(input_tokens=7))
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route_to_embeddings),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hi")

        # Fell back to the request-derived measurer (extractor returned None).
        assert confirm.call_args.args[2].input_tokens == 7
        assert report.call_args.args[0].input_tokens == 7

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_unobservable_quantity_reports_none_and_skips_confirm(self) -> None:
        client, _ = _sync_client()
        client.embeddings.create.return_value = SimpleNamespace(usage=None)
        solwyn = _build_sync(client)
        # Both hooks report nothing observable -> quantity stays None (never $0).
        spec = _spec(measure=lambda _kwargs: None)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route_to_embeddings),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hi")

        confirm.assert_not_called()  # no observed quantity -> never settle a real $0 price
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.token_details is None
        assert event.input_tokens == 0

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_budget_denied_raises_and_reports_without_dispatch(self) -> None:
        client, _ = _sync_client()
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_deny()),
            patch.object(solwyn._reporter, "report") as report,
            pytest.raises(BudgetExceededError),
        ):
            solwyn._media_call(_spec(), model="text-embedding-3-small", input="hi")

        client.embeddings.create.assert_not_called()
        assert report.call_args.args[0].status == CallStatus.BUDGET_DENIED

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_unsupported_surface_reports_error_then_raises(self) -> None:
        client, _ = _sync_client()
        solwyn = _build_sync(client)
        # No prepare_media_call patch -> the real OpenAI adapter raises (foundation).
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._reporter, "report") as report,
            pytest.raises(UnsupportedSurfaceError),
        ):
            solwyn._media_call(_spec(), model="text-embedding-3-small", input="hi")

        event = report.call_args.args[0]
        assert event.status == CallStatus.ERROR
        assert event.failover_error_class == "UnsupportedSurfaceError"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


def _async_client() -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI"
    client.with_options.return_value = client
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=99))
    client.embeddings.create = AsyncMock(return_value=resp)
    return client, resp


@pytest.mark.unit
class TestMediaCallAsync:
    @pytest.mark.asyncio
    async def test_success_confirms_and_reports_primary_only(self) -> None:
        client, resp = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        with (
            patch.object(
                solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow())
            ) as check,
            patch.object(solwyn._budget, "confirm_cost", new=AsyncMock()) as confirm,
            patch.object(solwyn._reporter, "report") as report,
            patch.object(solwyn._runtimes[0].adapter, "prepare_media_call", _route_to_embeddings),
        ):
            result = await solwyn._media_call(
                _spec(), model="text-embedding-3-small", input="hello"
            )

        assert result is resp
        client.embeddings.create.assert_awaited_once()
        assert check.call_args.kwargs["provider"] == "openai"
        confirm.assert_awaited_once()
        assert confirm.call_args.args[2].input_tokens == 99
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.is_model_fallback is False

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_budget_denied_raises_without_dispatch(self) -> None:
        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._budget, "check_budget", new=AsyncMock(return_value=_deny())),
            patch.object(solwyn._reporter, "report") as report,
            pytest.raises(BudgetExceededError),
        ):
            await solwyn._media_call(_spec(), model="text-embedding-3-small", input="hi")

        client.embeddings.create.assert_not_awaited()
        assert report.call_args.args[0].status == CallStatus.BUDGET_DENIED

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
