"""Tests for the ``_media_call`` lifecycle (non-chat media surfaces).

The lifecycle is FOUNDATION: no proxy or real surface exists yet, so these
tests drive ``_media_call`` directly with a fake ``MediaSurfaceSpec`` and a
patched ``prepare_media_call`` seam. They pin the lean shape the plan mandates:
estimate -> budget check -> primary-only call -> extract/measure -> confirm +
report, with ``is_model_fallback`` always False and no candidate walk.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

import solwyn as solwyn_pkg
from solwyn._base import MediaSurfaceSpec
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, CallStatus, MediaUsage
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import (
    BudgetExceededError,
    ProviderUnavailableError,
    RunStoppedError,
    UnsupportedSurfaceError,
)

# A failover window small enough that a slow budget pre-flight outlives it, and
# a pre-flight that reliably does. Only the SLEEP can overshoot, so the expiry
# is deterministic in both directions.
_TINY_WINDOW = 0.05
_SLOW_PREFLIGHT = 0.15


def _extract_prompt_tokens(response: object) -> TokenDetails | None:
    """A fake per-surface extractor: read prompt_tokens off the usage block."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenDetails(input_tokens=usage.prompt_tokens)


def _spec(
    *, surface="embeddings", extract=_extract_prompt_tokens, measure=lambda _kwargs: None
) -> MediaSurfaceSpec:
    return MediaSurfaceSpec(
        surface=surface,
        modality="embedding",
        extract_usage=extract,
        measure_request=measure,
    )


def _media_spec(
    *,
    extract=lambda _r: None,
    measure=lambda _kwargs: None,
    measure_media=None,
    estimate_media=None,
) -> MediaSurfaceSpec:
    """A spec with the optional non-token media hooks wired."""
    return MediaSurfaceSpec(
        surface="images",
        modality="image",
        extract_usage=extract,
        measure_request=measure,
        measure_media=measure_media,
        estimate_media=estimate_media,
    )


def _route_to_embeddings(surface, client, kwargs, *, timeout, max_retries):
    """Stand-in ``prepare_media_call``: route the embeddings surface to the SDK."""
    return client.embeddings.create, dict(kwargs)


def _route_to_images(surface, client, kwargs, *, timeout, max_retries):
    """Stand-in ``prepare_media_call``: route the images surface to the SDK."""
    return client.images.generate, dict(kwargs)


def _allow(
    reservation_id: str | None = "res_media",
    *,
    failover_tuning_allowed: bool | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
        failover_tuning_allowed=failover_tuning_allowed,
    )


def _deny(denied_by_period: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        allowed=False,
        reservation_id=None,
        price_hints=None,
        project_id=VALID_PROJECT_ID,
        budget_limit=10.0,
        current_usage=10.0,
        denied_by_period=denied_by_period,
        deny_source="server",
        deny_reason=denied_by_period or "monthly",
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
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    return solwyn


@pytest.mark.unit
class TestMediaCallSync:
    def test_directive_refreshes_media_deadline_before_dispatch(self) -> None:
        client, resp = _sync_client()
        solwyn = _build_sync(client, failover_total_timeout=91.0)

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                return_value=_allow(None, failover_tuning_allowed=False),
            ),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            result = solwyn._media_call(
                _spec(),
                model="text-embedding-3-small",
                input="hello world",
            )

        assert result is resp
        assert solwyn._solwyn_config.failover_total_timeout == 30.0
        assert 0.0 < client.with_options.call_args.kwargs["timeout"].connect <= 30.0
        assert client.with_options.call_args.kwargs["timeout"].read == 600.0
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_custom_hop_read_timeout_flows_to_media_dispatch(self) -> None:
        # The media dispatch site must pass the CONFIGURED per-hop read bound.
        # The assertions above only ever see the 600.0 default, so a hardcoded
        # 600.0 (or a read/connect swap) here would pass unnoticed.
        client, resp = _sync_client()
        solwyn = _build_sync(client, failover_hop_read_timeout=37.0)

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow(None)),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            result = solwyn._media_call(
                _spec(),
                model="text-embedding-3-small",
                input="hello world",
            )

        assert result is resp
        bound = client.with_options.call_args.kwargs["timeout"]
        assert bound.read == 37.0
        assert bound.write == 37.0
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_success_confirms_and_reports_primary_only(self) -> None:
        client, resp = _sync_client()
        solwyn = _build_sync(client)

        def create_after_project_learned(**_kwargs):
            assert solwyn._solwyn_reporter._breaker_project_id == VALID_PROJECT_ID
            return resp

        client.embeddings.create.side_effect = create_after_project_learned
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            solwyn_pkg.run("orchestrator") as parent_run_id,
            solwyn_pkg.run("sync-media", tags={"team": "platform"}) as run_id,
        ):
            result = solwyn._media_call(
                _spec(),
                model="text-embedding-3-small",
                input="hello world",
                solwyn_tags={"job": "embed"},
            )

        assert result is resp
        client.embeddings.create.assert_called_once()

        # Budget checked against the PRIMARY only — no failover chain hinted;
        # the spec's modality rides the check so the API prices the pending call.
        assert check.call_args.kwargs["provider"] == "openai"
        assert check.call_args.kwargs["modality"] == "embedding"
        assert check.call_args.kwargs["agent_run_id"] == run_id
        assert "fallback_providers" not in check.call_args.kwargs
        assert "fallback_models" not in check.call_args.kwargs

        # Settled off the hot path: confirm + event ride report_settlement as one
        # ordered item. Confirmed with the extractor's quantity, never a provider
        # fallback; the confirm carries the surface modality.
        settle.assert_called_once()
        confirm, event = settle.call_args.args
        assert confirm.token_details.input_tokens == 42
        assert confirm.is_provider_fallback is False
        assert confirm.modality == "embedding"

        # Metadata: SUCCESS, foundation flags off, modality carried.
        assert event.status == CallStatus.SUCCESS
        assert event.is_model_fallback is False
        assert event.is_provider_fallback is False
        assert event.input_tokens == 42
        assert event.modality == "embedding"
        assert event.agent_run_id == run_id
        assert event.parent_agent_run_id == parent_run_id
        assert event.tags == {"team": "platform", "job": "embed"}
        assert check.call_args.kwargs["tags"] == event.tags
        assert "solwyn_tags" not in client.embeddings.create.call_args.kwargs

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_request_derived_quantity_when_response_has_no_usage(self) -> None:
        client, _ = _sync_client()
        client.embeddings.create.return_value = SimpleNamespace(usage=None)
        solwyn = _build_sync(client)
        spec = _spec(measure=lambda _kwargs: TokenDetails(input_tokens=7))
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hi")

        # Fell back to the request-derived measurer (extractor returned None).
        confirm, event = settle.call_args.args
        assert confirm.token_details.input_tokens == 7
        assert event.input_tokens == 7

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_unobservable_quantity_reports_none_and_skips_confirm(self) -> None:
        client, _ = _sync_client()
        client.embeddings.create.return_value = SimpleNamespace(usage=None)
        solwyn = _build_sync(client)
        # Both hooks report nothing observable -> quantity stays None (never $0).
        spec = _spec(measure=lambda _kwargs: None)
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(solwyn._solwyn_reporter, "report") as report,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            solwyn._media_call(spec, model="text-embedding-3-small", input="hi")

        # No observed quantity -> never settle a real $0 price; the SUCCESS event
        # rides the plain report() path (no reservation to settle).
        settle.assert_not_called()
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.token_details is None
        assert event.input_tokens == 0

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_both_bases_flow_when_token_and_media_observable(self) -> None:
        # Native gpt-image: token usage (with image buckets) AND request-derived
        # MediaUsage BOTH ride the call. estimated_media rides the CHECK; both
        # bases ride the confirm + event. The server's card unit picks.
        client, _ = _sync_client()
        client.images.generate.return_value = SimpleNamespace(service_tier=None)
        solwyn = _build_sync(client)
        spec = _media_spec(
            extract=lambda _r: TokenDetails(
                input_tokens=222,
                image_input_tokens=194,
                output_tokens=1024,
                image_output_tokens=1024,
            ),
            measure_media=lambda _kwargs, _response: MediaUsage(
                image_count=2, resolution="1024x1024", quality="low"
            ),
            estimate_media=lambda _kwargs: MediaUsage(image_count=2, resolution="1024x1024"),
        )
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_images
            ),
        ):
            solwyn._media_call(spec, model="gpt-image-2", prompt="a cat", n=2)

        client.images.generate.assert_called_once()
        # estimated_media rides the CHECK (precise per-unit pre-flight).
        assert check.call_args.kwargs["estimated_media"].image_count == 2
        assert check.call_args.kwargs["modality"] == "image"
        # BOTH bases ride the confirm: token usage (image buckets) AND MediaUsage.
        settle.assert_called_once()
        confirm, event = settle.call_args.args
        assert confirm.token_details.image_input_tokens == 194
        assert confirm.media_usage.image_count == 2
        assert confirm.modality == "image"
        # And the metadata event carries both.
        assert event.status == CallStatus.SUCCESS
        assert event.modality == "image"
        assert event.token_details.image_output_tokens == 1024
        assert event.media_usage.image_count == 2

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_unexpressible_media_estimate_degrades_instead_of_killing_the_call(self) -> None:
        # A request-derived quantity past the wire model's bound must not raise
        # out of the pre-flight: the call proceeds with no media estimate.
        client, _ = _sync_client()
        client.images.generate.return_value = SimpleNamespace(service_tier=None)
        solwyn = _build_sync(client)
        spec = _media_spec(
            estimate_media=lambda _kwargs: MediaUsage(image_count=200_000_001),
        )
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._solwyn_reporter, "report_settlement"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_images
            ),
        ):
            solwyn._media_call(spec, model="gpt-image-2", prompt="a cat", n=200_000_001)

        client.images.generate.assert_called_once()
        assert check.call_args.kwargs["estimated_media"] is None

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_media_only_quantity_confirms_with_zeroed_token_details(self) -> None:
        # Compat image (Together FLUX): usage: null, so no TOKEN basis — but the
        # request-derived per-image MediaUsage IS observable. Confirm still fires
        # (never a silent $0) with a zeroed TokenDetails carrier + the MediaUsage.
        client, _ = _sync_client()
        client.images.generate.return_value = SimpleNamespace(usage=None, service_tier=None)
        solwyn = _build_sync(client)
        spec = _media_spec(
            extract=lambda _r: None,
            measure=lambda _kwargs: None,
            measure_media=lambda _kwargs, _response: MediaUsage(image_count=1),
            estimate_media=lambda _kwargs: MediaUsage(image_count=1),
        )
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_images
            ),
        ):
            solwyn._media_call(spec, model="black-forest-labs/FLUX.1-schnell", prompt="a cat")

        settle.assert_called_once()
        confirm, event = settle.call_args.args
        assert confirm.token_details == TokenDetails()  # zeroed token carrier
        assert confirm.media_usage.image_count == 1
        assert event.token_details is None  # no token basis observed
        assert event.input_tokens == 0
        assert event.media_usage.image_count == 1

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_budget_denied_raises_and_reports_without_dispatch(self) -> None:
        client, _ = _sync_client()
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_deny()) as check,
            patch.object(solwyn._solwyn_reporter, "report") as report,
            solwyn_pkg.run("denied-media", tags={"team": "platform"}) as run_id,
            pytest.raises(BudgetExceededError),
        ):
            solwyn._media_call(
                _media_spec(
                    estimate_media=lambda _kwargs: MediaUsage(
                        image_count=2,
                        resolution="1024x1024",
                        quality="hd",
                    )
                ),
                model="gpt-image-2",
                prompt="hi",
                n=2,
                solwyn_tags={"job": "embed"},
            )

        client.embeddings.create.assert_not_called()
        denied = report.call_args.args[0]
        assert denied.status == CallStatus.BUDGET_DENIED
        assert denied.modality == "image"  # the denied event carries it too
        assert denied.media_usage == check.call_args.kwargs["estimated_media"]
        assert denied.media_usage.image_count == 2
        assert denied.media_usage.resolution == "1024x1024"
        assert denied.media_usage.quality == "hd"
        assert denied.agent_run_id == run_id
        assert denied.tags == {"team": "platform", "job": "embed"}
        assert denied.deny_source == "server"
        assert denied.deny_reason == "monthly"
        assert denied.denied_by_period is None
        assert denied.estimated_output_bound == check.call_args.kwargs["estimated_output_bound"]

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_run_stopped_raises_typed_error_without_media_dispatch(self) -> None:
        client, _ = _sync_client()
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                return_value=_deny("run_stopped"),
            ),
            patch.object(solwyn._solwyn_reporter, "report") as report,
            solwyn_pkg.run("dashboard-stopped-media") as run_id,
            pytest.raises(RunStoppedError) as exc_info,
        ):
            solwyn._media_call(
                _media_spec(
                    estimate_media=lambda _kwargs: MediaUsage(
                        generation_count=1,
                        video_seconds=12.5,
                    )
                ),
                model="sora-2",
                prompt="hi",
            )

        client.embeddings.create.assert_not_called()
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

        error = exc_info.value
        assert type(error) is RunStoppedError
        assert not isinstance(error, BudgetExceededError)
        assert str(error) == f"Agent run {run_id} was stopped (server: run_stopped)"
        assert error.agent_run_id == run_id
        assert error.reason == "run_stopped"
        assert error.source == "server"
        denied = report.call_args.args[0]
        assert denied.modality == "image"
        assert denied.media_usage.generation_count == 1
        assert denied.media_usage.video_seconds == 12.5

    def test_pre_gate_media_stop_receipt_uses_admission_quantity(self) -> None:
        from solwyn._run_control import mark_terminated

        client, _ = _sync_client()
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._solwyn_budget, "check_budget") as check,
            patch.object(solwyn._solwyn_reporter, "report") as report,
            solwyn_pkg.run("pre-gated-audio") as run_id,
        ):
            mark_terminated(run_id, reason="velocity:repeat_size", source="local_velocity")
            with pytest.raises(RunStoppedError):
                solwyn._media_call(
                    MediaSurfaceSpec(
                        surface="audio.speech",
                        modality="audio",
                        extract_usage=lambda _response: None,
                        measure_request=lambda _kwargs: None,
                        estimate_media=lambda _kwargs: MediaUsage(input_characters=321),
                    ),
                    model="gpt-4o-mini-tts",
                    input="hi",
                )

        check.assert_not_called()
        denied = report.call_args.args[0]
        assert denied.modality == "audio"
        assert denied.media_usage.input_characters == 321
        assert (denied.deny_source, denied.deny_reason, denied.denied_by_period) == (
            "run_terminated",
            "velocity:repeat_size",
            "run_stopped",
        )
        client.audio.speech.create.assert_not_called()
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_unsupported_surface_reports_error_then_raises(self) -> None:
        client, _ = _sync_client()
        solwyn = _build_sync(client)
        # No prepare_media_call patch -> the real OpenAI adapter serves embeddings,
        # images, audio, and video but still raises for an unrecognized surface.
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._solwyn_reporter, "report") as report,
            pytest.raises(UnsupportedSurfaceError),
        ):
            solwyn._media_call(_spec(surface="translations"), model="whisper-1", input="hi")

        event = report.call_args.args[0]
        assert event.status == CallStatus.ERROR
        assert event.failover_error_class == "UnsupportedSurfaceError"

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()


@pytest.mark.unit
class TestMediaDeadlineExpiry:
    """No media I/O may START outside the failover window.

    PJ-8/R7 decoupled connect from read, so an expired window no longer bounds
    the call on its own: _hop_connect_slice falls to its 0.001s floor, which a
    WARM POOLED connection satisfies — and the hop would then read for the full
    (default 600s) bound. Before the split, the single whole-request timeout
    made that impossible. The chat walk rejects this exact state; media must
    too, rather than letting pool warmth decide.
    """

    def test_expired_preflight_never_calls_provider(self) -> None:
        # Arrange: the budget pre-flight outlives the whole failover window.
        client, _ = _sync_client()
        solwyn = _build_sync(client, failover_total_timeout=_TINY_WINDOW)

        def slow_check(**_kwargs):
            time.sleep(_SLOW_PREFLIGHT)
            return _allow()

        # Act + Assert
        with (
            patch.object(solwyn._solwyn_budget, "check_budget", side_effect=slow_check),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            pytest.raises(ProviderUnavailableError) as exc_info,
        ):
            solwyn._media_call(_spec(), model="text-embedding-3-small", input="hello world")

        assert "deadline expired" in str(exc_info.value)
        assert exc_info.value.attempted == ["openai"]
        # THE point of the test: no request was ever sent.
        client.embeddings.create.assert_not_called()

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_expired_preflight_releases_the_reservation(self) -> None:
        # Nothing will settle this call, so the reservation must be handed back
        # rather than stranded until the 900s sweep (parity with the chat gate
        # and with the media dispatch-error path).
        client, _ = _sync_client()
        solwyn = _build_sync(client, failover_total_timeout=_TINY_WINDOW)

        def slow_check(**_kwargs):
            time.sleep(_SLOW_PREFLIGHT)
            return _allow("res_media")

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", side_effect=slow_check),
            patch.object(solwyn._solwyn_budget, "release_reservation") as release,
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            pytest.raises(ProviderUnavailableError),
        ):
            solwyn._media_call(_spec(), model="text-embedding-3-small", input="hello world")

        release.assert_called_once()

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_budget_denial_still_wins_over_expiry(self) -> None:
        # Ordering control: a denied budget is the more specific answer, and the
        # gate must not mask it. Both conditions hold here.
        client, _ = _sync_client()
        solwyn = _build_sync(client, failover_total_timeout=_TINY_WINDOW)

        def slow_deny(**_kwargs):
            time.sleep(_SLOW_PREFLIGHT)
            return _deny()

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", side_effect=slow_deny),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            pytest.raises(BudgetExceededError),
        ):
            solwyn._media_call(_spec(), model="text-embedding-3-small", input="hello world")

        client.embeddings.create.assert_not_called()

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    def test_live_window_still_dispatches(self) -> None:
        # The guard on the guard: a healthy window must NOT be gated. Without
        # this, a gate that always fired would pass every assertion above.
        client, resp = _sync_client()
        solwyn = _build_sync(client, failover_total_timeout=30.0)

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", return_value=_allow(None)),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            result = solwyn._media_call(
                _spec(), model="text-embedding-3-small", input="hello world"
            )

        assert result is resp
        client.embeddings.create.assert_called_once()

        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()

    @pytest.mark.asyncio
    async def test_async_expired_preflight_never_calls_provider(self) -> None:
        # Async mirror — a separate _media_call implementation, so it needs its
        # own control rather than inheriting the sync one's coverage.
        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, failover_total_timeout=_TINY_WINDOW)

        async def slow_check(**_kwargs):
            await asyncio.sleep(_SLOW_PREFLIGHT)
            return _allow()

        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(side_effect=slow_check)
            ),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            pytest.raises(ProviderUnavailableError) as exc_info,
        ):
            await solwyn._media_call(_spec(), model="text-embedding-3-small", input="hello world")

        assert "deadline expired" in str(exc_info.value)
        assert exc_info.value.attempted == ["openai"]
        client.embeddings.create.assert_not_awaited()

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_expired_preflight_releases_the_reservation(self) -> None:
        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, failover_total_timeout=_TINY_WINDOW)

        async def slow_check(**_kwargs):
            await asyncio.sleep(_SLOW_PREFLIGHT)
            return _allow("res_media")

        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(side_effect=slow_check)
            ),
            patch.object(solwyn._solwyn_budget, "release_reservation") as release,
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
            pytest.raises(ProviderUnavailableError),
        ):
            await solwyn._media_call(_spec(), model="text-embedding-3-small", input="hello world")

        release.assert_called_once()

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_live_window_still_dispatches(self) -> None:
        client, resp = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, failover_total_timeout=30.0)

        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow(None))
            ),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            result = await solwyn._media_call(
                _spec(), model="text-embedding-3-small", input="hello world"
            )

        assert result is resp
        client.embeddings.create.assert_awaited_once()

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()


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
    async def test_directive_refreshes_media_deadline_before_dispatch(self) -> None:
        client, resp = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, failover_total_timeout=91.0)

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                new=AsyncMock(return_value=_allow(None, failover_tuning_allowed=False)),
            ),
            patch.object(solwyn._solwyn_reporter, "report"),
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            result = await solwyn._media_call(
                _spec(),
                model="text-embedding-3-small",
                input="hello world",
            )

        assert result is resp
        assert solwyn._solwyn_config.failover_total_timeout == 30.0
        assert 0.0 < client.with_options.call_args.kwargs["timeout"].connect <= 30.0
        assert client.with_options.call_args.kwargs["timeout"].read == 600.0
        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_success_confirms_and_reports_primary_only(self) -> None:
        client, resp = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)

        async def create_after_project_learned(**_kwargs):
            assert solwyn._solwyn_reporter._breaker_project_id == VALID_PROJECT_ID
            return resp

        client.embeddings.create.side_effect = create_after_project_learned
        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow())
            ) as check,
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_embeddings
            ),
        ):
            async with solwyn_pkg.run("orchestrator") as parent_run_id:
                async with solwyn_pkg.run("async-media", tags={"team": "platform"}) as run_id:
                    result = await solwyn._media_call(
                        _spec(),
                        model="text-embedding-3-small",
                        input="hello",
                        solwyn_tags={"job": "embed"},
                    )

        assert result is resp
        client.embeddings.create.assert_awaited_once()
        assert check.call_args.kwargs["provider"] == "openai"
        assert check.call_args.kwargs["modality"] == "embedding"
        assert check.call_args.kwargs["agent_run_id"] == run_id
        settle.assert_called_once()
        confirm, event = settle.call_args.args
        assert confirm.token_details.input_tokens == 99
        assert confirm.modality == "embedding"
        assert event.status == CallStatus.SUCCESS
        assert event.is_model_fallback is False
        assert event.modality == "embedding"
        assert event.agent_run_id == run_id
        assert event.parent_agent_run_id == parent_run_id
        assert event.tags == {"team": "platform", "job": "embed"}
        assert check.call_args.kwargs["tags"] == event.tags
        assert "solwyn_tags" not in client.embeddings.create.call_args.kwargs

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_both_bases_flow_when_token_and_media_observable(self) -> None:
        client, _ = _async_client()
        client.images.generate = AsyncMock(return_value=SimpleNamespace(service_tier=None))
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        spec = _media_spec(
            extract=lambda _r: TokenDetails(input_tokens=222, image_input_tokens=194),
            measure_media=lambda _kwargs, _response: MediaUsage(image_count=2),
            estimate_media=lambda _kwargs: MediaUsage(image_count=2),
        )
        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_allow())
            ) as check,
            patch.object(solwyn._solwyn_reporter, "report_settlement") as settle,
            patch.object(
                solwyn._solwyn_runtimes[0].adapter, "prepare_media_call", _route_to_images
            ),
        ):
            await solwyn._media_call(spec, model="gpt-image-2", prompt="a cat", n=2)

        client.images.generate.assert_awaited_once()
        assert check.call_args.kwargs["estimated_media"].image_count == 2
        settle.assert_called_once()
        confirm, event = settle.call_args.args
        assert confirm.token_details.image_input_tokens == 194
        assert confirm.media_usage.image_count == 2
        assert event.modality == "image"
        assert event.media_usage.image_count == 2

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_budget_denied_raises_without_dispatch(self) -> None:
        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(
                solwyn._solwyn_budget, "check_budget", new=AsyncMock(return_value=_deny())
            ) as check,
            patch.object(solwyn._solwyn_reporter, "report") as report,
            pytest.raises(BudgetExceededError),
        ):
            async with solwyn_pkg.run("async-denied-media") as run_id:
                await solwyn._media_call(
                    _media_spec(
                        estimate_media=lambda _kwargs: MediaUsage(
                            audio_seconds=7.25,
                            quality="standard",
                        )
                    ),
                    model="audio-model",
                    prompt="hi",
                )

        client.embeddings.create.assert_not_awaited()
        denied = report.call_args.args[0]
        assert denied.status == CallStatus.BUDGET_DENIED
        assert denied.agent_run_id == run_id
        assert denied.deny_source == "server"
        assert denied.deny_reason == "monthly"
        assert denied.denied_by_period is None
        assert denied.media_usage == check.call_args.kwargs["estimated_media"]
        assert denied.media_usage.audio_seconds == 7.25
        assert denied.estimated_output_bound == check.call_args.kwargs["estimated_output_bound"]

        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_run_stopped_raises_typed_error_without_media_dispatch(self) -> None:
        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, budget_mode=BudgetMode.HARD_DENY)

        with (
            patch.object(
                solwyn._solwyn_budget,
                "check_budget",
                new=AsyncMock(return_value=_deny("run_stopped")),
            ),
            patch.object(solwyn._solwyn_reporter, "report"),
        ):
            async with solwyn_pkg.run("dashboard-stopped-media-async") as run_id:
                with pytest.raises(RunStoppedError) as exc_info:
                    await solwyn._media_call(_spec(), model="text-embedding-3-small", input="hi")

        client.embeddings.create.assert_not_awaited()
        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()

        error = exc_info.value
        assert type(error) is RunStoppedError
        assert not isinstance(error, BudgetExceededError)
        assert str(error) == f"Agent run {run_id} was stopped (server: run_stopped)"
        assert error.agent_run_id == run_id
        assert error.reason == "run_stopped"
        assert error.source == "server"

    @pytest.mark.asyncio
    async def test_pre_gate_media_stop_receipt_carries_deny_attribution(self) -> None:
        from solwyn._run_control import mark_terminated

        client, _ = _async_client()
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY, budget_mode=BudgetMode.HARD_DENY)

        with (
            patch.object(solwyn._solwyn_budget, "check_budget", new=AsyncMock()) as check,
            patch.object(solwyn._solwyn_reporter, "report") as report,
        ):
            async with solwyn_pkg.run("async-pre-gated-audio") as run_id:
                mark_terminated(run_id, reason="velocity:repeat_size", source="local_velocity")
                with pytest.raises(RunStoppedError):
                    await solwyn._media_call(
                        MediaSurfaceSpec(
                            surface="audio.speech",
                            modality="audio",
                            extract_usage=lambda _response: None,
                            measure_request=lambda _kwargs: None,
                            estimate_media=lambda _kwargs: MediaUsage(input_characters=321),
                        ),
                        model="gpt-4o-mini-tts",
                        input="hi",
                    )

        check.assert_not_awaited()
        denied = report.call_args.args[0]
        assert denied.modality == "audio"
        assert denied.media_usage.input_characters == 321
        assert (denied.deny_source, denied.deny_reason, denied.denied_by_period) == (
            "run_terminated",
            "velocity:repeat_size",
            "run_stopped",
        )
        client.audio.speech.create.assert_not_called()
        await solwyn._solwyn_budget._http.aclose()
        await solwyn._solwyn_reporter._http.aclose()
