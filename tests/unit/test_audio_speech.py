"""Tests for the audio.speech (TTS) media surface (openai dialect).

``client.audio.speech.create()`` routes through the media lifecycle: TTS
responses carry NO usage metadata, so the sole billable basis is the request's
``input`` character count, measured in the firewall (``input_characters``) and
priced server-side. Token-billed TTS models (gpt-4o-mini-tts) publish no usage
at all — those calls are CARVED OUT: one warning per process, then a direct
pass-through to the raw client, untracked (a warned pass-through, never a silent
$0).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._proxies import _speech_spec
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, CallStatus
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import BudgetExceededError
from solwyn.providers.openai import (
    _AUDIO_OP_KEY,
    _is_untracked_tts_model,
    _reset_untracked_tts_warning,
)

_OPENAI_LOGGER = "solwyn.providers.openai"


@pytest.fixture(autouse=True)
def _reset_warn_latches() -> None:
    """Reset both warn-once latches so warn tests stay order-independent."""
    _reset_unmetered_spend_warnings()
    _reset_untracked_tts_warning()


# ---------------------------------------------------------------------------
# Spec + carve-out units
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpeechSpecAndCarveOut:
    def test_spec_has_no_token_basis_and_measures_input_characters(self) -> None:
        spec = _speech_spec()
        assert spec.surface == "audio"
        assert spec.modality == "audio"
        # TTS responses carry no usage -> no token basis, ever.
        assert spec.extract_usage(SimpleNamespace(usage=SimpleNamespace(type="tokens"))) is None
        assert spec.measure_request({"input": "hello"}) is None
        # The billable basis is the request's input character count.
        assert spec.measure_media is not None
        media = spec.measure_media({"input": "hello world"}, object())
        assert media is not None
        assert media.input_characters == len("hello world")
        assert media.is_estimated is False
        # estimate_media mirrors measure_media so the check carries the exact cost.
        assert spec.estimate_media is not None
        estimate = spec.estimate_media({"input": "hello world"})
        assert estimate is not None
        assert estimate.input_characters == len("hello world")

    def test_untracked_model_prefix_matches_family_and_snapshots(self) -> None:
        assert _is_untracked_tts_model("gpt-4o-mini-tts") is True
        assert _is_untracked_tts_model("gpt-4o-mini-tts-2026-01-01") is True
        # Char-billed models and non-strings are NOT carved out.
        assert _is_untracked_tts_model("tts-1") is False
        assert _is_untracked_tts_model("tts-1-hd") is False
        assert _is_untracked_tts_model(None) is False
        assert _is_untracked_tts_model(123) is False


# ---------------------------------------------------------------------------
# Sync proxy integration
# ---------------------------------------------------------------------------


def _mock_speech_client(response: object) -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    client.audio.speech.create.return_value = response
    return client


def _allow(reservation_id: str | None = "res_tts") -> SimpleNamespace:
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


def _build_sync(client: MagicMock, **overrides) -> Solwyn:
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, api_key=VALID_API_KEY, **overrides)
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    return solwyn


def _close_sync(solwyn: Solwyn) -> None:
    solwyn._reporter._http.close()
    solwyn._budget._http.close()


@pytest.mark.unit
class TestAudioSpeechProxy:
    def test_char_billed_model_reports_input_characters_and_confirms(self) -> None:
        client = _mock_speech_client(b"raw-audio-bytes")
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            result = solwyn.audio.speech.create(model="tts-1", voice="alloy", input="hello world")

        assert result is client.audio.speech.create.return_value
        client.audio.speech.create.assert_called_once()
        # The op marker routed speech.create but is stripped before the SDK call.
        assert _AUDIO_OP_KEY not in client.audio.speech.create.call_args.kwargs
        # Budget checked on the audio modality with the EXACT pre-flight char count.
        assert check.call_args.kwargs["modality"] == "audio"
        assert check.call_args.kwargs["estimated_media"].input_characters == len("hello world")
        # Confirm settles on the char basis with a zeroed TokenDetails carrier
        # (TTS has no token basis; never a silent $0).
        confirm.assert_called_once()
        assert confirm.call_args.args[2] == TokenDetails()
        assert confirm.call_args.kwargs["media_usage"].input_characters == len("hello world")
        assert confirm.call_args.kwargs["modality"] == "audio"
        # Metadata event carries the char basis + audio modality, no token basis.
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.modality == "audio"
        assert event.token_details is None
        assert event.input_tokens == 0
        assert event.media_usage.input_characters == len("hello world")
        _close_sync(solwyn)

    def test_untracked_token_billed_model_warns_once_and_passes_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # gpt-4o-mini-tts publishes no usage -> carved out: warn once, then a
        # direct pass-through to the raw client, with NO budget check / confirm /
        # report, and NO op marker injected.
        client = _mock_speech_client(b"raw-audio-bytes")
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget") as check,
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
            caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER),
        ):
            first = solwyn.audio.speech.create(model="gpt-4o-mini-tts", input="hi", voice="alloy")
            second = solwyn.audio.speech.create(model="gpt-4o-mini-tts", input="again")

        assert first is client.audio.speech.create.return_value
        assert second is client.audio.speech.create.return_value
        # Untracked: the lifecycle never runs.
        check.assert_not_called()
        confirm.assert_not_called()
        report.assert_not_called()
        assert client.audio.speech.create.call_count == 2
        # Raw kwargs only — no injected op marker on the carve-out path.
        assert _AUDIO_OP_KEY not in client.audio.speech.create.call_args_list[0].kwargs
        # One warning per process, however many carved-out calls are made.
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "untracked" in message.lower()
        assert "unobservable" in message.lower()
        # Never echoes request content.
        assert "hi" not in message.split()
        _close_sync(solwyn)

    def test_dated_snapshot_of_untracked_model_is_also_carved_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _mock_speech_client(b"audio")
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget") as check,
            patch.object(solwyn._reporter, "report") as report,
            caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER),
        ):
            solwyn.audio.speech.create(model="gpt-4o-mini-tts-2026-01-01", input="hi")

        check.assert_not_called()
        report.assert_not_called()
        client.audio.speech.create.assert_called_once()
        assert len(caplog.records) == 1
        _close_sync(solwyn)

    def test_speech_getattr_passthrough_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _mock_speech_client(b"audio")
        client.audio.speech.with_streaming_response = MagicMock(return_value="streamed")
        solwyn = _build_sync(client)
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            assert solwyn.audio.speech.with_streaming_response() == "streamed"
        assert caplog.records == []
        _close_sync(solwyn)

    def test_budget_denied_short_circuits(self) -> None:
        client = _mock_speech_client(b"audio")
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_deny()),
            patch.object(solwyn._reporter, "report") as report,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.audio.speech.create(model="tts-1", input="blocked")

        client.audio.speech.create.assert_not_called()
        denied = report.call_args.args[0]
        assert denied.status == CallStatus.BUDGET_DENIED
        assert denied.modality == "audio"
        _close_sync(solwyn)


# ---------------------------------------------------------------------------
# Async proxy integration
# ---------------------------------------------------------------------------


def _mock_async_speech_client(response: object) -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI"
    client.with_options.return_value = client
    client.audio.speech.create = AsyncMock(return_value=response)
    return client


@pytest.mark.unit
class TestAsyncAudioSpeechProxy:
    @pytest.mark.asyncio
    async def test_async_char_billed_model_intercepted(self) -> None:
        client = _mock_async_speech_client(b"raw-audio")
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        with (
            patch.object(solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow())),
            patch.object(solwyn._budget, "confirm_cost", new=AsyncMock()) as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            await solwyn.audio.speech.create(model="tts-1-hd", input="hello world", voice="nova")

        client.audio.speech.create.assert_awaited_once()
        assert _AUDIO_OP_KEY not in client.audio.speech.create.await_args.kwargs
        confirm.assert_awaited_once()
        assert confirm.call_args.kwargs["media_usage"].input_characters == len("hello world")
        event = report.call_args.args[0]
        assert event.modality == "audio"
        assert event.media_usage.input_characters == len("hello world")
        assert event.token_details is None
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_untracked_model_warns_and_passes_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _mock_async_speech_client(b"raw-audio")
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        with (
            patch.object(solwyn._budget, "check_budget", new=AsyncMock()) as check,
            patch.object(solwyn._reporter, "report") as report,
            caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER),
        ):
            result = await solwyn.audio.speech.create(model="gpt-4o-mini-tts", input="hi")

        assert result == b"raw-audio"
        check.assert_not_called()
        report.assert_not_called()
        client.audio.speech.create.assert_awaited_once()
        assert len(caplog.records) == 1
        assert "untracked" in caplog.records[0].getMessage().lower()
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
