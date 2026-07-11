"""Tests for the audio.transcriptions media surface (openai dialect).

``client.audio.transcriptions.create()`` routes through the media lifecycle:
token-billed models (gpt-4o-transcribe family) settle on ``audio_input_tokens``,
the duration-billed model (whisper-1) on ``audio_seconds``, and a non-JSON
response_format (text/srt/vtt) is tracked UNPRICED with a one-time hint. The
sibling ``speech`` sub-surface is intercepted too (see test_audio_speech.py);
``translations`` warns-once and passes through.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._token_details import TokenDetails
from solwyn._types import BudgetMode, CallStatus
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import BudgetExceededError
from solwyn.providers.openai import (
    _extract_transcription_usage,
    _measure_transcription_media,
    _reset_transcription_unpriced_warning,
    _transcription_usage_basis,
)

_OPENAI_LOGGER = "solwyn.providers.openai"


@pytest.fixture(autouse=True)
def _reset_warn_latches() -> None:
    """Reset both warn-once latches so warn tests stay order-independent."""
    _reset_unmetered_spend_warnings()
    _reset_transcription_unpriced_warning()


# ---------------------------------------------------------------------------
# Response builders (duck-typed; no provider SDK import)
# ---------------------------------------------------------------------------


def _token_usage(*, input_tokens: int = 100, audio_tokens: int = 30, output_tokens: int = 20):
    """gpt-4o-transcribe token-usage block: input_token_details has text+audio."""
    return SimpleNamespace(
        type="tokens",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details=SimpleNamespace(
            text_tokens=input_tokens - audio_tokens,
            audio_tokens=audio_tokens,
        ),
    )


def _duration_usage(seconds: int = 7):
    """whisper-1 duration-usage block: seconds is the whole-second billed basis."""
    return SimpleNamespace(type="duration", seconds=seconds)


def _token_response():
    return SimpleNamespace(text="hello there", usage=_token_usage())


def _duration_response():
    return SimpleNamespace(text="hello there", usage=_duration_usage(7))


# ---------------------------------------------------------------------------
# Extractor / discriminator units
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscriptionUsageExtraction:
    def test_token_usage_maps_audio_input_tokens(self) -> None:
        details = _extract_transcription_usage(_token_response())
        assert details is not None
        # input_tokens is the total (text + audio); audio_tokens rides the audio
        # subset, the text side is derived server-side (input - audio).
        assert details.input_tokens == 100
        assert details.output_tokens == 20
        assert details.audio_input_tokens == 30
        assert details.is_estimated is False
        # Token models carry no MediaUsage basis.
        assert _measure_transcription_media({}, _token_response()) is None

    def test_duration_usage_maps_audio_seconds(self) -> None:
        # whisper-1: the duration basis rides MediaUsage.audio_seconds; no tokens.
        assert _extract_transcription_usage(_duration_response()) is None
        media = _measure_transcription_media({}, _duration_response())
        assert media is not None
        assert media.audio_seconds == 7.0
        assert media.is_estimated is False

    def test_duration_present_on_default_json_not_only_verbose(self) -> None:
        # The billed basis is usage.seconds (integer), present on default json;
        # a fractional top-level `duration` field is NEVER preferred.
        response = SimpleNamespace(text="hi", duration=7.84, usage=_duration_usage(8))
        media = _measure_transcription_media({}, response)
        assert media is not None
        assert media.audio_seconds == 8.0  # usage.seconds, not the 7.84 duration

    def test_non_json_string_response_is_unpriced_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # text/srt/vtt return a plain string with no usage -> both bases None +
        # a single hint to use a JSON response_format.
        response = "just the transcript text"
        with caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER):
            token, media = _transcription_usage_basis(response)
            # A second call must not re-warn (per-process latch).
            _transcription_usage_basis(response)

        assert token is None
        assert media is None
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "unpriced" in message.lower()
        assert "response_format" in message
        assert "json" in message.lower()
        # Never echoes response content.
        assert "just the transcript text" not in message

    def test_unknown_usage_type_is_unpriced_without_json_hint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A usage block with an unrecognized `type` is unpriced but is NOT a
        # non-JSON response_format case, so it must not emit the JSON hint.
        response = SimpleNamespace(usage=SimpleNamespace(type="mystery", widgets=3))
        with caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER):
            assert _transcription_usage_basis(response) == (None, None)
        assert caplog.records == []

    def test_zero_token_counts_yield_none(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                type="tokens",
                input_tokens=0,
                output_tokens=0,
                input_token_details=SimpleNamespace(text_tokens=0, audio_tokens=0),
            )
        )
        assert _extract_transcription_usage(response) is None

    def test_garbage_seconds_yields_none(self) -> None:
        for seconds in (0, -3, "5", True, None):
            response = SimpleNamespace(usage=SimpleNamespace(type="duration", seconds=seconds))
            assert _measure_transcription_media({}, response) is None

    def test_never_raises_on_arbitrary_response(self) -> None:
        # Duck-typed and total: any shape degrades to (None, None), never raises.
        for response in (None, object(), 42, SimpleNamespace()):
            assert _transcription_usage_basis(response) == (None, None)


# ---------------------------------------------------------------------------
# Sync proxy integration
# ---------------------------------------------------------------------------


def _mock_transcription_client(response: object) -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    client.audio.transcriptions.create.return_value = response
    return client


def _allow(reservation_id: str | None = "res_audio") -> SimpleNamespace:
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
class TestAudioTranscriptionsProxy:
    def test_token_model_reports_audio_input_tokens_and_confirms(self) -> None:
        client = _mock_transcription_client(_token_response())
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()) as check,
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            result = solwyn.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=b"audio-bytes"
            )

        assert result is client.audio.transcriptions.create.return_value
        client.audio.transcriptions.create.assert_called_once()
        # Budget checked on the audio modality.
        assert check.call_args.kwargs["modality"] == "audio"
        # Confirm settles on the token basis with the audio subset.
        confirm.assert_called_once()
        assert confirm.call_args.args[2].audio_input_tokens == 30
        assert confirm.call_args.kwargs["modality"] == "audio"
        assert confirm.call_args.kwargs["media_usage"] is None
        # Metadata event carries the token basis + audio modality.
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.modality == "audio"
        assert event.input_tokens == 100
        assert event.token_details.audio_input_tokens == 30
        assert event.media_usage is None
        _close_sync(solwyn)

    def test_duration_model_reports_audio_seconds_and_confirms(self) -> None:
        client = _mock_transcription_client(_duration_response())
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            solwyn.audio.transcriptions.create(model="whisper-1", file=b"audio-bytes")

        # No token basis; confirm still fires on the MediaUsage duration basis
        # with a zeroed TokenDetails carrier (never a silent $0).
        confirm.assert_called_once()
        assert confirm.call_args.args[2] == TokenDetails()
        assert confirm.call_args.kwargs["media_usage"].audio_seconds == 7.0
        assert confirm.call_args.kwargs["modality"] == "audio"
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.modality == "audio"
        assert event.token_details is None
        assert event.input_tokens == 0
        assert event.media_usage.audio_seconds == 7.0
        _close_sync(solwyn)

    def test_non_json_response_is_unpriced_tracked_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A plain-string (text/srt/vtt) response: no usage -> both bases None ->
        # SUCCESS reported but NO confirm (unpriced), plus one JSON-hint warning.
        client = _mock_transcription_client("plain transcript")
        solwyn = _build_sync(client)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow()),
            patch.object(solwyn._budget, "confirm_cost") as confirm,
            patch.object(solwyn._reporter, "report") as report,
            caplog.at_level(logging.WARNING, logger=_OPENAI_LOGGER),
        ):
            solwyn.audio.transcriptions.create(model="whisper-1", file=b"x", response_format="text")

        confirm.assert_not_called()  # no observed basis -> never settle a real $0
        event = report.call_args.args[0]
        assert event.status == CallStatus.SUCCESS
        assert event.modality == "audio"
        assert event.token_details is None
        assert event.media_usage is None
        assert event.input_tokens == 0
        assert len(caplog.records) == 1
        assert "response_format" in caplog.records[0].getMessage()
        _close_sync(solwyn)

    def test_transcriptions_getattr_passthrough_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _mock_transcription_client(_token_response())
        client.audio.transcriptions.with_raw_response = MagicMock(return_value="raw")
        solwyn = _build_sync(client)
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            assert solwyn.audio.transcriptions.with_raw_response() == "raw"
        assert caplog.records == []
        _close_sync(solwyn)

    def test_budget_denied_short_circuits(self) -> None:
        client = _mock_transcription_client(_token_response())
        solwyn = _build_sync(client, budget_mode=BudgetMode.HARD_DENY)
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_deny()),
            patch.object(solwyn._reporter, "report") as report,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.audio.transcriptions.create(model="gpt-4o-transcribe", file=b"x")

        client.audio.transcriptions.create.assert_not_called()
        denied = report.call_args.args[0]
        assert denied.status == CallStatus.BUDGET_DENIED
        assert denied.modality == "audio"
        _close_sync(solwyn)

    def test_translations_warns_once_and_passes_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _mock_transcription_client(_token_response())
        client.audio.translations = MagicMock()
        solwyn = _build_sync(client)
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            first = solwyn.audio.translations
            second = solwyn.audio.translations

        assert first is client.audio.translations
        assert second is client.audio.translations
        assert len(caplog.records) == 1
        assert "surface 'translations'" in caplog.records[0].getMessage()
        _close_sync(solwyn)

    def test_accessing_audio_attribute_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The audio attribute is intercepted machinery now: touching it (and its
        # transcriptions/speech sub-proxies) is silent; only translations warns.
        client = _mock_transcription_client(_token_response())
        solwyn = _build_sync(client)
        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            _ = solwyn.audio
            _ = solwyn.audio.transcriptions
            _ = solwyn.audio.speech
        assert caplog.records == []
        _close_sync(solwyn)


# ---------------------------------------------------------------------------
# Async proxy integration
# ---------------------------------------------------------------------------


def _mock_async_transcription_client(response: object) -> MagicMock:
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "AsyncOpenAI"
    client.with_options.return_value = client
    client.audio.transcriptions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.unit
class TestAsyncAudioTranscriptionsProxy:
    @pytest.mark.asyncio
    async def test_async_token_model_intercepted(self) -> None:
        client = _mock_async_transcription_client(_token_response())
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        with (
            patch.object(solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow())),
            patch.object(solwyn._budget, "confirm_cost", new=AsyncMock()) as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            await solwyn.audio.transcriptions.create(model="gpt-4o-transcribe", file=b"audio")

        client.audio.transcriptions.create.assert_awaited_once()
        confirm.assert_awaited_once()
        assert confirm.call_args.args[2].audio_input_tokens == 30
        event = report.call_args.args[0]
        assert event.modality == "audio"
        assert event.token_details.audio_input_tokens == 30
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.asyncio
    async def test_async_duration_model_intercepted(self) -> None:
        client = _mock_async_transcription_client(_duration_response())
        solwyn = AsyncSolwyn(client, api_key=VALID_API_KEY)
        with (
            patch.object(solwyn._budget, "check_budget", new=AsyncMock(return_value=_allow())),
            patch.object(solwyn._budget, "confirm_cost", new=AsyncMock()) as confirm,
            patch.object(solwyn._reporter, "report") as report,
        ):
            await solwyn.audio.transcriptions.create(model="whisper-1", file=b"audio")

        confirm.assert_awaited_once()
        assert confirm.call_args.kwargs["media_usage"].audio_seconds == 7.0
        event = report.call_args.args[0]
        assert event.modality == "audio"
        assert event.media_usage.audio_seconds == 7.0
        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
