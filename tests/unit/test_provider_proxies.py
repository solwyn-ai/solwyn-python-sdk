"""Tests for provider-native API surface interception.

Ensures client.messages.create() (Anthropic) and
client.models.generate_content() (Google) route through _intercepted_call,
not __getattr__ pass-through.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock as AsyncMockFn
from unittest.mock import MagicMock, patch

import pytest
from conftest import (
    ALLOW_BUDGET_RESPONSE,
    VALID_API_KEY,
    VALID_PROJECT_ID,
    foreground_records,
)

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import BudgetExceededError
from solwyn.providers.openai import _IMAGE_OP_KEY


@pytest.fixture(autouse=True)
def _reset_spend_surface_latch() -> None:
    """Reset the per-process warn-once latch so warn tests stay order-independent."""
    _reset_unmetered_spend_warnings()


def _mock_anthropic_client():
    """Create a mock that looks like anthropic.Anthropic."""
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    # Per-hop with_options(timeout, max_retries) must return the same client so
    # the configured .messages.create mock is the one actually invoked.
    client.with_options.return_value = client
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="Hello")],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
        ),
    )
    return client


def _mock_google_client():
    """Create a mock that looks like google.genai.Client."""
    client = MagicMock()
    client.__class__.__module__ = "google.genai._client"
    # Per-hop with_options must return the same client (see _mock_anthropic_client).
    client.with_options.return_value = client
    client.models.generate_content.return_value = SimpleNamespace(
        text="Hello",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=0,
            cached_content_token_count=0,
            tool_use_prompt_token_count=0,
        ),
    )
    return client


def _mock_google_embeddings_client(*, prompt_token_count: int = 55, usage: bool = True):
    """Create a mock that looks like google.genai.Client for embed_content.

    With ``usage=True`` the response carries ``usage_metadata.prompt_token_count``
    (the snake_case attribute the SDK would expose); with ``usage=False`` it omits
    usage entirely, mirroring today's real ``EmbedContentResponse`` (which surfaces
    no usage) so the request-side estimator drives billing.
    """
    client = MagicMock()
    client.__class__.__module__ = "google.genai._client"
    # Per-hop with_options must return the same client (see _mock_anthropic_client).
    client.with_options.return_value = client
    fields: dict[str, Any] = {"embeddings": [SimpleNamespace(values=[0.1, 0.2])]}
    if usage:
        fields["usage_metadata"] = SimpleNamespace(prompt_token_count=prompt_token_count)
    client.models.embed_content.return_value = SimpleNamespace(**fields)
    return client


def _mock_google_images_client():
    """Create a mock that looks like google.genai.Client for generate_images.

    imagen responses expose NO usage; the SDK derives the billable per-image
    quantity from ``config.number_of_images``. The mock's return value carries
    only generated images, no usage_metadata.
    """
    client = MagicMock()
    client.__class__.__module__ = "google.genai._client"
    client.with_options.return_value = client
    client.models.generate_images.return_value = SimpleNamespace(
        generated_images=[SimpleNamespace(image=SimpleNamespace(image_bytes=b"png"))]
    )
    return client


def _mock_google_videos_client():
    """Create a mock that looks like google.genai.Client for generate_videos.

    generate_videos is asynchronous: it returns a long-running OPERATION object
    (no usage), which the SDK passes back untouched for the caller to poll. The
    billable per-second quantity is derived from ``config.duration_seconds`` at
    ``config.resolution``.
    """
    client = MagicMock()
    client.__class__.__module__ = "google.genai._client"
    client.with_options.return_value = client
    client.models.generate_videos.return_value = SimpleNamespace(
        name="operations/veo-123", done=False, response=None
    )
    return client


def _mock_openai_embeddings_client(*, prompt_tokens: int = 42, usage: bool = True):
    """Create a mock that looks like openai.OpenAI() with an embeddings surface."""
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    # Per-hop with_options must return the same client (see _mock_anthropic_client).
    client.with_options.return_value = client
    usage_block = SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)],
        usage=usage_block if usage else None,
    )
    return client


def _image_usage(*, native: bool):
    """gpt-image usage block (native) or None (compat/dall-e usage-less)."""
    if not native:
        return None
    return SimpleNamespace(
        input_tokens=222,
        output_tokens=1024,
        total_tokens=1246,
        input_tokens_details=SimpleNamespace(image_tokens=194, text_tokens=28),
        output_tokens_details=SimpleNamespace(image_tokens=1024, text_tokens=0),
    )


def _mock_openai_images_client(*, native: bool = True):
    """Mock openai.OpenAI() exposing an images surface (generate + edit).

    ``native=True`` -> gpt-image token usage carrying image buckets; ``native=
    False`` -> a compat/dall-e response with ``usage=None`` (per-image only).
    """
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json="aGVsbG8=", url=None)],
        usage=_image_usage(native=native),
    )
    client.images.generate.return_value = response
    client.images.edit.return_value = response
    return client


def _mock_openai_videos_client():
    """Mock openai.OpenAI() exposing a videos surface (Sora).

    videos.create is asynchronous: it returns a video JOB object (no usage) which
    the SDK passes back untouched for the caller to poll. The billable per-second
    quantity is derived from the top-level ``seconds`` at the ``size``-derived
    resolution label.
    """
    client = MagicMock()
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    client.with_options.return_value = client
    client.videos.create.return_value = SimpleNamespace(id="video_123", status="queued", progress=0)
    return client


def _make_solwyn(client, **overrides):
    defaults = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    return Solwyn(client, **defaults)


def _mock_budget(solwyn, response=None):
    """Patch the budget enforcer to return an allow response."""
    resp = MagicMock()
    resp.json.return_value = response or ALLOW_BUDGET_RESPONSE
    resp.raise_for_status = MagicMock()
    return patch.object(solwyn._budget._http, "post", return_value=resp)


# ---------------------------------------------------------------------------
# Anthropic: client.messages.create()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicMessagesProxy:
    """client.messages.create() routes through _intercepted_call."""

    def test_messages_create_is_intercepted(self) -> None:
        client = _mock_anthropic_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.messages.create(
                model="claude-sonnet-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Should have called the underlying Anthropic client
        client.messages.create.assert_called_once()
        # Should have reported metadata
        assert len(reported) == 1
        assert reported[0].status == "success"
        assert reported[0].input_tokens == 100
        # Should return the Anthropic response
        assert result.content[0].text == "Hello"

        solwyn.close()

    def test_messages_create_checks_budget(self) -> None:
        client = _mock_anthropic_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.messages.create(
                model="claude-sonnet-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        # LLM should NOT have been called
        client.messages.create.assert_not_called()
        solwyn.close()

    def test_messages_getattr_passthrough(self) -> None:
        """Non-create attributes pass through to underlying client."""
        client = _mock_anthropic_client()
        client.messages.count_tokens = MagicMock(return_value=42)
        solwyn = _make_solwyn(client)
        assert solwyn.messages.count_tokens() == 42
        solwyn.close()


# ---------------------------------------------------------------------------
# Google: client.models.generate_content()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoogleModelsProxy:
    """client.models.generate_content() routes through _intercepted_call."""

    def test_generate_content_is_intercepted(self) -> None:
        client = _mock_google_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.models.generate_content(
                model="gemini-3.5-flash",
                contents="Hello",
            )

        client.models.generate_content.assert_called_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        assert result.text == "Hello"
        solwyn.close()

    def test_generate_content_stream_dispatches_correctly(self) -> None:
        """generate_content_stream() calls the correct underlying method via _force_stream."""
        client = _mock_google_client()
        # A real generate_content_stream yields chunk objects (not a single
        # complete response). First-chunk materialization iterates it, so
        # the mock must be an ITERABLE of chunks.
        chunk = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Hello")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=0,
                cached_content_token_count=0,
                tool_use_prompt_token_count=0,
            ),
        )
        client.models.generate_content_stream.return_value = iter([chunk])

        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.generate_content_stream(
                model="gemini-3.5-flash",
                contents="Hello",
            )

        # Should have called generate_content_stream, NOT generate_content
        client.models.generate_content_stream.assert_called_once()
        client.models.generate_content.assert_not_called()
        solwyn.close()

    def test_models_getattr_passthrough(self) -> None:
        """Non-generate attributes pass through."""
        client = _mock_google_client()
        client.models.list = MagicMock(return_value=["gemini-2.5-pro"])
        solwyn = _make_solwyn(client)
        assert solwyn.models.list() == ["gemini-2.5-pro"]
        solwyn.close()

    def test_generate_videos_is_intercepted_and_reports_video_seconds(self) -> None:
        # generate_videos (veo) routes through the media lifecycle, NOT the
        # __getattr__ warn-once pass-through. The call returns a long-running
        # operation with no usage, so the billable basis is the request-derived
        # per-second MediaUsage, ALWAYS is_estimated=True (settles at initiation).
        client = _mock_google_videos_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt="a cat",
                config={"duration_seconds": 8, "resolution": "720p"},
            )

        client.models.generate_videos.assert_called_once()
        assert len(reported) == 1
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        # veo reports no token usage — media is the sole billable basis.
        assert event.token_details is None
        assert event.input_tokens == 0
        assert event.media_usage.video_seconds == 8.0
        assert event.media_usage.resolution == "720p"
        assert event.media_usage.is_estimated is True
        # The long-running operation object is passed back untouched.
        assert result.name == "operations/veo-123"
        assert result.done is False
        solwyn.close()

    def test_generate_videos_reads_config_object_shape(self) -> None:
        # google-genai callers pass a GenerateVideosConfig OBJECT; duration_seconds
        # and resolution are read duck-typed via attribute access (not only dict).
        client = _mock_google_videos_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt="a dog",
                config=SimpleNamespace(duration_seconds=6, resolution="1080p"),
            )

        assert reported[0].media_usage.video_seconds == 6.0
        assert reported[0].media_usage.resolution == "1080p"
        solwyn.close()

    def test_generate_videos_absent_duration_tracked_unpriced(self) -> None:
        # No documented default duration -> absent stays None (unpriced-tracked),
        # never a guessed duration. The call is still tracked (modality=video).
        client = _mock_google_videos_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.generate_videos(model="veo-3.0-generate-001", prompt="a bird")

        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        assert event.media_usage.video_seconds is None
        assert event.media_usage.is_estimated is True
        solwyn.close()

    def test_generate_videos_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # generate_videos is intercepted, not an untracked pass-through —
        # it must never emit the warn-once "coming soon" surface warning.
        client = _mock_google_videos_client()
        solwyn = _make_solwyn(client)

        with caplog.at_level(logging.WARNING, logger="solwyn._base"), _mock_budget(solwyn):
            solwyn.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt="a cat",
                config={"duration_seconds": 8},
            )

        assert foreground_records(caplog) == []
        solwyn.close()

    def test_generate_videos_checks_budget_and_denies(self) -> None:
        # An oversized video request is denied BEFORE the provider call: the
        # precise per-second pre-flight cost (duration x resolution rate) is
        # priced server-side, so hard-deny short-circuits generate_videos.
        client = _mock_google_videos_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt="a cat",
                config={"duration_seconds": 8, "resolution": "4k"},
            )

        # Hard-deny short-circuits before the provider call.
        client.models.generate_videos.assert_not_called()
        solwyn.close()

    def test_embed_content_is_intercepted_and_reports_prompt_tokens(self) -> None:
        # embed_content routes through the media lifecycle, NOT the
        # __getattr__ warn-once pass-through: budget-checked, confirmed, reported.
        client = _mock_google_embeddings_client(prompt_token_count=55)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.models.embed_content(
                model="gemini-embedding-001",
                contents="hello world",
            )

        client.models.embed_content.assert_called_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        # Billable basis is usage_metadata.prompt_token_count; embeddings emit NO
        # output tokens (a TRUE zero) and the count is provider-reported.
        assert reported[0].input_tokens == 55
        assert reported[0].output_tokens == 0
        assert reported[0].token_details.is_estimated is False
        assert result.embeddings[0].values == [0.1, 0.2]
        solwyn.close()

    def test_embed_content_missing_usage_falls_back_to_estimated(self) -> None:
        # Today's EmbedContentResponse surfaces no usage — request-derived estimate
        # off contents= (google 4.0 char/token ratio), explicitly is_estimated=True.
        client = _mock_google_embeddings_client(usage=False)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.embed_content(
                model="gemini-embedding-001",
                contents="a" * 40,  # 40 chars / 4.0 google ratio -> 10 tokens
            )

        event = reported[0]
        assert event.status == "success"
        assert event.token_details.is_estimated is True
        assert event.input_tokens == 10
        assert event.output_tokens == 0
        solwyn.close()

    def test_embed_content_checks_budget_and_denies(self) -> None:
        client = _mock_google_embeddings_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.models.embed_content(model="gemini-embedding-001", contents="hi")

        # Hard-deny short-circuits before the provider call.
        client.models.embed_content.assert_not_called()
        solwyn.close()

    def test_embed_content_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # embed_content is intercepted, not an untracked pass-through — it must
        # never emit the warn-once "coming soon" surface warning.
        client = _mock_google_embeddings_client()
        solwyn = _make_solwyn(client)

        with caplog.at_level(logging.WARNING, logger="solwyn._base"), _mock_budget(solwyn):
            solwyn.models.embed_content(model="gemini-embedding-001", contents="hi")

        assert foreground_records(caplog) == []
        solwyn.close()

    def test_generate_images_is_intercepted_and_reports_image_count(self) -> None:
        # generate_images (imagen) routes through the media lifecycle, NOT
        # the __getattr__ warn-once pass-through. imagen has no token usage, so the
        # billable basis is the request-derived per-image MediaUsage.
        client = _mock_google_images_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt="a cat",
                config={"number_of_images": 4},
            )

        client.models.generate_images.assert_called_once()
        assert len(reported) == 1
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "image"
        # imagen reports no token usage — media is the sole billable basis.
        assert event.token_details is None
        assert event.input_tokens == 0
        assert event.media_usage.image_count == 4
        assert event.media_usage.is_estimated is False
        assert result.generated_images[0].image.image_bytes == b"png"
        solwyn.close()

    def test_generate_images_reads_config_object_shape(self) -> None:
        # google-genai callers pass a GenerateImagesConfig OBJECT; number_of_images
        # is read duck-typed via attribute access (not only the dict shape).
        client = _mock_google_images_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt="a dog",
                config=SimpleNamespace(number_of_images=2),
            )

        assert reported[0].media_usage.image_count == 2
        solwyn.close()

    def test_generate_images_missing_number_defaults_to_one(self) -> None:
        # imagen's contract defaults number_of_images to 1 — a TRUE known
        # quantity, never a zero-as-default.
        client = _mock_google_images_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.models.generate_images(model="imagen-3.0-generate-002", prompt="a bird")

        assert reported[0].media_usage.image_count == 1
        solwyn.close()

    def test_generate_images_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # generate_images is intercepted, not an untracked pass-through —
        # it must never emit the warn-once "coming soon" surface warning.
        client = _mock_google_images_client()
        solwyn = _make_solwyn(client)

        with caplog.at_level(logging.WARNING, logger="solwyn._base"), _mock_budget(solwyn):
            solwyn.models.generate_images(
                model="imagen-3.0-generate-002", prompt="a cat", config={"number_of_images": 1}
            )

        assert foreground_records(caplog) == []
        solwyn.close()

    def test_generate_images_checks_budget_and_denies(self) -> None:
        client = _mock_google_images_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.models.generate_images(
                model="imagen-3.0-generate-002", prompt="a cat", config={"number_of_images": 2}
            )

        # Hard-deny short-circuits before the provider call.
        client.models.generate_images.assert_not_called()
        solwyn.close()

    def test_list_passthrough_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # models.list is an unrelated surface — passes through with no warning.
        client = _mock_google_client()
        client.models.list = MagicMock(return_value=["gemini-2.5-pro"])
        solwyn = _make_solwyn(client)

        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            assert solwyn.models.list() == ["gemini-2.5-pro"]

        assert foreground_records(caplog) == []
        solwyn.close()

    def test_openai_models_is_not_proxied(self) -> None:
        """For OpenAI clients, .models passes through to the raw client."""
        client = MagicMock()
        client.__class__.__module__ = "openai._client"
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hi"))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )
        client.models.list.return_value = ["gpt-5.5"]
        solwyn = _make_solwyn(client)
        # Should pass through to OpenAI's models.list(), not our proxy
        assert solwyn.models.list() == ["gpt-5.5"]
        solwyn.close()


# ---------------------------------------------------------------------------
# OpenAI dialect: client.embeddings.create()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingsProxy:
    """client.embeddings.create() routes through the media lifecycle (_media_call)."""

    def test_embeddings_create_is_intercepted_and_reports_prompt_tokens(self) -> None:
        client = _mock_openai_embeddings_client(prompt_tokens=42)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.embeddings.create(
                model="text-embedding-3-small",
                input="hello world",
            )

        # Routed to the underlying embeddings surface, not the raw passthrough.
        client.embeddings.create.assert_called_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        # Billable basis is usage.prompt_tokens; embeddings emit NO output tokens
        # (a TRUE zero, not a zero-as-default) and the count is provider-reported.
        assert reported[0].input_tokens == 42
        assert reported[0].output_tokens == 0
        assert reported[0].token_details.is_estimated is False
        assert result.usage.prompt_tokens == 42
        solwyn.close()

    def test_embeddings_create_checks_budget_and_denies(self) -> None:
        client = _mock_openai_embeddings_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.embeddings.create(model="text-embedding-3-small", input="hi")

        # Hard-deny short-circuits before the provider call.
        client.embeddings.create.assert_not_called()
        solwyn.close()

    def test_embeddings_missing_usage_falls_back_to_estimated(self) -> None:
        # A compat-style response with no usage block -> request-derived estimate,
        # explicitly flagged is_estimated=True (never a silent zero).
        client = _mock_openai_embeddings_client(usage=False)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.embeddings.create(
                model="text-embedding-3-small",
                input="a" * 40,  # 40 chars / 4.0 openai ratio -> 10 tokens
            )

        event = reported[0]
        assert event.status == "success"
        assert event.token_details.is_estimated is True
        assert event.input_tokens == 10
        assert event.output_tokens == 0
        solwyn.close()

    def test_embeddings_getattr_passthrough(self) -> None:
        """Non-create attributes pass through to the underlying embeddings surface."""
        client = _mock_openai_embeddings_client()
        client.embeddings.some_helper = MagicMock(return_value="ok")
        solwyn = _make_solwyn(client)
        assert solwyn.embeddings.some_helper() == "ok"
        solwyn.close()


# ---------------------------------------------------------------------------
# OpenAI dialect: client.images.generate() / client.images.edit()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImagesProxy:
    """client.images.generate()/edit() route through the media lifecycle (_media_call)."""

    def test_native_gpt_image_reports_token_and_media_bases(self) -> None:
        # Native gpt-image sends BOTH bases: token usage (with image buckets) AND
        # the request-derived per-image MediaUsage. The server's card unit picks.
        client = _mock_openai_images_client(native=True)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.images.generate(
                model="gpt-image-2", prompt="a cat", n=1, size="1024x1024", quality="low"
            )

        client.images.generate.assert_called_once()
        client.images.edit.assert_not_called()
        # The private op marker never reaches the SDK.
        assert _IMAGE_OP_KEY not in client.images.generate.call_args.kwargs
        assert len(reported) == 1
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "image"
        # Token basis with image buckets (image ⊂ input, image_output ⊂ output).
        assert event.input_tokens == 222
        assert event.token_details.image_input_tokens == 194
        assert event.token_details.image_output_tokens == 1024
        assert event.token_details.is_estimated is False
        # Media basis: request-derived config quantities.
        assert event.media_usage.image_count == 1
        assert event.media_usage.resolution == "1024x1024"
        assert event.media_usage.quality == "low"
        solwyn.close()

    def test_compat_or_dalle_image_reports_media_only(self) -> None:
        # A usage-less image response (compat FLUX / dall-e): no TOKEN basis, but
        # the per-image MediaUsage IS observed -> reported (never a silent $0).
        client = _mock_openai_images_client(native=False)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.images.generate(model="dall-e-3", prompt="a cat", n=2, size="1792x1024")

        event = reported[0]
        assert event.status == "success"
        assert event.modality == "image"
        assert event.token_details is None  # no token basis observed
        assert event.input_tokens == 0
        assert event.media_usage.image_count == 2
        assert event.media_usage.resolution == "1792x1024"
        solwyn.close()

    def test_edit_routes_to_images_edit_with_marker_stripped(self) -> None:
        client = _mock_openai_images_client(native=True)
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.images.edit(model="gpt-image-2", prompt="add a hat", image=b"png-bytes")

        client.images.edit.assert_called_once()
        client.images.generate.assert_not_called()
        # The op marker routed the hop but is stripped before the SDK call.
        assert _IMAGE_OP_KEY not in client.images.edit.call_args.kwargs
        assert reported[0].status == "success"
        assert reported[0].media_usage.image_count == 1
        solwyn.close()

    def test_images_create_variation_passes_through(self) -> None:
        # create_variation is neither generate nor edit -> passes through untracked.
        client = _mock_openai_images_client()
        client.images.create_variation = MagicMock(return_value="varied")
        solwyn = _make_solwyn(client)
        assert solwyn.images.create_variation() == "varied"
        solwyn.close()

    def test_images_generate_checks_budget_and_denies(self) -> None:
        client = _mock_openai_images_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.images.generate(model="gpt-image-2", prompt="a cat")

        # Hard-deny short-circuits before the provider call.
        client.images.generate.assert_not_called()
        solwyn.close()


# ---------------------------------------------------------------------------
# OpenAI dialect: client.videos.create() (Sora)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVideosProxy:
    """client.videos.create() (Sora) routes through the media lifecycle (_media_call)."""

    def test_videos_create_is_intercepted_and_reports_video_seconds(self) -> None:
        # videos.create (Sora) routes through the media lifecycle, NOT a warn-once
        # pass-through. The call returns an async video job with no usage, so the
        # billable basis is the request-derived per-second MediaUsage, ALWAYS
        # is_estimated=True (settles at initiation). size normalizes to a label.
        client = _mock_openai_videos_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            result = solwyn.videos.create(
                model="sora-2", prompt="a cat", seconds="8", size="1280x720"
            )

        client.videos.create.assert_called_once()
        assert len(reported) == 1
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        # Sora reports no token usage — media is the sole billable basis.
        assert event.token_details is None
        assert event.input_tokens == 0
        assert event.media_usage.video_seconds == 8.0
        assert event.media_usage.resolution == "720p"
        assert event.media_usage.is_estimated is True
        # The async video job object is passed back untouched.
        assert result.id == "video_123"
        assert result.status == "queued"
        solwyn.close()

    def test_videos_create_defaults_seconds_and_size(self) -> None:
        # A bare call (no seconds/size) settles OpenAI's documented defaults
        # (4 seconds at 720p) — priced, never a silent $0.
        client = _mock_openai_videos_client()
        solwyn = _make_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_budget(solwyn):
            solwyn.videos.create(model="sora-2", prompt="a dog")

        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        assert event.media_usage.video_seconds == 4.0
        assert event.media_usage.resolution == "720p"
        assert event.media_usage.is_estimated is True
        solwyn.close()

    def test_videos_create_checks_budget_and_denies(self) -> None:
        # An oversized video request is denied BEFORE the provider call: the
        # precise per-second pre-flight cost (seconds x resolution rate) is priced
        # server-side, so hard-deny short-circuits videos.create.
        client = _mock_openai_videos_client()
        solwyn = _make_solwyn(client, budget_mode="hard_deny")

        deny_response = {
            **ALLOW_BUDGET_RESPONSE,
            "allowed": False,
            "remaining_budget": 0.0,
            "mode": "hard_deny",
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        with _mock_budget(solwyn, deny_response), pytest.raises(BudgetExceededError):
            solwyn.videos.create(model="sora-2", prompt="a cat", seconds="12", size="1792x1024")

        # Hard-deny short-circuits before the provider call.
        client.videos.create.assert_not_called()
        solwyn.close()

    def test_videos_create_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # videos is intercepted, not an untracked pass-through — it must never emit
        # the warn-once "coming soon" surface warning (it graduated out of the
        # openai unshipped-spend set).
        client = _mock_openai_videos_client()
        solwyn = _make_solwyn(client)

        with caplog.at_level(logging.WARNING, logger="solwyn._base"), _mock_budget(solwyn):
            solwyn.videos.create(model="sora-2", prompt="a cat", seconds="4")

        assert foreground_records(caplog) == []
        solwyn.close()

    def test_videos_non_create_passes_through(self) -> None:
        # retrieve is not create -> passes through untracked to the client's videos.
        client = _mock_openai_videos_client()
        client.videos.retrieve = MagicMock(return_value="job")
        solwyn = _make_solwyn(client)
        assert solwyn.videos.retrieve() == "job"
        solwyn.close()


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


def _make_async_solwyn(client, **overrides):
    defaults = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    return AsyncSolwyn(client, **defaults)


def _mock_async_budget(solwyn, response=None):
    resp = MagicMock()
    resp.json.return_value = response or ALLOW_BUDGET_RESPONSE
    resp.raise_for_status = MagicMock()
    return patch.object(solwyn._budget._http, "post", return_value=resp)


@pytest.mark.unit
class TestAsyncAnthropicMessagesProxy:
    """Async client.messages.create() routes through _intercepted_call."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_messages_create_is_intercepted(self) -> None:
        client = _mock_anthropic_client()
        client.messages.create = AsyncMockFn(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(text="Hello")],
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                ),
            )
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.messages.create(
                model="claude-sonnet-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        client.messages.create.assert_called_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        assert result.content[0].text == "Hello"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


@pytest.mark.unit
class TestAsyncGoogleModelsProxy:
    """Async client.models.generate_content() routes through _intercepted_call."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_generate_content_is_intercepted(self) -> None:
        client = _mock_google_client()
        client.models.generate_content = AsyncMockFn(
            return_value=SimpleNamespace(
                text="Hello",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=50,
                    thoughts_token_count=0,
                    cached_content_token_count=0,
                    tool_use_prompt_token_count=0,
                ),
            )
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.models.generate_content(
                model="gemini-3.5-flash",
                contents="Hello",
            )

        client.models.generate_content.assert_called_once()
        assert len(reported) == 1
        assert result.text == "Hello"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_generate_videos_is_intercepted(self) -> None:
        # Mirror of the sync generate_videos interception: veo returns a
        # long-running operation with no usage, so the per-second MediaUsage is
        # the sole billable basis, ALWAYS is_estimated=True (settles at initiation).
        client = _mock_google_videos_client()
        operation = SimpleNamespace(name="operations/veo-async", done=False, response=None)
        client.models.generate_videos = AsyncMockFn(return_value=operation)
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.models.generate_videos(
                model="veo-3.0-generate-001",
                prompt="a cat",
                config={"duration_seconds": 4, "resolution": "720p"},
            )

        client.models.generate_videos.assert_awaited_once()
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        assert event.token_details is None
        assert event.media_usage.video_seconds == 4.0
        assert event.media_usage.is_estimated is True
        # The long-running operation is passed back untouched.
        assert result is operation

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_embed_content_is_intercepted(self) -> None:
        client = _mock_google_embeddings_client(prompt_token_count=77)
        client.models.embed_content = AsyncMockFn(
            return_value=SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])],
                usage_metadata=SimpleNamespace(prompt_token_count=77),
            )
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.models.embed_content(
                model="gemini-embedding-001",
                contents="hello",
            )

        client.models.embed_content.assert_awaited_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        assert reported[0].input_tokens == 77
        assert reported[0].output_tokens == 0
        assert result.embeddings[0].values == [0.1, 0.2]

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_generate_images_is_intercepted(self) -> None:
        # Mirror of the sync generate_images interception: imagen carries
        # no token usage, so the per-image MediaUsage is the sole billable basis.
        client = _mock_google_images_client()
        client.models.generate_images = AsyncMockFn(
            return_value=SimpleNamespace(
                generated_images=[SimpleNamespace(image=SimpleNamespace(image_bytes=b"png"))]
            )
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            await solwyn.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt="a cat",
                config={"number_of_images": 3},
            )

        client.models.generate_images.assert_awaited_once()
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "image"
        assert event.token_details is None
        assert event.media_usage.image_count == 3

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


@pytest.mark.unit
class TestAsyncEmbeddingsProxy:
    """Async client.embeddings.create() routes through the async media lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_embeddings_create_is_intercepted(self) -> None:
        client = _mock_openai_embeddings_client(prompt_tokens=99)
        client.__class__.__name__ = "AsyncOpenAI"
        client.embeddings.create = AsyncMockFn(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)],
                usage=SimpleNamespace(prompt_tokens=99, total_tokens=99),
            )
        )
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.embeddings.create(
                model="text-embedding-3-small",
                input="hello",
            )

        client.embeddings.create.assert_awaited_once()
        assert len(reported) == 1
        assert reported[0].status == "success"
        assert reported[0].input_tokens == 99
        assert reported[0].output_tokens == 0
        assert result.usage.prompt_tokens == 99

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


@pytest.mark.unit
class TestAsyncImagesProxy:
    """Async client.images.generate()/edit() route through the async media lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_native_gpt_image_reports_both_bases(self) -> None:
        client = _mock_openai_images_client(native=True)
        client.__class__.__name__ = "AsyncOpenAI"
        client.images.generate = AsyncMockFn(return_value=client.images.generate.return_value)
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            await solwyn.images.generate(model="gpt-image-2", prompt="a cat", n=1, size="1024x1024")

        client.images.generate.assert_awaited_once()
        assert _IMAGE_OP_KEY not in client.images.generate.call_args.kwargs
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "image"
        assert event.token_details.image_input_tokens == 194
        assert event.media_usage.image_count == 1
        assert event.media_usage.resolution == "1024x1024"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_edit_routes_to_images_edit(self) -> None:
        client = _mock_openai_images_client(native=True)
        client.__class__.__name__ = "AsyncOpenAI"
        edit_response = client.images.edit.return_value
        client.images.edit = AsyncMockFn(return_value=edit_response)
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            await solwyn.images.edit(model="gpt-image-2", prompt="add a hat", image=b"png")

        client.images.edit.assert_awaited_once()
        assert _IMAGE_OP_KEY not in client.images.edit.call_args.kwargs
        assert reported[0].media_usage.image_count == 1

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


@pytest.mark.unit
class TestAsyncVideosProxy:
    """Async client.videos.create() (Sora) routes through the async media lifecycle."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_videos_create_is_intercepted(self) -> None:
        # Mirror of the sync videos interception: Sora returns an async video job
        # with no usage, so the per-second MediaUsage is the sole billable basis,
        # ALWAYS is_estimated=True (settles at initiation).
        client = _mock_openai_videos_client()
        client.__class__.__name__ = "AsyncOpenAI"
        job = SimpleNamespace(id="video_async", status="queued", progress=0)
        client.videos.create = AsyncMockFn(return_value=job)
        solwyn = _make_async_solwyn(client)
        reported: list = []
        solwyn._reporter.report = lambda e: reported.append(e)
        # Non-streaming settlement (with a reservation) rides report_settlement;
        # route its event into the same list so the SUCCESS event is observed.
        solwyn._reporter.report_settlement = lambda req, event: reported.append(event)

        with _mock_async_budget(solwyn):
            result = await solwyn.videos.create(
                model="sora-2", prompt="a cat", seconds="4", size="720x1280"
            )

        client.videos.create.assert_awaited_once()
        event = reported[0]
        assert event.status == "success"
        assert event.modality == "video"
        assert event.token_details is None
        assert event.media_usage.video_seconds == 4.0
        assert event.media_usage.resolution == "720p"
        assert event.media_usage.is_estimated is True
        # The async video job is passed back untouched.
        assert result is job

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
