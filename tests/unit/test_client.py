"""Tests for Solwyn and AsyncSolwyn client wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID

import solwyn as solwyn_pkg
from solwyn._privacy import estimate_content_length
from solwyn._types import BudgetMode, ProviderName
from solwyn.client import Solwyn
from solwyn.exceptions import BudgetExceededError, ProviderUnavailableError
from solwyn.providers import get_adapter_for_client
from solwyn.stream import AsyncStreamWrapper, SyncStreamWrapper


def _mock_openai_client():
    """Create a mock that looks like openai.OpenAI()."""
    client = MagicMock()
    # Set module to openai so auto-detection works
    client.__class__.__module__ = "openai._client"
    client.__class__.__name__ = "OpenAI"
    # Per-hop with_options(timeout, max_retries) returns the SAME client so the
    # configured .chat.completions.create mock is the one dispatch invokes.
    client.with_options.return_value = client

    # Mock response with usage
    mock_response = MagicMock()
    mock_response.usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
    )
    client.chat.completions.create.return_value = mock_response
    return client, mock_response


def _mock_anthropic_client():
    """Create a mock that looks like anthropic.Anthropic()."""
    client = MagicMock()
    client.__class__.__module__ = "anthropic._client"
    client.__class__.__name__ = "Anthropic"
    client.with_options.return_value = client

    mock_response = MagicMock()
    mock_response.usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
    )
    client.messages.create.return_value = mock_response
    return client, mock_response


def _make_solwyn(client, **overrides):
    """Create a Solwyn wrapper with mocked budget and reporter."""
    defaults = {
        "api_key": VALID_API_KEY,
    }
    defaults.update(overrides)

    # Patch the reporter background thread and budget HTTP calls
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)

    # Stop reporter thread
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)

    return solwyn


def _allow_budget_result() -> SimpleNamespace:
    """Minimal budget-check result for tests that bypass HTTP."""
    return SimpleNamespace(allowed=True, reservation_id=None, price_hints=None)


class _Status(Exception):
    """A duck-typed transport error carrying an HTTP status_code.

    classify_exception reads ``status_code``: 429 -> FAILOVER (advance the
    chain). A bare ``RuntimeError`` would instead classify as FAIL_FAST and
    stop the chain, so failover tests must use a status-bearing exception.
    """

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderDetection:
    """Auto-detect provider from client instance."""

    def test_detects_openai(self) -> None:
        client, _ = _mock_openai_client()
        assert ProviderName(get_adapter_for_client(client).name) == ProviderName.OPENAI

    def test_detects_anthropic(self) -> None:
        client, _ = _mock_anthropic_client()
        assert ProviderName(get_adapter_for_client(client).name) == ProviderName.ANTHROPIC

    def test_detects_google(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "google.generativeai._client"
        client.__class__.__name__ = "GenerativeModel"
        assert ProviderName(get_adapter_for_client(client).name) == ProviderName.GOOGLE

    def test_raises_on_unknown_client(self) -> None:
        client = MagicMock()
        client.__class__.__module__ = "some_other_lib"
        client.__class__.__name__ = "UnknownClient"
        with pytest.raises(ValueError, match="No provider adapter"):
            get_adapter_for_client(client)


# ---------------------------------------------------------------------------
# Text extraction for cost estimation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContentLengthEstimation:
    """estimate_content_length returns character counts without materializing joined text."""

    def test_openai_messages_length(self) -> None:
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello world"},
            ]
        }
        length = estimate_content_length(kwargs)
        assert length == len("You are helpful") + len("Hello world")

    def test_anthropic_content_blocks_length(self) -> None:
        kwargs = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello block"}]},
            ]
        }
        length = estimate_content_length(kwargs)
        assert length == len("Hello block")

    def test_anthropic_system_length(self) -> None:
        kwargs = {
            "system": "You are a helpful assistant",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        length = estimate_content_length(kwargs)
        assert length == len("You are a helpful assistant") + len("Hi")

    def test_empty_kwargs_returns_zero(self) -> None:
        assert estimate_content_length({}) == 0

    def test_google_contents_string_length(self) -> None:
        length = estimate_content_length({"contents": "Hello"})
        assert length == len("Hello")

    def test_google_contents_list_of_strings_length(self) -> None:
        length = estimate_content_length({"contents": ["Hello", "World"]})
        assert length == len("Hello") + len("World")

    def test_google_contents_list_of_part_dicts_length(self) -> None:
        length = estimate_content_length({"contents": [{"text": "Hello"}]})
        assert length == len("Hello")


# ---------------------------------------------------------------------------
# Basic wrapping: call goes through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicWrapping:
    """The underlying client's create() is called through the wrapper."""

    def test_openai_call_goes_through(self) -> None:
        client, mock_response = _mock_openai_client()
        solwyn = _make_solwyn(client)

        # Mock budget to allow
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            result = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is mock_response
        client.chat.completions.create.assert_called_once()
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_anthropic_call_goes_through(self) -> None:
        client, mock_response = _mock_anthropic_client()
        solwyn = _make_solwyn(client)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            result = solwyn.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is mock_response
        client.messages.create.assert_called_once()
        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Budget check happens before call
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBudgetCheckBeforeCall:
    """Budget is checked before the LLM call."""

    def test_budget_denied_raises_before_call(self) -> None:
        client, _ = _mock_openai_client()
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 0.0,
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 10.0,
            "current_usage": 10.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # LLM client should NOT have been called
        client.chat.completions.create.assert_not_called()
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_budget_denied_reports_metadata_with_estimated_tokens(self) -> None:
        """When hard-deny blocks a call, a budget_denied metadata event is still reported."""
        client, _ = _mock_openai_client()
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 0.0,
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 10.0,
            "current_usage": 10.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            patch.object(solwyn._reporter, "report") as mock_report,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # A metadata event should have been reported with budget_denied status
        mock_report.assert_called_once()
        event = mock_report.call_args[0][0]
        assert event.status == "budget_denied"
        assert event.input_tokens > 0  # estimated from "Hello"
        assert event.output_tokens == 0
        assert event.latency_ms == 0.0
        assert event.is_model_fallback is False

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_budget_denied_event_tags_agent_run_id(self) -> None:
        client, _ = _mock_openai_client()
        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 0.0,
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 10.0,
            "current_usage": 10.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            patch.object(solwyn._reporter, "report") as mock_report,
            solwyn_pkg.run("expensive-job") as run_id,
            pytest.raises(BudgetExceededError),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        mock_report.assert_called_once()
        event = mock_report.call_args[0][0]
        assert event.status == "budget_denied"
        assert event.agent_run_id == run_id
        assert event.agent_run_name == "expensive-job"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_google_budget_denied_reports_nonzero_input_tokens(self) -> None:
        """Hard-deny for a Google call with contents='Hello' reports input_tokens > 0."""
        client = MagicMock()
        client.__class__.__module__ = "google.genai._client"

        solwyn = _make_solwyn(client, budget_mode=BudgetMode.HARD_DENY)

        deny_response = {
            "allowed": False,
            "remaining_budget": 0.0,
            "reservation_id": None,
            "mode": "hard_deny",
            "budget_limit": 10.0,
            "current_usage": 10.0,
            "denied_by_period": "monthly",
            "project_id": VALID_PROJECT_ID,
        }
        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = deny_response
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            patch.object(solwyn._reporter, "report") as mock_report,
            pytest.raises(BudgetExceededError) as exc_info,
        ):
            solwyn.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        # LLM client should NOT have been called
        client.models.generate_content.assert_not_called()

        # A metadata event should have been reported with budget_denied status
        mock_report.assert_called_once()
        event = mock_report.call_args[0][0]
        assert event.status == "budget_denied"
        assert event.input_tokens > 0  # estimated from "Hello" via contents kwarg
        assert event.output_tokens == 0
        assert event.latency_ms == 0.0
        assert event.is_model_fallback is False

        # BudgetExceededError.estimated_cost should be non-zero
        assert exc_info.value.estimated_cost > 0

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Circuit breaker failover
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCircuitBreakerFailover:
    """When primary circuit opens, fallback provider is used."""

    def test_provider_unavailable_raises(self) -> None:
        client, _ = _mock_openai_client()
        solwyn = _make_solwyn(client)

        # Open the primary circuit breaker
        cb = solwyn._get_circuit_breaker("openai")
        for _ in range(3):
            cb.record_failure()

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            pytest.raises(ProviderUnavailableError),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# __getattr__ pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAttrPassThrough:
    """Non-intercepted attributes pass through to the underlying client."""

    def test_passthrough_to_underlying_client(self) -> None:
        client, _ = _mock_openai_client()
        client.models = MagicMock()
        client.models.list.return_value = ["gpt-4o"]

        solwyn = _make_solwyn(client)
        result = solwyn.models.list()
        assert result == ["gpt-4o"]

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextManager:
    """Solwyn supports with-statement."""

    def test_context_manager(self) -> None:
        client, _ = _mock_openai_client()
        with (
            patch("solwyn.reporter.MetadataReporter._flush_loop"),
            Solwyn(
                client,
                api_key=VALID_API_KEY,
            ) as solwyn,
        ):
            # Stop reporter thread
            solwyn._reporter._shutdown.set()
            solwyn._reporter._thread.join(timeout=2.0)
            assert solwyn._client is client


# ---------------------------------------------------------------------------
# Rich token extraction via adapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRichTokenExtraction:
    """Adapter extracts full TokenDetails and MetadataEvent carries token_details."""

    def test_openai_token_details_in_event(self) -> None:
        """Adapter extracts full TokenDetails from OpenAI response."""
        client, _ = _mock_openai_client()
        mock_response = MagicMock()
        mock_response.usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400, audio_tokens=0),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=0,
                audio_tokens=0,
                accepted_prediction_tokens=0,
                rejected_prediction_tokens=0,
            ),
        )
        client.chat.completions.create.return_value = mock_response
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        event = reported_events[0]
        assert event.token_details is not None
        assert event.token_details.cached_input_tokens == 400
        assert event.token_details.input_tokens == 1000
        # MetadataEvent no longer carries cost fields
        assert not hasattr(event, "actual_cost")

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_anthropic_token_details_in_event(self) -> None:
        """Adapter extracts TokenDetails including cache fields from Anthropic response."""
        client, _ = _mock_anthropic_client()
        mock_response = MagicMock()
        mock_response.usage = SimpleNamespace(
            input_tokens=800,
            output_tokens=200,
            cache_read_input_tokens=300,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=50,
                ephemeral_1h_input_tokens=25,
            ),
        )
        client.messages.create.return_value = mock_response
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            solwyn.chat.completions.create(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hello"}],
            )

        event = reported_events[0]
        assert event.token_details is not None
        # Anthropic: input_tokens is normalized sum of base + cache_read + cache writes
        assert event.token_details.input_tokens == 800 + 300 + 50 + 25
        assert event.token_details.cached_input_tokens == 300
        assert event.token_details.cache_creation_5m_tokens == 50
        assert event.token_details.cache_creation_1h_tokens == 25
        assert event.token_details.output_tokens == 200

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_openai_non_streaming_service_tier_in_event(self) -> None:
        """OpenAI service_tier reaches MetadataEvent on sync non-streaming calls."""
        client, mock_response = _mock_openai_client()
        mock_response.service_tier = "priority"
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert reported_events[0].service_tier == "priority"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_openai_non_streaming_call_tags_agent_run_id(self) -> None:
        client, _ = _mock_openai_client()
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()),
            solwyn_pkg.run("nightly-batch") as run_id,
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert len(reported_events) == 1
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "nightly-batch"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_anthropic_non_streaming_service_tier_none_in_event(self) -> None:
        """Providers without a service tier report None behaviorally through the client."""
        client, _ = _mock_anthropic_client()
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()):
            solwyn.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert reported_events[0].service_tier is None

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


@pytest.mark.unit
class TestSyncErrorAgentRunTagging:
    """Error and fallback-retry events preserve the active agent_run."""

    def test_primary_error_event_tags_agent_run_id(self) -> None:
        client, _ = _mock_openai_client()
        client.chat.completions.create.side_effect = RuntimeError("primary failed")
        solwyn = _make_solwyn(client)

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()),
            solwyn_pkg.run("doomed") as run_id,
            pytest.raises(RuntimeError, match="primary failed"),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert len(reported_events) == 1
        assert reported_events[0].status == "error"
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "doomed"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_fallback_retry_error_events_tag_agent_run_id(self) -> None:
        # Same-provider model-swap fallback: the SAME client serves both hops.
        # A 429-style error on the primary advances the chain to the gpt-4o-mini
        # swap; the swap also 429s, so the chain exhausts and re-raises.
        client, _ = _mock_openai_client()
        client.chat.completions.create.side_effect = [
            _Status(429, "primary failed"),
            _Status(429, "fallback failed"),
        ]
        solwyn = _make_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])

        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()),
            solwyn_pkg.run("retry-doomed") as run_id,
            pytest.raises(_Status, match="fallback failed"),
        ):
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert [event.status for event in reported_events] == ["error", "error"]
        assert all(event.agent_run_id == run_id for event in reported_events)
        assert all(event.agent_run_name == "retry-doomed" for event in reported_events)
        # Second hop swapped the model on the same client.
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Streaming interception
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSyncStreamingInterception:
    """Streaming calls return a wrapped iterator that reports usage on completion."""

    def test_streaming_call_returns_wrapper(self) -> None:
        client, _ = _mock_openai_client()
        # Make the provider return an iterable when stream=True
        mock_chunks = [
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
        client.chat.completions.create.return_value = mock_chunks

        solwyn = _make_solwyn(client)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            result = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

        assert isinstance(result, SyncStreamWrapper)
        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_streaming_reports_metadata_after_exhaustion(self) -> None:
        client, _ = _mock_openai_client()
        mock_chunks = [
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
        client.chat.completions.create.return_value = mock_chunks

        solwyn = _make_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            # Before exhaustion — no event yet
            assert len(reported_events) == 0

            # Exhaust stream
            chunks = list(stream)

        # After exhaustion — event reported
        assert len(reported_events) == 1
        event = reported_events[0]
        assert event.status == "success"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.token_details is not None

        # Chunks passed through unchanged
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hi"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_streaming_uses_run_snapshot_when_consumed_after_scope(self) -> None:
        client, _ = _mock_openai_client()
        mock_chunks = [
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
        client.chat.completions.create.return_value = mock_chunks

        solwyn = _make_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            solwyn_pkg.run("nightly") as run_id,
        ):
            stream = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

        list(stream)

        assert len(reported_events) == 1
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "nightly"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_streaming_openai_service_tier_in_event(self) -> None:
        """OpenAI service_tier reaches MetadataEvent on sync streaming calls."""
        client, _ = _mock_openai_client()
        client.chat.completions.create.return_value = [
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            ),
            SimpleNamespace(
                service_tier="flex",
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                choices=[],
            ),
        ]

        solwyn = _make_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget_result()):
            stream = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            list(stream)

        assert reported_events[0].service_tier == "flex"

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_streaming_confirms_budget_after_exhaustion(self) -> None:
        client, _ = _mock_openai_client()
        mock_chunks = [
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
        client.chat.completions.create.return_value = mock_chunks

        solwyn = _make_solwyn(client)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        # Confirm now goes through reporter.report_confirm (not budget.confirm_cost)
        confirm_calls: list = []
        solwyn._reporter.report_confirm = lambda request: confirm_calls.append(request)

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            # Before exhaustion — no confirm yet
            assert len(confirm_calls) == 0
            list(stream)

        # After exhaustion — reporter.report_confirm called once with the reservation_id
        assert len(confirm_calls) == 1
        assert confirm_calls[0].reservation_id == "res_123"
        # budget.confirm_cost must NOT have been called directly
        assert not hasattr(solwyn._budget, "_direct_confirm_calls")

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_streaming_injects_stream_options_for_openai(self) -> None:
        client, _ = _mock_openai_client()
        client.chat.completions.create.return_value = []

        solwyn = _make_solwyn(client)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            list(stream)  # exhaust

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_google_generate_content_stream_reports_metadata(self) -> None:
        """Google models.generate_content_stream() wraps and reports after exhaustion."""
        client = MagicMock()
        client.__class__.__module__ = "google.genai._client"
        client.with_options.return_value = client

        mock_chunks = [
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=30,
                    thoughts_token_count=0,
                ),
            ),
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=80,
                    thoughts_token_count=10,
                    cached_content_token_count=0,
                    tool_use_prompt_token_count=0,
                ),
            ),
        ]
        client.models.generate_content_stream.return_value = mock_chunks

        solwyn = _make_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = solwyn.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello",
            )
            assert len(reported_events) == 0
            chunks = list(stream)

        assert len(reported_events) == 1
        event = reported_events[0]
        assert event.status == "success"
        assert event.input_tokens == 100
        # output = candidates + thoughts from last chunk
        assert event.output_tokens == 80 + 10
        assert event.token_details.reasoning_tokens == 10
        assert len(chunks) == 2

        # Verify correct SDK method was called
        client.models.generate_content_stream.assert_called_once()
        client.models.generate_content.assert_not_called()

        solwyn._reporter._http.close()
        solwyn._budget._http.close()

    def test_anthropic_messages_stream_reports_metadata(self) -> None:
        """Anthropic messages.create(stream=True) wraps and reports after exhaustion."""
        client = MagicMock()
        client.__class__.__module__ = "anthropic._client"
        client.with_options.return_value = client

        mock_events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=150,
                        cache_read_input_tokens=50,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="Hello"),
            ),
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=83),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        client.messages.create.return_value = mock_events

        solwyn = _make_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = solwyn.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            assert len(reported_events) == 0
            chunks = list(stream)

        assert len(reported_events) == 1
        event = reported_events[0]
        assert event.status == "success"
        # Anthropic: input = base + cache_read + cache_creation
        assert event.input_tokens == 150 + 50 + 0
        assert event.output_tokens == 83
        assert event.token_details.cached_input_tokens == 50
        assert len(chunks) == 4

        solwyn._reporter._http.close()
        solwyn._budget._http.close()


# ---------------------------------------------------------------------------
# Async streaming interception
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock as AsyncMockFn  # noqa: E402

from solwyn.client import AsyncSolwyn  # noqa: E402


def _make_async_solwyn(client, **overrides):
    """Create an AsyncSolwyn wrapper with mocked budget and reporter."""
    defaults = {
        "api_key": VALID_API_KEY,
    }
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)
    return solwyn


@pytest.mark.unit
class TestAsyncStreamingInterception:
    """Async streaming calls return an AsyncStreamWrapper."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_streaming_reports_metadata(self) -> None:
        client, _ = _mock_openai_client()

        async def async_stream():
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            )
            yield SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                choices=[],
            )

        # Make async create return our async generator
        client.chat.completions.create = AsyncMockFn(return_value=async_stream())

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            assert isinstance(stream, AsyncStreamWrapper)

            chunks = [c async for c in stream]

        assert len(chunks) == 2
        assert len(reported_events) == 1
        event = reported_events[0]
        assert event.status == "success"
        assert event.input_tokens == 100
        assert event.output_tokens == 50

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_streaming_uses_run_snapshot_when_consumed_after_scope(self) -> None:
        client, _ = _mock_openai_client()

        async def async_stream():
            yield SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                choices=[],
            )

        client.chat.completions.create = AsyncMockFn(return_value=async_stream())

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            async with solwyn_pkg.run("nightly") as run_id:
                stream = await solwyn.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

        _ = [chunk async for chunk in stream]

        assert len(reported_events) == 1
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "nightly"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_streaming_openai_service_tier_in_event(self) -> None:
        """OpenAI service_tier reaches MetadataEvent on async streaming calls."""
        client, _ = _mock_openai_client()

        async def async_stream():
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            )
            yield SimpleNamespace(
                service_tier="flex",
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
                choices=[],
            )

        client.chat.completions.create = AsyncMockFn(return_value=async_stream())

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=_allow_budget_result()),
        ):
            stream = await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            _ = [c async for c in stream]

        assert reported_events[0].service_tier == "flex"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_google_generate_content_stream(self) -> None:
        """Async Google models.generate_content_stream() wraps and reports."""
        client = MagicMock()
        client.__class__.__module__ = "google.genai._client"
        client.with_options.return_value = client

        async def async_google_stream():
            yield SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=80,
                    thoughts_token_count=10,
                    cached_content_token_count=0,
                    tool_use_prompt_token_count=0,
                ),
            )

        client.models.generate_content_stream = AsyncMockFn(return_value=async_google_stream())

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = await solwyn.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello",
            )
            chunks = [c async for c in stream]

        assert len(chunks) == 1
        assert len(reported_events) == 1
        assert reported_events[0].input_tokens == 100
        assert reported_events[0].output_tokens == 90  # 80 candidates + 10 thoughts
        client.models.generate_content_stream.assert_called_once()

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_anthropic_messages_stream(self) -> None:
        """Async Anthropic messages.create(stream=True) wraps and reports."""
        client = MagicMock()
        client.__class__.__module__ = "anthropic._client"
        client.with_options.return_value = client

        async def async_anthropic_stream():
            yield SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=150,
                        cache_read_input_tokens=0,
                    )
                ),
            )
            yield SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=83),
            )

        client.messages.create = AsyncMockFn(return_value=async_anthropic_stream())

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        with patch.object(solwyn._budget._http, "post", return_value=mock_budget_response):
            stream = await solwyn.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            chunks = [c async for c in stream]

        assert len(reported_events) == 1
        assert reported_events[0].input_tokens == 150
        assert reported_events[0].output_tokens == 83
        assert len(chunks) == 2

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()


# ---------------------------------------------------------------------------
# Async non-streaming interception
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsyncNonStreamingInterception:
    """Async non-streaming calls go through _intercepted_call."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_openai_non_streaming_call(self) -> None:
        client, mock_response = _mock_openai_client()
        # Make create return an awaitable (non-streaming)
        client.chat.completions.create = AsyncMockFn(return_value=mock_response)

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        mock_budget_response = MagicMock()
        mock_budget_response.json.return_value = ALLOW_BUDGET_RESPONSE
        mock_budget_response.raise_for_status = MagicMock()

        mock_confirm_response = AsyncMockFn()
        mock_confirm_response.raise_for_status = MagicMock()

        with (
            patch.object(solwyn._budget._http, "post", return_value=mock_budget_response),
            patch.object(solwyn._budget, "confirm_cost", new_callable=AsyncMockFn) as mock_confirm,
        ):
            result = await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Response passes through
        assert result is mock_response
        # Non-streaming calls confirm_cost directly (not via reporter)
        mock_confirm.assert_called_once()
        # Metadata event reported
        assert len(reported_events) == 1

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_budget_denied_event_tags_agent_run_id(self) -> None:
        client, _ = _mock_openai_client()
        client.chat.completions.create = AsyncMockFn()

        solwyn = _make_async_solwyn(client, budget_mode=BudgetMode.HARD_DENY)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        deny_result = SimpleNamespace(
            allowed=False,
            project_id=VALID_PROJECT_ID,
            budget_limit=10.0,
            current_usage=10.0,
            mode=BudgetMode.HARD_DENY,
            reservation_id=None,
            price_hints=None,
        )

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=deny_result),
        ):
            async with solwyn_pkg.run("async-expensive-job") as run_id:
                with pytest.raises(BudgetExceededError):
                    await solwyn.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        client.chat.completions.create.assert_not_called()
        assert len(reported_events) == 1
        assert reported_events[0].status == "budget_denied"
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "async-expensive-job"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_openai_non_streaming_service_tier_in_event(self) -> None:
        """OpenAI service_tier reaches MetadataEvent on async non-streaming calls."""
        client, mock_response = _mock_openai_client()
        mock_response.service_tier = "batch"
        client.chat.completions.create = AsyncMockFn(return_value=mock_response)

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=_allow_budget_result()),
        ):
            await solwyn.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert reported_events[0].service_tier == "batch"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_openai_non_streaming_call_tags_agent_run_id(self) -> None:
        client, mock_response = _mock_openai_client()
        client.chat.completions.create = AsyncMockFn(return_value=mock_response)

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=_allow_budget_result()),
        ):
            async with solwyn_pkg.run("async-nightly-batch") as run_id:
                await solwyn.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                )

        assert len(reported_events) == 1
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "async-nightly-batch"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_primary_error_event_tags_agent_run_id(self) -> None:
        client, _ = _mock_openai_client()
        client.chat.completions.create = AsyncMockFn(side_effect=RuntimeError("primary failed"))

        solwyn = _make_async_solwyn(client)
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=_allow_budget_result()),
        ):
            async with solwyn_pkg.run("async-doomed") as run_id:
                with pytest.raises(RuntimeError, match="primary failed"):
                    await solwyn.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        assert len(reported_events) == 1
        assert reported_events[0].status == "error"
        assert reported_events[0].agent_run_id == run_id
        assert reported_events[0].agent_run_name == "async-doomed"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_fallback_retry_error_events_tag_agent_run_id(self) -> None:
        # Same-provider model-swap fallback on the SAME client; both hops 429.
        client, _ = _mock_openai_client()
        client.chat.completions.create = AsyncMockFn(
            side_effect=[
                _Status(429, "primary failed"),
                _Status(429, "fallback failed"),
            ]
        )

        solwyn = _make_async_solwyn(client, model="gpt-4o", fallback=[(client, "gpt-4o-mini")])
        reported_events: list = []
        solwyn._reporter.report = lambda e: reported_events.append(e)

        with patch.object(
            solwyn._budget,
            "check_budget",
            new=AsyncMockFn(return_value=_allow_budget_result()),
        ):
            async with solwyn_pkg.run("async-retry-doomed") as run_id:
                with pytest.raises(_Status, match="fallback failed"):
                    await solwyn.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        assert [event.status for event in reported_events] == ["error", "error"]
        assert all(event.agent_run_id == run_id for event in reported_events)
        assert all(event.agent_run_name == "async-retry-doomed" for event in reported_events)
        assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "gpt-4o-mini"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
