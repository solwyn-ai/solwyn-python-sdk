"""Tests for provider-native API surface interception.

Ensures client.messages.create() (Anthropic) and
client.models.generate_content() (Google) route through _intercepted_call,
not __getattr__ pass-through.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock as AsyncMockFn
from unittest.mock import MagicMock, patch

import pytest
from conftest import ALLOW_BUDGET_RESPONSE, VALID_API_KEY, VALID_PROJECT_ID

from solwyn.client import AsyncSolwyn, Solwyn


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

        with _mock_budget(solwyn):
            result = solwyn.messages.create(
                model="claude-sonnet-4-5",
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
        from solwyn.exceptions import BudgetExceededError

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
                model="claude-sonnet-4-5",
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

        with _mock_budget(solwyn):
            result = solwyn.models.generate_content(
                model="gemini-2.0-flash",
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
        # complete response). First-chunk materialization (§6.6) iterates it, so
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

        with _mock_budget(solwyn):
            solwyn.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        # Should have called generate_content_stream, NOT generate_content
        client.models.generate_content_stream.assert_called_once()
        client.models.generate_content.assert_not_called()
        solwyn.close()

    def test_models_getattr_passthrough(self) -> None:
        """Non-generate attributes pass through."""
        client = _mock_google_client()
        client.models.list = MagicMock(return_value=["gemini-pro"])
        solwyn = _make_solwyn(client)
        assert solwyn.models.list() == ["gemini-pro"]
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
        client.models.list.return_value = ["gpt-4o"]
        solwyn = _make_solwyn(client)
        # Should pass through to OpenAI's models.list(), not our proxy
        assert solwyn.models.list() == ["gpt-4o"]
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

        with _mock_async_budget(solwyn):
            result = await solwyn.messages.create(
                model="claude-sonnet-4-5",
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

        with _mock_async_budget(solwyn):
            result = await solwyn.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )

        client.models.generate_content.assert_called_once()
        assert len(reported) == 1
        assert result.text == "Hello"

        await solwyn._budget._http.aclose()
        await solwyn._reporter._http.aclose()
