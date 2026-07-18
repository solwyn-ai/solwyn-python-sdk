"""E2E: failover over real HTTP — real 429s, real fallback dispatch.

Every failover behavior here was previously proven only against hand-built
fake clients raising fake exceptions; these tests prove classification,
chain-walk, attribution, and settlement against actual wire traffic.
"""

from __future__ import annotations

import logging
import os

import pytest
from conftest import WireRecorder
from fake_provider import RESPONSE_CONTENT, FakeProviderServer

from solwyn._types import CallStatus, FailoverReason, ProviderName

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.fixture
def fallback_spec(fake_provider_fallback: FakeProviderServer):
    """(client, model, default_params, provider) fallback tuple -> second fake."""
    openai = pytest.importorskip("openai")
    inner = openai.OpenAI(base_url=fake_provider_fallback.base_url, api_key="sk-fake-fallback-key")
    return (inner, "gpt-4o-mini", {}, "groq")


@pytest.mark.integration
class TestSameDialectFailover:
    """Primary returns a REAL 429 -> the call lands on the fallback server."""

    @pytest.mark.integration
    def test_429_fails_over_and_attributes_serving_provider(
        self,
        make_wrapped_client,
        fallback_spec,
        fake_provider: FakeProviderServer,
        fake_provider_fallback: FakeProviderServer,
    ) -> None:
        # Arrange
        client = make_wrapped_client(fallback=[fallback_spec])
        recorder = WireRecorder().attach(client)
        fake_provider.fail_next(429)  # no Retry-After header

        # Act
        response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        # Assert — the caller sees the FALLBACK server's response
        assert response.choices[0].message.content == RESPONSE_CONTENT
        assert response.usage.prompt_tokens == 77  # fallback's counts, not primary's

        # Assert — exactly one request per server; the fallback got ITS entry model
        assert fake_provider.request_count == 1  # also proves per-hop max_retries=0
        assert fake_provider_fallback.request_count == 1
        assert fake_provider_fallback.requests[0].body["model"] == "gpt-4o-mini"

        # Assert — confirm settles against the SERVING provider
        assert len(recorder.confirms) == 1
        confirm = recorder.confirms[0]
        assert confirm["provider"] == "groq"
        assert confirm["model"] == "gpt-4o-mini"
        assert confirm["token_details"].input_tokens == 77
        assert confirm["token_details"].output_tokens == 33
        assert confirm["token_details"].is_estimated is False

        # Assert — one ERROR event (primary 429) + one SUCCESS event (fallback)
        error_events = [e for e in recorder.events if e.status == CallStatus.ERROR]
        success_events = [e for e in recorder.events if e.status == CallStatus.SUCCESS]
        assert len(error_events) == 1
        assert error_events[0].provider == ProviderName.OPENAI_COMPATIBLE
        assert len(success_events) == 1
        success = success_events[0]
        assert success.provider == ProviderName.GROQ
        assert success.is_provider_fallback is True
        assert success.failover_reason == FailoverReason.PRIMARY_ERROR
        assert success.attempt_index == 1
        assert success.requested_provider == ProviderName.OPENAI_COMPATIBLE
        assert success.requested_model == "gpt-4o"
        assert success.input_tokens == 77

    @pytest.mark.integration
    def test_429_with_retry_after_still_fails_over_under_default_config(
        self,
        make_wrapped_client,
        fallback_spec,
        fake_provider: FakeProviderServer,
        fake_provider_fallback: FakeProviderServer,
    ) -> None:
        # Default config (same_provider_retries=0): the header must not delay
        # or disrupt failover — the call lands on the fallback immediately.
        client = make_wrapped_client(fallback=[fallback_spec])
        recorder = WireRecorder().attach(client)
        fake_provider.fail_next(429, retry_after="0")

        response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        assert response.usage.prompt_tokens == 77  # fallback served
        assert fake_provider.request_count == 1  # no same-provider re-attempt
        assert fake_provider_fallback.request_count == 1
        assert recorder.confirms[0]["provider"] == "groq"

    @pytest.mark.integration
    def test_free_tier_directive_suppresses_same_provider_retries(
        self,
        make_wrapped_client,
        fallback_spec,
        fake_provider: FakeProviderServer,
        fake_provider_fallback: FakeProviderServer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # same_provider_retries=1 asks the SDK to honor Retry-After before
        # failing over, but the server's failover-tuning directive (free tier:
        # failover_tuning_allowed=False) forces SDK defaults — so the call
        # must fail over immediately with NO second primary attempt. This
        # wire-proves entitlement enforcement; the retry-honoring path itself
        # stays unit-covered until the harness can provision an entitled
        # (team/scale-tier) account.
        if os.environ.get("SOLWYN_TEST_API_KEY"):
            pytest.skip(
                "externally provided key may carry the failover-tuning entitlement; "
                "this test asserts the free-tier suppression path"
            )
        client = make_wrapped_client(fallback=[fallback_spec], same_provider_retries=1)
        recorder = WireRecorder().attach(client)
        fake_provider.fail_next(429, retry_after="0")

        with caplog.at_level(logging.WARNING, logger="solwyn._base"):
            response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        # The directive — not a broken retry path — is why the primary was not
        # re-attempted: both outcomes land on the fallback, so the suppression
        # warning is the distinguishing evidence.
        assert "Custom failover tuning is unavailable" in caplog.text
        assert response.usage.prompt_tokens == 77  # fallback served
        assert fake_provider.request_count == 1  # retry suppressed: one primary attempt
        assert fake_provider_fallback.request_count == 1
        assert recorder.confirms[0]["provider"] == "groq"
        success = [e for e in recorder.events if e.status == CallStatus.SUCCESS]
        assert len(success) == 1
        assert success[0].is_provider_fallback is True

    @pytest.mark.integration
    def test_chain_exhaustion_reraises_last_provider_error(
        self,
        make_wrapped_client,
        fallback_spec,
        fake_provider: FakeProviderServer,
        fake_provider_fallback: FakeProviderServer,
    ) -> None:
        openai = pytest.importorskip("openai")
        client = make_wrapped_client(fallback=[fallback_spec])
        recorder = WireRecorder().attach(client)
        fake_provider.fail_next(429)
        fake_provider_fallback.fail_next(429)

        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        assert fake_provider.request_count == 1
        assert fake_provider_fallback.request_count == 1
        assert recorder.confirms == []
        error_events = [e for e in recorder.events if e.status == CallStatus.ERROR]
        assert len(error_events) == 2  # one per attempted hop
        assert {e.provider for e in error_events} == {
            ProviderName.OPENAI_COMPATIBLE,
            ProviderName.GROQ,
        }

    @pytest.mark.integration
    def test_streaming_failover_settles_against_fallback(
        self,
        make_wrapped_client,
        fallback_spec,
        fake_provider: FakeProviderServer,
        fake_provider_fallback: FakeProviderServer,
    ) -> None:
        client = make_wrapped_client(fallback=[fallback_spec])
        recorder = WireRecorder().attach(client)
        fake_provider.fail_next(429)

        stream = client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta
        )

        assert content == RESPONSE_CONTENT
        assert fake_provider_fallback.request_count == 1
        # Streaming settles exactly once, via report_settlement, against the fallback
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.token_details.input_tokens == 77
        assert confirm_request.token_details.output_tokens == 33
        assert event.status == CallStatus.SUCCESS
        assert event.provider == ProviderName.GROQ
        assert event.is_provider_fallback is True
        assert recorder.confirms == []
