"""E2E: real Solwyn(OpenAI(...)) wrapper against the live API + fake provider.

First tests ever to run the full interception pipeline (check -> provider call
-> usage extraction -> confirm -> metadata) over real HTTP on both sides.
"""

from __future__ import annotations

import openai
import pytest
from conftest import WireRecorder
from fake_provider import RESPONSE_CONTENT, FakeProviderServer

from solwyn._types import CallStatus, ProviderName

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.mark.integration
class TestSyncHappyPath:
    """check -> provider -> extract -> confirm -> report, non-streaming."""

    @pytest.mark.integration
    def test_full_lifecycle_with_exact_usage(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        # Arrange
        client = make_wrapped_client()
        recorder = WireRecorder().attach(client)

        # Act
        response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        # Assert — the caller sees the provider response untouched
        assert response.choices[0].message.content == RESPONSE_CONTENT
        assert response.usage.prompt_tokens == fake_provider.prompt_tokens

        # Assert — the provider was called exactly once with the caller's model
        assert fake_provider.request_count == 1
        assert fake_provider.requests[0].body["model"] == "gpt-4o"

        # Assert — the settlement carried the provider's exact usage (not an
        # estimate): confirm + event ride report_settlement as one ordered unit.
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.provider.value == "openai_compatible"
        assert confirm_request.token_details.input_tokens == fake_provider.prompt_tokens
        assert confirm_request.token_details.output_tokens == fake_provider.completion_tokens
        assert confirm_request.token_details.is_estimated is False
        assert confirm_request.reservation_id

        # Assert — the success metadata event travels WITH the confirm (no
        # separate report() fired) and carries matching wire token counts
        assert recorder.events == []
        assert event.status == CallStatus.SUCCESS
        assert event.provider == ProviderName.OPENAI_COMPATIBLE
        assert event.input_tokens == fake_provider.prompt_tokens
        assert event.output_tokens == fake_provider.completion_tokens
        assert event.call_id == confirm_request.call_id

    @pytest.mark.integration
    def test_streaming_settles_exact_usage_on_completion(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client()
        recorder = WireRecorder().attach(client)

        stream = client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta
        )

        assert content == RESPONSE_CONTENT
        # Streaming settles via report_settlement (confirm+event as one unit) —
        # the same single settlement path every reservation-backed success uses.
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.token_details.input_tokens == fake_provider.prompt_tokens
        assert confirm_request.token_details.output_tokens == fake_provider.completion_tokens
        assert confirm_request.token_details.is_estimated is False
        assert event.status == CallStatus.SUCCESS
        assert recorder.events == []

    @pytest.mark.integration
    def test_close_flushes_pending_wire_traffic(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        # Long flush interval: nothing flushes until close() forces it — the
        # same live-ingest close contract test_metadata_ingest exercises for
        # the bare reporter, now through the wrapper. Both queues must be
        # NON-EMPTY before close, or the drained-after assertions are vacuous:
        # a consumed stream parks its settlement (confirm+event) in
        # _settlement_queue; a provider ERROR queues a plain metadata event on
        # _queue via the report path (a reservation-backed SUCCESS now settles,
        # so it would land in _settlement_queue, not _queue).
        client = make_wrapped_client(reporter_flush_interval=60.0)
        stream = client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        for _chunk in stream:
            pass
        fake_provider.fail_next(429)
        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert len(client._reporter._settlement_queue) == 1
        assert len(client._reporter._queue) == 1

        client.close()

        assert len(client._reporter._queue) == 0
        assert len(client._reporter._settlement_queue) == 0
