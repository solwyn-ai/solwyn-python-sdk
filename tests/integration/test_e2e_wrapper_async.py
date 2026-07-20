"""E2E: real AsyncSolwyn(AsyncOpenAI(...)) against the live API + fake provider."""

from __future__ import annotations

import openai
import pytest
from conftest import WireRecorder
from fake_provider import RESPONSE_CONTENT, FakeProviderServer

from solwyn._types import CallStatus, ProviderName

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.mark.integration
class TestAsyncHappyPath:
    """The async pipeline mirrors sync: check -> provider -> confirm -> report."""

    @pytest.mark.integration
    async def test_full_lifecycle_with_exact_usage(
        self, make_async_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client()
        recorder = WireRecorder().attach(client)

        response = await client.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert response.choices[0].message.content == RESPONSE_CONTENT
        assert fake_provider.request_count == 1

        # A non-streaming reservation-backed success settles via
        # report_settlement (confirm + event as one unit); no separate report().
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.provider.value == "openai_compatible"
        assert confirm_request.token_details.input_tokens == fake_provider.prompt_tokens
        assert confirm_request.token_details.output_tokens == fake_provider.completion_tokens
        assert confirm_request.token_details.is_estimated is False

        assert recorder.events == []
        assert event.status == CallStatus.SUCCESS
        assert event.provider == ProviderName.OPENAI_COMPATIBLE

    @pytest.mark.integration
    async def test_close_flushes_pending_wire_traffic(
        self, make_async_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        # Mirror of the sync close-flush contract: park a real settlement (a
        # fully consumed stream) in _settlement_queue AND a plain metadata event
        # (a provider ERROR via the report path) in _queue before close, so the
        # drained-after assertions are not vacuously true. A reservation-backed
        # SUCCESS now settles, so it would land in _settlement_queue, not _queue.
        client = await make_async_wrapped_client(reporter_flush_interval=60.0)
        stream = await client.chat.completions.create(
            model="gpt-5.5", messages=MESSAGES, stream=True
        )
        async for _chunk in stream:
            pass
        fake_provider.fail_next(429)
        with pytest.raises(openai.RateLimitError):
            await client.chat.completions.create(model="gpt-5.5", messages=MESSAGES)
        assert len(client._reporter._settlement_queue) == 1
        assert len(client._reporter._queue) == 1

        await client.close()

        assert len(client._reporter._queue) == 0
        assert len(client._reporter._settlement_queue) == 0
