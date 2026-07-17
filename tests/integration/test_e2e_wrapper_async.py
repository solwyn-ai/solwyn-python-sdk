"""E2E: real AsyncSolwyn(AsyncOpenAI(...)) against the live API + fake provider."""

from __future__ import annotations

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

        response = await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        assert response.choices[0].message.content == RESPONSE_CONTENT
        assert fake_provider.request_count == 1

        assert len(recorder.confirms) == 1
        confirm = recorder.confirms[0]
        assert confirm["provider"] == "openai_compatible"
        assert confirm["token_details"].input_tokens == fake_provider.prompt_tokens
        assert confirm["token_details"].output_tokens == fake_provider.completion_tokens
        assert confirm["token_details"].is_estimated is False

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert event.status == CallStatus.SUCCESS
        assert event.provider == ProviderName.OPENAI_COMPATIBLE

    @pytest.mark.integration
    async def test_close_flushes_pending_wire_traffic(
        self, make_async_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        # Mirror of the sync close-flush contract: park a real settlement (a
        # fully consumed stream) AND a metadata event before close, so the
        # drained-after assertions are not vacuously true.
        client = await make_async_wrapped_client(reporter_flush_interval=60.0)
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, stream=True
        )
        async for _chunk in stream:
            pass
        await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
        assert len(client._reporter._settlement_queue) == 1
        assert len(client._reporter._queue) == 1

        await client.close()

        assert len(client._reporter._queue) == 0
        assert len(client._reporter._settlement_queue) == 0
