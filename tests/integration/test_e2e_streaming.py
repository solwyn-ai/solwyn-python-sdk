"""E2E: streaming edge paths over real HTTP — error settlement and estimation.

The happy path (terminal usage chunk -> report_settlement exactly once) is
covered in test_e2e_wrapper_sync.py; these tests cover the wire behaviors
mocks can't fake: a REAL mid-stream connection drop, and providers that
genuinely omit usage.
"""

from __future__ import annotations

import httpx
import pytest
from conftest import WireRecorder
from fake_provider import RESPONSE_CONTENT, FakeProviderServer

from solwyn._types import CallStatus

MESSAGES = [{"role": "user", "content": "Say hello in five words."}]


@pytest.mark.integration
class TestStreamErrorSettlement:
    """A provider dropping mid-stream settles as an ERROR exactly once."""

    @pytest.mark.integration
    def test_mid_stream_drop_settles_error_exactly_once(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client()
        recorder = WireRecorder().attach(client)
        fake_provider.drop_next_stream(after_chunks=1)

        stream = client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        with pytest.raises(httpx.RemoteProtocolError):
            for _chunk in stream:
                pass

        # ERROR settlement: reported via reporter.report, never report_settlement
        error_events = [e for e in recorder.events if e.status == CallStatus.ERROR]
        assert len(error_events) == 1
        assert recorder.settlements == []
        assert recorder.confirms == []

        # Exactly once: poking close() after the error must not settle again
        stream.close()
        assert len([e for e in recorder.events if e.status == CallStatus.ERROR]) == 1
        assert recorder.settlements == []


@pytest.mark.integration
class TestUsageEstimation:
    """A provider omitting usage yields is_estimated=True — never silent zeros."""

    @pytest.mark.integration
    def test_missing_usage_yields_estimate_non_streaming(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client()
        recorder = WireRecorder().attach(client)
        fake_provider.set_omit_usage(True)

        response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

        assert response.choices[0].message.content == RESPONSE_CONTENT
        assert len(recorder.confirms) == 1
        details = recorder.confirms[0]["token_details"]
        assert details.is_estimated is True
        assert details.input_tokens > 0  # length-based, never zero
        assert details.output_tokens > 0

    @pytest.mark.integration
    def test_missing_terminal_usage_chunk_yields_estimate_streaming(
        self, make_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client()
        recorder = WireRecorder().attach(client)
        fake_provider.set_omit_usage(True)

        stream = client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True)
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta
        )

        assert content == RESPONSE_CONTENT
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.token_details.is_estimated is True
        assert confirm_request.token_details.input_tokens > 0
        assert confirm_request.token_details.output_tokens > 0
        assert event.status == CallStatus.SUCCESS  # estimation degrades data, not the call


@pytest.mark.integration
class TestAsyncStreaming:
    """Async streaming settles exactly once with the terminal chunk's usage."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_streaming_settles_exact_usage_once(
        self, make_async_wrapped_client, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client()
        recorder = WireRecorder().attach(client)

        stream = await client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, stream=True
        )
        content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                content += chunk.choices[0].delta.content or ""

        assert content == RESPONSE_CONTENT
        assert len(recorder.settlements) == 1
        confirm_request, event = recorder.settlements[0]
        assert confirm_request.token_details.input_tokens == fake_provider.prompt_tokens
        assert confirm_request.token_details.output_tokens == fake_provider.completion_tokens
        assert confirm_request.token_details.is_estimated is False
        assert event.status == CallStatus.SUCCESS
        assert recorder.confirms == []

        # Exactly once: closing an exhausted stream must not settle again
        await stream.close()
        assert len(recorder.settlements) == 1
