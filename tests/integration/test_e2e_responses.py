"""E2E: native OpenAI Responses surfaces through an explicit provider pin."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import WireRecorder
from fake_provider import RESPONSE_CONTENT, FakeProviderServer
from pydantic import BaseModel

from solwyn._types import CallStatus, ProviderName

INPUT = "Return the deterministic fake response."
MODEL = "gpt-5.5"
STRUCTURED_FORMAT = {
    "type": "json_schema",
    "strict": True,
    "name": "FakeStructuredResponse",
    "schema": {
        "properties": {
            "answer": {"title": "Answer", "type": "string"},
            "score": {"title": "Score", "type": "integer"},
        },
        "required": ["answer", "score"],
        "title": "FakeStructuredResponse",
        "type": "object",
        "additionalProperties": False,
    },
}


class FakeStructuredResponse(BaseModel):
    answer: str
    score: int


def _assert_request_and_lifecycle(
    recorder: WireRecorder,
    server: FakeProviderServer,
    *,
    expected_body: dict[str, object],
) -> None:
    assert server.request_count == 1
    request = server.requests[0]
    assert request.path == "/v1/responses"
    assert request.body == expected_body

    assert len(recorder.budget_checks) == 1
    check = recorder.budget_checks[0]
    assert check["provider"] == "openai"
    assert check["fallback_providers"] == []
    assert check["fallback_models"] == []

    assert recorder.events == []
    assert len(recorder.settlements) == 1
    confirm, event = recorder.settlements[0]
    assert confirm.provider == ProviderName.OPENAI
    assert confirm.token_details.input_tokens == server.prompt_tokens
    assert confirm.token_details.output_tokens == server.completion_tokens
    assert confirm.token_details.is_estimated is False
    assert event.status == CallStatus.SUCCESS
    assert event.provider == ProviderName.OPENAI
    assert event.input_tokens == server.prompt_tokens
    assert event.output_tokens == server.completion_tokens
    assert event.service_tier == "default"
    assert event.call_id == confirm.call_id


def _assert_estimated_request_and_lifecycle(
    recorder: WireRecorder,
    server: FakeProviderServer,
    *,
    expected_body: dict[str, object],
) -> None:
    assert server.request_count == 1
    request = server.requests[0]
    assert request.path == "/v1/responses"
    assert request.body == expected_body

    assert len(recorder.budget_checks) == 1
    check = recorder.budget_checks[0]
    assert check["provider"] == "openai"
    assert check["estimated_input_tokens"] > 0

    assert recorder.events == []
    assert len(recorder.settlements) == 1
    confirm, event = recorder.settlements[0]
    assert confirm.provider == ProviderName.OPENAI
    assert confirm.token_details.input_tokens == check["estimated_input_tokens"]
    assert confirm.token_details.output_tokens == 0
    assert confirm.token_details.is_estimated is True
    assert event.token_details == confirm.token_details
    assert event.provider == ProviderName.OPENAI
    assert event.call_id == confirm.call_id


@pytest.mark.integration
class TestSyncResponsesPipeline:
    def test_create_nonstream_without_usage_settles_estimate(
        self, make_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        fake_provider.set_omit_usage(True)
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = client.responses.create(model=MODEL, input=INPUT)

        assert response.output_text == RESPONSE_CONTENT
        assert response.usage is None
        _assert_estimated_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT},
        )

    def test_parse_without_usage_settles_estimate(
        self, make_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        fake_provider.set_omit_usage(True)
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = client.responses.parse(
            model=MODEL,
            input=INPUT,
            text_format=FakeStructuredResponse,
        )

        assert response.output_parsed == FakeStructuredResponse(
            answer="structured fake response",
            score=7,
        )
        assert response.usage is None
        _assert_estimated_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={
                "model": MODEL,
                "input": INPUT,
                "text": {"format": STRUCTURED_FORMAT},
            },
        )

    def test_create_nonstream(
        self, make_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = client.responses.create(model=MODEL, input=INPUT)

        assert response.output_text == RESPONSE_CONTENT
        assert response.usage.input_tokens == fake_provider.prompt_tokens
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT},
        )

    def test_create_stream(
        self, make_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        events = list(client.responses.create(model=MODEL, input=INPUT, stream=True))

        assert [event.type for event in events] == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.completed",
        ]
        assert events[-1].response.output_text == RESPONSE_CONTENT
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT, "stream": True},
        )

    def test_parse(self, make_wrapped_client: Any, fake_provider: FakeProviderServer) -> None:
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = client.responses.parse(
            model=MODEL,
            input=INPUT,
            text_format=FakeStructuredResponse,
        )

        assert response.output_parsed == FakeStructuredResponse(
            answer="structured fake response",
            score=7,
        )
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={
                "model": MODEL,
                "input": INPUT,
                "text": {"format": STRUCTURED_FORMAT},
            },
        )

    def test_stream_helper_forwards_get_final_response(
        self, make_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = make_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        with client.responses.stream(model=MODEL, input=INPUT) as stream:
            final_response = stream.get_final_response()

        assert final_response.output_text == RESPONSE_CONTENT
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT, "stream": True},
        )


@pytest.mark.integration
class TestAsyncResponsesPipeline:
    async def test_create_nonstream_without_usage_settles_estimate(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        fake_provider.set_omit_usage(True)
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = await client.responses.create(model=MODEL, input=INPUT)

        assert response.output_text == RESPONSE_CONTENT
        assert response.usage is None
        _assert_estimated_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT},
        )

    async def test_parse_without_usage_settles_estimate(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        fake_provider.set_omit_usage(True)
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = await client.responses.parse(
            model=MODEL,
            input=INPUT,
            text_format=FakeStructuredResponse,
        )

        assert response.output_parsed == FakeStructuredResponse(
            answer="structured fake response",
            score=7,
        )
        assert response.usage is None
        _assert_estimated_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={
                "model": MODEL,
                "input": INPUT,
                "text": {"format": STRUCTURED_FORMAT},
            },
        )

    async def test_create_nonstream(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = await client.responses.create(model=MODEL, input=INPUT)

        assert response.output_text == RESPONSE_CONTENT
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT},
        )

    async def test_create_stream(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        stream = await client.responses.create(model=MODEL, input=INPUT, stream=True)
        events = [event async for event in stream]

        assert [event.type for event in events] == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.completed",
        ]
        assert events[-1].response.output_text == RESPONSE_CONTENT
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT, "stream": True},
        )

    async def test_parse(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        response = await client.responses.parse(
            model=MODEL,
            input=INPUT,
            text_format=FakeStructuredResponse,
        )

        assert response.output_parsed == FakeStructuredResponse(
            answer="structured fake response",
            score=7,
        )
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={
                "model": MODEL,
                "input": INPUT,
                "text": {"format": STRUCTURED_FORMAT},
            },
        )

    async def test_stream_helper_forwards_get_final_response(
        self, make_async_wrapped_client: Any, fake_provider: FakeProviderServer
    ) -> None:
        client = await make_async_wrapped_client(provider="openai")
        recorder = WireRecorder().attach(client)

        async with client.responses.stream(model=MODEL, input=INPUT) as stream:
            final_response = await stream.get_final_response()

        assert final_response.output_text == RESPONSE_CONTENT
        _assert_request_and_lifecycle(
            recorder,
            fake_provider,
            expected_body={"model": MODEL, "input": INPUT, "stream": True},
        )
