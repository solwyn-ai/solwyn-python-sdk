"""Smoke tests for the fake provider server itself (harness infrastructure)."""

from __future__ import annotations

import json

import httpx
import pytest
from fake_provider import (
    RESPONSE_CONTENT,
    FakeAnthropicServer,
    FakeProviderServer,
)
from pydantic import BaseModel

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("fake_provider_harness_only")]


class _FakeStructuredResponse(BaseModel):
    answer: str
    score: int


@pytest.mark.integration
class TestFakeProviderServer:
    """The fake server speaks enough OpenAI dialect for the SDK to consume."""

    @pytest.mark.integration
    def test_json_completion_with_usage(self) -> None:
        with FakeProviderServer(prompt_tokens=7, completion_tokens=3) as server:
            r = httpx.post(
                f"{server.base_url}/chat/completions",
                json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200
            data = r.json()
            assert data["choices"][0]["message"]["content"] == RESPONSE_CONTENT
            assert data["usage"] == {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            }
            assert server.request_count == 1
            assert server.requests[0].body["model"] == "gpt-5.5"

    @pytest.mark.integration
    def test_sse_stream_ends_with_usage_chunk(self) -> None:
        with FakeProviderServer() as server:
            with httpx.stream(
                "POST",
                f"{server.base_url}/chat/completions",
                json={"model": "gpt-5.5", "messages": [], "stream": True},
            ) as r:
                assert r.headers["content-type"].startswith("text/event-stream")
                lines = [
                    line.removeprefix("data: ")
                    for line in r.iter_lines()
                    if line.startswith("data: ")
                ]
            assert lines[-1] == "[DONE]"
            final = json.loads(lines[-2])
            assert final["usage"]["prompt_tokens"] == server.prompt_tokens

    @pytest.mark.integration
    def test_json_response_with_flat_usage_and_request_recording(self) -> None:
        with FakeProviderServer(prompt_tokens=7, completion_tokens=3) as server:
            body = {"model": "gpt-5.5", "input": "hello", "service_tier": "priority"}
            response = httpx.post(f"{server.base_url}/responses", json=body)

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "response"
            assert data["status"] == "completed"
            assert data["output"][0]["content"][0] == {
                "type": "output_text",
                "text": RESPONSE_CONTENT,
                "annotations": [],
                "logprobs": [],
            }
            assert data["usage"] == {
                "input_tokens": 7,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 10,
            }
            assert data["service_tier"] == "priority"
            assert server.requests == [type(server.requests[0])(path="/v1/responses", body=body)]

    @pytest.mark.integration
    def test_structured_response_returns_json_output_text(self) -> None:
        with FakeProviderServer() as server:
            response = httpx.post(
                f"{server.base_url}/responses",
                json={
                    "model": "gpt-5.5",
                    "input": "return a small object",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "fake_structured_response",
                            "schema": {"type": "object"},
                            "strict": True,
                        }
                    },
                },
            )

            output_text = response.json()["output"][0]["content"][0]["text"]
            assert json.loads(output_text) == {"answer": "structured fake response", "score": 7}

    @pytest.mark.integration
    def test_responses_sse_terminal_event_carries_full_response_and_usage(self) -> None:
        with FakeProviderServer(prompt_tokens=13, completion_tokens=5) as server:
            body = {"model": "gpt-5.5", "input": "hello", "stream": True}
            with httpx.stream("POST", f"{server.base_url}/responses", json=body) as response:
                assert response.headers["content-type"].startswith("text/event-stream")
                payloads = [
                    line.removeprefix("data: ")
                    for line in response.iter_lines()
                    if line.startswith("data: ")
                ]

            assert payloads[-1] == "[DONE]"
            events = [json.loads(payload) for payload in payloads[:-1]]
            assert [event["type"] for event in events] == [
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.completed",
            ]
            terminal = events[-1]
            assert terminal["response"]["output"][0]["content"][0]["text"] == RESPONSE_CONTENT
            assert terminal["response"]["usage"]["input_tokens"] == 13
            assert terminal["response"]["usage"]["output_tokens"] == 5
            assert terminal["response"]["usage"]["total_tokens"] == 18
            assert terminal["response"]["service_tier"] == "default"
            assert server.requests[0].path == "/v1/responses"
            assert server.requests[0].body == body

    @pytest.mark.integration
    def test_real_openai_stream_helper_consumes_responses_lifecycle(self) -> None:
        openai = pytest.importorskip("openai")

        with (
            FakeProviderServer() as server,
            openai.OpenAI(base_url=server.base_url, api_key="sk-fake") as client,
            client.responses.stream(model="gpt-5.5", input="hello") as stream,
        ):
            final_response = stream.get_final_response()

        assert final_response.output_text == RESPONSE_CONTENT
        assert server.requests[0].path == "/v1/responses"
        assert server.requests[0].body == {
            "model": "gpt-5.5",
            "input": "hello",
            "stream": True,
        }

    @pytest.mark.integration
    def test_real_openai_nonstream_and_parse_accept_omitted_responses_usage(self) -> None:
        openai = pytest.importorskip("openai")

        with (
            FakeProviderServer() as server,
            openai.OpenAI(base_url=server.base_url, api_key="sk-fake") as client,
        ):
            server.set_omit_usage(True)
            response = client.responses.create(model="gpt-5.5", input="hello")
            parsed = client.responses.parse(
                model="gpt-5.5",
                input="hello",
                text_format=_FakeStructuredResponse,
            )

        assert response.output_text == RESPONSE_CONTENT
        assert response.usage is None
        assert parsed.output_parsed == _FakeStructuredResponse(
            answer="structured fake response",
            score=7,
        )
        assert parsed.usage is None
        assert [request.path for request in server.requests] == [
            "/v1/responses",
            "/v1/responses",
        ]

    @pytest.mark.integration
    def test_real_openai_extra_body_overrides_named_responses_fields_on_the_wire(self) -> None:
        openai = pytest.importorskip("openai")

        with (
            FakeProviderServer() as server,
            openai.OpenAI(base_url=server.base_url, api_key="sk-fake") as client,
        ):
            client.responses.create(
                model="named-model",
                input="named input",
                instructions="named instructions",
                max_output_tokens=10,
                extra_body={
                    "model": "body-model",
                    "input": "body input",
                    "instructions": "body instructions",
                    "max_output_tokens": 99,
                },
            )

        assert server.requests[0].body == {
            "model": "body-model",
            "input": "body input",
            "instructions": "body instructions",
            "max_output_tokens": 99,
        }

    @pytest.mark.integration
    def test_fail_next_injects_error_then_recovers(self) -> None:
        with FakeProviderServer() as server:
            server.fail_next(500)
            body = {"model": "gpt-5.5", "messages": []}
            assert httpx.post(f"{server.base_url}/chat/completions", json=body).status_code == 500
            assert httpx.post(f"{server.base_url}/chat/completions", json=body).status_code == 200

    @pytest.mark.integration
    def test_fail_next_can_attach_retry_after_header(self) -> None:
        with FakeProviderServer() as server:
            server.fail_next(429, retry_after="0")
            r = httpx.post(
                f"{server.base_url}/chat/completions", json={"model": "m", "messages": []}
            )
            assert r.status_code == 429
            assert r.headers["retry-after"] == "0"
            # Next request recovers and carries no Retry-After
            r2 = httpx.post(
                f"{server.base_url}/chat/completions", json={"model": "m", "messages": []}
            )
            assert r2.status_code == 200

    @pytest.mark.integration
    def test_omit_usage_strips_usage_from_json_and_stream(self) -> None:
        with FakeProviderServer() as server:
            server.set_omit_usage(True)
            r = httpx.post(
                f"{server.base_url}/chat/completions", json={"model": "m", "messages": []}
            )
            assert "usage" not in r.json()
            with httpx.stream(
                "POST",
                f"{server.base_url}/chat/completions",
                json={"model": "m", "messages": [], "stream": True},
            ) as sr:
                payloads = [
                    line.removeprefix("data: ")
                    for line in sr.iter_lines()
                    if line.startswith("data: ")
                ]
            assert payloads[-1] == "[DONE]"
            assert all(json.loads(p).get("usage") is None for p in payloads[:-1])

            responses_json = httpx.post(
                f"{server.base_url}/responses",
                json={"model": "m", "input": "hello"},
            )
            assert "usage" not in responses_json.json()

            parsed_json = httpx.post(
                f"{server.base_url}/responses",
                json={
                    "model": "m",
                    "input": "hello",
                    "text": {"format": {"type": "json_schema", "schema": {}}},
                },
            )
            assert "usage" not in parsed_json.json()

            with httpx.stream(
                "POST",
                f"{server.base_url}/responses",
                json={"model": "m", "input": "hello", "stream": True},
            ) as responses_stream:
                response_payloads = [
                    line.removeprefix("data: ")
                    for line in responses_stream.iter_lines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
            terminal = json.loads(response_payloads[-1])
            assert terminal["type"] == "response.completed"
            assert "usage" not in terminal["response"]

            server.reset()
            r3 = httpx.post(
                f"{server.base_url}/chat/completions", json={"model": "m", "messages": []}
            )
            assert "usage" in r3.json()  # reset() restored usage

    @pytest.mark.integration
    def test_drop_next_stream_truncates_mid_stream(self) -> None:
        with FakeProviderServer() as server:
            server.drop_next_stream(after_chunks=1)
            received: list[str] = []
            with (
                pytest.raises(httpx.RemoteProtocolError),
                httpx.stream(
                    "POST",
                    f"{server.base_url}/chat/completions",
                    json={"model": "m", "messages": [], "stream": True},
                ) as sr,
            ):
                for line in sr.iter_lines():
                    if line.startswith("data: "):
                        received.append(line)
            assert len(received) == 1  # got the first event, then the wire died
            # One-shot: the next stream completes normally
            with httpx.stream(
                "POST",
                f"{server.base_url}/chat/completions",
                json={"model": "m", "messages": [], "stream": True},
            ) as sr:
                lines = [line for line in sr.iter_lines() if line.startswith("data: ")]
            assert lines[-1] == "data: [DONE]"


@pytest.mark.integration
class TestFakeAnthropicServer:
    """The anthropic fake speaks enough Messages dialect for one translated hop."""

    @pytest.mark.integration
    def test_messages_endpoint_returns_anthropic_shape(self) -> None:
        with FakeAnthropicServer(prompt_tokens=88, completion_tokens=44) as server:
            r = httpx.post(
                f"{server.base_url}/v1/messages",
                json={
                    "model": "claude-sonnet-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["content"] == [{"type": "text", "text": RESPONSE_CONTENT}]
            assert data["stop_reason"] == "end_turn"
            assert data["usage"] == {"input_tokens": 88, "output_tokens": 44}
            assert server.requests[0].body["max_tokens"] == 64

    @pytest.mark.integration
    def test_fail_next_and_streaming_rejection(self) -> None:
        with FakeAnthropicServer() as server:
            server.fail_next(429, retry_after="0")
            r = httpx.post(f"{server.base_url}/v1/messages", json={"model": "m"})
            assert r.status_code == 429
            assert r.headers["retry-after"] == "0"
            # Streaming is out of scope for the cross-dialect fake: fail LOUD.
            r2 = httpx.post(f"{server.base_url}/v1/messages", json={"model": "m", "stream": True})
            assert r2.status_code == 501
