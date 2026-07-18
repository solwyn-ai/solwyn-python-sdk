"""Smoke tests for the fake provider server itself (harness infrastructure)."""

from __future__ import annotations

import json

import httpx
import pytest
from fake_provider import RESPONSE_CONTENT, FakeAnthropicServer, FakeProviderServer


@pytest.mark.integration
class TestFakeProviderServer:
    """The fake server speaks enough OpenAI dialect for the SDK to consume."""

    @pytest.mark.integration
    def test_json_completion_with_usage(self) -> None:
        with FakeProviderServer(prompt_tokens=7, completion_tokens=3) as server:
            r = httpx.post(
                f"{server.base_url}/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
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
            assert server.requests[0].body["model"] == "gpt-4o"

    @pytest.mark.integration
    def test_sse_stream_ends_with_usage_chunk(self) -> None:
        with FakeProviderServer() as server:
            with httpx.stream(
                "POST",
                f"{server.base_url}/chat/completions",
                json={"model": "gpt-4o", "messages": [], "stream": True},
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
    def test_fail_next_injects_error_then_recovers(self) -> None:
        with FakeProviderServer() as server:
            server.fail_next(500)
            body = {"model": "gpt-4o", "messages": []}
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
                    "model": "claude-sonnet-4-5",
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
