"""Smoke tests for the fake provider server itself (harness infrastructure)."""

from __future__ import annotations

import json

import httpx
import pytest
from fake_provider import RESPONSE_CONTENT, FakeProviderServer


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
