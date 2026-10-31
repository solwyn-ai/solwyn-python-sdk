"""WS-13: under hard_deny the output bound is TRUE, or the call is refused locally.

Real-SDK smoke tests (the sanctioned exception, see tests/CLAUDE.md): GENUINE
provider SDK clients serialize each request onto an in-memory transport (httpx
``MockTransport``, httpx2 for anthropic 2.x) or a botocore ``Stubber``, so the
assertions read the body the provider would actually receive. Never network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from solwyn._lease import DEFAULT_OUTPUT_BOUND
from solwyn.exceptions import OutputBoundUnenforceableError, SolwynError
from solwyn.testing import FakeControlPlane

httpx = pytest.importorskip("httpx")

MESSAGES = [{"role": "user", "content": "hi"}]
QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
XAI_URL = "https://api.x.ai/v1"

_CHAT_BODY = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_GEMINI_BODY = {
    "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


class _Wire:
    """Records every serialized provider request body."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._response = response

    def handler(self, request: Any) -> Any:
        self.bodies.append(json.loads(request.content or b"{}"))
        return httpx.Response(200, json=self._response)

    async def async_handler(self, request: Any) -> Any:
        return self.handler(request)


def _openai(wire: _Wire, base_url: str | None = None) -> Any:
    openai = pytest.importorskip("openai")
    return openai.OpenAI(
        api_key="sk-test",
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
        max_retries=0,
    )


def _azure(wire: _Wire, api_version: str) -> Any:
    openai = pytest.importorskip("openai")
    return openai.AzureOpenAI(
        api_key="k",
        azure_endpoint="https://example.openai.azure.com",
        api_version=api_version,
        http_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
        max_retries=0,
    )


def _google(wire: _Wire, **http_options: Any) -> Any:
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    return genai.Client(
        api_key="k",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
            **http_options,
        ),
    )


def _bound(plane: FakeControlPlane) -> int | None:
    assert len(plane.checks) == 1
    return plane.checks[0].estimated_output_bound


def _refused(exc_info: pytest.ExceptionInfo[OutputBoundUnenforceableError], reason: str) -> None:
    error = exc_info.value
    assert error.code == "output_bound_unenforceable"
    assert error.reason == reason
    assert str(error).endswith(f"(output_bound_unenforceable: {reason})")


@pytest.mark.unit
class TestOpenAIDialect:
    def test_multiple_choices_reserve_cap_times_count(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, n=4, max_tokens=1000)

        assert _bound(plane) == 4000
        assert wire.bodies[0]["n"] == 4
        assert wire.bodies[0]["max_tokens"] == 1000

    def test_uncapped_multiple_choices_inject_the_per_choice_default(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, n=2)

        assert _bound(plane) == 2 * DEFAULT_OUTPUT_BOUND
        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND

    def test_malformed_choice_count_is_refused(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, n="2")

        _refused(caught, "malformed_candidate_count")
        assert wire.bodies == []

    def test_extra_body_cap_is_the_priced_cap(self) -> None:
        # The SDK applies extra_body after named arguments, so 10000 is what
        # the provider receives; the check must price exactly that.
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="gpt-4o",
                messages=MESSAGES,
                max_tokens=4096,
                extra_body={"max_tokens": 10_000},
            )

        assert _bound(plane) == 10_000
        assert wire.bodies[0]["max_tokens"] == 10_000

    def test_extra_body_choice_count_is_priced(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="gpt-4o", messages=MESSAGES, max_tokens=100, extra_body={"n": 3}
            )

        assert _bound(plane) == 300
        assert wire.bodies[0]["n"] == 3

    def test_cap_beyond_the_wire_range_is_refused(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(
                model="gpt-4o", messages=MESSAGES, max_tokens=200_000_000
            )

        _refused(caught, "bound_exceeded")
        assert wire.bodies == []

    def test_alert_only_sends_every_request_as_built(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane(mode="alert_only")
        with plane.wrap(_openai(wire), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, n="2")

        assert wire.bodies[0] == {"model": "gpt-4o", "messages": MESSAGES, "n": "2"}

    @pytest.mark.asyncio
    async def test_async_multiple_choices_reserve_cap_times_count(self) -> None:
        openai = pytest.importorskip("openai")
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        client = openai.AsyncOpenAI(
            api_key="sk-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.async_handler)),
            max_retries=0,
        )
        async with plane.wrap_async(client, lease_enabled=False) as solwyn:
            await solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, n=3)

        assert _bound(plane) == 3 * DEFAULT_OUTPUT_BOUND
        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND


@pytest.mark.unit
class TestReasoningOutsideTheCap:
    def test_qwen_thinking_budget_is_added_to_the_bound(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="qwen3.5-plus",
                messages=MESSAGES,
                extra_body={"enable_thinking": True, "thinking_budget": 12_000},
            )

        assert _bound(plane) == DEFAULT_OUTPUT_BOUND + 12_000
        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND
        assert wire.bodies[0]["thinking_budget"] == 12_000

    def test_qwen_explicit_completion_cap_still_prices_its_thinking(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="qwen3.5-plus",
                messages=MESSAGES,
                max_completion_tokens=4096,
                extra_body={"enable_thinking": True, "thinking_budget": 12_000},
            )

        assert _bound(plane) == 4096 + 12_000
        assert wire.bodies[0]["max_tokens"] == 4096

    def test_qwen_default_thinking_without_a_budget_is_refused(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="qwen3.5-plus", messages=MESSAGES)

        _refused(caught, "reasoning_without_budget")
        assert caught.value.provider == "qwen"
        assert wire.bodies == []

    def test_qwen_thinking_off_needs_no_budget(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="qwen3.5-plus", messages=MESSAGES, extra_body={"enable_thinking": False}
            )

        assert _bound(plane) == DEFAULT_OUTPUT_BOUND
        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND

    def test_qwen_alert_only_is_sent_as_built(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane(mode="alert_only")
        with plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="qwen3.5-plus", messages=MESSAGES)

        assert "max_tokens" not in wire.bodies[0]

    def test_xai_reasoning_model_is_refused(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire, XAI_URL), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="grok-4", messages=MESSAGES, max_tokens=100)

        _refused(caught, "reasoning_without_budget")
        assert wire.bodies == []

    def test_xai_non_reasoning_model_is_capped(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, XAI_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="grok-4-fast-non-reasoning", messages=MESSAGES)

        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND

    def test_lease_refusal_releases_the_reservation(self) -> None:
        import solwyn

        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, QWEN_URL)) as wrapped, solwyn.run("refused") as run_id:
            wrapped.chat.completions.create(
                model="qwen3.5-plus", messages=MESSAGES, extra_body={"enable_thinking": False}
            )
            with pytest.raises(OutputBoundUnenforceableError):
                wrapped.chat.completions.create(model="qwen3.5-plus", messages=MESSAGES)
            state = wrapped._solwyn_budget._lease._states[run_id]
            assert state.reservations == {}

        assert len(plane.lease_grants) == 1
        assert len(wire.bodies) == 1


@pytest.mark.unit
class TestAzureDeployments:
    def test_deployment_alias_gets_max_completion_tokens(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_azure(wire, "2024-10-21"), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="prod-reasoning", messages=MESSAGES)

        assert wire.bodies[0]["max_completion_tokens"] == DEFAULT_OUTPUT_BOUND
        assert "max_tokens" not in wire.bodies[0]

    def test_old_api_version_refuses_an_uncapped_call(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_azure(wire, "2024-06-01"), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="prod-reasoning", messages=MESSAGES)

        _refused(caught, "azure_api_version")
        assert wire.bodies == []

    def test_old_api_version_sends_an_explicit_cap_as_is(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_azure(wire, "2024-06-01"), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="prod-reasoning", messages=MESSAGES, max_tokens=300
            )

        assert wire.bodies[0]["max_tokens"] == 300

    @pytest.mark.parametrize("mode", ["hard_deny", "alert_only"])
    def test_completion_cap_is_never_weakened(self, mode: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane(mode=mode)
        with plane.wrap(_azure(wire, "2024-10-21"), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="my-gpt4o", messages=MESSAGES, max_completion_tokens=500
            )

        assert wire.bodies[0]["max_completion_tokens"] == 500
        assert "max_tokens" not in wire.bodies[0]

    def test_highest_layer_cap_key_wins_alone(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(
            _azure(wire, "2024-10-21"), lease_enabled=False, default_params={"max_tokens": 100}
        ) as solwyn:
            solwyn.chat.completions.create(
                model="my-gpt4o", messages=MESSAGES, max_completion_tokens=500
            )

        assert _bound(plane) == 500
        assert wire.bodies[0]["max_completion_tokens"] == 500
        assert "max_tokens" not in wire.bodies[0]


@pytest.mark.unit
class TestAnthropic:
    def test_extra_body_cap_is_the_priced_cap(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        httpx2 = pytest.importorskip("httpx2")
        bodies: list[dict[str, Any]] = []

        def handler(request: Any) -> Any:
            bodies.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json={
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "c",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        client = anthropic.Anthropic(
            api_key="k",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
            max_retries=0,
        )
        plane = FakeControlPlane()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.messages.create(
                model="claude-sonnet-4-5",
                messages=MESSAGES,
                max_tokens=4096,
                extra_body={"max_tokens": 10_000},
            )

        assert _bound(plane) == 10_000
        assert bodies[0]["max_tokens"] == 10_000


@pytest.mark.unit
class TestGoogle:
    def test_bound_uses_the_dispatch_field_level_merge(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with plane.wrap(
            _google(wire),
            lease_enabled=False,
            default_params={"config": {"max_output_tokens": 10_000}},
        ) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config={"temperature": 0.2}
            )

        assert _bound(plane) == 10_000
        assert wire.bodies[0]["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": 10_000}

    def test_generation_config_alias_cap_is_kept(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with plane.wrap(_google(wire), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", generation_config={"max_output_tokens": 500}
            )

        assert _bound(plane) == 500
        assert wire.bodies[0]["generationConfig"] == {"maxOutputTokens": 500}

    def test_camel_case_cap_is_kept_without_a_duplicate(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with plane.wrap(_google(wire), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config={"maxOutputTokens": 10_000}
            )

        assert _bound(plane) == 10_000
        assert wire.bodies[0]["generationConfig"] == {"maxOutputTokens": 10_000}

    def test_candidate_count_multiplies_the_bound(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with plane.wrap(_google(wire), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro",
                contents="hi",
                config={"candidateCount": 3, "max_output_tokens": 100},
            )

        assert _bound(plane) == 300
        assert wire.bodies[0]["generationConfig"] == {"candidateCount": 3, "maxOutputTokens": 100}

    def test_request_body_override_is_refused(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        override = {"extra_body": {"generationConfig": {"maxOutputTokens": 9000}}}
        with (
            plane.wrap(_google(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.models.generate_content(
                model="gemini-2.5-pro",
                contents="hi",
                config={"max_output_tokens": 100, "http_options": override},
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    def test_request_body_override_is_sent_on_alert_only(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane(mode="alert_only")
        override = {"extra_body": {"generationConfig": {"maxOutputTokens": 9000}}}
        with plane.wrap(_google(wire), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro",
                contents="hi",
                config={"max_output_tokens": 100, "http_options": override},
            )

        assert wire.bodies[0]["generationConfig"] == {"maxOutputTokens": 9000}

    def test_client_body_extension_is_refused(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        client = _google(wire, extra_body={"generationConfig": {"maxOutputTokens": 9000}})
        with (
            plane.wrap(client, lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.models.generate_content(model="gemini-2.5-pro", contents="hi")

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    @pytest.mark.parametrize("config", [17, []])
    def test_malformed_config_never_reaches_the_provider(self, config: object) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with plane.wrap(_google(wire), lease_enabled=False) as solwyn, pytest.raises(SolwynError):
            solwyn.models.generate_content(model="gemini-2.5-pro", contents="hi", config=config)

        assert wire.bodies == []


@pytest.mark.unit
class TestBedrock:
    def _client(self) -> tuple[Any, Any]:
        boto3 = pytest.importorskip("boto3")
        from botocore.stub import Stubber

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        return client, Stubber(client)

    @pytest.mark.parametrize("inference", [[], 17, "x"])
    def test_malformed_inference_config_is_refused(self, inference: object) -> None:
        client, stubber = self._client()
        plane = FakeControlPlane()
        with (
            plane.wrap(client, lease_enabled=False) as solwyn,
            stubber,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.converse(
                modelId="anthropic.claude",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
                inferenceConfig=inference,
            )

        _refused(caught, "malformed_cap_container")
        stubber.assert_no_pending_responses()

    def test_model_field_cap_override_is_refused(self) -> None:
        client, stubber = self._client()
        plane = FakeControlPlane()
        with (
            plane.wrap(client, lease_enabled=False) as solwyn,
            stubber,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.converse(
                modelId="anthropic.claude",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
                inferenceConfig={"maxTokens": 100},
                additionalModelRequestFields={"max_tokens": 9000},
            )

        _refused(caught, "unsupported_override")

    def test_thinking_budget_inside_max_tokens_is_sent(self) -> None:
        client, stubber = self._client()
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        thinking = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        stubber.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            },
            {
                "modelId": "anthropic.claude",
                "messages": messages,
                "inferenceConfig": {"maxTokens": 2000},
                "additionalModelRequestFields": thinking,
            },
        )
        plane = FakeControlPlane()
        with plane.wrap(client, lease_enabled=False) as solwyn, stubber:
            solwyn.converse(
                modelId="anthropic.claude",
                messages=messages,
                inferenceConfig={"maxTokens": 2000},
                additionalModelRequestFields=thinking,
            )

        stubber.assert_no_pending_responses()
        assert _bound(plane) == 2000


@pytest.mark.unit
class TestBodyExtensionAllowlist:
    """Under hard_deny a body extension may carry only keys the bound accounts for."""

    @pytest.mark.parametrize(
        "http_options",
        [
            {"extraBody": {"generationConfig": {"maxOutputTokens": 9000, "candidateCount": 3}}},
            {"extraBody": {"generation_config": {"maxOutputTokens": 9000}}},
            {"extra_body": {"GenerationConfig": {"maxOutputTokens": 9000}}},
            {"extra_body": {"labels": {"team": "a"}}},
        ],
    )
    def test_google_request_extension_is_refused_in_any_spelling(
        self, http_options: dict[str, Any]
    ) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        with (
            plane.wrap(_google(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.models.generate_content(
                model="gemini-2.5-pro",
                contents="hi",
                config={"max_output_tokens": 100, "http_options": http_options},
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    def test_google_client_extension_with_any_key_is_refused(self) -> None:
        wire, plane = _Wire(_GEMINI_BODY), FakeControlPlane()
        client = _google(wire, extra_body={"labels": {"team": "a"}})
        with (
            plane.wrap(client, lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config={"max_output_tokens": 100}
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    @pytest.mark.asyncio
    async def test_async_google_request_extension_is_refused(self) -> None:
        genai = pytest.importorskip("google.genai")
        from google.genai import types

        from solwyn.client import AsyncSolwyn

        wire = _Wire(_GEMINI_BODY)
        client = genai.Client(
            api_key="k",
            http_options=types.HttpOptions(
                httpx_client=httpx.Client(transport=httpx.MockTransport(wire.handler)),
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(wire.async_handler)
                ),
            ),
        )
        plane = FakeControlPlane()
        solwyn = AsyncSolwyn(
            client.aio,
            api_key=plane.api_key,
            api_url=plane.api_url,
            control_plane_transport=plane.transport,
            budget_check_cache_ttl=0,
            lease_enabled=False,
        )
        try:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.models.generate_content(
                    model="gemini-2.5-pro",
                    contents="hi",
                    config={
                        "max_output_tokens": 100,
                        "http_options": {
                            "extraBody": {"generationConfig": {"maxOutputTokens": 9000}}
                        },
                    },
                )
        finally:
            await solwyn.close()

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    def test_model_override_in_extra_body_is_refused(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire, XAI_URL), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(
                model="grok-4-fast-non-reasoning",
                messages=MESSAGES,
                max_tokens=100,
                extra_body={"model": "grok-4"},
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    @pytest.mark.asyncio
    async def test_async_model_override_in_extra_body_is_refused(self) -> None:
        openai = pytest.importorskip("openai")
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        client = openai.AsyncOpenAI(
            api_key="sk-test",
            base_url=XAI_URL,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.async_handler)),
            max_retries=0,
        )
        async with plane.wrap_async(client, lease_enabled=False) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.chat.completions.create(
                    model="grok-4-fast-non-reasoning",
                    messages=MESSAGES,
                    max_tokens=100,
                    extra_body={"model": "grok-4"},
                )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    @pytest.mark.parametrize("key", ["top_k", "Max_Tokens", "reasoning"])
    def test_unknown_openai_extra_body_key_is_refused(self, key: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(
                model="gpt-4o", messages=MESSAGES, max_tokens=100, extra_body={key: 1}
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    def test_qwen_thinking_keys_are_allowed_but_not_elsewhere(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        thinking = {"enable_thinking": True, "thinking_budget": 1000}
        with plane.wrap(_openai(wire, QWEN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="qwen3.5-plus", messages=MESSAGES, max_tokens=100, extra_body=thinking
            )
        assert wire.bodies[0]["thinking_budget"] == 1000

        other, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(other), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(
                model="gpt-4o", messages=MESSAGES, max_tokens=100, extra_body=thinking
            )
        _refused(caught, "unsupported_override")
        assert other.bodies == []

    def test_unknown_extra_body_key_is_sent_on_alert_only(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane(mode="alert_only")
        with plane.wrap(_openai(wire, XAI_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="grok-4-fast-non-reasoning",
                messages=MESSAGES,
                max_tokens=100,
                extra_body={"model": "grok-4"},
            )

        assert wire.bodies[0]["model"] == "grok-4"

    def test_unknown_responses_extra_body_key_is_refused(self) -> None:
        wire = _Wire(
            {
                "id": "r",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.5",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        plane = FakeControlPlane()
        with (
            plane.wrap(_openai(wire), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.responses.create(
                model="gpt-5.5", input="hi", max_output_tokens=100, extra_body={"vendor": 1}
            )

        _refused(caught, "unsupported_override")
        assert wire.bodies == []

    @pytest.mark.parametrize("fields", [{"top_k": 5}, {"anthropic_beta": ["x"]}, []])
    def test_bedrock_model_fields_outside_the_allowlist_are_refused(self, fields: object) -> None:
        boto3 = pytest.importorskip("boto3")
        from botocore.stub import Stubber

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        stubber = Stubber(client)
        plane = FakeControlPlane()
        with (
            plane.wrap(client, lease_enabled=False) as solwyn,
            stubber,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.converse(
                modelId="anthropic.claude",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
                inferenceConfig={"maxTokens": 100},
                additionalModelRequestFields=fields,
            )

        reason = "malformed_cap_container" if fields == [] else "unsupported_override"
        _refused(caught, reason)


def _get_weather(city: str) -> str:
    """Report the weather (an AFC-callable test tool)."""
    return f"sunny in {city}"


class _LoopingGemini:
    """google-genai wire double whose first response invites another generation.

    ``kind="continuation"`` ends the first response with CONTINUATION and a
    token, so automatic continuation would resend the request. ``kind="afc"``
    answers with a function call, so automatic function calling would run the
    tool and generate again. Every later response stops.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.bodies: list[dict[str, Any]] = []

    def _response(self) -> dict[str, Any]:
        first = len(self.bodies) == 1
        if first and self.kind == "afc":
            parts: list[dict[str, Any]] = [
                {"functionCall": {"name": "_get_weather", "args": {"city": "Oslo"}}}
            ]
        else:
            parts = [{"text": "ok"}]
        candidate: dict[str, Any] = {
            "content": {"parts": parts, "role": "model"},
            "finishReason": "CONTINUATION" if first and self.kind == "continuation" else "STOP",
        }
        if first and self.kind == "continuation":
            candidate["continuationToken"] = "dG9rZW4="
        return {
            "candidates": [candidate],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 100,
                "totalTokenCount": 101,
            },
        }

    def handler(self, request: Any) -> Any:
        self.bodies.append(json.loads(request.content or b"{}"))
        body = self._response()
        if "alt=sse" in str(request.url):
            return httpx.Response(
                200,
                text=f"data: {json.dumps(body)}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=body)

    async def async_handler(self, request: Any) -> Any:
        return self.handler(request)

    def client(self) -> Any:
        genai = pytest.importorskip("google.genai")
        from google.genai import types

        # Both transports are mocked whatever the caller's mode: AsyncSolwyn
        # may dispatch a genai.Client through its sync path, and a missing
        # mock would reach the real network.
        options = types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(self.async_handler)),
        )
        return genai.Client(api_key="k", http_options=options)


def _async_google_wrapper(plane: FakeControlPlane, client: Any) -> Any:
    from solwyn.client import AsyncSolwyn

    return AsyncSolwyn(
        client.aio,
        api_key=plane.api_key,
        api_url=plane.api_url,
        control_plane_transport=plane.transport,
        budget_check_cache_ttl=0,
        lease_enabled=False,
    )


@pytest.mark.unit
class TestGoogleSingleGeneration:
    """Under hard_deny a Google call makes exactly one model generation."""

    def _config(self, kind: str, **extra: Any) -> dict[str, Any]:
        config: dict[str, Any] = {"max_output_tokens": 100, **extra}
        if kind == "afc":
            config["tools"] = [_get_weather]
        return config

    @pytest.mark.parametrize("kind", ["continuation", "afc"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_default_loop_is_turned_off(self, kind: str, streaming: bool) -> None:
        provider, plane = _LoopingGemini(kind), FakeControlPlane()
        with plane.wrap(provider.client(), lease_enabled=False) as solwyn:
            if streaming:
                list(
                    solwyn.models.generate_content_stream(
                        model="gemini-2.5-pro", contents="hi", config=self._config(kind)
                    )
                )
            else:
                solwyn.models.generate_content(
                    model="gemini-2.5-pro", contents="hi", config=self._config(kind)
                )

        assert _bound(plane) == 100
        assert len(provider.bodies) == 1
        assert provider.bodies[0]["generationConfig"] == {"maxOutputTokens": 100}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["continuation", "afc"])
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_async_default_loop_is_turned_off(self, kind: str, streaming: bool) -> None:
        provider, plane = _LoopingGemini(kind), FakeControlPlane()
        solwyn = _async_google_wrapper(plane, provider.client())
        try:
            if streaming:
                stream = await solwyn.models.generate_content_stream(
                    model="gemini-2.5-pro", contents="hi", config=self._config(kind)
                )
                _ = [chunk async for chunk in stream]
            else:
                await solwyn.models.generate_content(
                    model="gemini-2.5-pro", contents="hi", config=self._config(kind)
                )
        finally:
            await solwyn.close()

        assert _bound(plane) == 100
        assert len(provider.bodies) == 1

    @pytest.mark.parametrize(
        ("kind", "extra"),
        [
            ("continuation", {"automatic_continuation": True}),
            ("continuation", {"automaticContinuation": True}),
            ("afc", {"automatic_function_calling": {"disable": False}}),
            ("afc", {"automatic_function_calling": {"maximum_remote_calls": 3}}),
        ],
    )
    def test_explicitly_enabled_loop_is_refused(self, kind: str, extra: dict[str, Any]) -> None:
        provider, plane = _LoopingGemini(kind), FakeControlPlane()
        with (
            plane.wrap(provider.client(), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config=self._config(kind, **extra)
            )

        _refused(caught, "multi_generation_loop")
        assert provider.bodies == []

    @pytest.mark.asyncio
    async def test_async_explicitly_enabled_loop_is_refused(self) -> None:
        provider, plane = _LoopingGemini("continuation"), FakeControlPlane()
        solwyn = _async_google_wrapper(plane, provider.client())
        try:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.models.generate_content(
                    model="gemini-2.5-pro",
                    contents="hi",
                    config=self._config("continuation", automatic_continuation=True),
                )
        finally:
            await solwyn.close()

        _refused(caught, "multi_generation_loop")
        assert provider.bodies == []

    @pytest.mark.parametrize(
        ("kind", "extra"),
        [
            ("continuation", {"automatic_continuation": False}),
            ("afc", {"automatic_function_calling": {"disable": True}}),
            ("afc", {"automatic_function_calling": {"maximum_remote_calls": 0}}),
        ],
    )
    def test_explicitly_disabled_loop_is_sent_once(self, kind: str, extra: dict[str, Any]) -> None:
        provider, plane = _LoopingGemini(kind), FakeControlPlane()
        with plane.wrap(provider.client(), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config=self._config(kind, **extra)
            )

        assert len(provider.bodies) == 1

    @pytest.mark.parametrize("kind", ["continuation", "afc"])
    def test_alert_only_keeps_the_sdk_loop(self, kind: str) -> None:
        provider, plane = _LoopingGemini(kind), FakeControlPlane(mode="alert_only")
        with plane.wrap(provider.client(), lease_enabled=False) as solwyn:
            solwyn.models.generate_content(
                model="gemini-2.5-pro", contents="hi", config=self._config(kind)
            )

        assert len(provider.bodies) == 2


def _anthropic_sse(data: dict[str, Any]) -> str:
    start = {
        **data,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "ok"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": data["stop_reason"], "stop_sequence": None},
                "usage": data["usage"],
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {kind}\ndata: {json.dumps(event)}\n\n" for kind, event in events)


class _AnthropicWire:
    """anthropic 2.x (httpx2) wire double.

    ``refuse_first`` answers the first request with a 200 refusal, which makes
    ``BetaRefusalFallbackMiddleware`` resend at its fallback's cap.
    ``fail_first`` answers the first request with a 500.
    """

    def __init__(
        self, *, refuse_first: bool = False, fail_first: bool = False, slow: bool = False
    ) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.refuse_first = refuse_first
        self.fail_first = fail_first
        # Hold each request open so concurrent sends genuinely overlap.
        self.slow = slow

    def handler(self, request: Any) -> Any:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.slow:
            import time

            time.sleep(0.2)
        return self._respond(request, body)

    def _respond(self, request: Any, body: dict[str, Any]) -> Any:
        httpx2 = pytest.importorskip("httpx2")
        first = len(self.bodies) == 1
        if first and self.fail_first:
            return httpx2.Response(
                500,
                request=request,
                json={"type": "error", "error": {"type": "api_error", "message": "x"}},
            )
        refused = first and self.refuse_first
        message: dict[str, Any] = {
            "id": f"msg_{len(self.bodies)}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [] if refused else [{"type": "text", "text": "ok"}],
            "stop_reason": "refusal" if refused else "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 0 if refused else 50},
        }
        if refused:
            message["stop_details"] = {"type": "refusal", "category": "cyber", "explanation": "x"}
        if body.get("stream"):
            return httpx2.Response(
                200,
                request=request,
                text=_anthropic_sse(message),
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(200, request=request, json=message)

    async def async_handler(self, request: Any) -> Any:
        if not self.slow:
            return self.handler(request)
        import asyncio

        body = json.loads(request.content)
        self.bodies.append(body)
        await asyncio.sleep(0.05)
        return self._respond(request, body)

    def client(self, *, is_async: bool = False, middleware: Any = None) -> Any:
        anthropic = pytest.importorskip("anthropic")
        httpx2 = pytest.importorskip("httpx2")
        options: dict[str, Any] = {"api_key": "k", "max_retries": 0}
        if middleware is not None:
            options["middleware"] = middleware
        if is_async:
            return anthropic.AsyncAnthropic(
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.async_handler)),
                **options,
            )
        return anthropic.Anthropic(
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handler)), **options
        )


_FALLBACK = [{"model": "claude-opus-4-8", "max_tokens": 9000}]


@pytest.mark.unit
class TestAnthropicGenerationLoops:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_server_side_fallbacks_are_refused(self, is_async: bool, streaming: bool) -> None:
        wire, plane = _AnthropicWire(), FakeControlPlane()
        wrapper = (plane.wrap_async if is_async else plane.wrap)(
            wire.client(is_async=is_async), lease_enabled=False
        )
        try:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                result = wrapper.beta.messages.create(
                    model="claude-fable-5",
                    messages=MESSAGES,
                    max_tokens=100,
                    stream=streaming,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks=_FALLBACK,
                )
                if is_async:
                    await result
        finally:
            if is_async:
                await wrapper.close()
            else:
                wrapper.close()

        _refused(caught, "multi_generation_loop")
        assert wire.bodies == []

    def test_fallbacks_in_extra_body_name_the_loop(self) -> None:
        wire, plane = _AnthropicWire(), FakeControlPlane()
        with (
            plane.wrap(wire.client(), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.messages.create(
                model="claude-fable-5",
                messages=MESSAGES,
                max_tokens=100,
                extra_body={"fallbacks": _FALLBACK},
            )

        _refused(caught, "multi_generation_loop")
        assert wire.bodies == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    async def test_native_fallback_middleware_is_refused_before_any_request(
        self, is_async: bool
    ) -> None:
        # The middleware pins the fallback in a shared BetaFallbackState, so
        # even a later call's FIRST request could be the 9000-token fallback.
        anthropic = pytest.importorskip("anthropic")
        wire, plane = _AnthropicWire(refuse_first=True), FakeControlPlane()
        middleware = [anthropic.BetaRefusalFallbackMiddleware(_FALLBACK)]
        wrapper = (plane.wrap_async if is_async else plane.wrap)(
            wire.client(is_async=is_async, middleware=middleware), lease_enabled=False
        )
        try:
            with anthropic.BetaFallbackState():
                for _ in range(2):
                    with pytest.raises(OutputBoundUnenforceableError) as caught:
                        result = wrapper.beta.messages.create(
                            model="claude-fable-5", messages=MESSAGES, max_tokens=100
                        )
                        if is_async:
                            await result
                    _refused(caught, "multi_generation_loop")
        finally:
            if is_async:
                await wrapper.close()
            else:
                wrapper.close()

        assert wire.bodies == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize(
        ("rewrite", "reason"),
        [
            ({"model": "claude-opus-4-8"}, "unsupported_override"),
            ({"max_tokens": 9000}, "bound_exceeded"),
            ({"fallbacks": _FALLBACK}, "multi_generation_loop"),
        ],
    )
    async def test_first_request_rewritten_by_middleware_is_refused(
        self, is_async: bool, rewrite: dict[str, Any], reason: str
    ) -> None:
        wire, plane = _AnthropicWire(), FakeControlPlane()

        def rewritten(request: Any) -> Any:
            return request.copy(body={**request.json, **rewrite})

        def sync_rewriter(request: Any, call_next: Any) -> Any:
            return call_next(rewritten(request))

        async def async_rewriter(request: Any, call_next: Any) -> Any:
            return await call_next(rewritten(request))

        middleware = [async_rewriter if is_async else sync_rewriter]
        wrapper = (plane.wrap_async if is_async else plane.wrap)(
            wire.client(is_async=is_async, middleware=middleware), lease_enabled=False
        )
        try:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                result = wrapper.messages.create(
                    model="claude-fable-5", messages=MESSAGES, max_tokens=100
                )
                if is_async:
                    await result
        finally:
            if is_async:
                await wrapper.close()
            else:
                wrapper.close()

        _refused(caught, reason)
        assert wire.bodies == []

    @pytest.mark.asyncio
    async def test_concurrent_async_sends_cannot_both_generate(self) -> None:
        import asyncio

        wire, plane = _AnthropicWire(slow=True), FakeControlPlane()

        async def hedge(request: Any, call_next: Any) -> Any:
            first, _second = await asyncio.gather(call_next(request), call_next(request))
            return first

        async with plane.wrap_async(
            wire.client(is_async=True, middleware=[hedge]), lease_enabled=False
        ) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.messages.create(
                    model="claude-fable-5", messages=MESSAGES, max_tokens=100
                )

        _refused(caught, "multi_generation_loop")
        assert len(wire.bodies) == 1

    def test_concurrent_sync_sends_cannot_both_generate(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        wire, plane = _AnthropicWire(slow=True), FakeControlPlane()

        def hedge(request: Any, call_next: Any) -> Any:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(call_next, request) for _ in range(2)]
                results = [future.result() for future in futures]
            return results[0]

        with (
            plane.wrap(wire.client(middleware=[hedge]), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.messages.create(model="claude-fable-5", messages=MESSAGES, max_tokens=100)

        _refused(caught, "multi_generation_loop")
        assert len(wire.bodies) == 1

    def test_retry_after_a_failed_attempt_still_passes(self) -> None:
        wire, plane = _AnthropicWire(fail_first=True), FakeControlPlane()

        def retry_once(request: Any, call_next: Any) -> Any:
            response = call_next(request)
            return call_next(request) if response.status_code >= 500 else response

        with plane.wrap(wire.client(middleware=[retry_once]), lease_enabled=False) as solwyn:
            solwyn.messages.create(model="claude-fable-5", messages=MESSAGES, max_tokens=100)

        assert len(wire.bodies) == 2

    def test_alert_only_keeps_the_middleware_loop(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        wire, plane = _AnthropicWire(refuse_first=True), FakeControlPlane(mode="alert_only")
        middleware = [anthropic.BetaRefusalFallbackMiddleware(_FALLBACK)]
        with (
            plane.wrap(wire.client(middleware=middleware), lease_enabled=False) as solwyn,
            anthropic.BetaFallbackState(),
        ):
            solwyn.beta.messages.create(model="claude-fable-5", messages=MESSAGES, max_tokens=100)

        assert [body["max_tokens"] for body in wire.bodies] == [100, 9000]


@pytest.mark.unit
class TestCurrentProviderHosts:
    @pytest.mark.parametrize("region", ["ap-southeast-1", "cn-beijing", "us-east-1"])
    def test_qwen_workspace_host_is_qwen(self, region: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        base_url = f"https://ws-123.{region}.maas.aliyuncs.com/compatible-mode/v1"
        with (
            plane.wrap(_openai(wire, base_url), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="qwen3.8-max", messages=MESSAGES, max_tokens=100)

        _refused(caught, "reasoning_without_budget")
        assert caught.value.provider == "qwen"
        assert wire.bodies == []

    @pytest.mark.parametrize("host", ["eu-west-1.api.x.ai", "us.api.x.ai"])
    def test_xai_regional_host_is_xai(self, host: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire, f"https://{host}/v1"), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="grok-4", messages=MESSAGES, max_tokens=100)

        _refused(caught, "reasoning_without_budget")
        assert caught.value.provider == "xai"


UNKNOWN_URL = "https://llm.internal.example/v1"


@pytest.mark.unit
class TestModelFamilyReasoning:
    """Under hard_deny an adapter with no reasoning rule classifies the model id."""

    @pytest.mark.parametrize(
        "host",
        [
            "cn-hongkong.dashscope.aliyuncs.com",
            "coding.dashscope.aliyuncs.com",
            "trial.cn-hongkong.maas.aliyuncs.com",
        ],
    )
    def test_documented_alibaba_hosts_are_qwen(self, host: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(
                _openai(wire, f"https://{host}/compatible-mode/v1"), lease_enabled=False
            ) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model="qwen3.8-max", messages=MESSAGES, max_tokens=100)

        _refused(caught, "reasoning_without_budget")
        assert caught.value.provider == "qwen"
        assert wire.bodies == []

    @pytest.mark.parametrize(
        ("base_url", "model", "provider"),
        [
            (UNKNOWN_URL, "qwen3.8-max", "openai_compatible"),
            (UNKNOWN_URL, "QwQ-32B", "openai_compatible"),
            ("https://dashscope.aliyuncs.com.evil.example/v1", "qwen3.8-max", "openai_compatible"),
            ("https://api.together.xyz/v1", "Qwen/Qwen3-235B-A22B", "together"),
            (UNKNOWN_URL, "grok-4", "openai_compatible"),
            ("https://openrouter.ai/api/v1", "x-ai/grok-4", "openrouter"),
        ],
    )
    def test_family_model_on_a_host_without_a_rule_is_refused(
        self, base_url: str, model: str, provider: str
    ) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with (
            plane.wrap(_openai(wire, base_url), lease_enabled=False) as solwyn,
            pytest.raises(OutputBoundUnenforceableError) as caught,
        ):
            solwyn.chat.completions.create(model=model, messages=MESSAGES, max_tokens=100)

        _refused(caught, "reasoning_without_budget")
        assert caught.value.provider == provider
        assert "set the provider explicitly" in caught.value.hint
        assert wire.bodies == []

    def test_qwen_model_on_unknown_host_with_a_budget_is_priced_and_sent(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, UNKNOWN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model="qwen3.8-max",
                messages=MESSAGES,
                max_tokens=100,
                extra_body={"thinking_budget": 2000},
            )

        assert _bound(plane) == 2100
        assert wire.bodies[0]["thinking_budget"] == 2000

    @pytest.mark.parametrize("model", ["qwen3.8-max", "grok-4"])
    def test_explicit_pin_opts_out_of_model_classification(self, model: str) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(
            _openai(wire, UNKNOWN_URL), lease_enabled=False, provider="openai_compatible"
        ) as solwyn:
            solwyn.chat.completions.create(model=model, messages=MESSAGES)

        assert _bound(plane) == DEFAULT_OUTPUT_BOUND
        assert wire.bodies[0]["max_tokens"] == DEFAULT_OUTPUT_BOUND

    @pytest.mark.parametrize(
        ("model", "extra"),
        [
            ("qwen3.8-max", {"enable_thinking": False}),
            ("x-ai/grok-4-fast-non-reasoning", {}),
            ("llama-3.3-70b", {}),
        ],
    )
    def test_bounded_or_non_family_models_are_sent(self, model: str, extra: dict[str, Any]) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        with plane.wrap(_openai(wire, UNKNOWN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(
                model=model, messages=MESSAGES, max_tokens=100, extra_body=extra or None
            )

        assert _bound(plane) == 100
        assert wire.bodies[0]["max_tokens"] == 100

    def test_alert_only_sends_a_qwen_model_as_built(self) -> None:
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane(mode="alert_only")
        with plane.wrap(_openai(wire, UNKNOWN_URL), lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="qwen3.8-max", messages=MESSAGES)

        assert "max_tokens" not in wire.bodies[0]

    @pytest.mark.asyncio
    async def test_async_qwen_model_on_unknown_host_is_refused(self) -> None:
        openai = pytest.importorskip("openai")
        wire, plane = _Wire(_CHAT_BODY), FakeControlPlane()
        client = openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire.async_handler)),
            max_retries=0,
        )
        async with plane.wrap_async(client, lease_enabled=False) as solwyn:
            with pytest.raises(OutputBoundUnenforceableError) as caught:
                await solwyn.chat.completions.create(
                    model="qwen3.8-max", messages=MESSAGES, max_tokens=100
                )

        _refused(caught, "reasoning_without_budget")
        assert wire.bodies == []
