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
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(wire.async_handler)
                )
            ),
        )
        plane = FakeControlPlane()
        solwyn = AsyncSolwyn(
            client,
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
