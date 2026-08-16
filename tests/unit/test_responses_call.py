"""Public Responses proxies and their shared interception pipeline tests.

The public sync/async suites prove native and capability-gated compatible proxy
routing plus guarded raw leaves. The lower suites drive
``_surface="responses"`` directly for detailed dispatch, budgeting, routing,
streaming, and settlement coverage. A focused adjacent regression also
protects the default chat routing.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID
from openai import NOT_GIVEN, omit

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._proxies import _AsyncResponsesProxy, _SyncResponsesProxy
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError, ProviderUnavailableError, UnsupportedSurfaceError
from solwyn.providers.openai_compatible import COMPAT_PROFILES


@pytest.fixture(autouse=True)
def _reset_unmetered_warning_state() -> Iterator[None]:
    _reset_unmetered_spend_warnings()
    yield
    _reset_unmetered_spend_warnings()


class OpenAI:
    """Minimal native OpenAI client spec used by provider detection."""

    __module__ = "openai._client"
    chat: object
    responses: object

    def with_options(self, **kwargs: object) -> OpenAI: ...


class AsyncOpenAI:
    """Minimal native async OpenAI client spec used by provider detection."""

    __module__ = "openai._client"
    chat: object
    responses: object

    def with_options(self, **kwargs: object) -> AsyncOpenAI: ...


class Anthropic:
    """Minimal native Anthropic client spec used by provider detection."""

    __module__ = "anthropic._client"
    messages: object

    def with_options(self, **kwargs: object) -> Anthropic: ...


class AsyncAnthropic:
    """Minimal native async Anthropic client spec used by provider detection."""

    __module__ = "anthropic._client"
    messages: object

    def with_options(self, **kwargs: object) -> AsyncAnthropic: ...


def _sync_call(**kwargs: object) -> object:
    raise NotImplementedError


async def _async_call(**kwargs: object) -> object:
    raise NotImplementedError


class _SyncResponsesResource:
    """Realistic OpenAI Responses resource with observable provider calls."""

    __module__ = "openai.resources.responses.responses"

    def __init__(self, response: object) -> None:
        self.response = response
        self.create_calls: list[dict[str, object]] = []
        self.parse_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.retrieve_calls: list[str] = []

    def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return self.response

    def parse(self, **kwargs: object) -> object:
        self.parse_calls.append(kwargs)
        return self.response

    def stream(self, **kwargs: object) -> object:
        self.stream_calls.append(kwargs)
        order = getattr(self.response, "order", None)
        if order is not None:
            order.append("manager.create")
        return self.response

    def retrieve(self, response_id: str) -> object:
        self.retrieve_calls.append(response_id)
        return self.response


class _AsyncResponsesResource:
    """Async counterpart of ``_SyncResponsesResource``."""

    __module__ = "openai.resources.responses.responses"

    def __init__(self, response: object) -> None:
        self.response = response
        self.create_calls: list[dict[str, object]] = []
        self.parse_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.retrieve_calls: list[str] = []

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return self.response

    async def parse(self, **kwargs: object) -> object:
        self.parse_calls.append(kwargs)
        return self.response

    def stream(self, **kwargs: object) -> object:
        """Match OpenAI: async ``responses.stream`` returns a manager directly."""
        self.stream_calls.append(kwargs)
        order = getattr(self.response, "order", None)
        if order is not None:
            order.append("manager.create")
        return self.response

    async def retrieve(self, response_id: str) -> object:
        self.retrieve_calls.append(response_id)
        return self.response


class _NativeOpenAIClient:
    """Minimal native OpenAI shell whose namespace shape matches the SDK."""

    __module__ = "openai._client"

    def __init__(self, response: object) -> None:
        self._responses = _SyncResponsesResource(response)

    @functools.cached_property
    def responses(self) -> _SyncResponsesResource:
        return self._responses

    def with_options(self, **_kwargs: object) -> _NativeOpenAIClient:
        return self


class _AsyncNativeOpenAIClient:
    """Minimal native async OpenAI shell whose namespace shape matches the SDK."""

    __module__ = "openai._client"

    def __init__(self, response: object) -> None:
        self._responses = _AsyncResponsesResource(response)

    @functools.cached_property
    def responses(self) -> _AsyncResponsesResource:
        return self._responses

    def with_options(self, **_kwargs: object) -> _AsyncNativeOpenAIClient:
        return self


class AzureOpenAI(_NativeOpenAIClient):
    """Azure SDK-shaped sync client detected by class name and endpoint."""

    __module__ = "openai._client"
    base_url = "https://solwyn-test.openai.azure.com/openai"


class AsyncAzureOpenAI(_AsyncNativeOpenAIClient):
    """Azure SDK-shaped async client detected by class name and endpoint."""

    __module__ = "openai._client"
    base_url = "https://solwyn-test.cognitiveservices.azure.com/openai"


class _GroqClient(_NativeOpenAIClient):
    """OpenAI SDK shape targeting Groq through ``base_url`` detection."""

    __module__ = "openai._client"
    base_url = "https://api.groq.com/openai/v1"


class _AsyncGroqClient(_AsyncNativeOpenAIClient):
    """Async OpenAI SDK shape targeting Groq through ``base_url`` detection."""

    __module__ = "openai._client"
    base_url = "https://api.groq.com/openai/v1"


_COMPAT_PROFILE_URLS = {
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zai": "https://api.z.ai/api/paas/v4",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "perplexity": "https://api.perplexity.ai",
    "azure_openai": "https://solwyn-test.openai.azure.com/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai_compatible": "https://api.future-provider.example/v1",
}

_RESPONSES_CRITICAL_EXTRA_BODY_KEYS = (
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "stream",
)
_RESPONSES_SPEND_OPERATIONS = ("create", "parse", "create_stream", "stream_helper")


def _profile_client(profile_name: str, response: object, *, async_mode: bool = False) -> object:
    base = _AsyncNativeOpenAIClient if async_mode else _NativeOpenAIClient
    if profile_name == "azure_openai":
        class_name = "AsyncAzureOpenAI" if async_mode else "AzureOpenAI"
    else:
        class_name = "AsyncOpenAI" if async_mode else "OpenAI"
    client_type = type(
        class_name,
        (base,),
        {
            "__module__": "openai._client",
            "base_url": _COMPAT_PROFILE_URLS[profile_name],
        },
    )
    return client_type(response)


def _responses_usage(
    *,
    input_tokens: int = 120,
    output_tokens: int = 45,
    cached_tokens: int = 10,
    reasoning_tokens: int = 5,
) -> SimpleNamespace:
    """Mirror the complete Responses token-usage shape consumed by Solwyn."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=3,
            audio_tokens=2,
        ),
        output_tokens_details=SimpleNamespace(
            reasoning_tokens=reasoning_tokens,
            audio_tokens=1,
            accepted_prediction_tokens=4,
            rejected_prediction_tokens=6,
        ),
    )


def _responses_response(
    *, service_tier: str = "priority", include_usage: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        object="response",
        status="completed",
        model="gpt-5.5",
        output=[],
        usage=_responses_usage() if include_usage else None,
        service_tier=service_tier,
    )


def _empty_responses_usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        input_tokens_details=None,
        output_tokens_details=None,
    )


def _chat_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl_1",
        object="chat.completion",
        model="gpt-5.5",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=9,
            total_tokens=30,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        service_tier="default",
    )


def _mock_openai_client() -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock(spec=OpenAI)
    type(client).__module__ = "openai._client"
    type(client).__name__ = "OpenAI"
    response = _responses_response()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=MagicMock(spec=_sync_call)))
    client.responses = SimpleNamespace(
        create=MagicMock(spec=_sync_call, return_value=response),
        parse=MagicMock(spec=_sync_call, return_value=response),
        stream=MagicMock(spec=_sync_call),
    )
    client.with_options.return_value = client
    return client, response


def _mock_async_openai_client() -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock(spec=AsyncOpenAI)
    type(client).__module__ = "openai._client"
    type(client).__name__ = "AsyncOpenAI"
    response = _responses_response()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(spec=_async_call)))
    client.responses = SimpleNamespace(
        create=AsyncMock(spec=_async_call, return_value=response),
        parse=AsyncMock(spec=_async_call, return_value=response),
        stream=MagicMock(spec=_sync_call),
    )
    client.with_options.return_value = client
    return client, response


def _mock_anthropic_client() -> MagicMock:
    client = MagicMock(spec=Anthropic)
    type(client).__module__ = "anthropic._client"
    type(client).__name__ = "Anthropic"
    client.messages = SimpleNamespace(create=MagicMock(spec=_sync_call))
    client.with_options.return_value = client
    return client


def _mock_async_anthropic_client() -> MagicMock:
    client = MagicMock(spec=AsyncAnthropic)
    type(client).__module__ = "anthropic._client"
    type(client).__name__ = "AsyncAnthropic"
    client.messages = SimpleNamespace(create=AsyncMock(spec=_async_call))
    client.with_options.return_value = client
    return client


def _allow_budget() -> SimpleNamespace:
    """Complete allowed reservation result used by the interception lifecycle."""
    return SimpleNamespace(
        allowed=True,
        remaining_budget=100.0,
        reservation_id="res_1",
        lease_id=None,
        lease_claim_token=None,
        project_id=VALID_PROJECT_ID,
        mode=SimpleNamespace(value="alert_only"),
        budget_limit=100.0,
        current_usage=0.0,
        denied_by_period=None,
        price_hints=None,
        failover_directive=None,
        failover_tuning_allowed=None,
    )


def _allow_lease() -> SimpleNamespace:
    result = _allow_budget()
    result.reservation_id = None
    result.lease_id = "lease_1"
    result.lease_claim_token = 7
    return result


@contextmanager
def _sync_solwyn(client: object, **overrides: object) -> Iterator[Solwyn]:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY, "model": "gpt-5.5"}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter._shutdown.set()
    solwyn._solwyn_reporter._thread.join(timeout=2.0)
    solwyn._solwyn_reporter.report = MagicMock(spec=solwyn._solwyn_reporter.report)
    solwyn._solwyn_reporter.report_settlement = MagicMock(
        spec=solwyn._solwyn_reporter.report_settlement
    )
    try:
        yield solwyn
    finally:
        solwyn._solwyn_reporter._http.close()
        solwyn._solwyn_budget._http.close()


@asynccontextmanager
async def _async_solwyn(client: object, **overrides: object) -> AsyncIterator[AsyncSolwyn]:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY, "model": "gpt-5.5"}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._solwyn_reporter.report = MagicMock(spec=solwyn._solwyn_reporter.report)
    solwyn._solwyn_reporter.report_settlement = MagicMock(
        spec=solwyn._solwyn_reporter.report_settlement
    )
    try:
        yield solwyn
    finally:
        await solwyn._solwyn_reporter._http.aclose()
        await solwyn._solwyn_budget._http.aclose()


class _Status(Exception):
    """Duck-typed provider error that the routing classifier may fail over."""

    def __init__(self, status_code: int, message: str = "provider unavailable") -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeSyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.close_calls = 0

    def __iter__(self) -> Iterator[object]:
        yield from self._events

    def close(self) -> None:
        self.close_calls += 1


class _FakeAsyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.aclose_calls = 0

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _FakeSyncResponseStream(_FakeSyncStream):
    """SDK inner stream shape, including helpers that consume it directly."""

    def __init__(self, events: list[object], final_response: object) -> None:
        super().__init__(events)
        self._iterator = iter(events)
        self.final_response = final_response
        self.get_final_response_calls = 0

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        return next(self._iterator)

    def get_final_response(self) -> object:
        self.get_final_response_calls += 1
        list(self)
        return self.final_response


class _FakeAsyncResponseStream(_FakeAsyncStream):
    """Async SDK inner stream shape with its coroutine helper."""

    def __init__(self, events: list[object], final_response: object) -> None:
        super().__init__(events)
        self._iterator = self._iterate(events)
        self.final_response = final_response
        self.get_final_response_calls = 0

    @staticmethod
    async def _iterate(events: list[object]) -> AsyncIterator[object]:
        for event in events:
            yield event

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        return await anext(self._iterator)

    async def get_final_response(self) -> object:
        self.get_final_response_calls += 1
        async for _event in self:
            pass
        return self.final_response


class _FakeSyncResponseStreamManager:
    """OpenAI ResponseStreamManager lifecycle without importing the SDK."""

    def __init__(
        self,
        stream: _FakeSyncResponseStream,
        *,
        order: list[str] | None = None,
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.stream = stream
        self.order = order
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.enter_calls = 0
        self.exit_calls: list[tuple[object, ...]] = []
        self._entered = False

    def __enter__(self) -> _FakeSyncResponseStream:
        self.enter_calls += 1
        if self.order is not None:
            self.order.append("manager.open")
        if self.enter_error is not None:
            raise self.enter_error
        self._entered = True
        return self.stream

    def __exit__(self, *args: object) -> None:
        self.exit_calls.append(args)
        if self._entered:
            self.stream.close()
        if self.exit_error is not None:
            raise self.exit_error


class _FakeAsyncResponseStreamManager:
    """OpenAI AsyncResponseStreamManager lifecycle without importing the SDK."""

    def __init__(
        self,
        stream: _FakeAsyncResponseStream,
        *,
        order: list[str] | None = None,
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.stream = stream
        self.order = order
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.enter_calls = 0
        self.exit_calls: list[tuple[object, ...]] = []
        self._entered = False

    async def __aenter__(self) -> _FakeAsyncResponseStream:
        self.enter_calls += 1
        if self.order is not None:
            self.order.append("manager.open")
        if self.enter_error is not None:
            raise self.enter_error
        self._entered = True
        return self.stream

    async def __aexit__(self, *args: object) -> None:
        self.exit_calls.append(args)
        if self._entered:
            await self.stream.aclose()
        if self.exit_error is not None:
            raise self.exit_error


def _terminal_event(*, service_tier: str = "flex") -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id="resp_stream_1",
            object="response",
            status="completed",
            model="gpt-5.5",
            output=[],
            usage=_responses_usage(
                input_tokens=30,
                output_tokens=12,
                cached_tokens=7,
                reasoning_tokens=4,
            ),
            service_tier=service_tier,
        ),
    )


@pytest.mark.unit
class TestResponsesPublicProxySync:
    @pytest.mark.parametrize("client_type", [_NativeOpenAIClient, AzureOpenAI])
    @pytest.mark.parametrize("operation", _RESPONSES_SPEND_OPERATIONS)
    @pytest.mark.parametrize("override_key", _RESPONSES_CRITICAL_EXTRA_BODY_KEYS)
    @pytest.mark.parametrize("source", ["caller", "default"])
    def test_metering_critical_extra_body_override_is_refused_before_budget_or_dispatch(
        self,
        client_type: type[_NativeOpenAIClient],
        operation: str,
        override_key: str,
        source: str,
    ) -> None:
        response = _responses_response()
        client = client_type(response)
        extra_body = {override_key: object()}
        defaults = {"extra_body": extra_body} if source == "default" else None
        call_kwargs: dict[str, object] = {
            "model": "gpt-5.5",
            "input": "hello",
        }
        if source == "caller":
            call_kwargs["extra_body"] = extra_body

        with _sync_solwyn(client, default_params=defaults) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                if operation == "parse":
                    solwyn.responses.parse(**call_kwargs, text_format=dict)
                elif operation == "create_stream":
                    solwyn.responses.create(**call_kwargs, stream=True)
                elif operation == "stream_helper":
                    solwyn.responses.stream(**call_kwargs)
                else:
                    solwyn.responses.create(**call_kwargs)

            if operation == "parse" and override_key == "stream":
                # Parse resolves effective streaming (extra_body included) and
                # refuses it before the override check ever runs.
                assert exc_info.value.field == "stream"
            else:
                assert exc_info.value.field == "extra_body"
                assert override_key in str(exc_info.value)
            check.assert_not_called()
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == []
            assert client._responses.stream_calls == []

    @pytest.mark.parametrize("client_type", [_NativeOpenAIClient, AzureOpenAI])
    def test_vendor_extra_body_passes_through_unchanged(
        self, client_type: type[_NativeOpenAIClient]
    ) -> None:
        response = _responses_response()
        client = client_type(response)
        extra_body = {"vendor_extension": {"mode": "safe"}}

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    extra_body=extra_body,
                )

            assert result is response
            check.assert_called_once()
            assert client._responses.create_calls == [
                {"model": "gpt-5.5", "input": "hello", "extra_body": extra_body}
            ]

    @pytest.mark.parametrize("leaf", ["create", "parse"])
    def test_pinned_native_nonstream_without_usage_estimates_and_holds_lease_floor(
        self, leaf: str
    ) -> None:
        response = _responses_response(include_usage=False)
        client = _profile_client("openai_compatible", response)
        client.base_url = "http://localhost:9999/v1"  # type: ignore[attr-defined]

        with (
            patch(
                "solwyn._registry.get_adapter_for_client",
                side_effect=AssertionError("provider detection must not run"),
            ) as detector,
            _sync_solwyn(client, provider="openai") as solwyn,
        ):
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            kwargs: dict[str, object] = {
                "model": "gpt-5.5",
                "input": "12345678",
                "max_output_tokens": 50,
            }
            if leaf == "parse":
                kwargs["text_format"] = dict
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                result = getattr(solwyn.responses, leaf)(**kwargs)

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert result is response
            assert event.provider.value == "openai"
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

        detector.assert_not_called()

    def test_openai_pin_on_local_gateway_bypasses_detection_and_meters_responses(self) -> None:
        response = _responses_response(service_tier="priority")
        client = _profile_client("openai_compatible", response)
        client.base_url = "http://localhost:9999/v1"  # type: ignore[attr-defined]

        with (
            patch(
                "solwyn._registry.get_adapter_for_client",
                side_effect=AssertionError("provider detection must not run"),
            ) as detect,
            _sync_solwyn(client, provider="openai") as solwyn,
        ):
            assert isinstance(solwyn.responses, _SyncResponsesProxy)
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.create(model="gpt-5.5", input="hello")

            assert result is response
            detect.assert_not_called()
            assert solwyn._solwyn_adapter.name == "openai"
            assert check.call_args.kwargs["provider"] == "openai"
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "openai"

    def test_azure_create_uses_responses_defaults_and_azure_attribution(self) -> None:
        response = _responses_response(service_tier="priority")
        client = AzureOpenAI(response)

        with _sync_solwyn(
            client,
            model="deployment",
            default_params={
                "instructions": "default instruction",
                "temperature": 0.2,
                "max_output_tokens": 321,
                "max_tokens": 999,
            },
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.create(
                    model="deployment",
                    input="hello",
                    temperature=0.7,
                )

            assert result is response
            check.assert_called_once()
            assert check.call_args.kwargs["provider"] == "azure_openai"
            assert check.call_args.kwargs["fallback_providers"] == []
            assert check.call_args.kwargs["estimated_output_bound"] == 321
            assert client._responses.create_calls == [
                {
                    "model": "deployment",
                    "input": "hello",
                    "instructions": "default instruction",
                    "temperature": 0.7,
                    "max_output_tokens": 321,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.service_tier == "priority"
            assert confirm.token_details == event.token_details

    def test_azure_parse_dispatches_to_parse_and_settles_once(self) -> None:
        response = _responses_response()
        client = AzureOpenAI(response)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.parse(
                    model="deployment",
                    input="hello",
                    text_format=dict,
                )

            assert result is response
            check.assert_called_once()
            assert check.call_args.kwargs["provider"] == "azure_openai"
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == [
                {"model": "deployment", "input": "hello", "text_format": dict}
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.parametrize("usage_state", ["missing", "zero"])
    @pytest.mark.parametrize("leaf", ["create", "parse"])
    def test_azure_nonstream_empty_usage_estimates_and_holds_lease_floor(
        self, usage_state: str, leaf: str
    ) -> None:
        response = _responses_response(include_usage=usage_state == "zero")
        if usage_state == "zero":
            response.usage = _empty_responses_usage()
        client = AzureOpenAI(response)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            kwargs: dict[str, object] = {
                "model": "deployment",
                "input": "12345678",
                "max_output_tokens": 50,
            }
            if leaf == "parse":
                kwargs["text_format"] = dict
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                result = getattr(solwyn.responses, leaf)(**kwargs)

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert result is response
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    def test_azure_nonstream_measured_usage_trues_up_lease_without_floor(self) -> None:
        response = _responses_response()
        client = AzureOpenAI(response)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                result = solwyn.responses.create(model="deployment", input="12345678")

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                165,
                claim_token=7,
                floor_at_reservation=False,
            )
            assert result is response
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.token_details.is_estimated is False

    def test_azure_create_stream_settles_terminal_provider_usage(self) -> None:
        terminal = _terminal_event(service_tier="priority")
        provider_stream = _FakeSyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="hello"), terminal]
        )
        client = AzureOpenAI(provider_stream)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = solwyn.responses.create(
                    model="deployment",
                    input="hello",
                    stream=True,
                )
                assert list(stream)[-1] is terminal

            check.assert_called_once()
            assert client._responses.create_calls == [
                {"model": "deployment", "input": "hello", "stream": True}
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "priority"
            assert event.token_details.is_estimated is False

    def test_azure_stream_without_usage_estimates_and_holds_lease_floor(self) -> None:
        delta = SimpleNamespace(type="response.output_text.delta", delta="hello")
        provider_stream = _FakeSyncStream([delta])
        client = AzureOpenAI(provider_stream)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                stream = solwyn.responses.create(
                    model="deployment",
                    input="12345678",
                    stream=True,
                    max_output_tokens=50,
                )
                assert list(stream) == [delta]

            call_id = check.call_args.kwargs["call_id"]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            # The accumulator's own length estimate is kept as telemetry, but it
            # cannot account for all output, so the lease holds its bound.
            assert event.token_details.is_estimated is True
            assert event.input_tokens == 2
            assert confirm.token_details == event.token_details
            true_up.assert_called_once_with(
                call_id,
                event.token_details.total_tokens,
                claim_token=7,
                floor_at_reservation=True,
            )

    def test_azure_stream_helper_settles_terminal_usage_once(self) -> None:
        terminal = _terminal_event(service_tier="flex")
        inner = _FakeSyncResponseStream([terminal], terminal.response)
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = AzureOpenAI(provider_manager)

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                solwyn.responses.stream(model="deployment", input="hello") as events,
            ):
                assert list(events) == [terminal]
                assert events.get_final_response() is terminal.response

            check.assert_called_once()
            assert client._responses.stream_calls == [{"model": "deployment", "input": "hello"}]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "flex"

    def test_azure_existing_response_stream_keeps_raw_manager_identity(self) -> None:
        provider_manager = object()
        client = AzureOpenAI(provider_manager)

        with _sync_solwyn(
            client,
            model="deployment",
            default_params={"instructions": "must-not-leak"},
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.stream(response_id="resp_123", starting_after=7)

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"response_id": "resp_123", "starting_after": 7}
            ]
            check.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()

    def test_azure_background_refusal_happens_before_budget_or_provider(self) -> None:
        client = AzureOpenAI(_responses_response())

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.create(
                    model="deployment",
                    input="hello",
                    background=True,
                )

            assert exc_info.value.field == "background"
            check.assert_not_called()
            assert client._responses.create_calls == []

    def test_azure_provider_error_releases_reservation_without_fallback(self) -> None:
        client = AzureOpenAI(_responses_response())
        error = _Status(429)
        client._responses.create = MagicMock(side_effect=error)  # type: ignore[method-assign]

        with _sync_solwyn(client, model="deployment") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("azure_openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
            ):
                solwyn.responses.create(model="deployment", input="hello")

            assert exc_info.value is error
            check.assert_called_once()
            client._responses.create.assert_called_once()
            release.assert_called_once()
            breaker_failure.assert_called_once_with()

    def test_native_create_intercepts_once_dispatches_and_settles(self) -> None:
        response = _responses_response()
        client = _NativeOpenAIClient(response)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.create(model="gpt-5.5", input="hello")

            assert result is response
            check.assert_called_once()
            assert client._responses.create_calls == [{"model": "gpt-5.5", "input": "hello"}]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    def test_native_parse_reuses_defaults_budgeting_dispatch_and_flat_settlement(self) -> None:
        # Arrange.
        response = _responses_response()
        client = _NativeOpenAIClient(response)

        class ParsedAnswer:
            value: str

        # Act.
        with _sync_solwyn(
            client,
            default_params={"instructions": "12345678", "max_output_tokens": 321},
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.parse(
                    model="gpt-5.5",
                    input="abcd",
                    text_format=ParsedAnswer,
                    temperature=0.4,
                )

            # Assert.
            assert result is response
            check.assert_called_once()
            assert check.call_args.kwargs["estimated_input_tokens"] == 3
            assert check.call_args.kwargs["estimated_output_bound"] == 321
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "abcd",
                    "text_format": ParsedAnswer,
                    "temperature": 0.4,
                    "instructions": "12345678",
                    "max_output_tokens": 321,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.service_tier == "priority"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    @pytest.mark.parametrize(
        ("default_params", "call_kwargs"),
        [
            ({}, {"stream": True}),
            ({"stream": True}, {}),
            ({"stream": False}, {"extra_body": {"stream": True}}),
        ],
    )
    def test_native_parse_refuses_effective_streaming_before_budget_or_provider(
        self,
        default_params: dict[str, object],
        call_kwargs: dict[str, object],
    ) -> None:
        # Arrange.
        client = _NativeOpenAIClient(_responses_response())

        # Act.
        with _sync_solwyn(client, default_params=default_params) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.parse(
                    model="gpt-5.5",
                    input="hello",
                    text_format=dict,
                    **call_kwargs,
                )

            # Assert.
            assert exc_info.value.field == "stream"
            check.assert_not_called()
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == []

    def test_extra_body_stream_false_is_still_refused_for_parse(self) -> None:
        # Arrange.
        response = _responses_response()
        client = _NativeOpenAIClient(response)

        # Act.
        with _sync_solwyn(client, default_params={"stream": True}) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.parse(
                    model="gpt-5.5",
                    input="hello",
                    text_format=dict,
                    extra_body={"stream": False},
                )

            # Assert.
            assert exc_info.value.field == "extra_body"
            assert "stream" in str(exc_info.value)
            check.assert_not_called()
            assert client._responses.parse_calls == []

    def test_native_stream_helper_opens_after_one_budget_check_and_settles_terminal_usage(
        self,
    ) -> None:
        order: list[str] = []
        delta = SimpleNamespace(type="response.output_text.delta", delta="hello")
        terminal = _terminal_event()
        inner = _FakeSyncResponseStream([delta, terminal], terminal.response)
        provider_manager = _FakeSyncResponseStreamManager(inner, order=order)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(
            client,
            default_params={"instructions": "12345678", "max_output_tokens": 321},
        ) as solwyn:
            check = MagicMock(
                spec=solwyn._solwyn_budget.check_budget,
                side_effect=lambda **_kwargs: (order.append("budget.check"), _allow_budget())[1],
            )
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    model="gpt-5.5",
                    input="abcd",
                    temperature=0.4,
                )
                assert order == ["budget.check", "manager.create"]
                solwyn._solwyn_reporter.report_settlement.assert_not_called()

                with manager as events:
                    assert list(events) == [delta, terminal]

            assert order == ["budget.check", "manager.create", "manager.open"]
            check.assert_called_once()
            assert client._responses.stream_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "abcd",
                    "temperature": 0.4,
                    "instructions": "12345678",
                    "max_output_tokens": 321,
                }
            ]
            assert provider_manager.enter_calls == 1
            assert len(provider_manager.exit_calls) == 1
            assert inner.close_calls == 1
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "flex"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == "flex"

    def test_native_existing_response_stream_is_reviewed_raw_retrieval(self) -> None:
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(
            client,
            default_params={"instructions": "must-not-leak", "max_output_tokens": 321},
            on_unmetered="raise",
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    response_id="resp_123",
                    starting_after=7,
                    text_format=dict,
                )

            assert manager is provider_manager
            assert client._responses.stream_calls == [
                {
                    "response_id": "resp_123",
                    "starting_after": 7,
                    "text_format": dict,
                }
            ]
            check.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()

    @pytest.mark.parametrize(
        "selector_kwargs",
        [
            {"response_id": None},
            {"response_id": ""},
            {"response_id": "resp_123", "starting_after": 0},
        ],
    )
    def test_given_falsey_existing_response_selector_remains_raw(
        self, selector_kwargs: dict[str, object]
    ) -> None:
        provider_manager = object()
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client, on_unmetered="raise") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(**selector_kwargs)

            assert manager is provider_manager
            assert client._responses.stream_calls == [selector_kwargs]
            check.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()

    @pytest.mark.parametrize(
        ("sentinel_key", "sentinel"),
        [
            ("response_id", omit),
            ("response_id", NOT_GIVEN),
            ("starting_after", omit),
            ("starting_after", NOT_GIVEN),
        ],
    )
    def test_omitted_existing_response_selector_still_meters_new_response_stream(
        self,
        sentinel_key: str,
        sentinel: object,
    ) -> None:
        terminal = _terminal_event()
        inner = _FakeSyncResponseStream([terminal], terminal.response)
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    model="gpt-5.5",
                    input="1234",
                    **{sentinel_key: sentinel},
                )
                assert manager is not provider_manager
                with manager as events:
                    assert list(events) == [terminal]

            check.assert_called_once()
            assert client._responses.stream_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "1234",
                    sentinel_key: sentinel,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    def test_native_stream_helper_abandonment_estimates_and_holds_lease_floor(self) -> None:
        delta = SimpleNamespace(type="response.output_text.delta", delta="partial")
        inner = _FakeSyncResponseStream([delta], object())
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
                solwyn.responses.stream(
                    model="gpt-5.5",
                    input="12345678",
                    max_output_tokens=50,
                ) as events,
            ):
                assert next(events) is delta

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    def test_native_stream_helper_forwards_get_final_response_without_double_settlement(
        self,
    ) -> None:
        terminal = _terminal_event()
        inner = _FakeSyncResponseStream([terminal], terminal.response)
        client = _NativeOpenAIClient(_FakeSyncResponseStreamManager(inner))

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                solwyn.responses.stream(model="gpt-5.5", input="1234") as events,
            ):
                result = events.get_final_response()

            assert result is terminal.response
            assert hasattr(events, "until_done")
            assert hasattr(events, "get_final_response")
            assert inner.get_final_response_calls == 1
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "flex"
            assert confirm.token_details == event.token_details

    def test_native_stream_helper_close_before_enter_releases_and_closes_manager(self) -> None:
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_success = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_success",
                    new=breaker_success,
                ),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                manager.close()
                manager.close()

            # The manager never dispatched: no spend to confirm, no provider
            # verdict to record — only the reservation goes back.
            assert provider_manager.enter_calls == 0
            assert len(provider_manager.exit_calls) == 1
            release.assert_called_once()
            breaker_success.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()

    def test_native_stream_helper_enter_failure_releases_reservation(self) -> None:
        error = _Status(503, "stream open failed")
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner, enter_error=error)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
                solwyn.responses.stream(model="gpt-5.5", input="1234"),
            ):
                pass

            assert exc_info.value is error
            release.assert_called_once()
            # 5xx is post-send ambiguous: a health failure whose charge the
            # server must reconcile.
            breaker_failure.assert_called_once_with()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_called_once()
            event = solwyn._solwyn_reporter.report.call_args.args[0]
            assert event.possibly_succeeded is True
            assert event.failover_error_class == "_Status"

    def test_native_stream_helper_fail_fast_enter_failure_spares_the_breaker(self) -> None:
        error = _Status(400, "invalid request")
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner, enter_error=error)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
                solwyn.responses.stream(model="gpt-5.5", input="1234"),
            ):
                pass

            assert exc_info.value is error
            # A request-shaped 4xx is not a provider-health signal — the same
            # status on create(stream=True) records nothing either.
            breaker_failure.assert_not_called()
            release.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_called_once()
            event = solwyn._solwyn_reporter.report.call_args.args[0]
            assert event.possibly_succeeded is None
            assert event.failover_error_class == "_Status"

    def test_native_stream_helper_close_failure_does_not_mask_body_exception(self) -> None:
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(
            inner,
            exit_error=RuntimeError("close failed"),
        )
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ValueError, match="application failed"),
                solwyn.responses.stream(model="gpt-5.5", input="1234"),
            ):
                raise ValueError("application failed")

            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    def test_native_stream_helper_wrap_failure_releases_and_closes_once(self) -> None:
        original = ValueError("wrapping failed")
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _NativeOpenAIClient(provider_manager)

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(solwyn, "_wrap_stream", side_effect=original),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                with pytest.raises(ValueError, match="wrapping failed") as exc_info:
                    manager.__enter__()
                manager.close()

            assert exc_info.value is original
            release.assert_called_once()
            breaker_failure.assert_called_once_with()
            solwyn._solwyn_reporter.report.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert len(provider_manager.exit_calls) == 1
            assert inner.close_calls == 1

    def test_native_create_emits_no_unmetered_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _NativeOpenAIClient(_responses_response())

        with _sync_solwyn(client, report_untracked_surfaces=False) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                solwyn.responses.create(model="gpt-5.5", input="hello")

        assert all("untracked surface" not in record.getMessage() for record in caplog.records)

    def test_background_create_is_refused_before_budget_or_provider_dispatch(self) -> None:
        client = _NativeOpenAIClient(_responses_response())

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.create(model="gpt-5.5", input="hello", background=True)

            assert exc_info.value.field == "background"
            assert "create-time usage" in str(exc_info.value)
            check.assert_not_called()
            assert client._responses.create_calls == []

    def test_extra_body_background_overrides_false_and_is_refused_before_dispatch(
        self,
    ) -> None:
        client = _NativeOpenAIClient(_responses_response())

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    background=False,
                    extra_body={"background": True},
                )

            assert exc_info.value.field == "background"
            check.assert_not_called()
            assert client._responses.create_calls == []

    def test_default_background_create_is_refused_before_budget_or_provider_dispatch(
        self,
    ) -> None:
        client = _NativeOpenAIClient(_responses_response())

        with _sync_solwyn(client, default_params={"background": True}) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                solwyn.responses.create(model="gpt-5.5", input="hello")

            assert exc_info.value.field == "background"
            assert "create-time usage" in str(exc_info.value)
            check.assert_not_called()
            assert client._responses.create_calls == []

    def test_extra_body_background_false_overrides_true_default_and_is_intercepted(
        self,
    ) -> None:
        response = _responses_response()
        client = _NativeOpenAIClient(response)

        with _sync_solwyn(client, default_params={"background": True}) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    extra_body={"background": False},
                )

            assert result is response
            check.assert_called_once()
            assert client._responses.create_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "hello",
                    "background": True,
                    "extra_body": {"background": False},
                }
            ]

    def test_retrieve_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = object()
        client = _NativeOpenAIClient(response)

        with _sync_solwyn(client, report_untracked_surfaces=False) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.retrieve("resp_1")

            assert result is response
            assert client._responses.retrieve_calls == ["resp_1"]
            check.assert_not_called()
            assert any(
                "untracked surface 'responses.retrieve'" in record.getMessage()
                for record in caplog.records
            )

    def test_compat_create_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = object()
        client = _GroqClient(response)

        with _sync_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.create(model="llama-3.3-70b-versatile", input="hello")

            assert result is response
            assert client._responses.create_calls == [
                {"model": "llama-3.3-70b-versatile", "input": "hello"}
            ]
            check.assert_not_called()
            assert any(
                "untracked surface 'responses.create'" in record.getMessage()
                for record in caplog.records
            )

    def test_compat_parse_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange.
        response = object()
        client = _GroqClient(response)

        # Act.
        with _sync_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.parse(
                    model="llama-3.3-70b-versatile",
                    input="hello",
                    text_format=dict,
                )

            # Assert.
            assert result is response
            assert client._responses.parse_calls == [
                {
                    "model": "llama-3.3-70b-versatile",
                    "input": "hello",
                    "text_format": dict,
                }
            ]
            check.assert_not_called()
            assert any(
                "untracked surface 'responses.parse'" in record.getMessage()
                for record in caplog.records
            )

    def test_compat_stream_helper_remains_warned_raw_manager_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inner = _FakeSyncResponseStream([], object())
        provider_manager = _FakeSyncResponseStreamManager(inner)
        client = _GroqClient(provider_manager)

        with _sync_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.stream(
                    model="llama-3.3-70b-versatile",
                    input="hello",
                )

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"model": "llama-3.3-70b-versatile", "input": "hello"}
            ]
            check.assert_not_called()
            assert any(
                "untracked surface 'responses.stream'" in record.getMessage()
                for record in caplog.records
            )

    def test_compat_existing_response_stream_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider_manager = object()
        client = _GroqClient(provider_manager)

        with _sync_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            default_params={"instructions": "must-not-leak"},
            report_untracked_surfaces=False,
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.stream(response_id="resp_123", starting_after=7)

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"response_id": "resp_123", "starting_after": 7}
            ]
            check.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert any(
                "untracked surface 'responses.stream'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.parametrize(("profile_name", "base_url"), _COMPAT_PROFILE_URLS.items())
    def test_every_compat_profile_agrees_on_proxy_precheck_and_dispatch_capability(
        self,
        profile_name: str,
        base_url: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from solwyn.client import _responses_preparer

        assert [profile.name for profile in COMPAT_PROFILES] == list(_COMPAT_PROFILE_URLS)
        response = _responses_response()
        client = _profile_client(profile_name, response)
        assert vars(type(client))["base_url"] == base_url
        expects_metering = profile_name == "azure_openai"

        with _sync_solwyn(
            client,
            model="deployment",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                preparer = _responses_preparer(solwyn._solwyn_adapter)
                namespace = solwyn.responses

                assert (preparer is not None) is expects_metering
                assert isinstance(namespace, _SyncResponsesProxy) is expects_metering

                result = namespace.create(model="deployment", input="hello")
                assert result is response

                if expects_metering:
                    check.assert_called_once()
                    assert check.call_args.kwargs["provider"] == "azure_openai"
                    assert all(
                        "untracked surface 'responses.create'" not in record.getMessage()
                        for record in caplog.records
                    )
                else:
                    check.assert_not_called()
                    assert any(
                        "untracked surface 'responses.create'" in record.getMessage()
                        for record in caplog.records
                    )
                    with pytest.raises(UnsupportedSurfaceError) as exc_info:
                        solwyn._intercepted_call(
                            _surface="responses",
                            model="deployment",
                            input="hello",
                        )
                    assert exc_info.value.provider == profile_name
                    check.assert_not_called()

    def test_strict_acknowledgment_rejects_create_but_allows_retrieve(self) -> None:
        create_client = _NativeOpenAIClient(object())

        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(
                create_client,
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.create"},
            )

        assert exc_info.value.field == "acknowledge_untracked"

        response = object()
        retrieve_client = _NativeOpenAIClient(response)
        with _sync_solwyn(
            retrieve_client,
            on_unmetered="raise",
            acknowledge_untracked={"responses.retrieve"},
        ) as solwyn:
            assert solwyn.responses.retrieve("resp_1") is response
            assert retrieve_client._responses.retrieve_calls == ["resp_1"]

    def test_strict_acknowledgment_rejects_native_parse(self) -> None:
        # Arrange and act.
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(
                _NativeOpenAIClient(object()),
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.parse"},
            )

        # Assert.
        assert exc_info.value.field == "acknowledge_untracked"

    def test_strict_acknowledgment_rejects_native_stream(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            Solwyn(
                _NativeOpenAIClient(object()),
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.stream"},
            )

        assert exc_info.value.field == "acknowledge_untracked"

    def test_namespace_is_cached_for_native_and_compat(self) -> None:
        native = _NativeOpenAIClient(_responses_response())
        compat = _GroqClient(object())

        with (
            _sync_solwyn(native) as native_solwyn,
            _sync_solwyn(compat, model="llama-3.3-70b-versatile") as compat_solwyn,
        ):
            native_responses = native_solwyn.responses
            compat_responses = compat_solwyn.responses

            assert native_responses is native_solwyn.responses
            assert compat_responses is compat_solwyn.responses

    def test_native_client_without_responses_preserves_attribute_error(self) -> None:
        with _sync_solwyn(OpenAI()) as solwyn, pytest.raises(AttributeError):
            _ = solwyn.responses


@pytest.mark.unit
class TestResponsesPublicProxyAsync:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("client_type", [_AsyncNativeOpenAIClient, AsyncAzureOpenAI])
    @pytest.mark.parametrize("operation", _RESPONSES_SPEND_OPERATIONS)
    @pytest.mark.parametrize("override_key", _RESPONSES_CRITICAL_EXTRA_BODY_KEYS)
    @pytest.mark.parametrize("source", ["caller", "default"])
    async def test_metering_critical_extra_body_override_is_refused_before_budget_or_dispatch(
        self,
        client_type: type[_AsyncNativeOpenAIClient],
        operation: str,
        override_key: str,
        source: str,
    ) -> None:
        response = _responses_response()
        client = client_type(response)
        extra_body = {override_key: object()}
        defaults = {"extra_body": extra_body} if source == "default" else None
        call_kwargs: dict[str, object] = {
            "model": "gpt-5.5",
            "input": "hello",
        }
        if source == "caller":
            call_kwargs["extra_body"] = extra_body

        async with _async_solwyn(client, default_params=defaults) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                if operation == "parse":
                    await solwyn.responses.parse(**call_kwargs, text_format=dict)
                elif operation == "create_stream":
                    await solwyn.responses.create(**call_kwargs, stream=True)
                elif operation == "stream_helper":
                    await solwyn.responses.stream(**call_kwargs).__aenter__()
                else:
                    await solwyn.responses.create(**call_kwargs)

            if operation == "parse" and override_key == "stream":
                # Parse resolves effective streaming (extra_body included) and
                # refuses it before the override check ever runs.
                assert exc_info.value.field == "stream"
            else:
                assert exc_info.value.field == "extra_body"
                assert override_key in str(exc_info.value)
            check.assert_not_awaited()
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == []
            assert client._responses.stream_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("client_type", [_AsyncNativeOpenAIClient, AsyncAzureOpenAI])
    async def test_vendor_extra_body_passes_through_unchanged(
        self, client_type: type[_AsyncNativeOpenAIClient]
    ) -> None:
        response = _responses_response()
        client = client_type(response)
        extra_body = {"vendor_extension": {"mode": "safe"}}

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    extra_body=extra_body,
                )

            assert result is response
            check.assert_awaited_once()
            assert client._responses.create_calls == [
                {"model": "gpt-5.5", "input": "hello", "extra_body": extra_body}
            ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("leaf", ["create", "parse"])
    async def test_pinned_native_nonstream_without_usage_estimates_and_holds_lease_floor(
        self, leaf: str
    ) -> None:
        response = _responses_response(include_usage=False)
        client = _profile_client("openai_compatible", response, async_mode=True)
        client.base_url = "http://localhost:9999/v1"  # type: ignore[attr-defined]

        with patch(
            "solwyn._registry.get_adapter_for_client",
            side_effect=AssertionError("provider detection must not run"),
        ) as detector:
            async with _async_solwyn(client, provider="openai") as solwyn:
                check = AsyncMock(
                    spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease()
                )
                true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
                kwargs: dict[str, object] = {
                    "model": "gpt-5.5",
                    "input": "12345678",
                    "max_output_tokens": 50,
                }
                if leaf == "parse":
                    kwargs["text_format"] = dict
                with (
                    patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                    patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
                ):
                    result = await getattr(solwyn.responses, leaf)(**kwargs)

                call_id = check.call_args.kwargs["call_id"]
                true_up.assert_called_once_with(
                    call_id,
                    2,
                    claim_token=7,
                    floor_at_reservation=True,
                )
                confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
                assert result is response
                assert event.provider.value == "openai"
                assert event.input_tokens == 2
                assert event.output_tokens == 0
                assert event.token_details.is_estimated is True
                assert confirm.token_details == event.token_details

        detector.assert_not_called()

    @pytest.mark.asyncio
    async def test_openai_pin_on_local_gateway_bypasses_detection_and_meters_responses(
        self,
    ) -> None:
        response = _responses_response(service_tier="priority")
        client = _profile_client("openai_compatible", response, async_mode=True)
        client.base_url = "http://localhost:9999/v1"  # type: ignore[attr-defined]

        with patch(
            "solwyn._registry.get_adapter_for_client",
            side_effect=AssertionError("provider detection must not run"),
        ) as detect:
            async with _async_solwyn(client, provider="openai") as solwyn:
                assert isinstance(solwyn.responses, _AsyncResponsesProxy)
                check = AsyncMock(
                    spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget()
                )
                with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                    result = await solwyn.responses.create(model="gpt-5.5", input="hello")

                assert result is response
                detect.assert_not_called()
                assert solwyn._solwyn_adapter.name == "openai"
                assert check.call_args.kwargs["provider"] == "openai"
                solwyn._solwyn_reporter.report_settlement.assert_called_once()
                _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
                assert event.provider.value == "openai"

    @pytest.mark.asyncio
    async def test_azure_create_budget_dispatch_and_settlement_use_azure_attribution(
        self,
    ) -> None:
        response = _responses_response()
        client = AsyncAzureOpenAI(response)

        async with _async_solwyn(
            client,
            model="deployment",
            default_params={"instructions": "default", "max_output_tokens": 222},
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.create(
                    model="deployment",
                    input="hello",
                    max_output_tokens=333,
                )

            assert result is response
            check.assert_awaited_once()
            assert check.call_args.kwargs["provider"] == "azure_openai"
            assert check.call_args.kwargs["estimated_output_bound"] == 333
            assert client._responses.create_calls == [
                {
                    "model": "deployment",
                    "input": "hello",
                    "instructions": "default",
                    "max_output_tokens": 333,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 120
            assert event.output_tokens == 45

    @pytest.mark.asyncio
    async def test_azure_parse_dispatches_to_async_parse_and_settles_once(self) -> None:
        response = _responses_response()
        client = AsyncAzureOpenAI(response)

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.parse(
                    model="deployment",
                    input="hello",
                    text_format=dict,
                )

            assert result is response
            check.assert_awaited_once()
            assert check.call_args.kwargs["provider"] == "azure_openai"
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == [
                {"model": "deployment", "input": "hello", "text_format": dict}
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("usage_state", ["missing", "zero"])
    @pytest.mark.parametrize("leaf", ["create", "parse"])
    async def test_azure_nonstream_empty_usage_estimates_and_holds_lease_floor(
        self, usage_state: str, leaf: str
    ) -> None:
        response = _responses_response(include_usage=usage_state == "zero")
        if usage_state == "zero":
            response.usage = _empty_responses_usage()
        client = AsyncAzureOpenAI(response)

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            kwargs: dict[str, object] = {
                "model": "deployment",
                "input": "12345678",
                "max_output_tokens": 50,
            }
            if leaf == "parse":
                kwargs["text_format"] = dict
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                result = await getattr(solwyn.responses, leaf)(**kwargs)

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert result is response
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    @pytest.mark.asyncio
    async def test_azure_nonstream_measured_usage_trues_up_lease_without_floor(self) -> None:
        response = _responses_response()
        client = AsyncAzureOpenAI(response)

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                result = await solwyn.responses.create(model="deployment", input="12345678")

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                165,
                claim_token=7,
                floor_at_reservation=False,
            )
            assert result is response
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.token_details.is_estimated is False

    @pytest.mark.asyncio
    async def test_azure_stream_helper_abandonment_estimates_and_settles_once(self) -> None:
        delta = SimpleNamespace(type="response.output_text.delta", delta="partial")
        inner = _FakeAsyncResponseStream([delta], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = AsyncAzureOpenAI(provider_manager)

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    model="deployment",
                    input="12345678",
                )
                async with manager as events:
                    assert await anext(events) is delta

            check.assert_awaited_once()
            assert client._responses.stream_calls == [{"model": "deployment", "input": "12345678"}]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.provider.value == "azure_openai"
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    @pytest.mark.asyncio
    async def test_azure_stream_helper_without_usage_holds_lease_floor(self) -> None:
        delta = SimpleNamespace(type="response.output_text.delta", delta="partial")
        inner = _FakeAsyncResponseStream([delta], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = AsyncAzureOpenAI(provider_manager)

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                async with solwyn.responses.stream(
                    model="deployment",
                    input="12345678",
                    max_output_tokens=50,
                ) as events:
                    assert [event async for event in events] == [delta]

            call_id = check.call_args.kwargs["call_id"]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            # The accumulator's own length estimate is kept as telemetry, but it
            # cannot account for all output, so the lease holds its bound.
            assert event.token_details.is_estimated is True
            assert event.input_tokens == 2
            assert confirm.token_details == event.token_details
            true_up.assert_called_once_with(
                call_id,
                event.token_details.total_tokens,
                claim_token=7,
                floor_at_reservation=True,
            )

    @pytest.mark.asyncio
    async def test_azure_existing_response_stream_keeps_raw_async_manager_identity(
        self,
    ) -> None:
        provider_manager = object()
        client = AsyncAzureOpenAI(provider_manager)

        async with _async_solwyn(
            client,
            model="deployment",
            default_params={"instructions": "must-not-leak"},
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn.responses.stream(response_id="resp_123", starting_after=7)

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"response_id": "resp_123", "starting_after": 7}
            ]
            check.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_azure_parse_streaming_refusal_precedes_budget_and_provider(self) -> None:
        client = AsyncAzureOpenAI(_responses_response())

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.parse(
                    model="deployment",
                    input="hello",
                    text_format=dict,
                    stream=True,
                )

            assert exc_info.value.field == "stream"
            check.assert_not_awaited()
            assert client._responses.parse_calls == []

    @pytest.mark.asyncio
    async def test_azure_provider_error_releases_reservation_without_fallback(self) -> None:
        client = AsyncAzureOpenAI(_responses_response())
        error = _Status(429)
        client._responses.create = AsyncMock(side_effect=error)  # type: ignore[method-assign]

        async with _async_solwyn(client, model="deployment") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("azure_openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
            ):
                await solwyn.responses.create(model="deployment", input="hello")

            assert exc_info.value is error
            check.assert_awaited_once()
            client._responses.create.assert_awaited_once()
            release.assert_called_once()
            breaker_failure.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_native_create_intercepts_once_dispatches_and_settles(self) -> None:
        response = _responses_response()
        client = _AsyncNativeOpenAIClient(response)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.create(model="gpt-5.5", input="hello")

            assert result is response
            check.assert_awaited_once()
            assert client._responses.create_calls == [{"model": "gpt-5.5", "input": "hello"}]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_parse_reuses_defaults_budgeting_dispatch_and_flat_settlement(
        self,
    ) -> None:
        # Arrange.
        response = _responses_response()
        client = _AsyncNativeOpenAIClient(response)

        class ParsedAnswer:
            value: str

        # Act.
        async with _async_solwyn(
            client,
            default_params={"instructions": "12345678", "max_output_tokens": 321},
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.parse(
                    model="gpt-5.5",
                    input="abcd",
                    text_format=ParsedAnswer,
                    temperature=0.4,
                )

            # Assert.
            assert result is response
            check.assert_awaited_once()
            assert check.call_args.kwargs["estimated_input_tokens"] == 3
            assert check.call_args.kwargs["estimated_output_bound"] == 321
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "abcd",
                    "text_format": ParsedAnswer,
                    "temperature": 0.4,
                    "instructions": "12345678",
                    "max_output_tokens": 321,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.service_tier == "priority"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("default_params", "call_kwargs"),
        [
            ({}, {"stream": True}),
            ({"stream": True}, {}),
            ({"stream": False}, {"extra_body": {"stream": True}}),
        ],
    )
    async def test_native_parse_refuses_effective_streaming_before_budget_or_provider(
        self,
        default_params: dict[str, object],
        call_kwargs: dict[str, object],
    ) -> None:
        # Arrange.
        client = _AsyncNativeOpenAIClient(_responses_response())

        # Act.
        async with _async_solwyn(client, default_params=default_params) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.parse(
                    model="gpt-5.5",
                    input="hello",
                    text_format=dict,
                    **call_kwargs,
                )

            # Assert.
            assert exc_info.value.field == "stream"
            check.assert_not_awaited()
            assert client._responses.create_calls == []
            assert client._responses.parse_calls == []

    @pytest.mark.asyncio
    async def test_extra_body_stream_false_is_still_refused_for_parse(self) -> None:
        # Arrange.
        response = _responses_response()
        client = _AsyncNativeOpenAIClient(response)

        # Act.
        async with _async_solwyn(client, default_params={"stream": True}) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.parse(
                    model="gpt-5.5",
                    input="hello",
                    text_format=dict,
                    extra_body={"stream": False},
                )

            # Assert.
            assert exc_info.value.field == "extra_body"
            assert "stream" in str(exc_info.value)
            check.assert_not_awaited()
            assert client._responses.parse_calls == []

    @pytest.mark.asyncio
    async def test_native_stream_helper_opens_after_one_budget_check_and_settles_terminal_usage(
        self,
    ) -> None:
        order: list[str] = []
        delta = SimpleNamespace(type="response.output_text.delta", delta="hello")
        terminal = _terminal_event()
        inner = _FakeAsyncResponseStream([delta, terminal], terminal.response)
        provider_manager = _FakeAsyncResponseStreamManager(inner, order=order)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(
            client,
            default_params={"instructions": "12345678", "max_output_tokens": 321},
        ) as solwyn:

            async def budget_check(**_kwargs: object) -> object:
                order.append("budget.check")
                return _allow_budget()

            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, side_effect=budget_check)
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    model="gpt-5.5",
                    input="abcd",
                    temperature=0.4,
                )
                assert order == []
                solwyn._solwyn_reporter.report_settlement.assert_not_called()

                async with manager as events:
                    assert [event async for event in events] == [delta, terminal]

            assert order == ["budget.check", "manager.create", "manager.open"]
            check.assert_awaited_once()
            assert client._responses.stream_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "abcd",
                    "temperature": 0.4,
                    "instructions": "12345678",
                    "max_output_tokens": 321,
                }
            ]
            assert provider_manager.enter_calls == 1
            assert len(provider_manager.exit_calls) == 1
            assert inner.aclose_calls == 1
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "flex"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == "flex"

    @pytest.mark.asyncio
    async def test_native_existing_response_stream_is_reviewed_raw_retrieval(self) -> None:
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(
            client,
            default_params={"instructions": "must-not-leak", "max_output_tokens": 321},
            on_unmetered="raise",
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    response_id="resp_123",
                    starting_after=7,
                    text_format=dict,
                )

            assert manager is provider_manager
            assert client._responses.stream_calls == [
                {
                    "response_id": "resp_123",
                    "starting_after": 7,
                    "text_format": dict,
                }
            ]
            check.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "selector_kwargs",
        [
            {"response_id": None},
            {"response_id": ""},
            {"response_id": "resp_123", "starting_after": 0},
        ],
    )
    async def test_given_falsey_existing_response_selector_remains_raw(
        self, selector_kwargs: dict[str, object]
    ) -> None:
        provider_manager = object()
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client, on_unmetered="raise") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(**selector_kwargs)

            assert manager is provider_manager
            assert client._responses.stream_calls == [selector_kwargs]
            check.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sentinel_key", "sentinel"),
        [
            ("response_id", omit),
            ("response_id", NOT_GIVEN),
            ("starting_after", omit),
            ("starting_after", NOT_GIVEN),
        ],
    )
    async def test_omitted_existing_response_selector_still_meters_new_response_stream(
        self,
        sentinel_key: str,
        sentinel: object,
    ) -> None:
        terminal = _terminal_event()
        inner = _FakeAsyncResponseStream([terminal], terminal.response)
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(
                    model="gpt-5.5",
                    input="1234",
                    **{sentinel_key: sentinel},
                )
                assert manager is not provider_manager
                async with manager as events:
                    assert [event async for event in events] == [terminal]

            check.assert_awaited_once()
            assert client._responses.stream_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "1234",
                    sentinel_key: sentinel,
                }
            ]
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_helper_defers_pipeline_creation_until_context_entry(
        self,
    ) -> None:
        terminal = _terminal_event()
        inner = _FakeAsyncResponseStream([terminal], terminal.response)
        client = _AsyncNativeOpenAIClient(_FakeAsyncResponseStreamManager(inner))

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            intercepted = MagicMock(side_effect=solwyn._intercepted_call)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn, "_intercepted_call", new=intercepted),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                intercepted.assert_not_called()

                async with manager as events:
                    assert [event async for event in events] == [terminal]

            intercepted.assert_called_once()
            check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_native_stream_helper_abandonment_estimates_and_holds_lease_floor(
        self,
    ) -> None:
        delta = SimpleNamespace(type="response.output_text.delta", delta="partial")
        inner = _FakeAsyncResponseStream([delta], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                async with solwyn.responses.stream(
                    model="gpt-5.5",
                    input="12345678",
                    max_output_tokens=50,
                ) as events:
                    assert await anext(events) is delta

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 2
            assert event.output_tokens == 0
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    @pytest.mark.asyncio
    async def test_native_stream_helper_forwards_get_final_response_without_double_settlement(
        self,
    ) -> None:
        terminal = _terminal_event()
        inner = _FakeAsyncResponseStream([terminal], terminal.response)
        client = _AsyncNativeOpenAIClient(_FakeAsyncResponseStreamManager(inner))

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                async with solwyn.responses.stream(model="gpt-5.5", input="1234") as events:
                    result = await events.get_final_response()

            assert result is terminal.response
            assert hasattr(events, "until_done")
            assert hasattr(events, "get_final_response")
            assert inner.get_final_response_calls == 1
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.service_tier == "flex"
            assert confirm.token_details == event.token_details

    @pytest.mark.asyncio
    async def test_native_stream_helper_close_before_enter_is_inert(
        self,
    ) -> None:
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                await manager.close()
                await manager.close()

            assert provider_manager.enter_calls == 0
            assert provider_manager.exit_calls == []
            assert client._responses.stream_calls == []
            check.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()

    @pytest.mark.asyncio
    async def test_native_stream_helper_enter_failure_releases_reservation(self) -> None:
        error = _Status(503, "stream open failed")
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner, enter_error=error)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
            ):
                async with solwyn.responses.stream(model="gpt-5.5", input="1234"):
                    pass

            assert exc_info.value is error
            release.assert_called_once()
            # 5xx is post-send ambiguous: a health failure whose charge the
            # server must reconcile.
            breaker_failure.assert_called_once_with()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_called_once()
            event = solwyn._solwyn_reporter.report.call_args.args[0]
            assert event.possibly_succeeded is True
            assert event.failover_error_class == "_Status"

    @pytest.mark.asyncio
    async def test_native_stream_helper_fail_fast_enter_failure_spares_the_breaker(self) -> None:
        error = _Status(400, "invalid request")
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner, enter_error=error)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
                pytest.raises(_Status) as exc_info,
            ):
                async with solwyn.responses.stream(model="gpt-5.5", input="1234"):
                    pass

            assert exc_info.value is error
            # A request-shaped 4xx is not a provider-health signal — the same
            # status on create(stream=True) records nothing either.
            breaker_failure.assert_not_called()
            release.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            solwyn._solwyn_reporter.report.assert_called_once()
            event = solwyn._solwyn_reporter.report.call_args.args[0]
            assert event.possibly_succeeded is None
            assert event.failover_error_class == "_Status"

    @pytest.mark.asyncio
    async def test_native_stream_helper_close_failure_does_not_mask_body_exception(self) -> None:
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(
            inner,
            exit_error=RuntimeError("close failed"),
        )
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ValueError, match="application failed"),
            ):
                async with solwyn.responses.stream(model="gpt-5.5", input="1234"):
                    raise ValueError("application failed")

            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("opened_before_wait", [False, True])
    async def test_native_stream_helper_entry_cancellation_reconciles_and_closes_once(
        self,
        opened_before_wait: bool,
    ) -> None:
        inner = _FakeAsyncResponseStream([], object())
        enter_started = asyncio.Event()
        wait_forever = asyncio.Event()

        class CancellableManager:
            def __init__(self) -> None:
                self.enter_calls = 0
                self.exit_calls: list[tuple[object, ...]] = []
                self.opened = False

            async def __aenter__(self):
                self.enter_calls += 1
                self.opened = opened_before_wait
                enter_started.set()
                await wait_forever.wait()
                self.opened = True
                return inner

            async def __aexit__(self, *args: object) -> None:
                self.exit_calls.append(args)
                if self.opened:
                    await inner.aclose()

        provider_manager = CancellableManager()
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                enter_task = asyncio.create_task(manager.__aenter__())
                await enter_started.wait()
                enter_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await enter_task
                await manager.close()
                await manager.close()

            check.assert_awaited_once()
            release.assert_called_once()
            breaker_failure.assert_called_once_with()
            solwyn._solwyn_reporter.report.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert provider_manager.enter_calls == 1
            assert len(provider_manager.exit_calls) == 1
            exit_args = provider_manager.exit_calls[0]
            assert exit_args[0] is asyncio.CancelledError
            assert isinstance(exit_args[1], asyncio.CancelledError)
            assert inner.aclose_calls == int(opened_before_wait)

    @pytest.mark.asyncio
    async def test_native_stream_helper_nested_reentry_keeps_first_entry_closable(self) -> None:
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                first = await manager.__aenter__()
                with pytest.raises(RuntimeError, match="already entered"):
                    await manager.__aenter__()
                await manager.close()
                await manager.close()

            assert first is not None
            assert provider_manager.enter_calls == 1
            assert len(provider_manager.exit_calls) == 1
            assert inner.aclose_calls == 1
            release.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_helper_concurrent_reentry_keeps_first_entry_closable(
        self,
    ) -> None:
        inner = _FakeAsyncResponseStream([], object())
        enter_started = asyncio.Event()
        allow_enter = asyncio.Event()

        class BlockingOneShotManager:
            def __init__(self) -> None:
                self.enter_calls = 0
                self.exit_calls: list[tuple[object, ...]] = []

            async def __aenter__(self):
                self.enter_calls += 1
                if self.enter_calls > 1:
                    raise RuntimeError("one-shot manager reused")
                enter_started.set()
                await allow_enter.wait()
                return inner

            async def __aexit__(self, *args: object) -> None:
                self.exit_calls.append(args)
                await inner.aclose()

        provider_manager = BlockingOneShotManager()
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                first_task = asyncio.create_task(manager.__aenter__())
                await enter_started.wait()
                second_task = asyncio.create_task(manager.__aenter__())
                await asyncio.sleep(0)
                assert provider_manager.enter_calls == 1
                allow_enter.set()

                first = await first_task
                with pytest.raises(RuntimeError, match="already entered"):
                    await second_task
                await manager.close()

            assert first is not None
            assert provider_manager.enter_calls == 1
            assert len(provider_manager.exit_calls) == 1
            assert inner.aclose_calls == 1
            release.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_helper_cancelled_close_is_publicly_retryable(self) -> None:
        inner = _FakeAsyncResponseStream([], object())
        first_exit_started = asyncio.Event()
        second_exit_started = asyncio.Event()
        allow_exit = asyncio.Event()

        class RetryableExitManager:
            def __init__(self) -> None:
                self.exit_attempts = 0
                self.completed_exits = 0
                self.active_exits = 0
                self.max_active_exits = 0

            async def __aenter__(self):
                return inner

            async def __aexit__(self, *_args: object) -> None:
                self.exit_attempts += 1
                self.active_exits += 1
                self.max_active_exits = max(self.max_active_exits, self.active_exits)
                if self.exit_attempts == 1:
                    first_exit_started.set()
                else:
                    second_exit_started.set()
                try:
                    await allow_exit.wait()
                    await inner.aclose()
                    self.completed_exits += 1
                finally:
                    self.active_exits -= 1

        provider_manager = RetryableExitManager()
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                await manager.__aenter__()
                first_close = asyncio.create_task(manager.close())
                await first_exit_started.wait()
                second_close = asyncio.create_task(manager.close())
                await asyncio.sleep(0)
                assert provider_manager.exit_attempts == 1

                first_close.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first_close
                try:
                    await asyncio.wait_for(second_exit_started.wait(), timeout=1)
                finally:
                    allow_exit.set()
                await second_close
                await manager.close()

            check.assert_awaited_once()
            assert provider_manager.exit_attempts == 2
            assert provider_manager.completed_exits == 1
            assert provider_manager.max_active_exits == 1
            assert inner.aclose_calls == 1
            release.assert_not_called()
            solwyn._solwyn_reporter.report.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_native_stream_helper_wrap_failure_releases_and_closes_once(self) -> None:
        original = ValueError("wrapping failed")
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncNativeOpenAIClient(provider_manager)

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            breaker_failure = MagicMock()
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                patch.object(solwyn, "_wrap_stream_async", side_effect=original),
                patch.object(
                    solwyn._get_circuit_breaker("openai"),
                    "record_failure",
                    new=breaker_failure,
                ),
            ):
                manager = solwyn.responses.stream(model="gpt-5.5", input="1234")
                with pytest.raises(ValueError, match="wrapping failed") as exc_info:
                    await manager.__aenter__()
                await manager.close()

            assert exc_info.value is original
            release.assert_called_once()
            breaker_failure.assert_called_once_with()
            solwyn._solwyn_reporter.report.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert len(provider_manager.exit_calls) == 1
            assert inner.aclose_calls == 1

    @pytest.mark.asyncio
    async def test_native_create_emits_no_unmetered_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _AsyncNativeOpenAIClient(_responses_response())

        async with _async_solwyn(client, report_untracked_surfaces=False) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                await solwyn.responses.create(model="gpt-5.5", input="hello")

        assert all("untracked surface" not in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_background_create_is_refused_before_budget_or_provider_dispatch(self) -> None:
        client = _AsyncNativeOpenAIClient(_responses_response())

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.create(model="gpt-5.5", input="hello", background=True)

            assert exc_info.value.field == "background"
            assert "create-time usage" in str(exc_info.value)
            check.assert_not_awaited()
            assert client._responses.create_calls == []

    @pytest.mark.asyncio
    async def test_extra_body_background_overrides_false_and_is_refused_before_dispatch(
        self,
    ) -> None:
        client = _AsyncNativeOpenAIClient(_responses_response())

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    background=False,
                    extra_body={"background": True},
                )

            assert exc_info.value.field == "background"
            check.assert_not_awaited()
            assert client._responses.create_calls == []

    @pytest.mark.asyncio
    async def test_default_background_create_is_refused_before_budget_or_provider_dispatch(
        self,
    ) -> None:
        client = _AsyncNativeOpenAIClient(_responses_response())

        async with _async_solwyn(client, default_params={"background": True}) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                await solwyn.responses.create(model="gpt-5.5", input="hello")

            assert exc_info.value.field == "background"
            assert "create-time usage" in str(exc_info.value)
            check.assert_not_awaited()
            assert client._responses.create_calls == []

    @pytest.mark.asyncio
    async def test_extra_body_background_false_overrides_true_default_and_is_intercepted(
        self,
    ) -> None:
        response = _responses_response()
        client = _AsyncNativeOpenAIClient(response)

        async with _async_solwyn(client, default_params={"background": True}) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn.responses.create(
                    model="gpt-5.5",
                    input="hello",
                    extra_body={"background": False},
                )

            assert result is response
            check.assert_awaited_once()
            assert client._responses.create_calls == [
                {
                    "model": "gpt-5.5",
                    "input": "hello",
                    "background": True,
                    "extra_body": {"background": False},
                }
            ]

    @pytest.mark.asyncio
    async def test_retrieve_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = object()
        client = _AsyncNativeOpenAIClient(response)

        async with _async_solwyn(client, report_untracked_surfaces=False) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = await solwyn.responses.retrieve("resp_1")

            assert result is response
            assert client._responses.retrieve_calls == ["resp_1"]
            check.assert_not_awaited()
            assert any(
                "untracked surface 'responses.retrieve'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_compat_create_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = object()
        client = _AsyncGroqClient(response)

        async with _async_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = await solwyn.responses.create(
                    model="llama-3.3-70b-versatile", input="hello"
                )

            assert result is response
            assert client._responses.create_calls == [
                {"model": "llama-3.3-70b-versatile", "input": "hello"}
            ]
            check.assert_not_awaited()
            assert any(
                "untracked surface 'responses.create'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_compat_parse_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange.
        response = object()
        client = _AsyncGroqClient(response)

        # Act.
        async with _async_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = await solwyn.responses.parse(
                    model="llama-3.3-70b-versatile",
                    input="hello",
                    text_format=dict,
                )

            # Assert.
            assert result is response
            assert client._responses.parse_calls == [
                {
                    "model": "llama-3.3-70b-versatile",
                    "input": "hello",
                    "text_format": dict,
                }
            ]
            check.assert_not_awaited()
            assert any(
                "untracked surface 'responses.parse'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_compat_stream_helper_remains_warned_raw_manager_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inner = _FakeAsyncResponseStream([], object())
        provider_manager = _FakeAsyncResponseStreamManager(inner)
        client = _AsyncGroqClient(provider_manager)

        async with _async_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.stream(
                    model="llama-3.3-70b-versatile",
                    input="hello",
                )

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"model": "llama-3.3-70b-versatile", "input": "hello"}
            ]
            check.assert_not_awaited()
            assert any(
                "untracked surface 'responses.stream'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_compat_existing_response_stream_remains_warned_raw_passthrough(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider_manager = object()
        client = _AsyncGroqClient(provider_manager)

        async with _async_solwyn(
            client,
            model="llama-3.3-70b-versatile",
            default_params={"instructions": "must-not-leak"},
            report_untracked_surfaces=False,
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                result = solwyn.responses.stream(response_id="resp_123", starting_after=7)

            assert result is provider_manager
            assert client._responses.stream_calls == [
                {"response_id": "resp_123", "starting_after": 7}
            ]
            check.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert any(
                "untracked surface 'responses.stream'" in record.getMessage()
                for record in caplog.records
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("profile_name", "base_url"), _COMPAT_PROFILE_URLS.items())
    async def test_every_compat_profile_async_proxy_and_precheck_use_same_capability(
        self,
        profile_name: str,
        base_url: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from solwyn.client import _responses_preparer

        response = _responses_response()
        client = _profile_client(profile_name, response, async_mode=True)
        assert vars(type(client))["base_url"] == base_url
        expects_metering = profile_name == "azure_openai"

        async with _async_solwyn(
            client,
            model="deployment",
            report_untracked_surfaces=False,
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                caplog.at_level(logging.WARNING, logger="solwyn._base"),
            ):
                preparer = _responses_preparer(solwyn._solwyn_adapter)
                namespace = solwyn.responses

                assert (preparer is not None) is expects_metering
                assert isinstance(namespace, _AsyncResponsesProxy) is expects_metering

                result = await namespace.create(model="deployment", input="hello")
                assert result is response

                if expects_metering:
                    check.assert_awaited_once()
                    assert check.call_args.kwargs["provider"] == "azure_openai"
                    assert all(
                        "untracked surface 'responses.create'" not in record.getMessage()
                        for record in caplog.records
                    )
                else:
                    check.assert_not_awaited()
                    assert any(
                        "untracked surface 'responses.create'" in record.getMessage()
                        for record in caplog.records
                    )
                    with pytest.raises(UnsupportedSurfaceError) as exc_info:
                        await solwyn._intercepted_call(
                            _surface="responses",
                            model="deployment",
                            input="hello",
                        )
                    assert exc_info.value.provider == profile_name
                    check.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_strict_acknowledgment_rejects_create_but_allows_retrieve(self) -> None:
        create_client = _AsyncNativeOpenAIClient(object())

        with pytest.raises(ConfigurationError) as exc_info:
            AsyncSolwyn(
                create_client,
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.create"},
            )

        assert exc_info.value.field == "acknowledge_untracked"

        response = object()
        retrieve_client = _AsyncNativeOpenAIClient(response)
        async with _async_solwyn(
            retrieve_client,
            on_unmetered="raise",
            acknowledge_untracked={"responses.retrieve"},
        ) as solwyn:
            assert await solwyn.responses.retrieve("resp_1") is response
            assert retrieve_client._responses.retrieve_calls == ["resp_1"]

    def test_strict_acknowledgment_rejects_native_parse(self) -> None:
        # Arrange and act.
        with pytest.raises(ConfigurationError) as exc_info:
            AsyncSolwyn(
                _AsyncNativeOpenAIClient(object()),
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.parse"},
            )

        # Assert.
        assert exc_info.value.field == "acknowledge_untracked"

    def test_strict_acknowledgment_rejects_native_stream(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            AsyncSolwyn(
                _AsyncNativeOpenAIClient(object()),
                api_key=VALID_API_KEY,
                on_unmetered="raise",
                acknowledge_untracked={"responses.stream"},
            )

        assert exc_info.value.field == "acknowledge_untracked"

    @pytest.mark.asyncio
    async def test_namespace_is_cached_for_native_and_compat(self) -> None:
        native = _AsyncNativeOpenAIClient(_responses_response())
        compat = _AsyncGroqClient(object())

        async with (
            _async_solwyn(native) as native_solwyn,
            _async_solwyn(compat, model="llama-3.3-70b-versatile") as compat_solwyn,
        ):
            native_responses = native_solwyn.responses
            compat_responses = compat_solwyn.responses

            assert native_responses is native_solwyn.responses
            assert compat_responses is compat_solwyn.responses

    @pytest.mark.asyncio
    async def test_native_client_without_responses_preserves_attribute_error(self) -> None:
        async with _async_solwyn(AsyncOpenAI()) as solwyn:
            with pytest.raises(AttributeError):
                _ = solwyn.responses


@pytest.mark.unit
class TestResponsesInterceptedCallSync:
    def test_default_surface_preserves_chat_interception(self) -> None:
        client, _ = _mock_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        messages = [{"role": "user", "content": "hello"}]

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn._intercepted_call(model="gpt-5.5", messages=messages)

            assert result is response
            client.chat.completions.create.assert_called_once_with(
                model="gpt-5.5", messages=messages
            )
            client.responses.create.assert_not_called()
            check.assert_called_once()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 21
            assert event.output_tokens == 9
            assert event.service_tier == "default"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    def test_chat_stream_does_not_gain_responses_manager_helpers(self) -> None:
        client, _ = _mock_openai_client()
        client.chat.completions.create.return_value = _FakeSyncStream([])

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    model="gpt-5.5",
                    messages=[],
                    stream=True,
                )

            assert not hasattr(stream, "until_done")
            assert not hasattr(stream, "get_final_response")
            with pytest.raises(AttributeError):
                _ = stream.until_done
            with pytest.raises(AttributeError):
                _ = stream.get_final_response
            stream.close()

    def test_sync_dispatch_defaults_to_chat(self) -> None:
        client, _ = _mock_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        kwargs: dict[str, object] = {"model": "gpt-5.5", "messages": []}

        with _sync_solwyn(client) as solwyn:
            result = solwyn._sync_dispatch(
                solwyn._solwyn_runtimes[0],
                kwargs,
                is_streaming=False,
                timeout=1.0,
                read_timeout=2.0,
                max_retries=0,
            )

        assert result is response
        client.chat.completions.create.assert_called_once_with(**kwargs)
        client.responses.create.assert_not_called()

    def test_non_streaming_dispatches_to_responses_and_settles_flat_usage(self) -> None:
        client, response = _mock_openai_client()

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="hi there",
                )

            assert result is response
            client.responses.create.assert_called_once()
            client.chat.completions.create.assert_not_called()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.token_details.cached_input_tokens == 10
            assert event.token_details.cache_creation_5m_tokens == 3
            assert event.token_details.reasoning_tokens == 5
            assert event.service_tier == "priority"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    def test_budget_uses_responses_estimate_text_modality_and_no_fallback_hints(self) -> None:
        client, _ = _mock_openai_client()
        fallback = _mock_anthropic_client()

        with _sync_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    instructions="abcd",
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "12345678"}],
                        }
                    ],
                )

            budget_kwargs = check.call_args.kwargs
            assert budget_kwargs["estimated_input_tokens"] == 3
            assert budget_kwargs["modality"] == "text"
            assert budget_kwargs["fallback_providers"] == []
            assert budget_kwargs["fallback_models"] == []

    @pytest.mark.parametrize(
        (
            "entry_instructions",
            "caller_instructions",
            "expected_instructions",
            "expected_tokens",
        ),
        [
            (None, None, "12345678", 3),
            ("abcdefghijkl", None, "abcdefghijkl", 4),
            ("abcdefghijkl", "abcdefghijklmnop", "abcdefghijklmnop", 5),
        ],
    )
    def test_responses_estimation_uses_effective_default_instructions(
        self,
        entry_instructions: str | None,
        caller_instructions: str | None,
        expected_instructions: str,
        expected_tokens: int,
    ) -> None:
        client, _ = _mock_openai_client()

        with _sync_solwyn(
            client,
            default_params={"instructions": "12345678"},
        ) as solwyn:
            if entry_instructions is not None:
                solwyn._solwyn_runtimes[0].entry.default_params["instructions"] = entry_instructions
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "abcd"}
            if caller_instructions is not None:
                call_kwargs["instructions"] = caller_instructions
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_input_tokens"] == expected_tokens
            assert client.responses.create.call_args.kwargs["instructions"] == expected_instructions

    @pytest.mark.parametrize(
        ("entry_cap", "caller_cap", "expected_cap"),
        [(None, None, 111), (222, None, 222), (222, 333, 333)],
    )
    def test_responses_output_bound_uses_effective_default_cap(
        self,
        entry_cap: int | None,
        caller_cap: int | None,
        expected_cap: int,
    ) -> None:
        client, _ = _mock_openai_client()

        with _sync_solwyn(
            client,
            lease_output_bound_default=777,
            default_params={"max_output_tokens": 111},
        ) as solwyn:
            if entry_cap is not None:
                solwyn._solwyn_runtimes[0].entry.default_params["max_output_tokens"] = entry_cap
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_cap is not None:
                call_kwargs["max_output_tokens"] = caller_cap
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_output_bound"] == expected_cap
            assert client.responses.create.call_args.kwargs["max_output_tokens"] == expected_cap

    @pytest.mark.parametrize(("cap", "expected"), [(512, 512), (None, 777)])
    def test_responses_output_bound_uses_cap_or_configured_default(
        self, cap: int | None, expected: int
    ) -> None:
        client, _ = _mock_openai_client()

        with _sync_solwyn(
            client,
            lease_output_bound_default=777,
            default_params={"max_tokens": 999},
        ) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if cap is not None:
                call_kwargs["max_output_tokens"] = cap
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_output_bound"] == expected

    def test_responses_defaults_merge_filters_only_chat_keys_and_caller_wins(self) -> None:
        client, _ = _mock_openai_client()
        global_defaults = {
            "temperature": 0.1,
            "top_p": 0.2,
            "max_tokens": 100,
            "stream_options": {"include_usage": True},
            "solwyn_tags": {"source": "global-default"},
            "future_responses_parameter": "pass-through",
        }

        with _sync_solwyn(client, default_params=global_defaults) as solwyn:
            solwyn._solwyn_runtimes[0].entry.default_params.update(
                {
                    "temperature": 0.4,
                    "top_p": 0.6,
                    "max_completion_tokens": 200,
                }
            )
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                    temperature=0.8,
                )

            sent = client.responses.create.call_args.kwargs
            assert sent["temperature"] == 0.8
            assert sent["top_p"] == 0.6
            assert sent["future_responses_parameter"] == "pass-through"
            assert "max_tokens" not in sent
            assert "max_completion_tokens" not in sent
            assert "stream_options" not in sent
            assert "solwyn_tags" not in sent

    def test_provider_error_releases_reservation_without_fallback_or_translation(self) -> None:
        client, _ = _mock_openai_client()
        fallback = _mock_anthropic_client()
        client.responses.create.side_effect = _Status(429)

        with _sync_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                pytest.raises(_Status),
            ):
                solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            client.responses.create.assert_called_once()
            fallback.messages.create.assert_not_called()
            release.assert_called_once()

    @pytest.mark.parametrize(("force_stream", "caller_stream"), [(True, None), (False, True)])
    def test_streaming_adds_or_retains_stream_and_settles_nested_terminal_usage(
        self, force_stream: bool, caller_stream: bool | None
    ) -> None:
        client, _ = _mock_openai_client()
        delta = SimpleNamespace(type="response.output_text.delta", delta="hello")
        terminal = _terminal_event()
        inner = _FakeSyncStream([delta, terminal])
        client.responses.create.return_value = inner

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_stream is not None:
                call_kwargs["stream"] = caller_stream
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    _force_stream=force_stream,
                    **call_kwargs,
                )
            events = list(stream)

            assert events == [delta, terminal]
            assert not hasattr(stream, "until_done")
            assert not hasattr(stream, "get_final_response")
            with pytest.raises(AttributeError):
                _ = stream.until_done
            with pytest.raises(AttributeError):
                _ = stream.get_final_response
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.token_details.cached_input_tokens == 7
            assert event.token_details.reasoning_tokens == 4
            assert event.service_tier == "flex"
            assert confirm.service_tier == "flex"

    def test_default_stream_true_is_wrapped_and_settles_only_on_consumption(self) -> None:
        client, _ = _mock_openai_client()
        terminal = _terminal_event()
        inner = _FakeSyncStream([terminal])
        client.responses.create.return_value = inner

        with _sync_solwyn(client, default_params={"stream": True}) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert list(stream) == [terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12

    def test_abandoned_stream_settles_once_with_length_estimate(self) -> None:
        client, _ = _mock_openai_client()
        inner = _FakeSyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        client.responses.create.return_value = inner

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="12345678",
                    stream=True,
                )
            iterator = iter(stream)
            assert next(iterator).delta == "partial"
            stream.close()
            stream.close()

            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.token_details.input_tokens == 2
            assert event.token_details.is_estimated is True
            assert confirm.token_details == event.token_details

    def test_lease_backed_abandoned_stream_holds_reserved_floor(self) -> None:
        client, _ = _mock_openai_client()
        inner = _FakeSyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        client.responses.create.return_value = inner

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="12345678",
                    max_output_tokens=50,
                    stream=True,
                )
                assert next(iter(stream)).delta == "partial"
                stream.close()
                stream.close()

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    def test_adapter_without_responses_seam_fails_before_budget_check(self) -> None:
        client = _mock_anthropic_client()

        with _sync_solwyn(client, model="claude-x") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(UnsupportedSurfaceError) as exc_info,
            ):
                solwyn._intercepted_call(
                    _surface="responses",
                    model="claude-x",
                    input="x",
                )

            assert exc_info.value.surface == "responses.create"
            assert exc_info.value.provider == "anthropic"
            check.assert_not_called()
            client.messages.create.assert_not_called()

    def test_parse_capability_error_names_parse_before_budget_check(self) -> None:
        # Arrange.
        client = _mock_anthropic_client()

        # Act.
        with _sync_solwyn(client, model="claude-x") as solwyn:
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(UnsupportedSurfaceError) as exc_info,
            ):
                solwyn._intercepted_call(
                    _surface="responses",
                    _responses_leaf="parse",
                    model="claude-x",
                    input="x",
                    text_format=dict,
                )

            # Assert.
            assert exc_info.value.surface == "responses.parse"
            assert exc_info.value.provider == "anthropic"
            check.assert_not_called()
            client.messages.create.assert_not_called()

    def test_open_primary_with_only_fallback_candidate_releases_reservation(self) -> None:
        client, _ = _mock_openai_client()
        fallback = _mock_anthropic_client()

        with _sync_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            breaker = solwyn._get_circuit_breaker("openai")
            for _ in range(3):
                breaker.record_failure()
            check = MagicMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                pytest.raises(ProviderUnavailableError) as exc_info,
            ):
                solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            assert exc_info.value.attempted == []
            client.responses.create.assert_not_called()
            fallback.messages.create.assert_not_called()
            release.assert_called_once()


@pytest.mark.unit
class TestResponsesInterceptedCallAsync:
    @pytest.mark.asyncio
    async def test_default_surface_preserves_chat_interception(self) -> None:
        client, _ = _mock_async_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        messages = [{"role": "user", "content": "hello"}]

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn._intercepted_call(model="gpt-5.5", messages=messages)

            assert result is response
            client.chat.completions.create.assert_awaited_once_with(
                model="gpt-5.5", messages=messages
            )
            client.responses.create.assert_not_awaited()
            check.assert_awaited_once()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 21
            assert event.output_tokens == 9
            assert event.service_tier == "default"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    @pytest.mark.asyncio
    async def test_chat_stream_does_not_gain_responses_manager_helpers(self) -> None:
        client, _ = _mock_async_openai_client()
        client.chat.completions.create.return_value = _FakeAsyncStream([])

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    model="gpt-5.5",
                    messages=[],
                    stream=True,
                )

            assert not hasattr(stream, "until_done")
            assert not hasattr(stream, "get_final_response")
            with pytest.raises(AttributeError):
                _ = stream.until_done
            with pytest.raises(AttributeError):
                _ = stream.get_final_response
            await stream.close()

    @pytest.mark.asyncio
    async def test_async_dispatch_defaults_to_chat(self) -> None:
        client, _ = _mock_async_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        kwargs: dict[str, object] = {"model": "gpt-5.5", "messages": []}

        async with _async_solwyn(client) as solwyn:
            result = await solwyn._async_dispatch(
                solwyn._solwyn_runtimes[0],
                kwargs,
                is_streaming=False,
                timeout=1.0,
                read_timeout=2.0,
                max_retries=0,
            )

        assert result is response
        client.chat.completions.create.assert_awaited_once_with(**kwargs)
        client.responses.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_streaming_dispatches_to_responses_and_settles_flat_usage(self) -> None:
        client, response = _mock_async_openai_client()

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                result = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="hi there",
                )

            assert result is response
            client.responses.create.assert_awaited_once()
            client.chat.completions.create.assert_not_awaited()
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 120
            assert event.output_tokens == 45
            assert event.token_details.cached_input_tokens == 10
            assert event.token_details.cache_creation_5m_tokens == 3
            assert event.token_details.reasoning_tokens == 5
            assert event.service_tier == "priority"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    @pytest.mark.asyncio
    async def test_budget_uses_responses_estimate_text_modality_and_no_fallback_hints(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        fallback = _mock_async_anthropic_client()

        async with _async_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    instructions="abcd",
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "12345678"}],
                        }
                    ],
                )

            budget_kwargs = check.call_args.kwargs
            assert budget_kwargs["estimated_input_tokens"] == 3
            assert budget_kwargs["modality"] == "text"
            assert budget_kwargs["fallback_providers"] == []
            assert budget_kwargs["fallback_models"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "entry_instructions",
            "caller_instructions",
            "expected_instructions",
            "expected_tokens",
        ),
        [
            (None, None, "12345678", 3),
            ("abcdefghijkl", None, "abcdefghijkl", 4),
            ("abcdefghijkl", "abcdefghijklmnop", "abcdefghijklmnop", 5),
        ],
    )
    async def test_responses_estimation_uses_effective_default_instructions(
        self,
        entry_instructions: str | None,
        caller_instructions: str | None,
        expected_instructions: str,
        expected_tokens: int,
    ) -> None:
        client, _ = _mock_async_openai_client()

        async with _async_solwyn(
            client,
            default_params={"instructions": "12345678"},
        ) as solwyn:
            if entry_instructions is not None:
                solwyn._solwyn_runtimes[0].entry.default_params["instructions"] = entry_instructions
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "abcd"}
            if caller_instructions is not None:
                call_kwargs["instructions"] = caller_instructions
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                await solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_input_tokens"] == expected_tokens
            assert client.responses.create.call_args.kwargs["instructions"] == expected_instructions

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entry_cap", "caller_cap", "expected_cap"),
        [(None, None, 111), (222, None, 222), (222, 333, 333)],
    )
    async def test_responses_output_bound_uses_effective_default_cap(
        self,
        entry_cap: int | None,
        caller_cap: int | None,
        expected_cap: int,
    ) -> None:
        client, _ = _mock_async_openai_client()

        async with _async_solwyn(
            client,
            lease_output_bound_default=777,
            default_params={"max_output_tokens": 111},
        ) as solwyn:
            if entry_cap is not None:
                solwyn._solwyn_runtimes[0].entry.default_params["max_output_tokens"] = entry_cap
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_cap is not None:
                call_kwargs["max_output_tokens"] = caller_cap
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                await solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_output_bound"] == expected_cap
            assert client.responses.create.call_args.kwargs["max_output_tokens"] == expected_cap

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("cap", "expected"), [(512, 512), (None, 777)])
    async def test_responses_output_bound_uses_cap_or_configured_default(
        self, cap: int | None, expected: int
    ) -> None:
        client, _ = _mock_async_openai_client()

        async with _async_solwyn(
            client,
            lease_output_bound_default=777,
            default_params={"max_tokens": 999},
        ) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if cap is not None:
                call_kwargs["max_output_tokens"] = cap
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                await solwyn._intercepted_call(_surface="responses", **call_kwargs)

            assert check.call_args.kwargs["estimated_output_bound"] == expected

    @pytest.mark.asyncio
    async def test_responses_defaults_merge_filters_only_chat_keys_and_caller_wins(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        global_defaults = {
            "temperature": 0.1,
            "top_p": 0.2,
            "max_tokens": 100,
            "stream_options": {"include_usage": True},
            "solwyn_tags": {"source": "global-default"},
            "future_responses_parameter": "pass-through",
        }

        async with _async_solwyn(client, default_params=global_defaults) as solwyn:
            solwyn._solwyn_runtimes[0].entry.default_params.update(
                {
                    "temperature": 0.4,
                    "top_p": 0.6,
                    "max_completion_tokens": 200,
                }
            )
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                    temperature=0.8,
                )

            sent = client.responses.create.call_args.kwargs
            assert sent["temperature"] == 0.8
            assert sent["top_p"] == 0.6
            assert sent["future_responses_parameter"] == "pass-through"
            assert "max_tokens" not in sent
            assert "max_completion_tokens" not in sent
            assert "stream_options" not in sent
            assert "solwyn_tags" not in sent

    @pytest.mark.asyncio
    async def test_provider_error_releases_reservation_without_fallback_or_translation(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        fallback = _mock_async_anthropic_client()
        client.responses.create.side_effect = _Status(429)

        async with _async_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                pytest.raises(_Status),
            ):
                await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            client.responses.create.assert_awaited_once()
            fallback.messages.create.assert_not_awaited()
            release.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("force_stream", "caller_stream"), [(True, None), (False, True)])
    async def test_streaming_adds_or_retains_stream_and_settles_nested_terminal_usage(
        self, force_stream: bool, caller_stream: bool | None
    ) -> None:
        client, _ = _mock_async_openai_client()
        delta = SimpleNamespace(type="response.output_text.delta", delta="hello")
        terminal = _terminal_event()
        inner = _FakeAsyncStream([delta, terminal])
        client.responses.create.return_value = inner

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_stream is not None:
                call_kwargs["stream"] = caller_stream
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    _force_stream=force_stream,
                    **call_kwargs,
                )
            events = [event async for event in stream]

            assert events == [delta, terminal]
            assert not hasattr(stream, "until_done")
            assert not hasattr(stream, "get_final_response")
            with pytest.raises(AttributeError):
                _ = stream.until_done
            with pytest.raises(AttributeError):
                _ = stream.get_final_response
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12
            assert event.token_details.cached_input_tokens == 7
            assert event.token_details.reasoning_tokens == 4
            assert event.service_tier == "flex"
            assert confirm.service_tier == "flex"

    @pytest.mark.asyncio
    async def test_default_stream_true_is_wrapped_and_settles_only_on_consumption(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        terminal = _terminal_event()
        inner = _FakeAsyncStream([terminal])
        client.responses.create.return_value = inner

        async with _async_solwyn(client, default_params={"stream": True}) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            solwyn._solwyn_reporter.report_settlement.assert_not_called()
            assert [event async for event in stream] == [terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            _, event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12

    @pytest.mark.asyncio
    async def test_abandoned_stream_settles_once_with_length_estimate(self) -> None:
        client, _ = _mock_async_openai_client()
        inner = _FakeAsyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        client.responses.create.return_value = inner

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._solwyn_budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="12345678",
                    stream=True,
                )
            iterator = stream.__aiter__()
            event = await anext(iterator)
            assert event.delta == "partial"
            await stream.close()
            await stream.close()

            solwyn._solwyn_reporter.report_settlement.assert_called_once()
            confirm, settled_event = solwyn._solwyn_reporter.report_settlement.call_args.args
            assert settled_event.token_details.input_tokens == 2
            assert settled_event.token_details.is_estimated is True
            assert confirm.token_details == settled_event.token_details

    @pytest.mark.asyncio
    async def test_lease_backed_abandoned_stream_holds_reserved_floor(self) -> None:
        client, _ = _mock_async_openai_client()
        inner = _FakeAsyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        client.responses.create.return_value = inner

        async with _async_solwyn(client) as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._solwyn_budget._lease.true_up)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget._lease, "true_up", new=true_up),
            ):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="12345678",
                    max_output_tokens=50,
                    stream=True,
                )
                event = await anext(stream.__aiter__())
                assert event.delta == "partial"
                await stream.close()
                await stream.close()

            call_id = check.call_args.kwargs["call_id"]
            true_up.assert_called_once_with(
                call_id,
                2,
                claim_token=7,
                floor_at_reservation=True,
            )
            solwyn._solwyn_reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_without_responses_seam_fails_before_budget_check(self) -> None:
        client = _mock_async_anthropic_client()

        async with _async_solwyn(client, model="claude-x") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(UnsupportedSurfaceError) as exc_info,
            ):
                await solwyn._intercepted_call(
                    _surface="responses",
                    model="claude-x",
                    input="x",
                )

            assert exc_info.value.surface == "responses.create"
            assert exc_info.value.provider == "anthropic"
            check.assert_not_awaited()
            client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parse_capability_error_names_parse_before_budget_check(self) -> None:
        # Arrange.
        client = _mock_async_anthropic_client()

        # Act.
        async with _async_solwyn(client, model="claude-x") as solwyn:
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                pytest.raises(UnsupportedSurfaceError) as exc_info,
            ):
                await solwyn._intercepted_call(
                    _surface="responses",
                    _responses_leaf="parse",
                    model="claude-x",
                    input="x",
                    text_format=dict,
                )

            # Assert.
            assert exc_info.value.surface == "responses.parse"
            assert exc_info.value.provider == "anthropic"
            check.assert_not_awaited()
            client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_primary_with_only_fallback_candidate_releases_reservation(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        fallback = _mock_async_anthropic_client()

        async with _async_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            breaker = solwyn._get_circuit_breaker("openai")
            for _ in range(3):
                breaker.record_failure()
            check = AsyncMock(spec=solwyn._solwyn_budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._solwyn_budget.release_reservation)
            with (
                patch.object(solwyn._solwyn_budget, "check_budget", new=check),
                patch.object(solwyn._solwyn_budget, "release_reservation", new=release),
                pytest.raises(ProviderUnavailableError) as exc_info,
            ):
                await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            assert exc_info.value.attempted == []
            client.responses.create.assert_not_awaited()
            fallback.messages.create.assert_not_awaited()
            release.assert_called_once()
