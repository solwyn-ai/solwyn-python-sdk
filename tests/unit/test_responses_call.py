"""Internal Responses API interception pipeline tests.

Public ``responses`` proxies land separately.  These tests drive the private
``_surface="responses"`` branch directly so dispatch, budgeting, routing, and
settlement can be verified before the public surface contract changes.  A
focused adjacent regression also protects the default chat routing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID

from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ProviderUnavailableError, UnsupportedSurfaceError


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


def _sync_create(**kwargs: object) -> object:
    raise NotImplementedError


async def _async_create(**kwargs: object) -> object:
    raise NotImplementedError


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


def _responses_response(*, service_tier: str = "priority") -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        object="response",
        status="completed",
        model="gpt-5.5",
        output=[],
        usage=_responses_usage(),
        service_tier=service_tier,
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
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=MagicMock(spec=_sync_create)))
    client.responses = SimpleNamespace(create=MagicMock(spec=_sync_create, return_value=response))
    client.with_options.return_value = client
    return client, response


def _mock_async_openai_client() -> tuple[MagicMock, SimpleNamespace]:
    client = MagicMock(spec=AsyncOpenAI)
    type(client).__module__ = "openai._client"
    type(client).__name__ = "AsyncOpenAI"
    response = _responses_response()
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(spec=_async_create)))
    client.responses = SimpleNamespace(create=AsyncMock(spec=_async_create, return_value=response))
    client.with_options.return_value = client
    return client, response


def _mock_anthropic_client() -> MagicMock:
    client = MagicMock(spec=Anthropic)
    type(client).__module__ = "anthropic._client"
    type(client).__name__ = "Anthropic"
    client.messages = SimpleNamespace(create=MagicMock(spec=_sync_create))
    client.with_options.return_value = client
    return client


def _mock_async_anthropic_client() -> MagicMock:
    client = MagicMock(spec=AsyncAnthropic)
    type(client).__module__ = "anthropic._client"
    type(client).__name__ = "AsyncAnthropic"
    client.messages = SimpleNamespace(create=AsyncMock(spec=_async_create))
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
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    solwyn._reporter.report = MagicMock(spec=solwyn._reporter.report)
    solwyn._reporter.report_settlement = MagicMock(spec=solwyn._reporter.report_settlement)
    try:
        yield solwyn
    finally:
        solwyn._reporter._http.close()
        solwyn._budget._http.close()


@asynccontextmanager
async def _async_solwyn(client: object, **overrides: object) -> AsyncIterator[AsyncSolwyn]:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY, "model": "gpt-5.5"}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter.report = MagicMock(spec=solwyn._reporter.report)
    solwyn._reporter.report_settlement = MagicMock(spec=solwyn._reporter.report_settlement)
    try:
        yield solwyn
    finally:
        await solwyn._reporter._http.aclose()
        await solwyn._budget._http.aclose()


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
class TestResponsesInterceptedCallSync:
    def test_default_surface_preserves_chat_interception(self) -> None:
        client, _ = _mock_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        messages = [{"role": "user", "content": "hello"}]

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                result = solwyn._intercepted_call(model="gpt-5.5", messages=messages)

            assert result is response
            client.chat.completions.create.assert_called_once_with(
                model="gpt-5.5", messages=messages
            )
            client.responses.create.assert_not_called()
            check.assert_called_once()
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
            assert event.input_tokens == 21
            assert event.output_tokens == 9
            assert event.service_tier == "default"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    def test_sync_dispatch_defaults_to_chat(self) -> None:
        client, _ = _mock_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        kwargs: dict[str, object] = {"model": "gpt-5.5", "messages": []}

        with _sync_solwyn(client) as solwyn:
            result = solwyn._sync_dispatch(
                solwyn._runtimes[0],
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                result = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="hi there",
                )

            assert result is response
            client.responses.create.assert_called_once()
            client.chat.completions.create.assert_not_called()
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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
                solwyn._runtimes[0].entry.default_params["instructions"] = entry_instructions
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "abcd"}
            if caller_instructions is not None:
                call_kwargs["instructions"] = caller_instructions
            with patch.object(solwyn._budget, "check_budget", new=check):
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
                solwyn._runtimes[0].entry.default_params["max_output_tokens"] = entry_cap
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_cap is not None:
                call_kwargs["max_output_tokens"] = caller_cap
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if cap is not None:
                call_kwargs["max_output_tokens"] = cap
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            solwyn._runtimes[0].entry.default_params.update(
                {
                    "temperature": 0.4,
                    "top_p": 0.6,
                    "max_completion_tokens": 200,
                }
            )
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._budget.release_reservation)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget, "release_reservation", new=release),
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_stream is not None:
                call_kwargs["stream"] = caller_stream
            with patch.object(solwyn._budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    _force_stream=force_stream,
                    **call_kwargs,
                )
            events = list(stream)

            assert events == [delta, terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                stream = solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            solwyn._reporter.report_settlement.assert_not_called()
            assert list(stream) == [terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._reporter.report_settlement.assert_called_once()
            _, event = solwyn._reporter.report_settlement.call_args.args
            assert event.input_tokens == 30
            assert event.output_tokens == 12

    def test_abandoned_stream_settles_once_with_length_estimate(self) -> None:
        client, _ = _mock_openai_client()
        inner = _FakeSyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="partial")]
        )
        client.responses.create.return_value = inner

        with _sync_solwyn(client) as solwyn:
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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

            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
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
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._budget._lease.true_up)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget._lease, "true_up", new=true_up),
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
            solwyn._reporter.report_settlement.assert_called_once()

    def test_adapter_without_responses_seam_fails_before_budget_check(self) -> None:
        client = _mock_anthropic_client()

        with _sync_solwyn(client, model="claude-x") as solwyn:
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
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

    def test_open_primary_with_only_fallback_candidate_releases_reservation(self) -> None:
        client, _ = _mock_openai_client()
        fallback = _mock_anthropic_client()

        with _sync_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            breaker = solwyn._get_circuit_breaker("openai")
            for _ in range(3):
                breaker.record_failure()
            check = MagicMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._budget.release_reservation)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget, "release_reservation", new=release),
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                result = await solwyn._intercepted_call(model="gpt-5.5", messages=messages)

            assert result is response
            client.chat.completions.create.assert_awaited_once_with(
                model="gpt-5.5", messages=messages
            )
            client.responses.create.assert_not_awaited()
            check.assert_awaited_once()
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
            assert event.input_tokens == 21
            assert event.output_tokens == 9
            assert event.service_tier == "default"
            assert confirm.token_details == event.token_details
            assert confirm.service_tier == event.service_tier

    @pytest.mark.asyncio
    async def test_async_dispatch_defaults_to_chat(self) -> None:
        client, _ = _mock_async_openai_client()
        response = _chat_response()
        client.chat.completions.create.return_value = response
        kwargs: dict[str, object] = {"model": "gpt-5.5", "messages": []}

        async with _async_solwyn(client) as solwyn:
            result = await solwyn._async_dispatch(
                solwyn._runtimes[0],
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                result = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="hi there",
                )

            assert result is response
            client.responses.create.assert_awaited_once()
            client.chat.completions.create.assert_not_awaited()
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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
                solwyn._runtimes[0].entry.default_params["instructions"] = entry_instructions
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "abcd"}
            if caller_instructions is not None:
                call_kwargs["instructions"] = caller_instructions
            with patch.object(solwyn._budget, "check_budget", new=check):
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
                solwyn._runtimes[0].entry.default_params["max_output_tokens"] = entry_cap
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_cap is not None:
                call_kwargs["max_output_tokens"] = caller_cap
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if cap is not None:
                call_kwargs["max_output_tokens"] = cap
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            solwyn._runtimes[0].entry.default_params.update(
                {
                    "temperature": 0.4,
                    "top_p": 0.6,
                    "max_completion_tokens": 200,
                }
            )
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._budget.release_reservation)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget, "release_reservation", new=release),
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            call_kwargs: dict[str, object] = {"model": "gpt-5.5", "input": "x"}
            if caller_stream is not None:
                call_kwargs["stream"] = caller_stream
            with patch.object(solwyn._budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    _force_stream=force_stream,
                    **call_kwargs,
                )
            events = [event async for event in stream]

            assert events == [delta, terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._reporter.report_settlement.assert_called_once()
            confirm, event = solwyn._reporter.report_settlement.call_args.args
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
                stream = await solwyn._intercepted_call(
                    _surface="responses",
                    model="gpt-5.5",
                    input="x",
                )

            solwyn._reporter.report_settlement.assert_not_called()
            assert [event async for event in stream] == [terminal]
            sent = client.responses.create.call_args.kwargs
            assert sent["stream"] is True
            assert "stream_options" not in sent
            solwyn._reporter.report_settlement.assert_called_once()
            _, event = solwyn._reporter.report_settlement.call_args.args
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with patch.object(solwyn._budget, "check_budget", new=check):
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

            solwyn._reporter.report_settlement.assert_called_once()
            confirm, settled_event = solwyn._reporter.report_settlement.call_args.args
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
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_lease())
            true_up = MagicMock(spec=solwyn._budget._lease.true_up)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget._lease, "true_up", new=true_up),
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
            solwyn._reporter.report_settlement.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_without_responses_seam_fails_before_budget_check(self) -> None:
        client = _mock_async_anthropic_client()

        async with _async_solwyn(client, model="claude-x") as solwyn:
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
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
    async def test_open_primary_with_only_fallback_candidate_releases_reservation(
        self,
    ) -> None:
        client, _ = _mock_async_openai_client()
        fallback = _mock_async_anthropic_client()

        async with _async_solwyn(client, fallback=[(fallback, "claude-x")]) as solwyn:
            breaker = solwyn._get_circuit_breaker("openai")
            for _ in range(3):
                breaker.record_failure()
            check = AsyncMock(spec=solwyn._budget.check_budget, return_value=_allow_budget())
            release = MagicMock(spec=solwyn._budget.release_reservation)
            with (
                patch.object(solwyn._budget, "check_budget", new=check),
                patch.object(solwyn._budget, "release_reservation", new=release),
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
