"""Client-level tests for OpenAI-compatible providers.

End-to-end through Solwyn/AsyncSolwyn interception: attribution (the served
provider name on budget checks and metadata events), the ``provider=``
override, streaming usage injection policy at dispatch, the explicit
estimated-usage fallback, and failover semantics (same-dialect passthrough vs
cross-dialect translation).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import VALID_API_KEY, VALID_PROJECT_ID, make_mock_client

from solwyn._types import CallStatus, ProviderName
from solwyn.client import AsyncSolwyn, Solwyn
from solwyn.exceptions import ConfigurationError, UntranslatableRequestError
from solwyn.providers._errors import Disposition, classify_exception

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compat_client(base_url: str) -> MagicMock:
    client = make_mock_client()
    client.base_url = base_url
    return client


def _create_completion(**kwargs: object) -> object:
    raise NotImplementedError


class _ZaiClient:
    """Real client shell with spec'd mocks only at the OpenAI call boundary."""

    __module__ = "openai._client"

    def __init__(self, base_url: str = "https://api.z.ai/api/paas/v4") -> None:
        self.base_url = base_url
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(spec=_create_completion))
        )
        self.with_options = MagicMock(spec=self._with_options, return_value=self)

    def _with_options(self, **kwargs: object) -> _ZaiClient:
        return self


class _TypedTogetherCompletions:
    """Together v2-shaped call boundary with a deliberately strict signature."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
    ) -> object:
        self.calls.append({"model": model, "messages": messages, "stream": stream})
        return self.response


def _native_together_client(response: object) -> tuple[object, _TypedTogetherCompletions]:
    """Duck-typed native Together client without importing the provider SDK."""
    client_type = type("Together", (), {"__module__": "together"})
    completions = _TypedTogetherCompletions(response)
    client = client_type()
    client.chat = SimpleNamespace(completions=completions)
    return client, completions


def _allow_budget(reservation_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        allowed=True,
        reservation_id=reservation_id,
        project_id=VALID_PROJECT_ID,
        price_hints=None,
    )


def _make_solwyn(client: object, **overrides: object) -> Solwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    with patch("solwyn.reporter.MetadataReporter._flush_loop"):
        solwyn = Solwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter._shutdown.set()
    solwyn._reporter._thread.join(timeout=2.0)
    solwyn._reporter.report = MagicMock()
    solwyn._reporter.report_settlement = MagicMock()
    return solwyn


def _response(
    *,
    prompt_tokens: int | None = 11,
    completion_tokens: int | None = 7,
    cached_tokens: int = 0,
    content: str = "ok",
) -> SimpleNamespace:
    """A Chat Completions-shaped non-streaming response.

    ``prompt_tokens=None`` means NO usage block at all (the missing-usage case).
    """
    usage = None
    if prompt_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            completion_tokens_details=None,
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
    )


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)],
        usage=None,
    )


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


_REQUEST: dict[str, Any] = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "hello there"}],
}


def _reported_events(solwyn: Solwyn) -> list[Any]:
    return [call.args[0] for call in solwyn._reporter.report.call_args_list]


def _open_breaker(solwyn: Solwyn, provider: str) -> None:
    cb = solwyn._get_circuit_breaker(provider)
    for _ in range(cb.failure_threshold):
        cb.record_failure()


# ---------------------------------------------------------------------------
# Attribution: detected compat provider flows to budget + metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompatAttribution:
    def test_groq_base_url_attributes_to_groq(self) -> None:
        client = _compat_client("https://api.groq.com/openai/v1")
        client.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(client)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()) as check:
            solwyn.chat.completions.create(**_REQUEST)

        # Budget pre-flight names the detected provider, not "openai".
        assert check.call_args.kwargs["provider"] == "groq"
        events = _reported_events(solwyn)
        assert len(events) == 1
        assert events[0].provider == ProviderName.GROQ
        assert events[0].status == CallStatus.SUCCESS
        assert events[0].input_tokens == 11
        assert events[0].token_details.is_estimated is False

    def test_zai_base_url_attributes_usage_to_zai(self) -> None:
        client = _ZaiClient()
        client.chat.completions.create.return_value = _response(cached_tokens=4)
        solwyn = _make_solwyn(client)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()) as check:
            solwyn.chat.completions.create(**_REQUEST)

        assert check.call_args.kwargs["provider"] == "zai"
        event = _reported_events(solwyn)[0]
        assert event.provider == "zai"
        assert event.token_details.cached_input_tokens == 4

    def test_openrouter_model_slug_passes_through_verbatim(self) -> None:
        """Pricing identity: the API receives (openrouter, vendor/model)."""
        client = _compat_client("https://openrouter.ai/api/v1")
        client.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(client)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                model="anthropic/claude-sonnet-4.5",
                messages=[{"role": "user", "content": "hi"}],
            )

        event = _reported_events(solwyn)[0]
        assert event.provider == ProviderName.OPENROUTER
        assert event.model == "anthropic/claude-sonnet-4.5"

    def test_budget_chain_hint_carries_compat_fallbacks(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("https://openrouter.ai/api/v1")
        primary.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(fallback, "openrouter/auto")])

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()) as check:
            solwyn.chat.completions.create(**_REQUEST)

        assert check.call_args.kwargs["fallback_providers"] == ["openrouter"]
        assert check.call_args.kwargs["fallback_models"] == ["openrouter/auto"]


# ---------------------------------------------------------------------------
# provider= override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderOverride:
    def test_override_relabels_within_dialect(self) -> None:
        client = _compat_client("http://localhost:9999/v1")  # would be generic
        client.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(client, provider="vllm")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_REQUEST)

        assert _reported_events(solwyn)[0].provider == ProviderName.VLLM

    def test_zai_override_relabels_generic_compat_client(self) -> None:
        client = _ZaiClient("https://api.some-new-vendor.example/v1")
        client.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(client, provider="zai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_REQUEST)

        assert _reported_events(solwyn)[0].provider == "zai"

    def test_unknown_override_raises_configuration_error(self) -> None:
        client = _compat_client("http://localhost:9999/v1")
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            _make_solwyn(client, provider="not-a-provider")

    def test_pin_rejects_anthropic_client_through_surface_context_validation(self) -> None:
        client = make_mock_client(module="anthropic._client", name="Anthropic")
        client.messages.create = MagicMock()
        with pytest.raises(ConfigurationError) as exc_info:
            _make_solwyn(client, provider="groq")
        assert exc_info.value.field == "client"
        assert "groq/anthropic_sdk/sync" in str(exc_info.value)
        client.messages.create.assert_not_called()

    def test_pin_rejects_an_unrecognized_client_shape(self) -> None:
        UnknownClient = type("UnknownClient", (), {"__module__": "totally_unknown_sdk._client"})
        with pytest.raises(ConfigurationError) as exc_info:
            _make_solwyn(UnknownClient(), provider="groq")
        assert exc_info.value.field == "client"
        assert "undeclared_sdk" in str(exc_info.value)

    def test_fallback_spec_provider_override(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("http://localhost:9999/v1")
        solwyn = _make_solwyn(primary, fallback=[(fallback, "my-model", {}, "vllm")])

        assert solwyn._runtimes[1].entry.provider == ProviderName.VLLM

    def test_fallback_provider_pin_rejects_wrong_client_shape(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = make_mock_client(module="anthropic._client", name="Anthropic")

        with pytest.raises(ConfigurationError) as exc_info:
            _make_solwyn(primary, fallback=[(fallback, "my-model", {}, "vllm")])

        assert exc_info.value.field == "client"
        assert "vllm/anthropic_sdk/sync" in str(exc_info.value)

    def test_fallback_spec_non_string_provider_raises(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("http://localhost:9999/v1")
        with pytest.raises(ConfigurationError, match="provider"):
            _make_solwyn(primary, fallback=[(fallback, "my-model", {}, 42)])


# ---------------------------------------------------------------------------
# Streaming dispatch: per-provider include_usage policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingUsagePolicy:
    def _stream_call_kwargs(self, base_url: str) -> dict[str, Any]:
        client = _compat_client(base_url)
        client.chat.completions.create.return_value = iter([_text_chunk("hi"), _usage_chunk(5, 2)])
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**_REQUEST, stream=True)
            list(stream)
        return client.chat.completions.create.call_args.kwargs

    def test_groq_stream_injects_include_usage(self) -> None:
        kwargs = self._stream_call_kwargs("https://api.groq.com/openai/v1")
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_xai_stream_sends_no_stream_options(self) -> None:
        """xAI errors on stream_options — it must never reach the wire."""
        kwargs = self._stream_call_kwargs("https://api.x.ai/v1")
        assert "stream_options" not in kwargs

    def test_mistral_stream_sends_no_stream_options(self) -> None:
        kwargs = self._stream_call_kwargs("https://api.mistral.ai/v1")
        assert "stream_options" not in kwargs

    def test_zai_stream_injects_include_usage(self) -> None:
        kwargs = self._stream_call_kwargs("https://api.z.ai/api/paas/v4")
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_stream_usage_settles_from_final_chunk(self) -> None:
        client = _compat_client("https://api.groq.com/openai/v1")
        client.chat.completions.create.return_value = iter(
            [_text_chunk("hello"), _usage_chunk(9, 4)]
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            list(solwyn.chat.completions.create(**_REQUEST, stream=True))

        event = _reported_events(solwyn)[0]
        assert event.status == CallStatus.SUCCESS
        assert event.input_tokens == 9
        assert event.output_tokens == 4
        assert event.token_details.is_estimated is False


# ---------------------------------------------------------------------------
# Estimated-usage fallback (the explicit degradation path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimatedUsageFallback:
    def test_non_streaming_missing_usage_reports_flagged_estimate(self) -> None:
        client = _compat_client("http://localhost:1234/v1")  # lmstudio
        client.chat.completions.create.return_value = _response(
            prompt_tokens=None, content="z" * 80
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget("res_1")):
            solwyn.chat.completions.create(**_REQUEST)

        # A reservation was granted, so settlement rides report_settlement(confirm,
        # event) — the SUCCESS event travels WITH the confirm.
        solwyn._reporter.report_settlement.assert_called_once()
        confirm, event = solwyn._reporter.report_settlement.call_args.args
        assert event.token_details.is_estimated is True
        # Input from the pre-call length estimate ("hello there" = 11 chars).
        assert event.input_tokens > 0
        assert event.output_tokens == 20  # 80 chars / 4.0
        # The budget confirm settles on the SAME flagged estimate.
        assert confirm.token_details.is_estimated is True
        assert confirm.token_details.output_tokens == 20

    def test_streaming_missing_usage_reports_flagged_estimate(self) -> None:
        client = _compat_client("http://localhost:1234/v1")
        client.chat.completions.create.return_value = iter(
            [_text_chunk("a" * 30), _text_chunk("b" * 10)]
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            list(solwyn.chat.completions.create(**_REQUEST, stream=True))

        event = _reported_events(solwyn)[0]
        assert event.status == CallStatus.SUCCESS
        assert event.token_details.is_estimated is True
        assert event.input_tokens > 0  # pre-call estimate, not zero
        assert event.output_tokens == 10  # 40 chars / 4.0

    def test_zai_streaming_missing_usage_reports_flagged_estimate(self) -> None:
        client = _ZaiClient()
        client.chat.completions.create.return_value = iter(
            [_text_chunk("a" * 30), _text_chunk("b" * 10)]
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            list(solwyn.chat.completions.create(**_REQUEST, stream=True))

        event = _reported_events(solwyn)[0]
        assert event.provider == "zai"
        assert event.token_details.is_estimated is True
        assert event.input_tokens > 0
        assert event.output_tokens == 10

    def test_provider_reported_usage_is_never_overridden(self) -> None:
        client = _compat_client("http://localhost:1234/v1")
        client.chat.completions.create.return_value = _response(
            prompt_tokens=3, completion_tokens=1
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_REQUEST)

        event = _reported_events(solwyn)[0]
        assert event.token_details.is_estimated is False
        assert event.input_tokens == 3

    def test_negative_usage_degrades_to_estimate_not_raise(self) -> None:
        """A misbehaving endpoint reporting negative counts must not raise out
        of the success path (after record_success, before confirm) — it
        degrades to the flagged estimation tier and the caller gets the
        response."""
        client = _compat_client("http://localhost:1234/v1")
        client.chat.completions.create.return_value = _response(
            prompt_tokens=-1, completion_tokens=-1, content="z" * 80
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_REQUEST)

        assert result.choices[0].message.content == "z" * 80
        event = _reported_events(solwyn)[0]
        assert event.status == CallStatus.SUCCESS
        assert event.token_details.is_estimated is True
        assert event.output_tokens == 20  # 80 chars / 4.0

    def test_non_int_usage_degrades_to_estimate_not_raise(self) -> None:
        response = _response(content="z" * 80)
        response.usage.prompt_tokens = "abc"  # value-level garbage
        response.usage.completion_tokens = "def"
        client = _compat_client("http://localhost:1234/v1")
        client.chat.completions.create.return_value = response
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(**_REQUEST)

        event = _reported_events(solwyn)[0]
        assert event.status == CallStatus.SUCCESS
        assert event.token_details.is_estimated is True
        assert event.output_tokens == 20

    def test_streaming_negative_usage_falls_to_estimation_tier(self) -> None:
        """A streamed usage block with negative counts must not convert a
        deliverable stream into a settle-as-failure (breaker hit against a
        content-healthy provider) — it falls to the estimation tier."""
        client = _compat_client("https://api.groq.com/openai/v1")
        client.chat.completions.create.return_value = iter(
            [_text_chunk("x" * 40), _usage_chunk(-1, -2)]
        )
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            chunks = list(solwyn.chat.completions.create(**_REQUEST, stream=True))

        assert len(chunks) == 2  # the stream was delivered intact
        events = _reported_events(solwyn)
        assert len(events) == 1
        assert events[0].status == CallStatus.SUCCESS
        assert events[0].token_details.is_estimated is True
        assert events[0].output_tokens == 10  # 40 chars / 4.0


# ---------------------------------------------------------------------------
# Failover: same-dialect passthrough vs cross-dialect translation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompatFailover:
    def test_native_together_target_serves_same_dialect_model_swap(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        response = _response()
        together, completions = _native_together_client(response)
        solwyn = _make_solwyn(
            primary,
            fallback=[(together, "meta-llama/Llama-3.3-70B-Instruct-Turbo")],
        )
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_REQUEST)

        assert result is response
        assert primary.chat.completions.create.call_count == 0
        assert completions.calls == [
            {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "messages": _REQUEST["messages"],
                "stream": False,
            }
        ]
        event = _reported_events(solwyn)[0]
        assert event.provider == ProviderName.TOGETHER
        assert event.is_provider_fallback is True

    def test_native_together_failover_strips_caller_stream_options(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        together, completions = _native_together_client(iter([_usage_chunk(5, 2)]))
        solwyn = _make_solwyn(
            primary,
            fallback=[(together, "meta-llama/Llama-3.3-70B-Instruct-Turbo")],
        )
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            chunks = list(
                solwyn.chat.completions.create(
                    **_REQUEST,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )

        assert len(chunks) == 1
        assert completions.calls == [
            {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "messages": _REQUEST["messages"],
                "stream": True,
            }
        ]
        event = _reported_events(solwyn)[0]
        assert event.provider == ProviderName.TOGETHER
        assert event.input_tokens == 5

    def test_native_together_rejects_openai_only_kwarg_and_stops_chain(self) -> None:
        primary = _compat_client("https://api.groq.com/openai/v1")
        primary.chat.completions.create.side_effect = httpx.ConnectError("refused")
        together, completions = _native_together_client(_response())
        later = _compat_client("https://openrouter.ai/api/v1")
        later.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(
            primary,
            fallback=[
                (together, "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                (later, "openrouter/auto"),
            ],
        )

        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(TypeError, match="store") as exc_info,
        ):
            solwyn.chat.completions.create(**_REQUEST, store=True)

        assert classify_exception(exc_info.value) is Disposition.FAIL_FAST
        assert completions.calls == []
        later.chat.completions.create.assert_not_called()

    def test_same_dialect_failover_passes_tools_through(self) -> None:
        """Groq -> OpenRouter is native passthrough: tools survive, no
        canonical-subset restriction, even when streaming."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("https://openrouter.ai/api/v1")
        fallback.chat.completions.create.return_value = iter(
            [_text_chunk("hi"), _usage_chunk(6, 2)]
        )
        solwyn = _make_solwyn(primary, fallback=[(fallback, "openrouter/auto")])
        _open_breaker(solwyn, "groq")

        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            stream = solwyn.chat.completions.create(**_REQUEST, stream=True, tools=tools)
            chunks = list(stream)

        primary.chat.completions.create.assert_not_called()
        called = fallback.chat.completions.create.call_args.kwargs
        assert called["model"] == "openrouter/auto"
        assert called["tools"] == tools
        # OpenRouter profile: stream_options never sent (deprecated there).
        assert "stream_options" not in called
        # Same dialect: chunks pass through untranslated.
        assert chunks[0].choices[0].delta.content == "hi"

        event = _reported_events(solwyn)[0]
        assert event.provider == ProviderName.OPENROUTER
        assert event.is_provider_fallback is True
        assert event.requested_provider == ProviderName.GROQ
        assert event.input_tokens == 6

    def test_cross_dialect_failover_translates_to_anthropic(self) -> None:
        """Groq -> Anthropic crosses dialects: the translation contract runs
        and the response normalizes back to the caller's OpenAI shape."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        anthropic = make_mock_client(module="anthropic._client", name="Anthropic")
        anthropic.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="from claude")],
            stop_reason="end_turn",
            model="claude-sonnet-5",
            usage=SimpleNamespace(input_tokens=13, output_tokens=5),
        )
        solwyn = _make_solwyn(
            primary,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 128})],
        )
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            result = solwyn.chat.completions.create(**_REQUEST)

        anthropic.messages.create.assert_called_once()
        # Normalized back to the caller's (OpenAI) dialect.
        assert result.choices[0].message.content == "from claude"
        event = _reported_events(solwyn)[0]
        assert event.provider == ProviderName.ANTHROPIC
        assert event.is_provider_fallback is True

    def test_primary_caller_stream_options_not_stripped(self) -> None:
        """Drop-in contract: the caller's explicit stream_options for THEIR OWN
        endpoint reaches the wire even when the profile says unsupported."""
        client = _compat_client("https://api.x.ai/v1")
        client.chat.completions.create.return_value = iter([_usage_chunk(5, 2)])
        solwyn = _make_solwyn(client)
        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            list(
                solwyn.chat.completions.create(
                    **_REQUEST, stream=True, stream_options={"include_usage": True}
                )
            )
        called = client.chat.completions.create.call_args.kwargs
        assert called["stream_options"] == {"include_usage": True}

    def test_failover_hop_strips_stream_options_for_strict_target(self) -> None:
        """The same caller option IS stripped when a failover hop lands on a
        strict-validation provider it was never meant for."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("https://api.mistral.ai/v1")
        fallback.chat.completions.create.return_value = iter([_usage_chunk(5, 2)])
        solwyn = _make_solwyn(primary, fallback=[(fallback, "mistral-large-latest")])
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            list(
                solwyn.chat.completions.create(
                    **_REQUEST, stream=True, stream_options={"include_usage": True}
                )
            )
        called = fallback.chat.completions.create.call_args.kwargs
        assert "stream_options" not in called

    def test_same_dialect_failover_rewrites_max_completion_tokens(self) -> None:
        """kwargs authored for an OpenAI model (max_completion_tokens) must hit
        a compat target with the legacy max_tokens key — not 4xx the chain."""
        primary = make_mock_client()  # plain OpenAI (no base_url override)
        fallback = _compat_client("https://api.deepseek.com/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(fallback, "deepseek-chat")])
        _open_breaker(solwyn, "openai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=256,
            )

        called = fallback.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" not in called
        assert called["max_tokens"] == 256

    def test_caller_max_completion_tokens_beats_entry_default_max_tokens(self) -> None:
        """Per-call kwargs > per-entry default_params: the caller's output cap
        (a cost control) must survive the legacy-key rewrite, not be silently
        replaced by a larger fallback-entry default."""
        primary = make_mock_client()
        fallback = _compat_client("https://api.deepseek.com/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(fallback, "deepseek-chat", {"max_tokens": 4096})])
        _open_breaker(solwyn, "openai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=256,
            )

        called = fallback.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" not in called
        assert called["max_tokens"] == 256

    def test_caller_max_tokens_beats_entry_default_max_completion_tokens(self) -> None:
        """Mirror case of the test above: an entry-default max_completion_tokens
        must not collapse onto the per-call max_tokens during the legacy-key
        rewrite and silently raise the caller's output cap. Each kwargs layer
        is normalized BEFORE the precedence merge, so per-call wins regardless
        of which key either side used."""
        primary = make_mock_client()
        fallback = _compat_client("https://api.deepseek.com/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(
            primary,
            fallback=[(fallback, "deepseek-chat", {"max_completion_tokens": 4096})],
        )
        _open_breaker(solwyn, "openai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=256,
            )

        called = fallback.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" not in called
        assert called["max_tokens"] == 256

    def test_both_caps_per_call_modern_key_wins(self) -> None:
        """Both caps in the SAME source (per-call): max_completion_tokens wins,
        per the single-source rule documented on _with_legacy_max_tokens_key.
        OpenAI itself rejects both-present requests, so on a failover hop we
        keep the modern key's value rather than guess."""
        primary = make_mock_client()
        fallback = _compat_client("https://api.deepseek.com/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(fallback, "deepseek-chat")])
        _open_breaker(solwyn, "openai")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=256,
                max_tokens=512,
            )

        called = fallback.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" not in called
        assert called["max_tokens"] == 256

    def test_same_dialect_failover_strips_endpoint_scoped_params(self) -> None:
        """extra_headers/extra_query/extra_body carry gateway credentials and
        vendor extensions authored for the ORIGINAL target — they must never
        be forwarded to a different vendor on a same-dialect hop."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("https://openrouter.ai/api/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(fallback, "openrouter/auto")])
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                **_REQUEST,
                extra_headers={"Helicone-Auth": "Bearer sk-helicone-secret"},
                extra_query={"api-version": "2024-06-01"},
                extra_body={"vendor_knob": 1},
            )

        called = fallback.chat.completions.create.call_args.kwargs
        assert "extra_headers" not in called
        assert "extra_query" not in called
        assert "extra_body" not in called

    def test_primary_keeps_endpoint_scoped_params(self) -> None:
        """Drop-in contract: on the caller's own configured target their
        per-call transport params reach the wire untouched."""
        client = _compat_client("https://api.groq.com/openai/v1")
        client.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(client)

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                **_REQUEST,
                extra_headers={"Helicone-Auth": "Bearer sk-helicone-secret"},
                extra_query={"api-version": "2024-06-01"},
                extra_body={"vendor_knob": 1},
            )

        called = client.chat.completions.create.call_args.kwargs
        assert called["extra_headers"] == {"Helicone-Auth": "Bearer sk-helicone-secret"}
        assert called["extra_query"] == {"api-version": "2024-06-01"}
        assert called["extra_body"] == {"vendor_knob": 1}

    def test_same_provider_model_fallback_keeps_endpoint_scoped_params(self) -> None:
        """A same-provider model swap stays on the caller's own endpoint —
        their transport params still apply."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        swap = _compat_client("https://api.groq.com/openai/v1")
        primary.chat.completions.create.side_effect = httpx.ConnectError("refused")
        swap.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(primary, fallback=[(swap, "llama-3.1-8b-instant")])

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                **_REQUEST, extra_headers={"Helicone-Auth": "Bearer sk-helicone-secret"}
            )

        called = swap.chat.completions.create.call_args.kwargs
        assert called["model"] == "llama-3.1-8b-instant"
        assert called["extra_headers"] == {"Helicone-Auth": "Bearer sk-helicone-secret"}

    def test_same_dialect_failover_keeps_target_entry_default_extras(self) -> None:
        """The TARGET entry's own default_params extras were authored for that
        endpoint (e.g. OpenRouter attribution headers) and still apply."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        fallback = _compat_client("https://openrouter.ai/api/v1")
        fallback.chat.completions.create.return_value = _response()
        solwyn = _make_solwyn(
            primary,
            fallback=[(fallback, "openrouter/auto", {"extra_headers": {"X-Title": "MyApp"}})],
        )
        _open_breaker(solwyn, "groq")

        with patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()):
            solwyn.chat.completions.create(
                **_REQUEST, extra_headers={"Helicone-Auth": "Bearer sk-helicone-secret"}
            )

        called = fallback.chat.completions.create.call_args.kwargs
        # The caller's primary-scoped headers are gone; the entry's own stay.
        assert called["extra_headers"] == {"X-Title": "MyApp"}

    def test_cross_dialect_tool_stream_still_fails_loud(self) -> None:
        """Tool-using STREAMS still cannot cross dialects — fail before dispatch."""
        primary = _compat_client("https://api.groq.com/openai/v1")
        anthropic = make_mock_client(module="anthropic._client", name="Anthropic")
        solwyn = _make_solwyn(
            primary,
            fallback=[(anthropic, "claude-sonnet-5", {"max_tokens": 128})],
        )
        _open_breaker(solwyn, "groq")

        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        with (
            patch.object(solwyn._budget, "check_budget", return_value=_allow_budget()),
            pytest.raises(UntranslatableRequestError),
        ):
            solwyn.chat.completions.create(**_REQUEST, stream=True, tools=tools)
        anthropic.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Async mirror (the load-bearing subset)
# ---------------------------------------------------------------------------


def _async_iter(chunks: list[Any]) -> Any:
    async def _gen() -> Any:
        for c in chunks:
            yield c

    return _gen()


def _make_async_solwyn(client: object, **overrides: object) -> AsyncSolwyn:
    defaults: dict[str, object] = {"api_key": VALID_API_KEY}
    defaults.update(overrides)
    solwyn = AsyncSolwyn(client, **defaults)  # type: ignore[arg-type]
    solwyn._reporter.report = MagicMock()
    solwyn._reporter.report_settlement = MagicMock()
    return solwyn


def _async_reported_events(solwyn: AsyncSolwyn) -> list[Any]:
    return [call.args[0] for call in solwyn._reporter.report.call_args_list]


@pytest.mark.unit
class TestAsyncCompat:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_groq_attribution_and_usage(self) -> None:
        client = make_mock_client()
        client.base_url = "https://api.groq.com/openai/v1"
        client.chat.completions.create = AsyncMock(return_value=_response())
        solwyn = _make_async_solwyn(client)

        with patch.object(
            solwyn._budget, "check_budget", AsyncMock(return_value=_allow_budget())
        ) as check:
            await solwyn.chat.completions.create(**_REQUEST)

        assert check.call_args.kwargs["provider"] == "groq"
        event = _async_reported_events(solwyn)[0]
        assert event.provider == ProviderName.GROQ
        assert event.input_tokens == 11

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_streaming_missing_usage_estimates(self) -> None:
        client = make_mock_client()
        client.base_url = "http://localhost:11434/v1"  # ollama
        client.chat.completions.create = AsyncMock(
            return_value=_async_iter([_text_chunk("x" * 40)])
        )
        solwyn = _make_async_solwyn(client)

        with patch.object(solwyn._budget, "check_budget", AsyncMock(return_value=_allow_budget())):
            stream = await solwyn.chat.completions.create(**_REQUEST, stream=True)
            async for _ in stream:
                pass

        event = _async_reported_events(solwyn)[0]
        assert event.provider == ProviderName.OLLAMA
        assert event.token_details.is_estimated is True
        assert event.input_tokens > 0
        assert event.output_tokens == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_provider_override(self) -> None:
        client = make_mock_client(name="AsyncOpenAI")
        client.base_url = "http://localhost:9999/v1"
        client.chat.completions.create = AsyncMock(return_value=_response())
        solwyn = _make_async_solwyn(client, provider="vllm")

        with patch.object(solwyn._budget, "check_budget", AsyncMock(return_value=_allow_budget())):
            await solwyn.chat.completions.create(**_REQUEST)

        assert _async_reported_events(solwyn)[0].provider == ProviderName.VLLM
