"""WS-13: every per-call budget check carries a TRUE output bound.

The check sends ``estimated_output_bound``, computed exactly as the lease path
sizes its reservation (largest explicit cap across the chain, else
``lease_output_bound_default``). An uncapped hop gets that default in its
dialect's native cap field, so the call cannot spend more output than the
check priced. An explicit positive cap is never touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from conftest import VALID_PROJECT_ID, make_mock_client

from solwyn._base import _reset_unmetered_spend_warnings
from solwyn._constants import ORDINARY_TOKEN_COUNT_MAX
from solwyn._lease import DEFAULT_OUTPUT_BOUND
from solwyn._types import MediaUsage
from solwyn.budget import _BudgetEnforcerBase
from solwyn.exceptions import BudgetExceededError
from solwyn.testing import FakeControlPlane


@pytest.fixture(autouse=True)
def _reset_spend_surface_latch() -> None:
    _reset_unmetered_spend_warnings()


MESSAGES = [{"role": "user", "content": "hello"}]


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


def _responses_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_1",
        object="response",
        status="completed",
        model="gpt-5.5",
        output=[],
        usage=SimpleNamespace(
            input_tokens=21,
            output_tokens=9,
            total_tokens=30,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
        service_tier="default",
    )


def _openai_client(base_url: str | None = None) -> MagicMock:
    client = make_mock_client()
    if base_url is not None:
        client.base_url = base_url
    client.chat.completions.create.return_value = _chat_response()
    return client


def _sync_call(**_kwargs: object) -> object:
    return None


def _responses_client() -> MagicMock:
    # Responses admission reads the real SDK's capability graph.
    from openai import OpenAI

    client = MagicMock(spec=OpenAI)
    type(client).__module__ = "openai._client"
    type(client).__name__ = "OpenAI"
    client.responses = SimpleNamespace(
        create=MagicMock(spec=_sync_call, return_value=_responses_response()),
    )
    client.with_options.return_value = client
    return client


def _async_openai_client() -> MagicMock:
    client = make_mock_client(name="AsyncOpenAI")
    client.chat.completions.create = AsyncMock(return_value=_chat_response())
    return client


def _anthropic_client() -> MagicMock:
    client = make_mock_client("anthropic._client", "Anthropic")
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello")],
        stop_reason="end_turn",
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0),
    )
    return client


def _google_client() -> MagicMock:
    client = make_mock_client("google.genai._client", "Client")
    client.models.generate_content.return_value = SimpleNamespace(
        text="Hello",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=0,
            cached_content_token_count=0,
            tool_use_prompt_token_count=0,
        ),
    )
    return client


def _bedrock_client() -> MagicMock:
    client = make_mock_client("botocore.client", "BedrockRuntime")
    client.meta.service_model.service_name = "bedrock-runtime"
    client.meta.region_name = "us-east-1"
    client.converse.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
        "metrics": {"latencyMs": 50},
    }
    return client


_DIALECT_CALLS: dict[str, tuple[Any, Any, Any]] = {
    "anthropic": (
        _anthropic_client,
        lambda w: w._solwyn_client.messages.create,
        lambda w: w.messages.create(model="claude-sonnet-5", messages=MESSAGES),
    ),
    "google": (
        _google_client,
        lambda w: w._solwyn_client.models.generate_content,
        lambda w: w.models.generate_content(model="gemini-3-pro", contents="hello"),
    ),
    "bedrock": (
        _bedrock_client,
        lambda w: w._solwyn_client.converse,
        lambda w: w.converse(modelId="us.anthropic.claude-sonnet-4-5-v1:0", messages=MESSAGES),
    ),
    "responses": (
        _responses_client,
        lambda w: w._solwyn_client.responses.create,
        lambda w: w.responses.create(model="gpt-5.5", input="hello"),
    ),
}


def _sent_kwargs(method: MagicMock) -> dict[str, Any]:
    method.assert_called_once()
    return dict(method.call_args.kwargs)


def _only_check_bound(plane: FakeControlPlane) -> int | None:
    assert len(plane.checks) == 1
    return plane.checks[0].estimated_output_bound


@pytest.mark.unit
class TestCheckSendsTheBoundPerDialect:
    """The wire bound equals the explicit cap, or the default that is injected."""

    @pytest.mark.parametrize(
        ("model", "cap_key"),
        [("gpt-5.5", "max_completion_tokens"), ("gpt-4o", "max_tokens")],
    )
    def test_openai_chat_uncapped_sends_and_injects_default(self, model: str, cap_key: str) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model=model, messages=MESSAGES)

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.chat.completions.create)
        assert sent[cap_key] == DEFAULT_OUTPUT_BOUND
        assert ({"max_tokens", "max_completion_tokens"} & set(sent)) == {cap_key}

    @pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
    def test_openai_chat_explicit_cap_is_sent_as_is(self, key: str) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES, **{key: 10_000})

        assert _only_check_bound(plane) == 10_000
        sent = _sent_kwargs(client.chat.completions.create)
        # Existing normalization moves max_tokens to the modern key; the value
        # is the caller's, never the default.
        assert sent["max_completion_tokens"] == 10_000
        assert "max_tokens" not in sent

    @pytest.mark.parametrize("bad_cap", [0, -5, None, 1.5, "100", True])
    def test_openai_chat_malformed_cap_counts_as_uncapped(self, bad_cap: object) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-4o", messages=MESSAGES, max_tokens=bad_cap)

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.chat.completions.create)
        assert sent["max_tokens"] == DEFAULT_OUTPUT_BOUND
        assert "max_completion_tokens" not in sent

    def test_openai_compatible_uncapped_injects_legacy_key(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client(base_url="https://api.groq.com/openai/v1")
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="llama-3.3-70b-versatile", messages=MESSAGES)

        assert plane.checks[0].provider.value == "groq"
        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.chat.completions.create)
        assert sent["max_tokens"] == DEFAULT_OUTPUT_BOUND
        assert "max_completion_tokens" not in sent

    @pytest.mark.parametrize(("cap", "expected"), [(None, DEFAULT_OUTPUT_BOUND), (900, 900)])
    def test_openai_responses(self, cap: int | None, expected: int) -> None:
        plane = FakeControlPlane()
        client = _responses_client()
        extra = {} if cap is None else {"max_output_tokens": cap}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.responses.create(model="gpt-5.5", input="hello", **extra)

        assert _only_check_bound(plane) == expected
        assert _sent_kwargs(client.responses.create)["max_output_tokens"] == expected

    @pytest.mark.parametrize(("cap", "expected"), [(None, DEFAULT_OUTPUT_BOUND), (700, 700)])
    def test_anthropic(self, cap: int | None, expected: int) -> None:
        plane = FakeControlPlane()
        client = _anthropic_client()
        extra = {} if cap is None else {"max_tokens": cap}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.messages.create(model="claude-sonnet-5", messages=MESSAGES, **extra)

        assert _only_check_bound(plane) == expected
        assert _sent_kwargs(client.messages.create)["max_tokens"] == expected

    def test_google_uncapped_gets_config_cap(self) -> None:
        plane = FakeControlPlane()
        client = _google_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.models.generate_content(model="gemini-3-pro", contents="hello")

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.models.generate_content)
        assert sent["config"]["max_output_tokens"] == DEFAULT_OUTPUT_BOUND

    def test_google_config_without_cap_is_copied_not_mutated(self) -> None:
        plane = FakeControlPlane()
        client = _google_client()
        config = {"temperature": 0.1}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.models.generate_content(model="gemini-3-pro", contents="hello", config=config)

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.models.generate_content)
        assert sent["config"]["max_output_tokens"] == DEFAULT_OUTPUT_BOUND
        assert sent["config"]["temperature"] == 0.1
        assert config == {"temperature": 0.1}

    def test_google_explicit_cap_is_sent_as_is(self) -> None:
        plane = FakeControlPlane()
        client = _google_client()
        config = {"temperature": 0.1, "max_output_tokens": 300}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.models.generate_content(model="gemini-3-pro", contents="hello", config=config)

        assert _only_check_bound(plane) == 300
        assert _sent_kwargs(client.models.generate_content)["config"]["max_output_tokens"] == 300
        assert config == {"temperature": 0.1, "max_output_tokens": 300}

    def test_bedrock_uncapped_gets_inference_config(self) -> None:
        plane = FakeControlPlane()
        client = _bedrock_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.converse(modelId="us.anthropic.claude-sonnet-4-5-v1:0", messages=MESSAGES)

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        sent = _sent_kwargs(client.converse)
        assert sent["inferenceConfig"] == {"maxTokens": DEFAULT_OUTPUT_BOUND}

    def test_bedrock_inference_config_without_cap_is_copied_not_mutated(self) -> None:
        plane = FakeControlPlane()
        client = _bedrock_client()
        inference = {"temperature": 0.2}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.converse(
                modelId="us.anthropic.claude-sonnet-4-5-v1:0",
                messages=MESSAGES,
                inferenceConfig=inference,
            )

        sent = _sent_kwargs(client.converse)
        assert sent["inferenceConfig"] == {"temperature": 0.2, "maxTokens": DEFAULT_OUTPUT_BOUND}
        assert inference == {"temperature": 0.2}

    def test_bedrock_explicit_cap_is_sent_as_is(self) -> None:
        plane = FakeControlPlane()
        client = _bedrock_client()
        inference = {"maxTokens": 256, "temperature": 0.2}
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.converse(
                modelId="us.anthropic.claude-sonnet-4-5-v1:0",
                messages=MESSAGES,
                inferenceConfig=inference,
            )

        assert _only_check_bound(plane) == 256
        assert _sent_kwargs(client.converse)["inferenceConfig"] is inference

    def test_configured_default_is_both_sent_and_injected(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False, lease_output_bound_default=1_000) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert _only_check_bound(plane) == 1_000
        assert _sent_kwargs(client.chat.completions.create)["max_completion_tokens"] == 1_000

    def test_bound_is_the_largest_cap_across_the_chain(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(
            client,
            lease_enabled=False,
            fallback=[(client, "gpt-5.5-mini", {"max_completion_tokens": 8_000})],
        ) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        # The primary is uncapped (default) and the fallback carries 8000: the
        # check prices the largest hop, and the served primary is capped at
        # the default, which the bound covers.
        assert _only_check_bound(plane) == 8_000
        assert _sent_kwargs(client.chat.completions.create)["max_completion_tokens"] == (
            DEFAULT_OUTPUT_BOUND
        )

    def test_media_check_omits_the_bound(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[], model="text-embedding-3-small", usage=SimpleNamespace(prompt_tokens=3)
        )
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.embeddings.create(model="text-embedding-3-small", input="hello")

        assert plane.checks[0].modality == "embedding"
        assert _only_check_bound(plane) is None
        assert "max_tokens" not in _sent_kwargs(client.embeddings.create)

    @pytest.mark.asyncio
    async def test_async_openai_chat_uncapped(self) -> None:
        plane = FakeControlPlane()
        client = _async_openai_client()
        async with plane.wrap_async(client, lease_enabled=False) as solwyn:
            await solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        create = client.chat.completions.create
        create.assert_awaited_once()
        assert create.call_args.kwargs["max_completion_tokens"] == DEFAULT_OUTPUT_BOUND


@pytest.mark.unit
class TestInjectionOnlyUnderHardDeny:
    """The cap is injected only for a call Solwyn admitted under hard_deny."""

    def test_alert_only_check_sends_the_request_as_built(self) -> None:
        plane = FakeControlPlane(mode="alert_only")
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        # The bound is still priced; the provider request is untouched.
        assert _only_check_bound(plane) == DEFAULT_OUTPUT_BOUND
        assert _sent_kwargs(client.chat.completions.create) == {
            "model": "gpt-5.5",
            "messages": MESSAGES,
        }

    @pytest.mark.parametrize("dialect", ["anthropic", "google", "bedrock", "responses"])
    def test_alert_only_leaves_every_dialect_uncapped(self, dialect: str) -> None:
        plane = FakeControlPlane(mode="alert_only")
        with plane.wrap(_DIALECT_CALLS[dialect][0](), lease_enabled=False) as solwyn:
            method = _DIALECT_CALLS[dialect][1](solwyn)
            _DIALECT_CALLS[dialect][2](solwyn)
        sent = _sent_kwargs(method)
        assert not {"max_tokens", "max_output_tokens", "inferenceConfig"} & set(sent)
        assert "max_output_tokens" not in (sent.get("config") or {})

    def test_fail_open_outage_sends_the_request_as_built(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with (
            plane.wrap(client, lease_enabled=False, budget_mode="hard_deny") as solwyn,
            plane.outage(path="/api/v1/budgets/check"),
        ):
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        # Even with a hard_deny constructor mode, an outage admission is not
        # Solwyn authority, so nothing is injected.
        assert plane.checks == []
        assert _sent_kwargs(client.chat.completions.create) == {
            "model": "gpt-5.5",
            "messages": MESSAGES,
        }

    def test_cached_hard_deny_allow_still_injects(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False, budget_check_cache_ttl=60) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert len(plane.checks) == 1
        calls = client.chat.completions.create.call_args_list
        assert [call.kwargs["max_completion_tokens"] for call in calls] == [
            DEFAULT_OUTPUT_BOUND,
            DEFAULT_OUTPUT_BOUND,
        ]

    def test_lease_hard_deny_posture_injects(self) -> None:
        import solwyn

        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client) as wrapped, solwyn.run("bounded"):
            wrapped.chat.completions.create(model="gpt-5.5", messages=MESSAGES)
            wrapped.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert plane.checks == []
        assert len(plane.lease_grants) == 1
        calls = client.chat.completions.create.call_args_list
        assert [call.kwargs["max_completion_tokens"] for call in calls] == [
            DEFAULT_OUTPUT_BOUND,
            DEFAULT_OUTPUT_BOUND,
        ]

    def test_lease_alert_only_posture_does_not_inject(self) -> None:
        import solwyn

        plane = FakeControlPlane(mode="alert_only")
        client = _openai_client()
        with plane.wrap(client) as wrapped, solwyn.run("unbounded"):
            wrapped.chat.completions.create(model="gpt-5.5", messages=MESSAGES)

        assert plane.checks == []
        assert len(plane.lease_grants) == 1
        assert "max_completion_tokens" not in _sent_kwargs(client.chat.completions.create)

    def test_enforcer_flags_only_hard_deny_allows(self) -> None:
        from conftest import ALLOW_BUDGET_RESPONSE

        from solwyn._types import BudgetCheckResponse

        base = _BudgetEnforcerBase(api_url="https://api.test.solwyn.ai", api_key="k" * 8)
        hard = BudgetCheckResponse(**{**ALLOW_BUDGET_RESPONSE, "mode": "hard_deny"})
        alert = BudgetCheckResponse(**ALLOW_BUDGET_RESPONSE)
        assert base._build_result_from_response(hard).hard_deny_admitted is True
        assert base._build_result_from_response(alert).hard_deny_admitted is False
        assert base._build_fail_open_result().hard_deny_admitted is False


@pytest.mark.unit
class TestEnforcerResolution:
    def _base(self) -> _BudgetEnforcerBase:
        return _BudgetEnforcerBase(api_url="https://api.test.solwyn.ai", api_key="k" * 8)

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            (None, DEFAULT_OUTPUT_BOUND),
            (0, DEFAULT_OUTPUT_BOUND),
            (-3, DEFAULT_OUTPUT_BOUND),
            (300, 300),
            (ORDINARY_TOKEN_COUNT_MAX + 5, ORDINARY_TOKEN_COUNT_MAX),
        ],
    )
    def test_text_bound_resolves_like_the_lease(self, given: int | None, expected: int) -> None:
        bound = self._base()._check_output_bound(given, modality="text", estimated_media=None)
        assert bound == expected

    def test_media_sends_no_bound(self) -> None:
        base = self._base()
        assert base._check_output_bound(500, modality="image", estimated_media=None) is None
        assert (
            base._check_output_bound(
                500, modality="text", estimated_media=MediaUsage(image_count=1)
            )
            is None
        )


@pytest.mark.unit
class TestAllowCacheCohort:
    def test_allow_for_a_small_bound_does_not_admit_a_larger_one(self) -> None:
        plane = FakeControlPlane()
        client = _openai_client()
        with plane.wrap(client, lease_enabled=False, budget_check_cache_ttl=60) as solwyn:
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES, max_tokens=100)
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES, max_tokens=100)
            solwyn.chat.completions.create(model="gpt-5.5", messages=MESSAGES, max_tokens=10_000)

        # The repeat rides the cache; the larger bound pays its own check.
        assert [check.estimated_output_bound for check in plane.checks] == [100, 10_000]


# ---------------------------------------------------------------------------
# End to end against a priced stub control plane (hard_deny)
# ---------------------------------------------------------------------------

# Synthetic o1-pro-like card (USD per token). Not a live rate: the stub only
# needs output to be far pricier than input, as in the WS-9a repro.
_INPUT_RATE = 150 / 1_000_000
_OUTPUT_RATE = 600 / 1_000_000


class _PricedHardDenyPlane(httpx.BaseTransport):
    """Prices each check as input + bound x output rate against a hard cap.

    Mirrors the server contract: with a bound the reservation is input cost
    plus ``estimated_output_bound`` at the output rate; a reservation over the
    remainder is refused in hard_deny. Everything else (receipts, confirms)
    goes to the zero-network double.
    """

    def __init__(self, *, budget_limit: float) -> None:
        self.budget_limit = budget_limit
        self.reservations: list[float] = []
        self.plane = FakeControlPlane()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith("/budgets/check"):
            return self.plane.transport.handle_request(request)
        body = json.loads(request.content)
        bound = body.get("estimated_output_bound")
        if bound is None:
            # The legacy estimate the bound replaces: output assumed == input.
            bound = body["estimated_input_tokens"]
        reservation = body["estimated_input_tokens"] * _INPUT_RATE + bound * _OUTPUT_RATE
        self.reservations.append(reservation)
        remaining = self.budget_limit - sum(self.reservations[:-1])
        allowed = reservation <= remaining
        if not allowed:
            self.reservations[-1] = 0.0
        return httpx.Response(
            200,
            json={
                "allowed": allowed,
                "remaining_budget": remaining,
                "reservation_id": "res_stub" if allowed else None,
                "mode": "hard_deny",
                "budget_limit": self.budget_limit,
                "current_usage": self.budget_limit - remaining,
                "project_id": VALID_PROJECT_ID,
                **({} if allowed else {"denied_by_period": "monthly"}),
            },
        )


def _wrap_priced(stub: _PricedHardDenyPlane, client: object, **kwargs: Any) -> Any:
    from solwyn.client import Solwyn

    return Solwyn(
        client,
        api_key=stub.plane.api_key,
        api_url=stub.plane.api_url,
        control_plane_transport=stub,
        budget_check_cache_ttl=0,
        lease_enabled=False,
        **kwargs,
    )


@pytest.mark.unit
class TestHardDenyRefusesBeforeTheProvider:
    def test_capped_call_over_the_remainder_is_refused(self) -> None:
        stub = _PricedHardDenyPlane(budget_limit=1.00)
        client = _openai_client()
        with _wrap_priced(stub, client) as solwyn, pytest.raises(BudgetExceededError):
            solwyn.chat.completions.create(model="o1-pro", messages=MESSAGES, max_tokens=10_000)

        client.chat.completions.create.assert_not_called()
        assert stub.reservations == [0.0]

    def test_uncapped_call_is_priced_at_the_default_bound_and_refused(self) -> None:
        stub = _PricedHardDenyPlane(budget_limit=1.00)
        client = _openai_client()
        with _wrap_priced(stub, client) as solwyn, pytest.raises(BudgetExceededError):
            solwyn.chat.completions.create(model="o1-pro", messages=MESSAGES)

        client.chat.completions.create.assert_not_called()

    def test_call_within_the_remainder_is_sent_with_the_priced_cap(self) -> None:
        stub = _PricedHardDenyPlane(budget_limit=1.00)
        client = _openai_client()
        with _wrap_priced(stub, client, lease_output_bound_default=1_000) as solwyn:
            solwyn.chat.completions.create(model="o1-pro", messages=MESSAGES)

        # 1000 output tokens priced and allowed; the provider is held to them.
        assert 0 < stub.reservations[0] <= 1.00
        sent = _sent_kwargs(client.chat.completions.create)
        assert sent["max_completion_tokens"] == 1_000


class _LegacyRequest:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    @staticmethod
    def to_dict(request: _LegacyRequest, **_options: object) -> dict[str, Any]:
        return request.payload


class _LegacyGenerativeModel:
    """Duck-typed ``google.generativeai.GenerativeModel`` with constructor defaults."""

    def __init__(self, generation_config: dict[str, Any]) -> None:
        self._generation_config = generation_config

    def _prepare_request(self, **kwargs: Any) -> _LegacyRequest:
        merged = {**self._generation_config, **(kwargs.get("generation_config") or {})}
        return _LegacyRequest({"contents": kwargs["contents"], "generation_config": merged})


_LegacyGenerativeModel.__module__ = "google.generativeai.generative_models"
_LegacyGenerativeModel.__name__ = "GenerativeModel"


def _legacy_runtime(generation_config: dict[str, Any]) -> Any:
    return SimpleNamespace(
        adapter=SimpleNamespace(dialect="google", name="google"),
        sdk_client=_LegacyGenerativeModel(generation_config),
    )


@pytest.mark.unit
class TestLegacyGoogleInjection:
    def _cap(self, runtime: Any, kwargs: dict[str, object]) -> dict[str, object]:
        from solwyn._base import _with_default_output_cap

        return _with_default_output_cap(
            runtime=runtime,
            model="gemini-1.5-pro",
            kwargs=kwargs,
            surface="chat",
            default_bound=DEFAULT_OUTPUT_BOUND,
        )

    def test_constructor_cap_is_respected(self) -> None:
        kwargs: dict[str, object] = {"contents": "hello"}
        assert self._cap(_legacy_runtime({"max_output_tokens": 300}), kwargs) is kwargs

    def test_uncapped_model_gets_call_generation_config(self) -> None:
        capped = self._cap(_legacy_runtime({"temperature": 0.3}), {"contents": "hello"})
        assert capped["generation_config"] == {"max_output_tokens": DEFAULT_OUTPUT_BOUND}

    def test_call_generation_config_is_copied_not_mutated(self) -> None:
        generation = {"top_p": 0.9}
        capped = self._cap(
            _legacy_runtime({}), {"contents": "hello", "generation_config": generation}
        )
        assert capped["generation_config"] == {
            "top_p": 0.9,
            "max_output_tokens": DEFAULT_OUTPUT_BOUND,
        }
        assert generation == {"top_p": 0.9}
